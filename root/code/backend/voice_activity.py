from root.code.backend import settings

import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torchaudio.transforms as T
import librosa
import librosa.display
import soundfile as sf
import scipy.signal
import sounddevice as sd
from typing import Tuple, Optional

from scipy.signal import stft, istft
import time
import sounddevice as sd
from scipy.fft import rfft, irfft



def get_audio_data(file):
    # Get duration without loading the entire file
    duration = librosa.get_duration(path=file)
    
    # Get the native sample rate without loading the entire file
    original_sr = librosa.get_samplerate(file)
    
    return (duration, original_sr)

def load_audio(directory, start = None):
    # Function to load audio with proper handling of start and stop
    def load_audio_file(path, start):
        if start == None:
            try:
                f1, sr = sf.read(path, dtype='float32')
                return f1.T, sr
            except Exception as e:
                print(f'EXCEPTION EXCEPTION EXCEPTION: \n\t{path}\n\t{str({e})}')
                return (None, None)
        else:
            with sf.SoundFile(path, 'r', samplerate=None, channels=None, subtype=None, endian=None, format=None, closefd=None) as f:
                sr = f.samplerate

                # translate to native sr 
                read_start = int(start * (sr / settings.vad_resample)) 
                read_stop  = read_start + int(sr * 3)

                frames = f._prepare_read(read_start, read_stop, -1)
                f1 = f.read(frames, dtype='float32')
                
            return f1.T, sr
    
    data, sr = load_audio_file(directory, start)
    
    if data is None: # there is a seemingly random error and all it says is: Unspecified internal error.
        return (None, None)

    # Convert to mono if necessary
    if data.ndim > 1:
        data = librosa.to_mono(data)
    
    # Resample it if necessary
    if sr != settings.vad_resample:
        data = librosa.resample(data, orig_sr=sr, target_sr=settings.vad_resample)
        sr = settings.vad_resample
    
    return (data, sr)


def load_audio_startstop(full_path: str, start_stop: Tuple[float, float]):
    """
    Load an audio file with specified start and stop times in seconds.

    Parameters:
        full_path (str): Path to the audio file.
        start_stop (Tuple[float, float]): A tuple containing start and stop times in seconds.
            - start (float): Start time in seconds.
            - stop (float): Stop time in seconds. Must be greater than start.

    Returns:
        Tuple containing:
            - data (numpy.ndarray or None): The loaded audio data.
            - sr (int or None): The sample rate of the audio data.
    """
    
    start, stop = start_stop

    # Validate start and stop
    if start < 0 or stop <= start:
        print(f"Invalid start ({start}) and stop ({stop}) times. Ensure that 0 <= start < stop.")
        return None, None

    def load_audio_file(path: str, start: float, stop: float):
        try:
            with sf.SoundFile(path, 'r') as f:
                sr = f.samplerate

                # Calculate frame indices
                read_start = int(start * sr)
                read_stop = int(stop * sr)
                num_frames = read_stop - read_start

                # Ensure stop does not exceed total frames
                total_frames = len(f)
                if read_stop > total_frames:
                    print(f"Requested stop time ({stop}s) exceeds file duration. Adjusting to file's end.")
                    read_stop = total_frames
                    num_frames = read_stop - read_start

                # Seek to the start frame
                f.seek(read_start)

                # Read the desired number of frames
                data = f.read(num_frames, dtype='float32')
                
                if data.size == 0:
                    print(f"No data read from {path} between {start}s and {stop}s.")
                    return None, None

                return data.T, sr

        except Exception as e:
            print(f'EXCEPTION EXCEPTION EXCEPTION:\n\t{path}\n\t{str(e)}')
            return None, None

    data, sr = load_audio_file(full_path, start, stop)

    # Convert to mono if necessary
    if data.ndim > 1:
        data = librosa.to_mono(data)

    # Resample it if necessary
    if sr != settings.vad_resample:
        try:
            data = librosa.resample(data, orig_sr=sr, target_sr=settings.vad_resample)
            sr = settings.vad_resample
        except Exception as e:
            print(f'Resampling failed: {str(e)}')
            return None, None

    return data, sr




def wav_to_spec(data, trim_edges=True):
    D = np.abs(librosa.stft(data, n_fft = settings.n_fft, win_length = settings.win_length, hop_length = settings.hop_length))
    
    if trim_edges:
        D = D[..., 0:256, 0:256]
    
    return D

# create a reusable spectrogram layer to create the training targets. this is the same layer type the network uses internally
mel_spectrogram = T.MelSpectrogram(
    sample_rate=settings.vad_resample,
    n_fft=settings.n_fft *4,
    hop_length=settings.hop_length,
    win_length=settings.win_length,
    n_mels=128,
    f_max=8000
).to('cpu')

def wav_to_mel_spec_torch(data):
    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Ensure data is a torch tensor and move it to the target device
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(device)

    # If data is 1D, add a batch dimension
    if data.dim() == 1:
        data = data.unsqueeze(0)  # Shape: (1, num_samples)
    
    # Compute the mel spectrogram
    mel_spec = mel_spectrogram(data)  # Shape: (1, n_mels, time_frames)
    
    # Apply sqrt_log10_nonzero scaling
    mel_spec_scaled = torch.sqrt(torch.log10(mel_spec + 1))
    
    # Trim to desired time frames (e.g., 256 frames) because there are actually 259 spectrogram bins
    mel_spec_scaled = mel_spec_scaled[:, :, :256]

    return mel_spec_scaled.to('cpu')

def wav_to_mel_spec_batched(data):
    """
    data: NumPy array of shape (batch_size * 2, 66150) 
          (or however many waveforms you have concatenated)
    Returns: a torch.Tensor of shape (batch_size * 2, n_mels, time_frames)
    """
    device = 'cpu'  # or 'cuda' if GPU is allowed
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)

    # Use the same MelSpectrogram you defined globally, or create one here:
    #   n_fft, hop_length, etc. from your settings.
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=settings.vad_resample,
        n_fft=settings.n_fft * 4,  # or reduce this!
        hop_length=settings.hop_length,
        win_length=settings.win_length,
        n_mels=128,
        f_max=8000
    ).to(device)

    # Run the spectrogram transform (batched):
    mel = mel_spectrogram(data)  # => shape [batch_size*2, 128, time_frames]

    # Apply your sqrt(log10(x+1)) scaling
    mel = torch.sqrt(torch.log10(mel + 1))

    # Usually you’d slice to 256 frames if you want exactly 256:
    mel = mel[:, :, :256]

    return mel.cpu().numpy()  # or stay in torch if you want


def wav_to_mel_spec(data):
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=data,
        sr = settings.vad_resample,
        n_fft = settings.n_fft *4,
        hop_length = settings.hop_length,
        win_length = settings.win_length,
        n_mels = 128,
        fmax = 8000
    )

    # Trim to 256 frames
    mel_spec = mel_spec[:, :, :256]

    # Apply sqrt(log10(x + 1)) scaling
    mel_spec = np.sqrt(np.log10(mel_spec + 1))

    return mel_spec

def plot_audio(S_db):
    # spectrogram
    fig, ax = plt.subplots(figsize=(10,10))
    img = librosa.display.specshow(S_db,
                                   x_axis='time',
                                   y_axis='linear',
                                   # cmap='gray',
                                   ax=ax)

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()


# ESC-50 should all be greater than 3 seconds - double check for other datasets!
def load_environmental_noise(filename, start_idx):
    clean_wav, sr = load_audio(filename, start = start_idx)
    
    # If the environmental noise wave is less than 3 seconds, repeat it to fill the 3-second duration (must be longer than 0.5 seconds)
    if len(clean_wav) < 66150:
        # how many times should the clip be repeated?
        repetitions_needed = math.ceil(66150 / len(clean_wav))
        
        # process the aug clip
        clean_wav = np.tile(clean_wav, repetitions_needed)
        clean_wav = clean_wav[:66150]  # Ensure exact length

    return clean_wav


rng = np.random.default_rng()  

def load_peoples_speech(filenames):
    # Create a 3 second audio clip
    clean_clip = np.zeros(66150, dtype=np.float32)
    
    for (file_name, start_idx) in filenames:
        # the file will be minimum 3 seconds in length - could be longer - but only load 3
        clean_wav, sr = load_audio(file_name, start = start_idx)
        clean_clip[ : clean_wav.shape[0] ] += clean_wav
    return clean_clip

# load one of the spoken word recordings, and the noise we'll add to the recording later
def load_spoken_word(filenames):
    # Load the WAV file
    loaded_wavs = [load_audio(filename[0]) for filename in filenames]
    wave_idxs = [f[1] for f in filenames]
    
    # Create a 3 second audio clip
    clean_clip = np.zeros(66150).astype(np.float32)
    
    for i, (clean_wav, sr) in enumerate(loaded_wavs):
        # Insert the WAV file into the clip
        start_idx = wave_idxs[i]
        clean_clip[start_idx : start_idx + clean_wav.shape[0]] = clean_wav
    return clean_clip

def get_parameters(augmentations=[]):
    """
    Return random parameters for the specified augmentation functions using the Generator's uniform method.
    
    Parameters:
    - augmentations (list): List of augmentation function names.

    Returns:
    - list: List of dictionaries, each containing random parameters for a specific augmentation function.
    
    """
    
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(None)))
    
    # Define random parameter generation for each augmentation
    param_generators = {
        "change_pitch": {
            "pitch_factor": lambda: rng.uniform(-3, 3)
        },
        "change_speed": {
            "speed_factor": lambda: rng.uniform(0.80, 1.2)
        },
        "add_reverb": {
            "reverb_level": lambda: rng.uniform(0.2, 0.8),
            "decay_time": lambda: rng.uniform(0.2, 1.0),
            "num_delays": lambda: rng.uniform(1, 5)
        },
        "reduce_volume": {
            "reduction_level": lambda: rng.uniform(0.1, 0.99) 
        },
        "add_echo": {
            "echo_delay": lambda: rng.uniform(0.1, 0.7),
            "decay_factor": lambda: rng.uniform(0.3, 0.7)
        },
        "bandpass_filter": {
            "low_freq": lambda: rng.uniform(100, 500),
            "high_freq": lambda: rng.uniform(600, 2500)
        },
        "add_harmonic_distortion": {
            "distortion_level": lambda: rng.uniform(10, 50)
        },
        "compress_dynamics": {
            "compression_ratio": lambda: rng.uniform(1.5, 6.0),
            "threshold_level": lambda: rng.uniform(-40, -10)
        }
    }
    
    # Generate random parameters for the specified augmentations
    return_me = [{key: generator() for key, generator in param_generators[aug].items()} 
                 for aug in augmentations if aug in param_generators]
    
    return return_me

def apply_augmentations(audio, sr, augmentations=[], augmentation_params=[]):
    """
    Apply the specified audio augmentations in sequence.
    
    Parameters:
    - audio (numpy array): Input audio signal.
    - sr (int): Sampling rate of the audio signal.
    - augmentations (list): List of augmentation function names.
    - augmentation_params (list): List of dictionaries, each containing parameters for a specific augmentation function.

    Returns:
    - numpy array: Augmented audio signal.
    """
    func_map = {
        "change_pitch" : change_pitch,
        "change_speed" : change_speed,
        "add_reverb": add_reverb,
        "reduce_volume" : reduce_volume,
        "add_echo" : add_echo,
        "bandpass_filter" : bandpass_filter,
        "add_harmonic_distortion" : add_harmonic_distortion,
        "compress_dynamics" : compress_dynamics
    }
    
    for i, a in enumerate(augmentations):
        if a in func_map:
            # Apply the augmentation function with the provided parameters
            print(f'applying... {a}')
            audio = func_map[a](audio, sr, **augmentation_params[i])
        else:
            print(f"No such function: {a}")
    
    return audio

def pitch_shift(audio, n_steps):
    """
    Shifts the pitch of a batch of audio signals by a specified number of semitones.

    Parameters:
    - batch (np.ndarray): Input audio batch of shape (batch_size, 66150).
    - semitones (float): Number of semitones to shift the pitch. Positive for up, negative for down.

    Returns:
    - np.ndarray: Pitch-shifted audio batch of shape (batch_size, 66150).
    """
    # Calculate the pitch factor
    pitch_factor = 2 ** (n_steps / 12)

    # Original and desired number of samples
    original_length = 66150
    desired_length = 66150  # Keeping the same length

    # Generate the indices of the input samples
    # These indices are where we'll sample from the input to create the output
    input_indices = np.linspace(0, original_length, num=desired_length, endpoint=False) / pitch_factor

    # Floor and ceil indices for interpolation
    index_floor = np.floor(input_indices).astype(int)
    index_ceil = index_floor + 1

    # Clip indices to be within the valid range [0, original_length - 1]
    index_floor = np.clip(index_floor, 0, original_length - 1)
    index_ceil = np.clip(index_ceil, 0, original_length - 1)

    # Compute the weights for interpolation
    weights = input_indices - index_floor  # Fractional part

    # Expand indices and weights for batch processing
    # Shape: (batch_size, desired_length)
    batch_size = audio.shape[0]
    index_floor = index_floor[np.newaxis, :]  # Shape: (1, desired_length)
    index_ceil = index_ceil[np.newaxis, :]    # Shape: (1, desired_length)
    weights = weights[np.newaxis, :]          # Shape: (1, desired_length)

    # Repeat indices and weights for each item in the batch
    index_floor = np.repeat(index_floor, batch_size, axis=0)
    index_ceil = np.repeat(index_ceil, batch_size, axis=0)
    weights = np.repeat(weights, batch_size, axis=0)

    # Generate batch indices for advanced indexing
    batch_indices = np.arange(batch_size)[:, np.newaxis]

    # Gather the floor and ceil samples
    samples_floor = audio[batch_indices, index_floor]
    samples_ceil = audio[batch_indices, index_ceil]

    # Perform linear interpolation
    output = (1 - weights) * samples_floor + weights * samples_ceil

    return output

def change_pitch(audio_signal, sr, pitch_factor=0.0):
    """
    Change the pitch of the audio signal.
    
    Parameters:
    - audio_signal (numpy array): The input audio signal.
    - sr (int): Sampling rate of the audio signal.
    - pitch_factor (float): Factor to change the pitch of the audio.
                            Default is 0.0 (no change).
                            Represents the number of half-steps (semitones).
                            Negative values will lower the pitch.
                            Positive values will raise the pitch.

    Returns:
    - numpy array: Audio signal with altered pitch.
    """
    # Change the pitch of the audio
    # librosa.effects.pitch_shift(audio_signal, sr=sr, n_steps=pitch_factor)
    return pitch_shift(audio_signal, pitch_factor).astype(np.float32)





def stft(
    x: np.ndarray, 
    n_fft: int = 1024, 
    hop_length: int = 256, 
    window_fn=np.hanning
):
    """
    Compute the Short-Time Fourier Transform of a 1D signal x.
    Returns a 2D complex array of shape (num_frames, n_fft//2 + 1).
    """
    # Window
    window = window_fn(n_fft)

    # Number of frames needed
    num_frames = 1 + (len(x) - n_fft) // hop_length
    if num_frames < 1:
        # If the audio is too short, pad the signal so we can do at least one frame
        pad_amount = (n_fft + (0 if len(x) >= n_fft else n_fft - len(x))) - len(x)
        x = np.pad(x, (0, pad_amount), mode='constant')
        num_frames = 1 + (len(x) - n_fft) // hop_length

    stft_matrix = []
    idx = 0
    for _ in range(num_frames):
        frame = x[idx: idx + n_fft]
        # Apply window
        frame = frame * window
        # Compute real FFT
        spec = rfft(frame)
        stft_matrix.append(spec)
        idx += hop_length

    return np.array(stft_matrix, dtype=np.complex64)

def istft(
    stft_matrix: np.ndarray, 
    n_fft: int = 1024, 
    hop_length: int = 256, 
    window_fn=np.hanning
):
    """
    Inverse STFT for a 2D complex array of shape (num_frames, n_fft//2 + 1).
    Returns a 1D float array containing the reconstructed time-domain signal.
    """
    num_frames = stft_matrix.shape[0]
    window = window_fn(n_fft)

    # Output array length: hop_length * (num_frames - 1) + n_fft
    out_length = hop_length * (num_frames - 1) + n_fft
    x = np.zeros(out_length, dtype=np.float32)

    idx = 0
    for i in range(num_frames):
        # Get time-domain frame from spectrum
        spec = stft_matrix[i]
        # irfft expects n_fft//2+1 freq bins for real input transform
        frame = irfft(spec, n=n_fft).astype(np.float32)
        # Overlap-add
        x[idx: idx + n_fft] += frame * window
        idx += hop_length

    return x

def phase_vocoder(
    stft_matrix: np.ndarray,
    speed_factor: float,
    hop_length: int
):
    """
    Core Phase Vocoder routine.
    stft_matrix: (frames, freq_bins) complex array (from rfft).
    speed_factor: >1.0 speeds up, <1.0 slows down, 1.0 no change.
    hop_length: hop between frames in samples (used to adjust phase).
    Returns a new STFT matrix with time-stretched frames.
    """
    # If no speed change, just return
    if speed_factor == 1.0:
        return stft_matrix

    n_frames, n_freq = stft_matrix.shape
    # Calculate the new number of frames after stretching
    # e.g. if speed_factor > 1, we have fewer frames in the output
    new_n_frames = int(np.ceil(n_frames / speed_factor))

    # Output STFT array
    time_stretched = np.zeros((new_n_frames, n_freq), dtype=np.complex64)

    # Phase accumulator
    phase_acc = np.angle(stft_matrix[0])
    time_stretched[0] = stft_matrix[0]

    # For every frame in the new time base
    for t in range(1, new_n_frames):
        # Corresponding frame in original STFT
        orig_t = t * speed_factor
        # Index of the integer frames below and above
        int_t = int(np.floor(orig_t))
        frac_t = orig_t - int_t

        if int_t + 1 >= n_frames:
            # We've exceeded original number of frames
            break

        # Magnitude interpolation (linear in time)
        mag1 = np.abs(stft_matrix[int_t])
        mag2 = np.abs(stft_matrix[int_t + 1])
        mag = (1.0 - frac_t) * mag1 + frac_t * mag2

        # Phase advance
        # Original angles
        phase1 = np.angle(stft_matrix[int_t])
        phase2 = np.angle(stft_matrix[int_t + 1])
        # Instantaneous frequency
        dphase = phase2 - phase1
        # Wrap to [-pi, pi] range to handle phase ambiguity
        dphase = np.mod(dphase + np.pi, 2.0 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase * speed_factor

        # Build the complex output for frame t
        time_stretched[t] = mag * np.exp(1j * phase_acc)

    return time_stretched

def time_stretch_from_scratch(
    audio: np.ndarray,
    sr: int,
    speed_factor: float = 1.0,
    n_fft: int = 1024,
    hop_length: int = 256
) -> np.ndarray:
    """
    Time-stretch the audio (i.e., change speed) without altering pitch,
    returning the same number of samples as the input.
    
    Supports:
      - 1D: shape (num_samples,)
      - 2D batch: shape (batch_size, num_samples)
    """
    # Handle trivial case
    if speed_factor == 1.0:
        return audio

    # Make sure we handle batch vs. single audio
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # add batch dimension

    batch_size, num_samples = audio.shape
    output = np.zeros_like(audio)

    for i in range(batch_size):
        # 1) STFT
        X = stft(audio[i], n_fft=n_fft, hop_length=hop_length)

        # 2) Phase Vocoder
        X_stretched = phase_vocoder(X, speed_factor, hop_length)

        # 3) Inverse STFT
        y_stretched = istft(X_stretched, n_fft=n_fft, hop_length=hop_length)

        # 4) Truncate or pad to match original length
        if len(y_stretched) < num_samples:
            # Pad
            pad_amount = num_samples - len(y_stretched)
            y_stretched = np.pad(y_stretched, (0, pad_amount), mode='constant')
        else:
            # Truncate
            y_stretched = y_stretched[:num_samples]

        output[i] = y_stretched

    # If the original audio was 1D, return 1D
    if output.shape[0] == 1:
        output = output[0]

    return output.astype(np.float32)

def change_speed(
    audio_signal: np.ndarray, 
    sr: int, 
    speed_factor: float = 1.0
):
    """
    Wrapper function that behaves similarly to your original `change_speed`,
    but uses the custom time-stretch routine above (no librosa).
    """
    # If no speed change, return original
    if speed_factor == 1.0:
        return audio_signal
    
    # Use the custom time stretch
    speed_changed_output = time_stretch_from_scratch(
        audio_signal,
        sr,
        speed_factor=speed_factor,
        n_fft=1024,     # You can adjust these as needed
        hop_length=256
    )

    # # Debug code to listen to 10 samples (original vs. speed_changed)
    # for i in range(10):
    #     print(f'{i+1}/{10}')
    #     sd.play(audio_signal[i], sr)
    #     sd.wait()  # Wait until playback of the original finishes
    #     sd.play(speed_changed_output[i], sr)
    #     sd.wait()  # Wait until playback of the changed version finishes
    #     time.sleep(1)  # Pause 1 second before playing the next pair

    # print(f"CHANGE SPEED SHAPE: {speed_changed_output.shape}")
    return speed_changed_output


def __change_speed(audio_signal, sr, speed_factor=1.0):
    """
    Change the speed of the audio signal without altering its pitch.
    
    Parameters:
    - audio_signal (numpy array): The input audio signal.
    - sr (int): Sampling rate of the audio signal.
    - speed_factor (float): Factor to change the speed of the audio. 
                            Default is 1.0 (no change).
                            Values <1.0 will slow down the audio.
                            Values >1.0 will speed up the audio.

    Returns:
    - numpy array: Audio signal with altered speed (same number of samples as input).
    """
    
    # If no speed change is requested, return the original signal.
    if speed_factor == 1:
        return audio_signal

    # Setup an output array
    speed_changed_output = np.zeros_like(audio_signal)

    # Time-stretch the audio using librosa
    speed_changed = librosa.effects.time_stretch(audio_signal, rate=speed_factor).astype(np.float32)
    
    # Get the original and new lengths
    orig_len = len(audio_signal)
    new_len = len(speed_changed)
    
    if len(speed_changed) < orig_len:
        # When speeding up, the output is shorter.
        # Pad the end with zeros to reach the original length.
        pad_width = orig_len - new_len
        speed_changed_output = np.pad(speed_changed, (0, pad_width), mode='constant')

    if len(speed_changed) > orig_len:
        # When slowing down, the output is longer.
        # Truncate the extra samples so that the output matches the original length.
        speed_changed_output = speed_changed[:orig_len]

    # Debug code to listen to 10 samples (original vs. speed_changed)
    for i in range(10):
        print(f'MEAN: {speed_changed.mean()}')
        sd.play(audio_signal[i], sr)
        sd.wait()  # Wait until playback of the original finishes
        sd.play(speed_changed_output[i], sr)
        sd.wait()  # Wait until playback of the changed version finishes
        time.sleep(1)  # Pause 1 second before playing the next pair

    print(f'CHANGE SPEED SHAPE: { speed_changed_output.shape }')
    return speed_changed_output


def add_reverb(audio_signal, sr, reverb_level=0.5, decay_time=0.5, num_delays=5):
    """
    Add reverb to the audio signal to simulate distance.
    
    Parameters:
    - audio_signal (numpy array): The input audio signal.
    - sr (int): Sampling rate of the audio signal.
    - reverb_level (float): Level of reverb to be added. Range: 0.0 to 1.0.
    - decay_length (float): Length of the decay in seconds. Default is 0.5s.

    Returns:
    - numpy array: Audio signal with added reverb.
    """
    # # Generate the decay exponential
    # decay = np.exp(-np.arange(0, sr * decay_length) / float(sr * decay_length))
    
    # # Perform FFT-based convolution
    # signal_fft = np.fft.fft(audio_signal, n=audio_signal.size + decay.size - 1)
    # decay_fft = np.fft.fft(decay, n=audio_signal.size + decay.size - 1)
    # reverb_signal = np.fft.ifft(signal_fft * decay_fft).real[:audio_signal.size]
    
    # # Mix the original with the reverberated signal
    # return ((1-reverb_level) * audio_signal + reverb_level * reverb_signal).astype(np.float32)

    batch_size, num_samples = audio_signal.shape
    reverb = np.zeros_like(audio_signal)
    
    # Calculate maximum delay in samples
    max_delay = int(sr * decay_time)
    
    # Define delays: logarithmically spaced to simulate natural reverb
    delays = np.linspace(0, max_delay, int(num_delays + 1), dtype=int)[1:]  # Exclude zero delay
    
    # Calculate decay factors for each delay
    decay_factors = np.exp(-np.linspace(0, 3, int(num_delays)))  # Adjust the range as needed
    
    # Normalize decay factors so that their sum is 1
    decay_factors /= decay_factors.sum()
    
    # Apply each delay and its corresponding decay factor
    for delay, decay in zip(delays, decay_factors):
        if delay < num_samples:
            reverb[:, delay:] += audio_signal[:, :-delay] * decay
    
    # Mix the original signal with the reverberated signal
    output = (1 - reverb_level) * audio_signal + reverb_level * reverb
    
    # for i in range(100):
    #     sd.play(audio_signal[i], 22050)
    #     sd.wait()

    #     sd.play(output[i], 22050)
    #     sd.wait()

    return output.astype(np.float32)


def reduce_volume(audio_signal, sr, reduction_level=0.5):
    """
    Reduce the volume of the audio signal.
    
    Parameters:
    - audio_signal (numpy array): The input audio signal.
    - reduction_level (float): Level to reduce the volume. Range: 0.0 to 1.0.

    Returns:
    - numpy array: Audio signal with reduced volume.
    """
    return (audio_signal * reduction_level).astype(np.float32)


def add_echo(audio_signal, sr, echo_delay=0.5, decay_factor=0.5):
    """
    Add echo to the audio signal.
    
    Parameters:
    - audio_signal (numpy array): The input audio signal.
    - sr (int): Sampling rate of the audio signal.
    - echo_delay (float): Delay for the echo in seconds. 
    - decay_factor (float): Decay factor for the echo. Range: 0.0 to 1.0.

    Returns:
    - numpy array: Audio signal with added echo.
    """
    delay_samples = int(echo_delay * sr)
    echo_signal = np.zeros_like(audio_signal, dtype=np.float32)
    echo_signal[delay_samples:] = audio_signal[:-delay_samples] * decay_factor
    
    # Mix the original with the echo
    return (audio_signal + echo_signal).astype(np.float32)


def bandpass_filter(audio_signal, sr, low_freq, high_freq):
    """
    Apply a bandpass filter to the audio signal.
    
    Parameters:
    - audio_signal (numpy array): The input audio signal.
    - sr (int): Sampling rate of the audio signal.
    - low_freq (float): Low frequency cutoff for the bandpass filter.
    - high_freq (float): High frequency cutoff for the bandpass filter.

    Returns:
    - numpy array: Bandpass-filtered audio signal.
    """
    # Design the bandpass filter
    nyquist = 0.5 * sr
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = scipy.signal.butter(1, [low, high], btype='band')
    
    # Apply the filter
    return scipy.signal.lfilter(b, a, audio_signal).astype(np.float32)


def add_harmonic_distortion(audio_signal, sr, distortion_level=0.5):
    """
    Add harmonic distortion to the audio signal.
    
    Parameters:
    - audio_signal (numpy array): The input audio signal.
    - distortion_level (float): Level of harmonic distortion to be added. Range: 0.0 to 1.0.

    Returns:
    - numpy array: Audio signal with harmonic distortion.
    """
    # Applying a nonlinear function to create distortion
    return (np.sign(audio_signal) * (1.0 - np.exp(-distortion_level * np.abs(audio_signal)))).astype(np.float32)


def compress_dynamics(audio_signal, sr, compression_ratio=2.0, threshold_level=-20):
    """
    Apply dynamic range compression to the audio signal.
    
    Parameters:
    - audio_signal (numpy array): The input audio signal.
    - compression_ratio (float): Ratio of compression to be applied.
    - threshold_level (float): Threshold level in dB.

    Returns:
    - numpy array: Audio signal with compressed dynamics.
    """
    # Convert threshold level from dB to linear scale
    threshold_linear = 10**(threshold_level / 20.0)
    
    # Apply compression
    compressed_signal = np.where(
        np.abs(audio_signal) > threshold_linear,
        threshold_linear + (np.abs(audio_signal) - threshold_linear) / compression_ratio,
        audio_signal
    )
    
    return (np.sign(audio_signal) * compressed_signal).astype(np.float32)


def plot_speech_over_spec(spec, mask, snr=0):
    vad_starts, vad_stops = [], []
    in_region = False
    
    for i, v in enumerate(mask):
        if v == 1 and not in_region:
            vad_starts.append(i)
            in_region = True
        elif v == 0 and in_region:
            vad_stops.append(i)
            in_region = False
    
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    
    ax1.imshow(spec, origin='lower', aspect='auto', cmap='gray', interpolation='nearest')
    ax1.set_title('Spectrogram')
    ax2.imshow(spec, origin='lower', aspect='auto', cmap='gray', interpolation='nearest')
    ax2.set_title('Labeled Spectrogram')
    
    # plot the speech regions
    for start, stop in zip(vad_starts, vad_stops):
        # foreach region, plot with start and end idx
        ax2.axvspan(start, stop, alpha=0.3, color='green')
    
    plt.suptitle(f'SNR: {snr}', fontsize=14, fontweight='bold')
    plt.show()


def process_spec(spec):
    # create the spectrogram and scale between 0-1
    spec = spec - (-80)  # -80 is a special value predetermined
    spec = spec / 80  # use this to ensure same scaling over all windows
    
    # snip the edges to make it 256x256
    spec = spec[0:256, 2: -1]
    spec = np.expand_dims(spec, axis=-1)
    
    if spec.sum() == 256*256:
        spec = np.zeros(spec.shape)
    
    return spec 


def generate_1D_mask_pytorch(spec):
    # Check if there is any non-zero element in each column across the 128 rows for each batch.
    # This will produce a boolean tensor of shape (BATCH_SIZE, 256)
    mask = torch.any(spec != 0, dim=1).float()
    return mask


def generate_1D_mask(spec):
    # Check if there is any non-zero element in each column across the 128 rows for each batch
    mask = np.any(spec != 0, axis=1).astype(float)
    return mask


def test_workflow():
    file = 'I:\\FreesoundAudio\\PeoplesSpeech\\single words\\wav\\avenue_common_voice_en_18746016.wav'
    
    snr = 20
    
    clip, noisy_clip, sr = load_spoken_word(file, snr=snr)
    
    clean_spec = wav_to_spec(clip, 22050)
    clean_spec = process_spec(clean_spec)
    
    mask = generate_1D_mask(clean_spec)
    
    noisy_spec = wav_to_spec(noisy_clip, 22050)
    noisy_spec = process_spec(noisy_spec)
    plot_speech_over_spec(noisy_spec, mask, snr)
    

def experiment():
    # Load the wav file
    data, rate = librosa.load('I:\\FreesoundAudio\\ESC-50-master\\audio\\2-77346-A-46.wav')
    
    sd.play(data, rate)
    sd.wait()
    
    # Compute the STFT
    D = librosa.stft(data, n_fft=settings.n_fft, win_length=settings.win_length, hop_length=settings.hop_length)
    
    # Compute the amplitude in dB
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.imshow(S_db, origin='lower', cmap='gray')
    
    # Convert S_db back to amplitude
    S = librosa.db_to_amplitude(S_db)
    
    # Invert the spectrogram
    data_reconstructed = librosa.istft(S, win_length=settings.win_length, hop_length=settings.hop_length)
    
    # Play the inverted audio
    sd.play(data_reconstructed, rate)
    sd.wait()


def wav_to_spec_with_comments(data, trim_edges=True):
    D = librosa.stft(data, n_fft = settings.n_fft, win_length = settings.win_length, hop_length = settings.hop_length)
    # S_db = librosa.amplitude_to_db(np.abs(D ** 2), ref = np.max)
    magnitude = np.abs(D)
    
    if trim_edges:
        magnitude = magnitude[0:-1, 0: -3]
        # phase = np.angle(D)[0:-1, 0: -3]   # discard the phase for now! 
    
    return magnitude
