from root.code.backend.pytorch_neural_nets import SpecUNet_2D
from root.code.backend.voice_activity import get_audio_data
from root.code.backend import settings

import os
import math
import numpy as np
import torch
import logging

class NNDetector():
    """
    The detector should be started by giving it a list of audio files.
    The detector will analyze each file and keep track of the results.

    The project_manager knows where to load the detection file (.csv).
    Once all detections are done for a file, you can add them to the
    project's DataFrame and save.
    """

    def __init__(self, project_manager):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {self.device}")

        torch.set_num_threads(settings.cpu_threads)
        torch.set_grad_enabled(False)

        # settings coming from the UI
        self.project_manager = project_manager
        
        # the trained detection model
        self.model = SpecUNet_2D().to(self.device)
        self.load_checkpoint(self.model, os.path.join(settings.model_dir, settings.model_name))
        self.model.eval()
        
        # paths to all the audio files we want to process
        self.files_to_process = self.project_manager.get_unprocessed_list()
        
        # this dictionary will hold indexes or placeholders of planned work
        self.detections_project = {f: [] for f in self.files_to_process}

    def load_checkpoint(self, model, file_path='checkpoint.pth'):
        """
        Load model weights from disk if the checkpoint file exists.
        """
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            return start_epoch
        else:
            print("No checkpoint found. Starting training from scratch.")
            return -1

    def plan_detection_job(self):
        """
        Analyzes each file to figure out how many sliding windows we need,
        returning a dict {file_path: np.array of start_indexes} that can
        be used for batch inference.
        """
        p = self.detections_project
        
        for file in p.keys():
            logging.info(f"Analyzing file: {file}")
            (audio_len_seconds, _) = get_audio_data(file)
            
            sample_rate = settings.vad_resample  # Hz
            window_size = 3  # seconds
            step_size = settings.step_size  # e.g. 0.6
            
            # length with 3s of silence on either side
            audio_data_length = round(audio_len_seconds * sample_rate) + (window_size * 2 * sample_rate)
            
            samples_per_window = sample_rate * window_size
            samples_per_step = math.floor(sample_rate * step_size)
            
            num_windows = int(np.ceil((audio_data_length - samples_per_window) / samples_per_step))
            start_indexes = np.arange(num_windows) * samples_per_step
            
            p[file] = start_indexes
        
        return p

    def process_batch(self, audio_data, batch_indexes):
        """
        Given a padded audio array and a list of start indexes,
        run each 3s slice through the model.
        Returns (speech_pred, mask_pred) as numpy arrays.
        """
        audio_data_tensor = torch.tensor(audio_data, dtype=torch.float32).to(self.device)

        # Stack slices
        audio_slices = torch.stack([
            audio_data_tensor[idx : idx + settings.vad_resample * 3]
            for idx in batch_indexes
        ])

        with torch.no_grad():
            speech_pred, mask_pred = self.model(audio_slices)

        return speech_pred.cpu().numpy(), mask_pred.cpu().numpy()

    def find_speech_regions(self, averaged_detections, break_duration=0.5):
        """
        Finds continuous speech regions using the threshold in settings.threshold.
        averaged_detections: { file: [ (detection_value, time_str), ... ] }
        Returns { file: [ (start_time, end_time), ... ] }
        """
        threshold = settings.threshold
        speech_regions = {}

        for file, file_detections in averaged_detections.items():
            regions = []
            start_time = None
            end_time = None

            for detection, time in file_detections[file]:
                if detection > threshold:
                    if start_time is None:
                        start_time = time
                    end_time = time
                elif start_time is not None:
                    regions.append((start_time, end_time))
                    start_time = None

            if start_time is not None:
                regions.append((start_time, end_time))

            if len(regions) > 0:
                merged_regions = []
                current_region = regions[0]
                for next_region in regions[1:]:
                    if float(next_region[0]) - float(current_region[1]) <= break_duration:
                        current_region = (current_region[0], next_region[1])
                    else:
                        merged_regions.append(current_region)
                        current_region = next_region
                merged_regions.append(current_region)
                speech_regions[file] = merged_regions
            else:
                speech_regions[file] = []

        return speech_regions

    def extract_filename(self, file_path):
        """
        Returns the filename (without extension) from a file path.
        """
        full_filename = os.path.basename(file_path)
        filename_without_extension = full_filename.rsplit('.', 1)[0]
        return filename_without_extension

    def average_overlapping_detections(self, detections, audio_length_seconds, padding=0, min_count=1):
        """
        Averages overlapping window detections and returns a dictionary:
          { file: [ (average_detection_value, time_string), ...] }
        Also saves each file's raw detection array as a .pkl for debugging.
        """
        averaged_detections = {}

        for file, file_detections in detections.items():
            # detections_file_name = self.extract_filename(file)

            # # Save raw detection data for debugging
            # with open(os.path.join("./", f"detections_{detections_file_name}.pkl"), 'wb') as f:
            #     pickle.dump(file_detections, f)

            output_length = int(round(audio_length_seconds * 256 / 3))
            sum_detections = np.zeros(output_length + 2 * padding)
            count_detections = np.zeros(output_length + 2 * padding)

            time_resolution = 3 / 256  # 3 seconds / 256 time bins

            for i, window_detections in enumerate(file_detections):
                start_position = padding + int(round(i * settings.step_size / time_resolution))
                sum_detections[start_position : start_position + 256] += window_detections.reshape(-1)
                count_detections[start_position : start_position + 256] += 1

            # Build final list of (avg_value, time_str)
            # Only keep positions where count >= min_count
            results_for_file = []
            for idx, (s, c) in enumerate(zip(sum_detections, count_detections)):
                if c >= min_count:
                    avg_val = s / c
                    time_str = f"{idx / (256 / 3):.4f}"
                    results_for_file.append((avg_val, time_str))

            averaged_detections[file] = results_for_file

        return averaged_detections
