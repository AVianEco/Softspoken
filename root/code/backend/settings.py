import os

# STFT settings
n_fft = 512
win_length = n_fft
hop_length = win_length // 2

# controlling the window step size 
step_size = 0.6

# batches sent for predictions
prediction_batch_size = 32
threshold = 0.1

# the application operates at 22050 internally
vad_resample = 22050

# model settings
model_dir = '.\\root\\models\\spec_unet_2d_pytorch'
model_name = 'model_checkpoint.pth'

# project settings and results
project_dir = '.\\projects'

# detection duration must be longer than this to be seen for review
minimum_detection_len = 0.1 

# url for the online help
user_guide_url = 'https://github.com/AVianEco/Softspoken'

# use half the CPU
cpu_threads = os.cpu_count() // 2

