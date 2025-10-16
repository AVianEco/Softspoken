![Soft_Spoken_Logo_With_Text](https://github.com/user-attachments/assets/4ef88fef-70ad-4b65-98a2-a74b8dc01b90)

# SOFTSPOKEN
Sound Optimization for Fading Talkative Signals and Preserving Other Key Environmental Noises

A ML tool for detecting human speech in ecological acoustic datasets.

## USAGE
We introduce SOFTSPOKEN, a tool designed to automate removing human voices from ecological acoustic datasets to protect privacy. SOFTSPOKEN uses a machine-learned model trained to detect human speech while ignoring environmental and animal sounds. 

The simple graphical interface allows users to review detections and selectively silence the time-intervals where voices occur before sharing datasets publicly. 

Additionally, the app provides a CSV file output with all the detections. These CSV outputs are easily imported to data analysis scripts in programs like R, aiding studies of anthropogenic effects on animal behaviour.

## DOWNLOAD SOFTSPOKEN
### Requirements
To download and use Softspoken, you will need to first download Python Version 3.12 or newer. 

#### Python 3.12+ Installer
https://www.python.org/downloads/

### Using the Launcher - Easy Install
- Once the above requirements are installed, in this repo click on the green "<> Code" button then select download zip. 
- Unzip the downloaded repo to a location on your computer where you'd like Softspoken to live.
- Open the unzipped Softspoken folder, click into "Softspoken-main" and find the "softspoken_launcher.bat".
- Double-clicking on this .bat file will open a command-line interface to walk you through installing the dependancies for softspoken, then it will launch the app.

The first-time through the .bat file will take some time to download all the relevant requirements and set up the environment. Once through the first time, the .bat file can be clicked at any time to launch the app relatively quickly.

### For the Developers and Code Savvy
- You can clone this repo like any other.
- Install requirements.txt
- Use .bat file or from project root `python launch.py`

### Utilizing your GPU
On initial download, Softspoken is designed to run on CPU to allow broad compatibility for many users. If you would like to speed up the processing by utilizing you computer's NVIDIA GPU you can follow the instructions below.

- Ensure you have a NVIDIA GPU
- Ensure you have the latest drivers installed.
- Navigate to the Softspoken folder on your computer, type in the file path CMD and press enter.
- Run the follwing in the command window:

```
nvidia-smi
```
- Take note of the CUDA version.
- Activate the virtual environment by running:

```
venv\Scripts\activate
```

- Then unistall the CPU-only Torch, Torchvision, and Torchaudio by entering the following
```
pip uninstall torch torchvision torchaudio
```
- Go to [PYTORCH.ORG](https://pytorch.org/get-started/locally/)
- Enter your OS, and the CUDA version you noted above.
- Copy and paste the recommended command into the command window from Pytorch Website to install the GPU compatible versions of Torch


## PLATFORM SUPPORT
The script and the launcher have only been tested on Windows.

Application has been tested with:

- Python Version 3.12.7 and 3.13.2
- Windows 10/Windows 11 Home version

This may work as a python script on Mac machines, but we have not tested it.

If you try this on another machine or set-up like mac, please let us know how it goes!

## CONTACT
If you have questions or issues you may wish to submit an issue or contact us directly at info@avianeco.com.

## LICENSE
This application is licensed under an MIT license.
