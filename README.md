![Soft_Spoken_Logo_With_Text](https://github.com/user-attachments/assets/4ef88fef-70ad-4b65-98a2-a74b8dc01b90)

# SOFTSPOKEN
Sound Optimization for Fading Talkative Signals and Preserving Other Key Environmental Noises

A ML tool for detecting human speech in ecological acoustic datasets.

## USAGE
We introduce SOFTSPOKEN, a tool designed to automate removing human voices from ecological acoustic datasets to protect privacy. SOFTSPOKEN uses a machine-learned model trained to detect human speech while ignoring environmental and animal sounds. 

The simple graphical interface allows users to review detections and selectively silence the time-intervals where voices occur before sharing datasets publicly. 

Additionally, the app provides a CSV file output with all the detections. These CSV outputs are easily imported to data analysis scripts in programs like R, aiding studies of anthropogenic effects on animal behaviour.

## GETTING STARTED
### Table of Contents
- [Download Softspoken](#download-softspoken)
- [Quickstart](#quickstart)
- [Interface and Workflow Overview](#interface-and-workflow-overview)
- [Data Outputs](#data-outputs)
- [Platform Support](#platform-support)
- [Contact](#contact)
- [License](#license)

### DOWNLOAD SOFTSPOKEN
#### Requirements
To download and use Softspoken, first download Python Version 3.12 or newer. 

[Download Python 3.12+](https://www.python.org/downloads/)

#### Using the Launcher - Easy Install
- Once the above requirements are installed, in this repo click on the green `<> Code` button then select download zip. 
- Unzip the downloaded repo folder to a location on your computer where you'd like Softspoken to live.
- Open the unzipped Softspoken folder, click into `Softspoken-main` and find the `softspoken_launcher`.bat``.
- Double-clicking on this `.bat` file will open a command-line interface to walk you through installing the dependencies for softspoken, then it will launch the app.

The first-time through the `.bat` file will take some time to download all the relevant requirements and set up the environment. Once through the first time, the `.bat` file can be clicked at any time to launch the app relatively quickly.

#### For the Developers and Code Savvy
- You can clone this repo like any other.
- Install `requirements.txt`.
- Use `.bat` file or from project root `python launch.py`.

#### Utilizing your GPU
On initial download, Softspoken is designed to run on CPU to allow broad compatibility for many users. If you have an Nvidia GPU and would like to speed up the processing by utilizing this GPU, you can follow the instructions below.

- Ensure you have opened and subsequently closed Softspoken at least once using the `.bat` file, as per above instructions.
- Ensure you have a Nvidia GPU.
- Ensure your GPU drivers are up-to-date.
- Navigate to the Softspoken folder on your computer, type `CMD` into the address bar and press enter.
- Run the following in the command window:
```
nvidia-smi
```
- Take note of the CUDA version.
- Activate the virtual environment by running:
```
venv\Scripts\activate
```

- Then uninstall the CPU-only Torch, Torchvision, and Torchaudio by entering the following:
```
pip uninstall torch torchvision torchaudio
```

- Go to [PYTORCH.ORG](https://pytorch.org/get-started/locally/)
- Enter your OS, and the CUDA version you noted in the steps above.
- Copy and paste the recommended command from pytorch website into the command window to install the GPU compatible versions of Torch.
- Close command and open the Softspoken app using the `.bat` file as described in above sections.

### QUICKSTART

Below is a quick-start guide to get started. For more detail see the relevant sections below.

**1. Open user interface**
- Navigate to the `Softspoken-main` folder you unzipped during install.
- Double-click the `softspoken_launcher`.bat`` file.
- A command window will appear, followed by the Softspoken graphical interface.
		
**2.  Choose `Start New Project` and name it.**

**3. Click `add files` to select `.wav` files for the project.**

**4. Follow through the button prompts to `Run Voice Detector`, `Review Detections`, and `Silence Voices` in order.**
- A new window will appear for each.
- Follow through with the actions offered in the windows (e.g. `begin processing` in voice detector, and designating whether to keep or erase detections in `review detections`, etc.) before closing them to save.
- Detections need to be created in `Run Voice Detector` before review.
- Detections need to be flagged `erase` in the review screen prior to running silencing batch process.
		
**5. Find your data files**
		- Your silenced `.wav` will be in the location specified in the `silence detections` window.
		- Your data outputs (`.csv`, `.txt`) will be found in the softspoken folder under `projects` and the name of the project assigned in step 2.

### INTERFACE AND WORKFLOW OVERVIEW

#### Start-up Screen
When first opened the user is greeted with the Softspoken project launch screen.

![launch-screen-screenshot](https://github.com/AVianEco/Softspoken/blob/main/assets/launch-screen-screenshot.png)

In this screen the user can:

**Create a New Project**
- Press the green `Start New Project` button, or select `File > New Project`, or press `CTRL+N` on keyboard. 
- All the listed actions will bring up a dialogue that allows the user to enter the name of the new project, and press ok to create it.
- The new created project folder and any other projects can be found within the `Softspoken/projects`.

**Open an Existing Project**
- Press the green `Open Existing Project` button, or select `File > Open Project`, or press `CTRL+O` on the keyboard.
- All above actions will bring up a project picker dialog that lists all the projects under the `Softspoken/projects` folder.
- Alternatively, the user can choose to work on the project they were last reviewing in Softspoken by pressing `Open Last Project`.

**Get Help**
- `Help > User Guide` or CTRL+U brings up a web browser to this readme file.

#### Project Workspace Screen
Once a project is launched the user is taken to the Project Workspace Screen.

![project-workspace-screenshot](https://github.com/AVianEco/Softspoken/blob/main/assets/project-workspace-screenshot.png) 

In this screen the user can:

**Add or Delete .wav files to the Project**
- Press the green `Add Files` to bring up a file picker that will allow the user to navigate to the location where their files for review are saved.
- Select one or multiple `.wav` files to add them to the file list.
- Once files are added to the file list one or multiple can be selected from the list, then the `delete` button can be used to remove them.

**Activate Softspoken Voice Detector**
- Once files have been added to the project, the ML model can be activated to find voices using the `Run Voice Detector` green button.
- See the `Voice Detector Screen` section below for more details on this screen.

**Launch the Review Screen**
- Once the `Run Voice Detection` process is complete, `Review Detections` can be used to examine and edit the model's detections.
- See the `Review Detections Screen` section below for more details on this screen.

**Activate the Batch Silencing Screen**
- Once the detections are reviewed and flagged for removal in the `Review Detections` screen, `Silence Voices` can be used to create new `.wav` files with silenced voices.
- See the `Silence Voices Screen` section below for more details on this screen.

#### Voice Detector Screen
This screen is activated using the `Run Voice Detector` button in the Project Workspace.

![voice-detector-screenshot](https://github.com/AVianEco/Softspoken/blob/main/assets/voice-detector-screenshot.png)

In this screen the user can:

- Press `Begin Processing` to run the softspoken detection model over the audio files selected.
- `Stop processing` can pause the process at any time, select `Begin Processing` again to resume.
- Monitor the progress with the following provided information:
						*Total Files*: total number of files added to the project for review.
						*Processed*: number of files where detections are completed.
						*Currently Processing*: name of file that's under review by the model.
						*Total Audio Processed*: amount of recorded time in seconds processed by the model.
						*Processing Speed*: the ratio of recorded time in seconds processed to real time elapsed.
						*Processing bars*: File progress - for how much of the currently processing file is complete, and overall progress - for how much of the file list is complete.
						
- Once the detector has completed the review the user will receive a pop-up notification.
- The user can exit this screen once complete to move back to the project workspace.

#### Review Detections Screen
This screen is activated using the `Review Detections` button in the Project Workspace.

![review-detections-screenshot](https://github.com/AVianEco/Softspoken/blob/main/assets/review-detections-screenshot.png)

In this screen the user can:
**View Detections**
- A spectrogram image will appear, with a data table underneath. The spectrogram will have a red highlight over the detection selected in the table. 
- There will also be blue highlights over other detections visible in the timeline represented.
- The user can double click on any part of the table to edit it including start_time, end_time, erase, and user_comment.
- Adjust the scale of the spectrogram's x-axis by using `Time Axis` - plus to zoom in and minus to zoom out.
- There is a check box `Show Vertical Bars` toggling this will add or remove the pale blue dotted vertical lines at 0.5s increments.

**Playback**
- Play back the sound using the `Play All` - to play all visible sound, or `Play`- to play the highlighted detection or start to end time.
- `Stop` button stops the active playback.
- `Shift+Space` also play or stops the audio for the highlighted detection.
- The user can edit the `Start Time` and `End Time` values at the bottom left of the spectrogram to choose a section of time for playback without editing the detection start and end time in the table.

**Navigation**
- The table can be navigated with arrow keys as expected. Beginning to type will add values to the columns and rows you've navigated to.
- `Previous` navigates up to the previous detection in the table and shows it on the spectrogram.
- `Next` button goes down to the next detection in the table and shows it in the spectrogram.
- `Previous File` jumps up in the table to the last detection in the previous file name.
- `Next File` jumps down in the table to the first detection in the next file name.

**Grading Detections**
Before going to the `Silence Voices` screen, you will need to grade the detections to flag the ones desired for silencing.
- `Keep` button adds a `yes` to the erase column and updates the review_datetime column.
- `Erase` button leaves the erase column blank and updates the review_datetime column.
- `Yes` in the erase column signals the silencing tool to replace this section of the timeline with silence. Blanks in this column will be ignored and left in place.
- The review_datetime column signals to the user what has been reviewed, when, and what still needs review.
- Closing the review window updates the `project_review.csv` and the output files for other analysis programs. `project` in this case refers to the name given in the Start-up Screen.
See `Data Outputs` section for more information on the data outputs available and how they are created.

**Silence Voices Screen**
This screen is activated using the `Silence Voices` button in the Project Workspace.

![silence-voices-screenshot](https://github.com/AVianEco/Softspoken/blob/main/assets/silence-voices-screenshot.png)	

In this screen the user can:

- Select `Browse...` to open a file picker dialogue where the user can select the location where the altered `.wav` files will be saved.
- Once the location is selected, press `Start Silencing`. This will begin the process of creating new `.wav` files with silenced sections and a progress bar will appear.
- The process can be paused using `Stop` at any time.
		
### DATA OUTPUTS

Softspoken creates the following outputs:

#### project_detections.csv
After the voice detector is run, in `Softspoken/projects/Softspoken Outputs/[project-name]` this file will be created. This CSV contains all the detections as created by the voice detector
The file contains the following headers:
- *file_path*: the location where the original wav file is saved. 
- *file_name*: the name of the wav file where the detection occurs. 
- *start_time*: time since the beginning of the cited wav file where the detection starts.
- *end_time*: time since the beginning of the cited wav file where the detection ends.
- *erase*, *user_comment*, *review_datetime* - these are in the file to mirror the `project_review.csv` file, but the values will be null or zero.

#### project_review.csv
After opening and closing the review screen for the first time after the voice detector is run, in `Softspoken/projects/Softspoken Outputs/[project-name]` this file will be created. Every time the review screen is open and closed this file is updated.
This CSV contains all the detections either original or edited by the reviewer. It also includes all the reviewers designations on whether to keep or silence (erase) detections.
The file contains the following headers:
- *file_path*, *file_name*, *start_time*, *end_time*- these are the same as defined for the `project_detections.csv`.
- *erase*: a binary 1 (yes) or 0 (no) to indicate whether the user wants to keep (0) or erase (1) the detection at the silence voices step. By default before review all values are set to zero (0), i.e, keep.
- *user_comment*: a free-form text box where the user can add comments into during the review step.
- *review_datetime*: the date and time when the user designated keep or erase this detection in the review screen. By default before review, this field is blank or null.

#### project_files.txt
When files are added or removed from a project, the full list is tracked in `Softspoken/projects/Softspoken Outputs/[project-name]/projectname_files.txt` alongside the detection and review CSV outputs.

#### Audacity Outputs
After opening and closing the review screen for the first time after the voice detector is run, in `Softspoken/projects/Audacity Outputs` a project-named folder will be created.
For every project Softspoken creates a text file for each `.wav` file with human detections. 
The `.txt` file contains three tab-delimited rows: 
- first row for start time, 
- second row for end time, 
- and third row for the auto-label "Human".

**To view in Audacity:**
- `File > Open...` to select a `.wav` file that was reviewed by softspoken, or drag the `.wav` file into an Audacity window.
-  Once file is opened, select `File > Import > Labels...` 
- select the `.txt` file with the name that corresponds to the `.wav` file opened under the `Softspoken/projects/Audacity Outputs/project-name` folder.

#### Kaleidoscope Outputs
After opening and closing the review screen for the first time after the voice detector is run, in `Softspoken/projects/Kaleidoscope Outputs` a project-named folder will be created.
For every project Softspoken creates a `.csv` that summarizes the detections in all the `.wav` files scanned. 
The created `.csv` file contains the following headers:
- *INDIR*: the common file path for all files
- *FOLDER*: the remainder of the file path that is not common to all files but is relevant to the file cited in "IN FILE" for this row.
- *IN FILE*: the file name where the voice detection occurs.
- *OFFSET*: the time in seconds since the beginning of the `.wav` file where the detection starts.
- *DURATION*: the length of the voice detection in seconds.
- *TOP1MATCH*: the designation of "Human" from softspoken voice detection.
- *MANUAL ID*: a blank column that can be used in Kaleidoscope Pro or Lite by the user to add comments or annotations.
- *end_time*, *erase*, *review_datetime*: correspond to the same values for the `project_review.csv` output. 

**To view in Kaleidoscope (lite or pro)**
- From the main screen in Kaleidoscope select `File > Open Results...`
- navigate to the `Softspoken/projects/Kaleidoscope Outputs/project-name` folder and select the desired `.csv`. 
- a Kaleidoscope viewer and results window will open with your results, as it does for a cluster or batID analysis.
- To see all columns listed above you may need to select `File > Edit Columns...` in the results window and activate them for Kaleidoscope's table view.

#### Raven Outputs
After opening and closing the review screen for the first time after the voice detector is run, in `Softspoken/projects/Raven Outputs` a project-named folder will be created.
For every project Softspoken creates two `.txt` files. One file is a listfile summarizing all the `.wav` filenames where voice detections were found. The other is a `.txt` label file that summarizes the detections in all the `.wav` files scanned. 

The listfile is a simple list of files with their file paths. The detections `.txt` file contains the following headers:
- *Selection*: This is a unique ID number for ever detection.
- *View*: This tells Raven on which view to show the detection box. Default `Spectrogram 1` shows the detections on the spectrogram view.
- *Channel*: This tells Raven what channel the detection is in, it is set to 1 by default to accomodate single and multiple channel recordings.
- *Begin Time(s)*: the time in seconds since the beginning of the `.wav` files in the list file where the detection starts.
- *End Time (s)*: the time in seconds since the beginning of the `.wav` files in the list file where the detection ends.
- *Low Freq (Hz)*: denotes the lower bounds where Softspoken examines spectrograms for humans (0Hz). Helps raven draw the detection on the spectrogram.
- *High Freq (Hz)*: denotes the upper bounds where Softspoken examines spectrograms for humans (8kHz). Helps raven draw the annotation on the spectrogram.
- *Annotation*: by default is "Human" but can be edited in Raven Pro or Lite.
- *Begin Path*:  This corresponds to the “file_path” and “file_name” combined from the softspoken standard output. This column may only be visible in Raven Pro.
- *erase*, *user-comment*, *review_datetime*: correspond to the same values for the project_review`.csv` output. These columns will only be visible in Raven Pro.

**To view in Raven (lite or pro)**
After opening Raven, from the `Softspoken/projects/Raven Outputs/project-name` folder:
- The user can drag the listfile into Raven Pro or Lite to open all the files simultaneously.
- Then drag in the other `.txt` with the detections to get a selection table for review.

*Note that Softspoken can handle multiple files with different settings (e.g. sampling frequency 24kHz, 36kHz, etc.) together, but Raven cannot. Some listfiles may not open if `.wav` files input into Softspoken have varied settings recording settings used.

### PLATFORM SUPPORT
The script and the launcher have only been tested on Windows.

Application has been tested with:

- Python Version 3.12.7 and 3.13.2
- Windows 10/Windows 11 Home version

This may work as a python script on Mac machines, but we have not tested it.

If you try this on another machine or set-up like mac, please let us know how it goes!

### CONTACT
If you have questions or issues you may wish to submit an issue or contact us directly at info@avianeco.com.

### LICENSE
This application is licensed under an MIT license.

