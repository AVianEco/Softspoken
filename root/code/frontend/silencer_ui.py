import sys
import os
import json
import time
import logging
import pandas as pd
import numpy as np
import librosa 
import soundfile as sf
from datetime import datetime

from root.code.frontend.NNDetector import NNDetector
from root.code.frontend.review_detections import ReviewDetectionsScreen
from root.code.backend import settings
from root.code.backend.worker import ProcessWorker  

from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtCore import Qt, QObject, Signal, Slot, QThreadPool, QUrl, QRunnable
from PySide6.QtWidgets import (
    QDialog, QLineEdit, QMainWindow, QApplication, QPushButton, 
    QProgressBar, QWidget, QHBoxLayout, QLabel, QVBoxLayout, 
    QFileDialog, QSizePolicy, QListWidget, QVBoxLayout, QListWidget,
    QAbstractItemView, QMessageBox
)


class FilePickerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('File Dialog Example')
        self.setFixedSize(300, 200)

        self.open_button = QPushButton('Open Files', self)
        self.open_button.clicked.connect(self.show_file_dialog)
        self.open_button.resize(self.open_button.sizeHint())
        self.open_button.move(100, 80)

    def show_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self, "Select Files", "", "All Files (*);;Text Files (*.txt)", options=options)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec():
            files_list = file_dialog.selectedFiles()
            print("Selected files:", files_list)

class HomeScreen(QMainWindow):
    last_project_clicked = Signal()
    open_project_clicked = Signal()
    start_project_clicked = Signal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Sound Optimization for Fading Talkative Signals and Preserving Other Key Environment Noises')
        self.setFixedSize(750, 300)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        header_label = QLabel("S.O.F.T.S.P.O.K.E.N.", self)
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 24px; font-weight: bold;")

        new_project_button = QPushButton("Start New Project", self)
        new_project_button.clicked.connect(self.start_project_button_clicked)
        new_project_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        open_project_button = QPushButton("Open Existing Project", self)
        open_project_button.clicked.connect(self.open_project_button_clicked)
        open_project_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        last_project_link = QLabel("<a href='#'>Open Last Project</a>", self)
        last_project_link.setAlignment(Qt.AlignCenter)
        last_project_link.setOpenExternalLinks(False)
        last_project_link.linkActivated.connect(self.link_clicked)
        
        layout.addWidget(header_label)
        layout.addStretch(1)
        layout.addWidget(new_project_button)
        layout.addWidget(open_project_button)
        layout.addStretch(1)
        layout.addWidget(last_project_link)

        central_widget.setLayout(layout)

        # Add menu bar
        menu_bar = self.menuBar()

        # Create File menu
        file_menu = menu_bar.addMenu('File')
        
        # Add New Project action to File menu
        new_project_action = QAction('New Project', self)
        new_project_action.setShortcut('Ctrl+N')
        new_project_action.triggered.connect(self.start_project_button_clicked)
        file_menu.addAction(new_project_action)

        # Add Open Project action to File menu
        open_project_action = QAction('Open Project', self)
        open_project_action.setShortcut('Ctrl+O')
        open_project_action.triggered.connect(self.open_project_button_clicked)
        file_menu.addAction(open_project_action)

        # Add Close App action to File menu
        close_app_action = QAction('Close App', self)
        close_app_action.setShortcut('Ctrl+Q')
        close_app_action.triggered.connect(self.close)
        file_menu.addAction(close_app_action)


    def link_clicked(self, link):
        print("Open last project clicked")
        self.last_project_clicked.emit()

    def open_project_button_clicked(self):
        print("Open project clicked")
        self.open_project_clicked.emit()
    
    def start_project_button_clicked(self):
        print("Start project clicked")
        self.start_project_clicked.emit()

class VoiceDetectorScreen(QMainWindow):
    def __init__(self, project_manager, parent_app_screen ):
        super().__init__()
        self.project_manager = project_manager
        self.parent_app_screen = parent_app_screen 

        # UI Elements
        self.setWindowTitle("Voice Detector")
        self.threadpool = QThreadPool.globalInstance()  # or QThreadPool()

        self.detector = None
        self.processing = False
        self.stop_requested = False

        # Label: how many files total?
        self.total_label = QLabel("Total files: 0")
        self.processed_label = QLabel("Processed: 0 / 0")
        self.file_in_progress_label = QLabel("Currently processing: (none)")

        # -- NEW LABELS FOR METRICS --
        self.total_audio_processed_label = QLabel("Total Audio Processed (seconds): 0")
        self.processing_speed_label = QLabel("Processing Speed (audio sec / real sec): 0.00")

        # Keep track of total audio processed and when we started
        self.total_audio_processed = 0.0
        self.start_time = None       

        # Buttons
        self.begin_processing_button = QPushButton("Begin Processing")
        self.begin_processing_button.clicked.connect(self.begin_processing_button_click)

        self.pause_processing_button = QPushButton("Stop Processing")
        self.pause_processing_button.clicked.connect(self.stop_processing)

        # Progress bars
        self.file_progress_bar = QProgressBar()
        self.overall_progress_bar = QProgressBar()

        layout = QVBoxLayout()
        layout.addWidget(self.total_label)
        layout.addWidget(self.processed_label)
        layout.addWidget(self.file_in_progress_label)

        # Add the new labels into the layout
        layout.addWidget(self.total_audio_processed_label)
        layout.addWidget(self.processing_speed_label)        

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.begin_processing_button)
        hlayout.addWidget(self.pause_processing_button)
        layout.addLayout(hlayout)

        layout.addWidget(QLabel("File Progress:"))
        layout.addWidget(self.file_progress_bar)
        layout.addWidget(QLabel("Overall Progress:"))
        layout.addWidget(self.overall_progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Project file list / detection df
        self.detection_project = DetectionProject(project_manager)

        # Let’s quickly see how many files exist
        file_list_path = project_manager.current_project['file_list_file']
        self.file_list = self.load_unprocessed_list(file_list_path)
        self.total_label.setText(f"Total files: {len(self.file_list)}")
        self.processed_label.setText(f"Processed: 0 / {len(self.file_list)}")

    def load_unprocessed_list(self, file_location):
        lines_list = []
        try:
            with open(file_location, 'r') as file:
                for line in file:
                    lines_list.append(line.strip())
        except FileNotFoundError:
            print(f"File '{file_location}' not found.")
        return lines_list

    def begin_processing_button_click(self):
        """
        Create the NNDetector, plan the work, and launch a background worker.
        """
        if self.processing:
            QMessageBox.information(self, "Already Running", "Processing is already in progress.")
            return

        # Initialize the tracking of how many total audio seconds processed
        self.total_audio_processed = 0.0
        self.start_time = time.time()

        # Create the NNDetector
        self.detector = NNDetector(self.project_manager)

        # Plan the detection job
        planned_work = self.detector.plan_detection_job()  # { file: [indexes] }
        if not planned_work:
            QMessageBox.information(self, "No Work", "No files to process.")
            return

        # Create the worker
        self.worker = ProcessWorker(self.detector, self.detection_project, planned_work)
        # Connect signals to update UI
        self.worker.signals.fileProgressChanged.connect(self.on_file_progress)
        self.worker.signals.overallProgressChanged.connect(self.on_overall_progress)
        self.worker.signals.fileStarted.connect(self.on_file_started)
        self.worker.signals.fileDone.connect(self.on_file_done)
        self.worker.signals.finished.connect(self.on_all_done)

        # Start the worker in a separate thread
        self.threadpool.start(self.worker)
        self.processing = True
        self.stop_requested = False

    def stop_processing(self):
        """
        Called when user clicks 'Stop Processing'.
        """
        if not self.processing:
            return
        self.stop_requested = True
        self.worker.stop()  # Ask worker to stop
        # We won't immediately set self.processing=False until on_all_done

    # Slots to handle signals:
    def on_file_started(self, filepath):
        if self.parent_app_screen:
            self.parent_app_screen.refresh_step_status()
    
        filename = os.path.basename(filepath)
        self.file_in_progress_label.setText(f"Currently processing: {filename}")
        self.file_progress_bar.setValue(0)

    def on_file_progress(self, percent):
        self.file_progress_bar.setValue(int(percent))

    def on_file_done(self, filepath):
        """
        The worker finished an entire file.
        Update your new metrics here:
          - total_audio_processed
          - processing_speed
        """
        # 1) Determine how long the audio file is:
        try:
            duration_sec = librosa.get_duration(path=filepath)
        except Exception as e:
            print(f"Could not determine duration for {filepath}: {e}")
            duration_sec = 0.0

        # 2) Update the total audio processed
        self.total_audio_processed += duration_sec

        # 3) Compute elapsed real (wall-clock) time since processing started
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            speed = self.total_audio_processed / elapsed
        else:
            speed = 0

        # 4) Update the labels
        self.total_audio_processed_label.setText(
            f"Total Audio Processed (seconds): {self.total_audio_processed:.2f}"
        )
        self.processing_speed_label.setText(
            f"Processing Speed (audio sec / real sec): {speed:.2f}"
        )

    def on_overall_progress(self, percent):
        """
        Worker updates total progress across all files.
        percent is 0..100. Convert that to an integer for the QProgressBar.
        Also update "Processed X / Y" label if you want a numeric count.
        """
        self.overall_progress_bar.setValue(int(percent))

        total_files = len(self.file_list)
        # approximate number processed
        processed_float = (percent/100.0) * total_files
        processed_int = int(round(processed_float))
        self.processed_label.setText(f"Processed: {processed_int} / {total_files}")

    def on_all_done(self):
        """
        Worker is completely finished OR stopped early.
        """
        if self.parent_app_screen:
            self.parent_app_screen.refresh_step_status()

        self.processing = False
        self.file_in_progress_label.setText("Done.")
        self.file_progress_bar.setValue(100)

        if self.stop_requested:
            QMessageBox.information(self, "Stopped", "Processing was stopped by user.")
        else:
            QMessageBox.information(self, "Complete", "All files processed successfully.")

class AppScreen(QMainWindow):
    def __init__(self, project_manager):
        self.project_manager = project_manager
        super().__init__()
        self.init_ui()

    def showEvent(self, event):
        # Set the title of the window when the window is shown
        project_name = self.project_manager.current_project['name']
        self.setWindowTitle(f'Project Workspace: {project_name}')

        # Refresh the list of files and update the label
        self.refresh_file_list()

        # -- Refresh step statuses each time we show this window --
        self.refresh_step_status()

        # Call the base class implementation of showEvent
        super().showEvent(event)

    # figure out the textual status and color for each step
    def compute_step_status(self, csv_path):
        """
        Returns (status_text, colorStyle) based on the file’s existence & size.
        For example:
            - Not Started if file does not exist
            - In Progress if file exists but is empty
            - Complete if file exists and is non-empty
        You can add more nuanced checks (partial lines, data validity, etc.) as needed.
        """
        if not csv_path:
            return ("Not Started", "color: red;")  # no path at all

        if not os.path.exists(csv_path):
            return ("Not Started", "color: red;")

        size = os.path.getsize(csv_path)
        if size == 0:
            return ("In Progress", "color: orange;")
        else:
            return ("Complete", "color: green;")

    # updates the three step labels based on the presence/size of CSV files
    def refresh_step_status(self):
        # 1) Voice Detector step is tied to detections_file
        detections_path = self.project_manager.current_project['detections_file']
        status_text_1, color_style_1 = self.compute_step_status(detections_path)
        self.step_labels[0].setText(status_text_1)
        self.step_labels[0].setStyleSheet(color_style_1)

        # 2) Review Detections step is tied to review_file
        review_path = self.project_manager.current_project['review_file']
        status_text_2, color_style_2 = self.compute_step_status(review_path)
        self.step_labels[1].setText(status_text_2)
        self.step_labels[1].setStyleSheet(color_style_2)

        # 3) Silence Voices step – if you have a real file for this step, check it here.
        #    Otherwise, we'll just put "Not Started" for now:
        silence_path = getattr(self.project_manager.current_project, 'silence_file', None)
        if not silence_path:
            # If your project data doesn't have a field, we’ll just say "Not Started"
            self.step_labels[2].setText("Not Started")
            self.step_labels[2].setStyleSheet("color: red;")
        else:
            status_text_3, color_style_3 = self.compute_step_status(silence_path)
            self.step_labels[2].setText(status_text_3)
            self.step_labels[2].setStyleSheet(color_style_3)

    def refresh_file_list(self):
        """
        Helper method to refresh the file list display and update the file count.
        """
        full_file_list = self.project_manager.get_unprocessed_list()
        self.list_widget.clear()
        self.list_widget.addItems(full_file_list)
        self.file_count_label.setText(f"Total files: {len(full_file_list)}")

    def open_add_files(self):
        """
        Existing method for adding files – no changes to functionality.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self, "Select Files", "", "All Files (*);;Text Files (*.txt)", options=options)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec():
            files_list = file_dialog.selectedFiles()
            print("Selected files:", files_list)
            
            # update the project manager with the new files
            full_file_list = self.project_manager.update_file_list(files_list)
            
            # update the list on the UI
            self.list_widget.clear()
            self.list_widget.addItems(full_file_list)

            self.file_count_label.setText(f"Total files: {len(full_file_list)}")

    def remove_selected_files(self):
        """
        Remove selected files from the project list, with a confirmation popup.
        """
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return

        # Confirmation popup
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to remove {len(selected_items)} selected file(s) from the project?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Collect file paths from selected items
            files_to_remove = [item.text() for item in selected_items]
            
            # Actually remove them from the project file
            self._remove_files_from_project_manager(files_to_remove)
            self.refresh_file_list()

    def _remove_files_from_project_manager(self, files_list):
        """
        Helper method to remove files from the project's file list text file.
        """
        file_path = self.project_manager.current_project['file_list_file']
        
        # Read all existing lines
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_lines = [line.strip() for line in f.readlines()]
        else:
            existing_lines = []

        # Remove the specified files
        updated_lines = [line for line in existing_lines if line not in files_list]

        # Write back to the file
        with open(file_path, 'w') as f:
            for line in updated_lines:
                f.write(line + '\n')

    def launch_voice_detector_ui(self):
        self.voice_detector_screen = VoiceDetectorScreen(
            self.project_manager,
            parent_app_screen = self
        )

        add_common_menus(self.voice_detector_screen)
        self.voice_detector_screen.show()

    def launch_review_detections_ui(self):
        self.review_detections_screen = ReviewDetectionsScreen(
            self.project_manager,
            parent_app_screen = self
        )
        
        add_common_menus(self.review_detections_screen)
        self.review_detections_screen.show()

    def launch_silence_voices_ui(self):
        self.silence_voices_screen = SilenceVoicesScreen(
            self.project_manager,
            parent_app_screen = self
        )

        add_common_menus(self.silence_voices_screen)
        self.silence_voices_screen.show()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)  # Give some margin around edges
        main_layout.setSpacing(20)

        # Create a QVBoxLayout for the list widget + buttons
        list_button_layout = QVBoxLayout()
        list_button_layout.setSpacing(10)

        # a label above the list for total file count
        self.file_count_label = QLabel("Total files: 0")
        self.file_count_label.setStyleSheet("font-weight: bold;")  # Slight emphasis
        list_button_layout.addWidget(self.file_count_label)

        # Create the QListWidget
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setFixedWidth(400)  
        list_button_layout.addWidget(self.list_widget, alignment=Qt.AlignLeft)

        # Create a horizontal layout just for the "Add Files" and "Delete" buttons
        file_buttons_layout = QHBoxLayout()
        file_buttons_layout.setSpacing(10)

        new_button = QPushButton("Add Files", self)
        new_button.clicked.connect(self.open_add_files)
        file_buttons_layout.addWidget(new_button)

        delete_button = QPushButton("Delete", self)
        delete_button.clicked.connect(self.remove_selected_files)
        file_buttons_layout.addWidget(delete_button)

        list_button_layout.addLayout(file_buttons_layout)
        main_layout.addLayout(list_button_layout)
        
        # Right side: the workflow steps
        buttons = ['Run Voice Detector', 'Review Detections', 'Silence Voices']
        # labels = ['Not Started', 'Not Started', 'Not Started']
        self.step_labels = [QLabel("Not Started"), QLabel("Not Started"), QLabel("Not Started")]

        actions = [self.launch_voice_detector_ui, self.launch_review_detections_ui, self.launch_silence_voices_ui]

        for b_text, label, action in zip(buttons, self.step_labels, actions):
            button_label_layout = QVBoxLayout()
            button_label_layout.setContentsMargins(0, 0, 0, 0)  
            button_label_layout.setSpacing(0)
            
            button = QPushButton(b_text, self)
            if action is not None:
                button.clicked.connect(action)
            button_label_layout.addWidget(button, alignment=Qt.AlignCenter)

            # label = QLabel(l_text, self)
            label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            # If you want the "Not Done" text to stand out less, you could lighten it:
            label.setContentsMargins(0, 0, 0, 0)
            label.setStyleSheet("color: #666666; margin: 0px; padding: 0px;")
            button_label_layout.addWidget(label)

            main_layout.addLayout(button_label_layout)

        central_widget.setLayout(main_layout)

# tracks the audio files that are added to the project 
class ProjectManager:
    def __init__(self):
        # Define the path to the 'projects' folder and the 'projects.json' file
        self.projects_folder = settings.project_dir
        self.projects_file = os.path.join(self.projects_folder, 'projects.json')
        
        # all projects saved by the user
        self.projects_data = []
        
        # the settings for the project we're currently working on
        self.current_project = None
                
        # Check if the 'projects' folder exists, and create it if it doesn't
        if not os.path.exists(self.projects_folder):
            os.mkdir(self.projects_folder)
        
        # Check if the 'projects.json' file exists, and load it if it does
        if os.path.exists(self.projects_file):
            with open(self.projects_file, 'r') as f:
                self.projects_data = json.load(f)
        else:
            # If the file doesn't exist, create it with an empty JSON object
            self.write_projects_file()

    def get_unprocessed_list(self):
        if self.current_project is None:
            return []
        
        # Initialize an empty list to store the lines from the file
        file_location = self.current_project['file_list_file']
        lines_list = []
        
        try:
            # Open the file for reading
            with open(file_location, 'r') as file:
                # Read each line from the file and add it to the list
                for line in file:
                    lines_list.append(line.strip())  # Use .strip() to remove leading/trailing whitespace and newline characters
        except FileNotFoundError:
            print(f"File '{file_location}' not found.")
        
        # Return the list of lines
        return lines_list
    
    def update_file_list(self, files_list):
        # Define the path to the 'files.txt' file
        file_path = self.current_project['file_list_file']
        
        # Initialize a set to store the unique file names
        unique_files = set()
        
        # Read the existing file names from 'files.txt' if the file exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    unique_files.add(line.strip())
        
        # Add the new file names from files_list to the set (duplicates will be ignored)
        unique_files.update(files_list)
        
        # Convert the set to a sorted list
        sorted_files = sorted(unique_files)
        
        # Write the updated and sorted file names to 'files.txt'
        with open(file_path, 'w') as file:
            for file_name in sorted_files:
                file.write(file_name + '\n')
        
        # Return the updated list of file names
        return sorted_files
    
    def get_new_project_settings(self):
        return {
            'name': '', # unique name of the project
            'file_list_file': '_files.txt',  # list of the individual files
            'detections_file': '_detections.csv',  # thresholded detections
            'review_file': '_review.csv', 
            'last_accessed': ''
        }
    
    def get_current_datetime_str(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def list_projects_by_name(self):
        return [p['name'] for p in self.projects_data]
        
    def add_project(self, name):
        project_settings = self.get_new_project_settings()
        project_settings['name'] = name
        project_settings['file_list_file'] = os.path.join(self.projects_folder, name + project_settings['file_list_file'])
        project_settings['detections_file'] = os.path.join(self.projects_folder, name + project_settings['detections_file'])
        project_settings['review_file'] = os.path.join(self.projects_folder, name + project_settings['review_file'])
        project_settings['last_accessed'] = self.get_current_datetime_str()
        
        self.projects_data.append(project_settings)
        self.write_projects_file()
        
    def write_projects_file(self):
        # open the project file - save the project data
        with open(self.projects_file, 'w') as f:
            json.dump(self.projects_data, f)
    
    def set_active_project(self, project_name):
        # Use a generator expression to find the project with the given name, otherwise return None
        project = next((p for p in self.projects_data if p['name'] == project_name), None)
        self.current_project = project

    def activate_latest(self):
        if len(self.projects_data) == 0:
            return False

        project = sorted(self.projects_data, key=lambda x: x['last_accessed'], reverse=True)[0]
        self.set_active_project(project['name'])
        return True

class DetectionProject:
    def __init__(self, project_settings):
        self.settings = project_settings

        column_types = {
            'ID': 'int64',
            'file_path': str,
            'file_name': str,
            'start_time': str,
            'end_time': str,
            'erase': int,
            'user_comment': str,
            'review_datetime': 'datetime64[ns]'
        }
        
        # use these to setup a new detections df
        self.columns = column_types.keys()

        detections_path = self.settings.current_project['detections_file']
        if os.path.exists(detections_path):
            self.df = pd.read_csv(detections_path)

            if 'ID' not in self.df.columns:
                self.df.insert(0, 'ID', range(1, len(self.df) + 1))
            else:
                self.df['ID'] = pd.to_numeric(self.df['ID'], errors='coerce')
                missing_ids = self.df['ID'].isna()
                if missing_ids.any():
                    current_max = self.df['ID'].dropna().max()
                    start_id = int(current_max) if not np.isnan(current_max) else 0
                    for offset, idx in enumerate(self.df.index[missing_ids], start=start_id + 1):
                        self.df.at[idx, 'ID'] = offset
                self.df['ID'] = self.df['ID'].astype('int64')

            if 'review_datetime' in self.df.columns:
                self.df['review_datetime'] = pd.to_datetime(self.df['review_datetime'], errors='coerce')

            self.df = self.df.reindex(columns=self.columns).astype(column_types)
        else:
            self.df = pd.DataFrame(columns = self.columns).astype(column_types)
    
    def save_detections(self):
        self.df.to_csv(self.settings.current_project['detections_file'], index = False)

class NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super(NewProjectDialog, self).__init__(parent)
        
        # Set up the layout for the dialog
        layout = QVBoxLayout()
        
        # Create a QLabel to display a message to the user
        label = QLabel("Enter the name of the new project:")
        layout.addWidget(label)
        
        # Create a QLineEdit widget for the user to enter the project name
        self.project_name_input = QLineEdit()
        layout.addWidget(self.project_name_input)
        
        # Create a QHBoxLayout for the buttons
        button_layout = QHBoxLayout()
        
        # Create the "OK" button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        # Create the "Cancel" button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        # Add the QHBoxLayout to the QVBoxLayout
        layout.addLayout(button_layout)
        
        # Set the layout for the QDialog
        self.setLayout(layout)
        
    def accept(self):
        # Get the project name entered by the user
        self.project_name = self.project_name_input.text()
        super(NewProjectDialog, self).accept()

    def reject(self):
        super(NewProjectDialog, self).reject()

class ProjectSelectionDialog(QDialog):
    def __init__(self, items, parent=None):
        super(ProjectSelectionDialog, self).__init__(parent)
        
        # Set up the layout for the dialog
        layout = QVBoxLayout()
        
        # Create a QListWidget to display the list of items
        self.list_widget = QListWidget()
        self.list_widget.addItems(items)
        layout.addWidget(self.list_widget)
        
        # Create a QHBoxLayout for the buttons
        button_layout = QHBoxLayout()
        
        # Create the "OK" button (initially disabled)
        self.ok_button = QPushButton("OK")
        self.ok_button.setEnabled(False)
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        
        # Create the "Cancel" button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        # Add the QHBoxLayout to the QVBoxLayout
        layout.addLayout(button_layout)
        
        # Set the layout for the QDialog
        self.setLayout(layout)
        
        # Connect the QListWidget's itemSelectionChanged signal to enable/disable the OK button
        self.list_widget.itemSelectionChanged.connect(self.update_ok_button_state)
        
    def update_ok_button_state(self):
        # Enable the OK button if an item is selected, disable it otherwise
        self.ok_button.setEnabled(len(self.list_widget.selectedItems()) > 0)
        
    def accept(self):
        # Get the selected item text
        self.selected_item = self.list_widget.currentItem().text()
        super(ProjectSelectionDialog, self).accept()
    
    def reject(self):
        super(ProjectSelectionDialog, self).reject()

class SilenceWorkerSignals(QObject):
    """
    Defines signals emitted by SilenceWorker.
    """
    fileStarted   = Signal(str)  # Emitted when we begin processing a new file
    fileProgress  = Signal(int)  # For partial progress if desired
    fileComplete  = Signal(str)  # Emitted when a file is done
    overallProgress = Signal(int)
    finished = Signal()

class SilenceWorker(QRunnable):
    """
    Worker that silences segments from the CSV where erase=1,
    saves them to a new location, and emits signals for UI updates.
    """
    def __init__(self, review_df, output_dir, sr=44100):
        super().__init__()
        self.signals = SilenceWorkerSignals()
        self.review_df = review_df
        self.output_dir = output_dir
        self.stop_requested = False

    @Slot()
    def run(self):
        """
        Main process: group the CSV rows by (file_path, file_name),
        silence each file's segments, then write to new location.
        """
        # Group by the unique file references
        # We'll filter out only rows with erase=1
        erase_df = self.review_df[self.review_df['erase'] == 1]
        if erase_df.empty:
            # No segments to erase
            self.signals.finished.emit()
            return

        # Each group is basically one audio file
        grouped = erase_df.groupby(['file_path', 'file_name'])

        total_files = len(grouped)
        files_done = 0

        for (fpath, fname), group_rows in grouped:
            if self.stop_requested:
                break

            full_path = os.path.join(fpath, fname)
            self.signals.fileStarted.emit(full_path)

            # Load audio
            try:
                audio_data, sr = librosa.load(full_path, sr=None, mono=False) 
                # shape = (n_samples,) if mono, or (channels, n_samples) if stereo
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                # You could emit an error signal or skip
                files_done += 1
                self.signals.overallProgress.emit(int(files_done / total_files * 100))
                continue

            # Convert to a consistent shape so that indexing is simpler
            if audio_data.ndim == 1:
                # Mono => shape=(samples,) => expand to (1, samples)
                audio_data = np.expand_dims(audio_data, axis=0)

            # Zero out each row's start_time..end_time
            for _, row in group_rows.iterrows():
                st = float(row['start_time'])
                et = float(row['end_time'])
                start_index = int(round(st * sr))
                end_index   = int(round(et * sr))

                # Bound checks
                start_index = max(0, min(start_index, audio_data.shape[1]))
                end_index   = max(0, min(end_index,   audio_data.shape[1]))

                # Zero out that range for all channels
                audio_data[:, start_index:end_index] = 0.0

            # Build new filename: e.g. "myfile_silenced.wav"
            base, ext = os.path.splitext(fname)
            silenced_fname = f"{base}_silenced.wav"
            out_fullpath = os.path.join(self.output_dir, silenced_fname)

            # Write
            try:
                # If audio_data shape is (channels, samples), 
                # need to transpose for soundfile or reshape for librosa
                # We'll do soundfile with shape=(samples, channels).
                audio_data_for_save = audio_data.T  # shape => (samples, channels)
                sf.write(out_fullpath, audio_data_for_save, sr)
            except Exception as e:
                print(f"Error writing {out_fullpath}: {e}")

            # One file done
            self.signals.fileComplete.emit(out_fullpath)

            files_done += 1
            percent_done = int(files_done / total_files * 100)
            self.signals.overallProgress.emit(percent_done)

        self.signals.finished.emit()

    def stop(self):
        """
        Request stop from the UI (user clicked a stop/cancel button).
        """
        self.stop_requested = True

class SilenceVoicesScreen(QMainWindow):
    def __init__(self, project_manager, parent_app_screen):
        super().__init__()
        self.setWindowTitle("Silence Voices")
        self.setMinimumSize(600, 300)

        self.project_manager = project_manager
        self.parent_app_screen = parent_app_screen

        # We'll store the output directory path here
        self.output_dir = ""

        # Prepare the layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # 1) Show current CSV file path
        self.review_file_path = self.project_manager.current_project['review_file']
        self.csv_label = QLabel(f"Review CSV: {self.review_file_path}")
        main_layout.addWidget(self.csv_label)

        # 2) Let user pick the output folder
        folder_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select folder where silenced files will be saved...")
        folder_row.addWidget(self.output_dir_edit)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_output_dir)
        folder_row.addWidget(browse_button)
        main_layout.addLayout(folder_row)

        # 3) Show a label for how many segments are marked "erase=1"
        self.segments_label = QLabel("Segments to erase: 0")  # We'll fill it once we load the CSV
        main_layout.addWidget(self.segments_label)

        # 4) A Start + Stop button row
        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start Silencing")
        self.start_button.clicked.connect(self.start_silencing)
        button_row.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_silencing)
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.stop_button)
        main_layout.addLayout(button_row)

        # 5) A progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # 6) Threadpool + worker placeholders
        self.threadpool = QThreadPool.globalInstance()
        self.worker = None

        # Finally, load the CSV data so we can see how many segments
        self.load_review_data()

    def browse_output_dir(self):
        """
        Let user pick an output directory for the silenced files.
        """
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", os.getcwd()
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            self.output_dir = dir_path

    def load_review_data(self):
        """
        Read the review CSV, count how many segments have erase=1,
        display that to the user.
        """
        if not os.path.exists(self.review_file_path):
            QMessageBox.warning(self, "No Review File", f"Review file not found: {self.review_file_path}")
            return

        try:
            self.review_df = pd.read_csv(self.review_file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read CSV: {e}")
            self.review_df = pd.DataFrame()

        if not self.review_df.empty and 'erase' in self.review_df.columns:
            # Convert erase to numeric or bool if needed
            self.review_df['erase'] = pd.to_numeric(self.review_df['erase'], errors='coerce').fillna(0).astype(int)
            erase_count = (self.review_df['erase'] == 1).sum()
        else:
            erase_count = 0

        self.segments_label.setText(f"Segments to erase: {erase_count}")

    def start_silencing(self):
        """
        Creates and starts the SilenceWorker with the loaded CSV data, if any.
        """
        if self.output_dir_edit.text().strip() == "":
            QMessageBox.information(self, "Output Folder", "Please select an output folder first.")
            return

        if getattr(self, 'review_df', None) is None or self.review_df.empty:
            QMessageBox.information(self, "No Data", "No data in the review CSV or not loaded.")
            return

        # Create the worker
        self.worker = SilenceWorker(self.review_df, self.output_dir_edit.text().strip())
        # Connect signals
        self.worker.signals.fileStarted.connect(self.on_file_started)
        self.worker.signals.fileComplete.connect(self.on_file_complete)
        self.worker.signals.overallProgress.connect(self.on_overall_progress)
        self.worker.signals.finished.connect(self.on_finished)

        # Start
        self.threadpool.start(self.worker)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

    def stop_silencing(self):
        """
        Requests the worker to stop early.
        """
        if self.worker:
            self.worker.stop()

    # --- Slots to handle signals ---
    def on_file_started(self, file_path):
        """
        Let the user know which file is being processed.
        """
        self.csv_label.setText(f"Processing {file_path}...")

    def on_file_complete(self, out_path):
        """
        Called when a single file is fully silenced and saved.
        """
        # Optionally log or show status
        pass

    def on_overall_progress(self, percent):
        self.progress_bar.setValue(percent)

    def on_finished(self):
        """
        Worker is done (or user canceled).
        """
        self.csv_label.setText("Silencing complete.")
        self.progress_bar.setValue(100)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

def add_common_menus(main_window):
    """
    Add a 'Help' menu (with user guide link, etc.) to the given main_window's menu bar.
    """
    # Create or retrieve the menu bar
    menu_bar = main_window.menuBar()

    # Create 'Help' menu
    help_menu = menu_bar.addMenu("Help")

    # Create 'User Guide' action
    user_guide_action = QAction("User Guide", main_window)
    user_guide_action.setShortcut("Ctrl+U")
    user_guide_action.triggered.connect(
        lambda: QDesktopServices.openUrl(QUrl(settings.user_guide_url))
    )
    help_menu.addAction(user_guide_action)

    return menu_bar

def main():
    # the project manager hold a list of all projects kept by the user - plus the settings for the currently active project
    project_manager = ProjectManager()
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # ----------------------
    # Apply overall style
    # ----------------------
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f2f2f2;
        }
        QLabel {
            font-size: 14px;
        }
        QPushButton {
            background-color: #4CAF50; 
            color: white;
            font-size: 14px; 
            padding: 8px 16px; 
            border: none; 
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QListWidget {
            background-color: #ffffff;
            border: 1px solid #ccc;
        }
    """)

    # the purpose of the home screen is to allow the user to make a new project, or load an existing one
    # then load the project settings and get them into the app
    home_screen = HomeScreen()
    add_common_menus(home_screen)
    
    # the user types the name of a new project into here
    new_project_dialog = NewProjectDialog()
    
    # app screen contains workflow for the detections
    app_screen = AppScreen(project_manager)
    add_common_menus(app_screen)
    
    def open_app_screen():
        home_screen.close()

        # pick the last accessed project
        activated = app_screen.project_manager.activate_latest()
        if activated:
            app_screen.show()
    
    def start_new_project_screen():
        result = new_project_dialog.exec()
        
        # proceed on accept
        if result == QDialog.Accepted:
            print(f"New project name: {new_project_dialog.project_name}")
            
            # Get the entered project name
            project_name = new_project_dialog.project_name
            
            # add the new project and activate it
            project_manager.add_project(project_name)
            project_manager.set_active_project(project_name)
            
            # close the home screen and launch the analysis app
            home_screen.close()
            app_screen.show()
        
        else:
            # nothing to do on cancel
            pass
        
    def open_project_selection_dialog():
        # list of the projects by name
        items = project_manager.list_projects_by_name()
        
        # pick one of the projects - or cancel
        project_dialog = ProjectSelectionDialog(items)
        result = project_dialog.exec()
        
        if result == QDialog.Accepted:
            # get the selected item text
            selected_item = project_dialog.selected_item
            logging.info(f"Selected item: {selected_item}")
            
            # do something with the selected_item
            project_manager.set_active_project(selected_item)
            home_screen.close()
            app_screen.show()
            
    home_screen.last_project_clicked.connect(open_app_screen)
    home_screen.start_project_clicked.connect(start_new_project_screen)
    home_screen.open_project_clicked.connect(open_project_selection_dialog)

    home_screen.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
