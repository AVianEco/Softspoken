import io
import os
import math
import time
import pandas as pd
import numpy as np
import datetime
import librosa
import tempfile, soundfile as sf
from pathlib import Path
import matplotlib
matplotlib.use('agg') # non-interactive
import matplotlib.pyplot as plt

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import Qt, QByteArray, QTimer, Signal, QUrl
from PySide6.QtGui import QPixmap, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QSplitter, QWidget,
    QCheckBox, QAbstractItemView, QSpacerItem, QHeaderView, QSizePolicy, QDoubleSpinBox
)

from root.code.backend import voice_activity, settings

class DebouncedSplitter(QSplitter):
    # Custom signal that we will emit once the user has stopped dragging
    debouncedResize = Signal()

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        # A single-shot timer, just like your existing resize_timer
        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.on_debounce_timeout)

    def splitterMoved(self, event):
        # Called continuously while the user drags (or whenever the splitter is resized)
        super().resizeEvent(event)
        # Each time, restart the timer 
        self.debounce_timer.start(100)

    def on_debounce_timeout(self):
        # Once the timer expires, emit our custom signal
        self.debouncedResize.emit()

class ReviewDetectionsScreen(QMainWindow):
    def on_splitter_moved(self, pos, index):
        # Restart timer every time the splitter is moved
        self.splitter_timer.start(100)

    def save_review(self, persist=True):
        """
        Sync the QTableWidget data back to self.csv_data and write out the CSV.
        """
        start = time.time()

        if self.parent_app_screen:
            self.parent_app_screen.refresh_step_status()

        # 1) Update self.csv_data from table rows
        row_count = self.table.rowCount()
        col_count = self.table.columnCount()

        # Build a DataFrame from the table's current contents
        table_data = []
        headers = [self.table.horizontalHeaderItem(c).text() for c in range(col_count)]

        for row in range(row_count):
            row_values = []
            for col in range(col_count):
                cell_item = self.table.item(row, col)
                row_values.append(cell_item.text() if cell_item else "")
            table_data.append(row_values)

        # Convert to a pandas DataFrame
        df = pd.DataFrame(table_data, columns=headers)
        # Cast numeric columns as needed
        if "start_time" in df.columns:
            df["start_time"] = pd.to_numeric(df["start_time"], errors="coerce")
        if "end_time" in df.columns:
            df["end_time"] = pd.to_numeric(df["end_time"], errors="coerce")

        # convert back to bool (TODO: flakey and needs a more logical flow for save/load)
        if "erase" in df.columns:
            df["erase"] = df["erase"].apply(
                lambda x: 1 if x.strip().lower() == "yes" else 0
            )

        self.csv_data = df  # store it if you want to keep using self.csv_data

        # 2) Finally write out to CSV
        if persist:
            output_path = self.project_manager.current_project['review_file'] 
            df.to_csv(output_path, index=False)
            print(f"Review saved to {output_path}")

            import root.code.frontend.review_exporter as review_exporter
            exporter = review_exporter.ReviewExportManager(df)
            exporter.register_transform(review_exporter.AudacityTxtTransform())
            exporter.register_transform(review_exporter.KaleidoscopeCsvTransform())
            exporter.register_transform(review_exporter.RavenTxtTransform())

            exporter.export(
                "audacity",
                dst=".",                                       # not used by this transform
                base_dir=Path(output_path).parent,        # REQUIRED
                project_name=self.project_manager.current_project["name"]  # REQUIRED
            )

            exporter.export(
                "kaleidoscope",
                dst=".",                                       # ignored by this transform
                base_dir=Path(output_path).parent,        # REQUIRED
                project_name=self.project_manager.current_project["name"]  # REQUIRED
            )

            exporter.export(
                "raven",
                dst=".",                                    # ignored by this transform
                base_dir=Path(output_path).parent,      # REQUIRED
                project_name=self.project_manager.current_project["name"]  # REQUIRED
            )

        elapsed = time.time() - start
        print(f'save_review took: {elapsed}.  persist: {persist}')

    def __init__(self, project_manager, parent_app_screen):
        super().__init__()
        self.setWindowTitle(f"Review Detections: {project_manager.current_project['name']}")

        self.project_manager = project_manager
        self.parent_app_screen = parent_app_screen

        detections_path = self.project_manager.current_project['detections_file']
        review_path = self.project_manager.current_project['review_file'] 
        
        # load any existing review data if it exists, it won't on the first time opening
        if review_path is not None and os.path.exists(review_path):
            self.csv_data = pd.read_csv(review_path)
        # load the nn detections in lieu of the reviewed data, starting the review process from the beginning
        elif detections_path is not None and os.path.exists(detections_path):
            self.csv_data = pd.read_csv(detections_path)
            self.filter_by_minimum_detection_len()
        # Fallback if no csv_path provided or file doesn't exist
        else:
            self.csv_data = pd.DataFrame(columns=["file_path", "file_name", 
                                                  "start_time", "end_time", 
                                                  "erase", "user_comment", 
                                                  "review_datetime"])

        self.current_index = 0
        self.zoom_level = 1
        
        # DPI and window width in inches
        screen = QApplication.primaryScreen()  # Get the primary screen
        self.dpi = screen.physicalDotsPerInch()  # Get the DPI of the screen
        
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)  # The timer will auto-stop after the timeout.
        self.resize_timer.timeout.connect(self.refresh_spectrogram)

        # Add a timer for the splitter as well
        self.splitter_timer = QTimer(self)
        self.splitter_timer.setSingleShot(True)
        self.splitter_timer.timeout.connect(self.refresh_spectrogram)
       
        self.main_splitter = DebouncedSplitter(Qt.Vertical)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.splitterMoved.connect(self.on_splitter_moved) # start debounce timer
        self.main_splitter.debouncedResize.connect(self.refresh_spectrogram) # when timer expires do the work
        
        self.top_widget = QWidget()
        layout = QVBoxLayout(self.top_widget)
    
        self.spectrogram_label = QLabel()
        layout.addWidget(self.spectrogram_label)
    
        self.button_row = QHBoxLayout()
        layout.addLayout(self.button_row)
    
        self.time_axis_label = QLabel('Time Axis: ')
        self.zoom_in_button = QPushButton('+')
        self.zoom_out_button = QPushButton('-')
    
        self.scroll_next_button = QPushButton('Next')
        self.scroll_previous_button = QPushButton('Previous')
        self.scroll_next_file_button = QPushButton('Next File')
        self.scroll_previous_file_button = QPushButton('Previous File')
    
        self.silence_yes_button = QPushButton('Keep')
        self.silence_yes_button.setToolTip("Shift + K")

        self.silence_no_button = QPushButton('Erase')
        self.silence_no_button.setToolTip("Shift + E")
    
        self.left_button_layout = QHBoxLayout()
        self.left_button_layout.addWidget(self.time_axis_label)
        self.left_button_layout.addWidget(self.zoom_in_button)
        self.left_button_layout.addWidget(self.zoom_out_button)
    
        self.center_button_layout = QHBoxLayout()
        self.center_button_layout.addWidget(self.scroll_previous_button)
        self.center_button_layout.addWidget(self.scroll_next_button)
        self.center_button_layout.addWidget(self.scroll_previous_file_button)
        self.center_button_layout.addWidget(self.scroll_next_file_button)
    
        # Create layout for buttons
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.silence_yes_button)
        self.button_layout.addWidget(self.silence_no_button)
        
        # Create layout for the entire right side
        self.right_button_layout = QVBoxLayout()
        self.right_button_layout.addLayout(self.button_layout)


        #######  media  #######

        # Create a media player + audio output
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.playbackStateChanged.connect(self.on_playback_state_changed)

        # Create spinboxes + labels for adjusting start/end
        self.start_label = QLabel("Start time:")
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 99999)
        self.start_spin.setSingleStep(0.1)   # 0.1s increments

        self.stop_label = QLabel("End time:")
        self.stop_spin = QDoubleSpinBox()
        self.stop_spin.setRange(0, 99999)
        self.stop_spin.setSingleStep(0.1)

        self.play_all_button = QPushButton("Play All")
        self.play_all_button.setObjectName("playAllButton")        # unique name for styling
        self.play_all_button.setCursor(Qt.PointingHandCursor)      # pointer cursor
        self.play_all_button.setCheckable(True)                    # stay “pressed” while playing
        self.play_all_button.setStyleSheet("""
            /* Base */
            #playAllButton {
                background-color: lightgray;
                border: 1px solid #808080;
                border-radius: 4px;
                padding: 4px 8px;
            }
            /* Hover */
            #playAllButton:hover:enabled {
                background-color: #d0d0d0;
            }
            /* Pressed OR checked */
            #playAllButton:pressed:enabled,
            #playAllButton:checked:enabled {
                background-color: #a8a8a8;
            }
            /* Disabled */
            #playAllButton:disabled {
                background-color: #f0f0f0;
                color: #a0a0a0;
            }
        """)

        self.play_all_button.clicked.connect(self.play_window_audio)

        self.play_button = QPushButton("Play")
        self.play_button.setToolTip("Shift + Space")
        self.play_button.clicked.connect(self.play_selected_segment)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_playback)

        # Audio controls layout (two rows)
        self.audio_vlayout = QVBoxLayout()
        # 1) First row: Start time, End time
        self.audio_hlayout = QHBoxLayout()
        self.audio_hlayout.addWidget(self.start_label)
        self.audio_hlayout.addWidget(self.start_spin)
        self.audio_hlayout.addWidget(self.stop_label)
        self.audio_hlayout.addWidget(self.stop_spin)
        # add to vertical layout
        self.audio_vlayout.addLayout(self.audio_hlayout)

        # 2) Second row: Play All/Play/Stop buttons
        self.play_all_button.setFixedWidth(80)
        self.play_button.setFixedWidth(80)  # optional sizing
        self.stop_button.setFixedWidth(80)
        self.play_button_hbox = QHBoxLayout()
        self.play_button_hbox.addWidget(self.play_all_button)
        self.play_button_hbox.addWidget(self.play_button)
        self.play_button_hbox.addWidget(self.stop_button)
        self.play_button_hbox.addStretch()  # push the buttons left
        self.audio_vlayout.addLayout(self.play_button_hbox)

        # Now we have self.audio_vlayout containing:
        #   Row 1: Start time / End time
        #   Row 2: Play
    
        self.button_row.addLayout(self.audio_vlayout)
        self.button_row.addItem(QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))
        #######  media  #######

        self.button_row.addLayout(self.left_button_layout)
        self.button_row.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.button_row.addLayout(self.center_button_layout)
        self.button_row.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.button_row.addLayout(self.right_button_layout)

        self.bottom_widget = QWidget()
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)   # Force table to select entire rows, not individual cells
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)  # Allow only one row to be selected at a time
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive) # or... ResizeToContents ?

        self.bottom_widget.setLayout(QVBoxLayout())
        self.bottom_widget.layout().addWidget(self.table)

        self.main_splitter.addWidget(self.top_widget)
        self.main_splitter.addWidget(self.bottom_widget)
        self.setCentralWidget(self.main_splitter)
    
        self.zoom_in_button.clicked.connect(lambda: self.zoom(False))
        self.zoom_out_button.clicked.connect(lambda: self.zoom(True))
        self.scroll_next_button.clicked.connect(lambda: self.scroll("next"))
        self.scroll_previous_button.clicked.connect(lambda: self.scroll("previous"))
        self.scroll_next_file_button.clicked.connect(lambda: self.scroll("next_file"))
        self.scroll_previous_file_button.clicked.connect(lambda: self.scroll("previous_file"))
        self.silence_yes_button.clicked.connect(self.apply_keep)
        self.silence_no_button.clicked.connect(self.apply_erase)

        # Keyboard shortcuts
        keep_shortcut = QShortcut(QKeySequence("Shift+K"), self)
        keep_shortcut.activated.connect(self.apply_keep)
        erase_shortcut = QShortcut(QKeySequence("Shift+E"), self)
        erase_shortcut.activated.connect(self.apply_erase)
        # play the audio 
        play_audio_shortcut = QShortcut(QKeySequence("Shift+Space"), self)
        play_audio_shortcut.activated.connect(self.play_selected_segment)
        

        self.show_bars_checkbox = QCheckBox("Show Vertical Bars")
        self.show_bars_checkbox.setChecked(True)  # default = ON
        self.show_bars_checkbox.stateChanged.connect(self.refresh_spectrogram)
        self.left_button_layout.addWidget(self.show_bars_checkbox)

        self.populate_table()
        self.refresh_spectrogram()

        # populate the table before connecting these 
        self.table.itemClicked.connect(self.select_detection_from_table)
        self.table.itemChanged.connect(self.on_table_item_changed)
        # keyboard nav
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)

        QApplication.instance().aboutToQuit.connect(
            lambda: self.save_review(persist=True)
        )

    def play_selected_segment(self):
        # 1) Figure out the row. You can rely on self.current_index.
        row_idx = self.current_index
        if row_idx < 0 or row_idx >= len(self.csv_data):
            return  # nothing to play

        # 2) Build the full file path from table
        col_indexes = {self.table.horizontalHeaderItem(c).text(): c for c in range(self.table.columnCount())}
        file_path = self.table.item(row_idx, col_indexes["file_path"]).text()
        file_name = self.table.item(row_idx, col_indexes["file_name"]).text()
        full_path = os.path.join(file_path, file_name)

        # 3) Read the user’s chosen start/stop from spinboxes
        s = self.start_spin.value()
        e = self.stop_spin.value()
        if e <= s:
            print("End time must be > Start time!")
            return

        # 4) Load the subregion of audio
        data, sr = voice_activity.load_audio_startstop(full_path, (s, e))

        # 5) Write to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_wav_path = tmp.name
        sf.write(temp_wav_path, data, sr)

        # 6) Set the media player source to that temp WAV
        self.player.setSource(QUrl.fromLocalFile(temp_wav_path))

        # 7) Start playback
        self.player.play()
        self.stop_button.setEnabled(True)

    def play_window_audio(self):
        """Play all audio currently visible in the spectrogram window."""
        row_idx = self.current_index
        if row_idx < 0 or row_idx >= len(self.csv_data):
            return

        start = getattr(self, "visible_audio_start", None)
        end = getattr(self, "visible_audio_end", None)
        full_path = None

        if start is None or end is None or end <= start:
            return

        col_indexes = {self.table.horizontalHeaderItem(c).text(): c for c in range(self.table.columnCount())}
        file_path = self.table.item(row_idx, col_indexes["file_path"]).text()
        file_name = self.table.item(row_idx, col_indexes["file_name"]).text()
        full_path = os.path.join(file_path, file_name)

        data, sr = voice_activity.load_audio_startstop(full_path, (start, end))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_wav_path = tmp.name
        sf.write(temp_wav_path, data, sr)
        self.player.setSource(QUrl.fromLocalFile(temp_wav_path))
        self.player.play()
        self.play_all_button.setChecked(True)   # keep the button in “down” state
        self.stop_button.setEnabled(True)

    def stop_playback(self):
        self.player.stop()
        self.stop_button.setEnabled(False)
        self.play_all_button.setChecked(False)  # reset “Play All” appearance

    def on_playback_state_changed(self, state):
        if state != QMediaPlayer.PlayingState:
            self.stop_button.setEnabled(False)
            self.play_all_button.setChecked(False)   # un‑latch when audio stops

    def apply_keep(self):
        self.apply_label_to_current_detection(erase_flag=0)

    def apply_erase(self):
        self.apply_label_to_current_detection(erase_flag=1)

    def apply_label_to_current_detection(self, erase_flag):
        """
        Sets the 'erase' column to erase_flag (0=Keep, 1=Erase),
        marks the review_datetime,
        updates table + CSV,
        and (optionally) scrolls to the next detection.
        """
        if len(self.csv_data) == 0:
            return

        # Update the DataFrame
        self.csv_data.at[self.current_index, "erase"] = erase_flag
        self.csv_data.at[self.current_index, "review_datetime"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Temporarily block signals from the table 
        self.table.blockSignals(True)
        try:
            # Overwrite that row in the table with "Yes"/"No"
            #  1) Find the table column for "erase"
            erase_col = self.csv_data.columns.get_loc("erase")  
            #  2) Set the item text to Yes/No
            new_text = "Yes" if erase_flag == 1 else ""
            self.table.setItem(self.current_index, erase_col, QTableWidgetItem(new_text))

            # and the review datetime
            reviewdatetime_col = self.csv_data.columns.get_loc("review_datetime")
            self.table.setItem(self.current_index, reviewdatetime_col, QTableWidgetItem(self.csv_data.at[self.current_index, "review_datetime"]))
        finally:
            self.table.blockSignals(False)

        # Save to disk
        self.save_review(persist=True)

        # Auto-advance to the next detection for faster review
        self.scroll("next")

    def on_table_item_changed(self, item):
        """
        Triggered whenever a table cell is edited and the user commits the edit.
        1) Validate the edit.
        2) If valid, refresh the spectrogram.
        3) Save the latest changes to disk.
        """
        # Validate
        if self.validate_table_edits() is False:
            # Optionally revert the change if validation fails
            # or show a message to user. For now, we do nothing.
            return

        self.update_start_end_spinboxes(item.row())

        # Save changes (update df only, don't save to disk yet)
        self.save_review(persist=False) 

        # If valid, refresh the spectrogram
        self.refresh_spectrogram()

    def on_table_selection_changed(self):
        """
        Called whenever the user changes selected row(s), e.g. by arrow keys.
        We want to do the same action as a mouse-click selection.
        """
        # Get all selected rows. Often there's only one, but let's be safe:
        selected_rows = self.table.selectionModel().selectedRows()

        if not selected_rows:
            return

        # Just pick the first selected row
        row = selected_rows[0].row()
        self.select_detection(row)

    def validate_table_edits(self):
        """
        Perform validation on the table's current data.
        Return True if everything is valid, False if not.
        For now, always returns True.
        """
        # Example logic: always pass
        return True

    def filter_by_minimum_detection_len(self):
        """
        Removes rows from self.csv_data where (end_time - start_time) <= settings.minimum_detection_len
        and refreshes the table and spectrogram.
        """
        # Filter out detections shorter than the minimum length
        self.csv_data = self.csv_data[ (self.csv_data['end_time'] - self.csv_data['start_time']) > settings.minimum_detection_len ]

    def refresh_spectrogram(self):
        if len(self.csv_data) > 0:
            spectrogram, detection_start, detection_end, total_duration, audio_start, audio_end, file_name = self.load_audio()
            self.display_spectrogram(spectrogram, detection_start, detection_end, total_duration, audio_start, audio_end, file_name)
        
    def resizeEvent(self, event):
        self.resize_timer.start(100)  # Start/restart the timer with a delay
        self.spectrogram_label.setMinimumSize(1, 1)  # Reset the QLabel's minimum size

    def closeEvent(self, event):
        """
        Called automatically whenever the window is asked to close
        (user click, Alt‑F4, parent closes it, etc.).
        Ensures the latest table contents are flushed to disk.
        """
        # if the user was still typing in a cell, force‑commit the text
        if self.table.state() == QAbstractItemView.EditingState:
            current_item = self.table.currentItem()
            if current_item is not None:
                self.table.closePersistentEditor(current_item)
            self.table.clearFocus()              # also ends editing

        # persist the review file
        self.save_review(persist=True)

        # let the base‑class finish shutting the window down
        super().closeEvent(event)

    def load_audio(self):
        # 1) Build a dictionary of column indices by header name
        col_indexes = {}
        for col in range(self.table.columnCount()):
            header_text = self.table.horizontalHeaderItem(col).text()
            col_indexes[header_text] = col

        # 2) Retrieve the values from the table at the current_index
        row_idx = self.current_index
        file_path_col = col_indexes["file_path"]
        file_name_col = col_indexes["file_name"]
        start_col = col_indexes["start_time"]
        end_col   = col_indexes["end_time"]

        file_path = self.table.item(row_idx, file_path_col).text()
        file_name = self.table.item(row_idx, file_name_col).text()
        full_path = os.path.join(file_path, file_name)

        detection_start = float(self.table.item(row_idx, start_col).text())
        detection_end   = float(self.table.item(row_idx, end_col).text())
        
        
        # the total length of the detection
        detection_len = detection_end - detection_start

        # dpi = 96  # assuming a common DPI value
        window_width_in = self.width() / self.dpi
    
        # seconds per inch based on zoom level
        seconds_per_inch = self.zoom_level 
    
        # how many seconds long is this file?
        audio_duration = librosa.get_duration(path=full_path)
    
        # the clip must be exactly this long
        total_duration = math.ceil(window_width_in * seconds_per_inch)
        # total duration of audio to possibly load, not exceeding total length of the file
        load_duration = min(audio_duration, math.ceil(window_width_in * seconds_per_inch))
        
        # how many seconds before and after to load? assuming the gap can be loaded this centers the detection
        gap_size = (load_duration - detection_len) / 2

        # make the end a function of start
        adjust_start = 0
        if (detection_end + gap_size) > audio_duration:
            adjust_start = (detection_end + gap_size) - audio_duration

        # start and end times of audio to load
        audio_start = math.floor( max(0, detection_start - gap_size - adjust_start))
        audio_end = audio_start + load_duration

        # store for play-all functionality
        self.visible_audio_start = audio_start
        self.visible_audio_end = audio_end
        self.visible_file_path = full_path
    
        # load the audio data and compute the spectrogram
        data, sr = voice_activity.load_audio_startstop(full_path, start_stop = (audio_start, audio_end))

        # pad the audio to the full length if required
        if len(data) / sr != total_duration:
            temp = np.zeros((total_duration * sr,))
            temp[:len(data)] = data
            data = temp

        spectrogram = voice_activity.wav_to_spec(data, trim_edges=False)
    
        # this spectrogram covers the width of the window - the detection is only a sliceof that
        return spectrogram, detection_start - audio_start, detection_end - audio_start, total_duration, audio_start, (audio_start + total_duration), file_name

    def display_spectrogram(self, spectrogram, detection_start, detection_end, audio_duration, audio_start, audio_end, file_name):
        """
        spectrogram     : The raw spectrogram data.
        detection_start : Start of the detection, relative to the loaded clip.
        detection_end   : End of the detection, relative to the loaded clip.
        audio_duration  : Duration (in seconds) of the loaded audio clip.
        audio_start     : The position (in seconds) in the original audio where the clip starts.
        audio_end       : The position (in seconds) in the original audio where the clip ends.
        """
        # Convert the power spectrogram to dB
        spectrogram_db = librosa.amplitude_to_db(spectrogram**2, ref=np.max) 
        color_flipped_spec = np.abs(spectrogram_db)
        vmin_transformed = np.min(color_flipped_spec)
        vmax_transformed = np.max(color_flipped_spec)

        fig, ax1 = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=((self.width() / self.dpi), (self.height() / self.dpi / 2)),
            dpi=self.dpi
        )

        # Draw the spectrogram
        im1 = ax1.imshow(
            color_flipped_spec,
            origin='lower',
            aspect='auto',
            cmap='gray',
            vmin=vmin_transformed,
            vmax=vmax_transformed,
            extent=[audio_start, audio_end, 0, 11025]
        )

        # --------------------------------------------------------------------
        #  Plot all detections from the same file that overlap this time window
        # --------------------------------------------------------------------
        # Filter for same file
        file_detections = self.csv_data[self.csv_data['file_name'] == file_name]
        # Overlap if detection_end >= audio_start and detection_start <= audio_end
        in_window = file_detections[
            (file_detections['end_time'] >= audio_start) & 
            (file_detections['start_time'] <= audio_end)
        ]

        # For each detection in the window, plot a span
        for idx, row in in_window.iterrows():
            det_start = row['start_time']
            det_end   = row['end_time']

            # If this is the currently selected detection, draw in red
            if (
                abs(det_start - (detection_start + audio_start)) < 1e-6 and
                abs(det_end   - (detection_end   + audio_start)) < 1e-6
            ):
                ax1.axvspan(det_start, det_end, color='red', alpha=0.3)
            else:
                ax1.axvspan(det_start, det_end, color='blue', alpha=0.3)

        # -------------------------------------------------------------------------
        # Create half-second ticks (0.5s spacing)
        # -------------------------------------------------------------------------
        time_ticks = np.arange(
            np.floor(audio_start),
            np.ceil(audio_end) + 0.5,  # +0.5 ensures it includes the last interval
            0.5
        )

        # Only draw vertical lines if checkbox is checked
        if self.show_bars_checkbox.isChecked():
            for t_tick in time_ticks:
                ax1.axvline(x=t_tick, color='b', linestyle=':', alpha=0.3)

        # Set the ticks and labels at 0.5 second increments
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels([f"{t_tick:.1f}" for t_tick in time_ticks])
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        ax1.set_xlim(audio_start, audio_end)
        ax1.set_ylim(0, 11025)

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Frequency [Hz]')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)

        pixmap = QPixmap()
        pixmap.loadFromData(QByteArray(buf.getvalue()))

        self.spectrogram_label.setPixmap(
            pixmap.scaled(
                self.spectrogram_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def populate_table(self):
        # Clear existing rows/columns
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        
        # Sort detections by file_name, start_time
        self.csv_data.sort_values(by=['file_name', 'start_time'], ignore_index=True, inplace=True)
        
        # Round times to 3 decimal places
        self.csv_data[['start_time', 'end_time']] = self.csv_data[['start_time', 'end_time']].round(3)
        
        # Create columns
        self.table.setColumnCount(len(self.csv_data.columns))
        self.table.setHorizontalHeaderLabels(self.csv_data.columns)

        # Populate rows
        for i, row_series in self.csv_data.iterrows():
            self.table.insertRow(i)
            for j, col_name in enumerate(self.csv_data.columns):
                value = row_series[col_name]
                
                # Convert erase=0|1 -> “No”|“Yes”
                if col_name == "erase":
                    display_val = "Yes" if value == 1 else ""
                elif pd.isna(value):
                    display_val = ""  # For NaN (e.g. empty comment)
                else:
                    display_val = str(value)
                    
                cell_item = QTableWidgetItem(display_val)
                self.table.setItem(i, j, cell_item)

            review_dt = row_series.get("review_datetime", "")
            if pd.notna(review_dt) and str(review_dt) != "":
                color = QColor('lightblue')  # reviewed
            else:
                color = QColor('white')      # not reviewed

            for j in range(self.table.columnCount()):
                table_item = self.table.item(i, j)
                if table_item:
                    table_item.setBackground(color)

    def select_detection_from_table(self, row):
        clicked_row_idx = row.row()
        self.select_detection(clicked_row_idx)

    def highlight_all_rows(self):
        """
        Called after data changes or row selection changes.
        Paint all non-selected rows in white (unreviewed) or lightblue (reviewed).
        Let Qt's default highlight show for the selected row(s).
        """
        selected_rows = set(index.row() for index in self.table.selectionModel().selectedRows())
        
        # Temporarily block signals from the table
        self.table.blockSignals(True)
        for row_idx in range(self.table.rowCount()):
            # If row_idx is selected, let Qt handle the highlight color
            if row_idx in selected_rows:
                continue

            # reviewed if review_datetime is not empty/NaN
            review_val = self.csv_data.iloc[row_idx]['review_datetime']
            reviewed = pd.notna(review_val) and str(review_val) != ""
            row_color = QColor('lightblue') if reviewed else QColor('white')

            for col_idx in range(self.table.columnCount()):
                item = self.table.item(row_idx, col_idx)
                if item:
                    item.setBackground(row_color)
        self.table.blockSignals(False)

    def scroll(self, direction):
        if direction == "next":
            self.current_index += 1
        elif direction == "previous":
            self.current_index -= 1
        elif direction == "next_file":
            self.current_index += 10
        elif direction == "previous_file":
            self.current_index -= 10

        self.current_index = max(0, min(self.current_index, len(self.csv_data) - 1))
        self.table.selectRow(self.current_index)
        
        spectrogram, detection_start, detection_end, audio_duration, audio_start, audio_end, file_name = self.load_audio()
        self.display_spectrogram(spectrogram, detection_start, detection_end, audio_duration, audio_start, audio_end, file_name) 
        
        self.highlight_all_rows()

    def zoom(self, zoom_out):
        if zoom_out:
            if self.zoom_level == 0.5: # remove the half step if present
                self.zoom_level = 1
            else:
                self.zoom_level = self.zoom_level * 2
        else:
            # allowing for a bit of zoom in
            self.zoom_level = max(0.5, self.zoom_level - (self.zoom_level / 2))
        

        print(f'zoom level: {self.zoom_level}')

        spectrogram, detection_start, detection_end, audio_duration, audio_start, audio_end, file_name = self.load_audio()
        self.display_spectrogram(spectrogram, detection_start, detection_end, audio_duration, audio_start, audio_end, file_name)  

    def update_start_end_spinboxes(self, row_index):
        # refreshes the spinbox values given the selected row
        col_indexes = {}
        for col in range(self.table.columnCount()):
            header_text = self.table.horizontalHeaderItem(col).text()
            col_indexes[header_text] = col

        # get the start and end to play
        start_col = col_indexes["start_time"]
        end_col   = col_indexes["end_time"]
        detection_start = float(self.table.item(row_index, start_col).text())
        detection_end   = float(self.table.item(row_index, end_col).text())

        # Put those in the spinboxes
        self.start_spin.setValue(detection_start)
        self.stop_spin.setValue(detection_end)

    def select_detection(self, index):
        self.current_index = index

        self.update_start_end_spinboxes(self.current_index)

        # (Re)load and display spectrogram for that detection
        spectrogram, detection_start, detection_end, audio_duration, audio_start, audio_end, fname = self.load_audio()
        self.display_spectrogram(spectrogram, detection_start, detection_end, audio_duration, audio_start, audio_end, fname) 
        
    def highlight_row(self, i):
        for j in range(self.table.columnCount()):
            item = self.table.item(i, j)
            review_val = self.csv_data.iloc[i]['review_datetime']
            if pd.notna(review_val) and str(review_val) != "":
                item.setBackground(QColor('lightblue'))  # reviewed
            else:
                item.setBackground(QColor('white'))      # not reviewed

