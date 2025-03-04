# worker.py
from PySide6.QtCore import QObject, Signal

class WorkerSignals(QObject):
    fileProgressChanged = Signal(float)   # File-level progress [0..100]
    overallProgressChanged = Signal(float)# Overall progress [0..100]
    fileStarted = Signal(str)             # Name/path of file that just started
    fileDone = Signal(str)                # Name/path of file that just finished
    finished = Signal()                   # All files done
    message = Signal(str)                 # For status updates/logging

# worker.py
import time
import numpy as np
import pandas as pd
from PySide6.QtCore import QRunnable, QThreadPool

from root.code.backend.voice_activity import load_audio
from root.code.backend import settings

class ProcessWorker(QRunnable):
    """
    Worker that processes audio files in a background thread.
    Signals UI updates via WorkerSignals.
    """
    def __init__(self, detector, detection_project, planned_work, parent=None):
        super().__init__()
        self.signals = WorkerSignals()
        self.detector = detector
        self.detection_project = detection_project  # for saving
        self.planned_work = planned_work            # dict: { filepath: [indexes] }
        self.stop_requested = False

    def stop(self):
        """Let main thread request that we stop."""
        self.stop_requested = True

    def run(self):
        """
        Process each file in self.planned_work.
        - For each file, run NN detection in batches.
        - Update file-level progress each batch.
        - Save partial results after each file is fully processed.
        - If self.stop_requested is True, exit early.
        """
        total_files = len(self.planned_work)
        files_done = 0

        for ii, file in enumerate(self.planned_work.keys()):
            if self.stop_requested:
                break

            # Let UI know we started this file
            self.signals.fileStarted.emit(file)

            # Load audio (with extra 3s padding as in your original code)
            audio_data, original_sr = load_audio(file)
            sample_rate = settings.vad_resample
            padding_samples = sample_rate * 3
            padded = np.zeros(len(audio_data) + 2*padding_samples, dtype=audio_data.dtype)
            padded[padding_samples:padding_samples+len(audio_data)] = audio_data
            audio_data = padded

            # Plan batch processing
            indexes = self.planned_work[file]
            total_work_count = len(indexes)
            prediction_batch_size = settings.prediction_batch_size
            batch_predictions = []

            # For progress calculation
            for start_idx in range(0, total_work_count, prediction_batch_size):
                if self.stop_requested:
                    break
                end_idx = min(start_idx + prediction_batch_size, total_work_count)
                batch_indexes = indexes[start_idx:end_idx]

                # Perform batch inference
                speech_pred, mask_pred = self.detector.process_batch(audio_data, batch_indexes)
                batch_predictions.append(mask_pred)

                # Update file-level progress
                done = end_idx
                file_progress = (done / total_work_count) * 100.0
                self.signals.fileProgressChanged.emit(file_progress)

            if self.stop_requested:
                break

            # Post-process (averaging, speech regions, etc.)
            audio_length_seconds = len(audio_data) / sample_rate
            if len(batch_predictions) > 0:
                avg = self.detector.average_overlapping_detections({ file: np.vstack(batch_predictions) }, audio_length_seconds)
            else:
                avg = self.detector.average_overlapping_detections({ file: np.array([]) }, audio_length_seconds)

            # TODO: make the break duration configurable
            speech_regions = self.detector.find_speech_regions({ file: avg }, break_duration = 0.5)

            # Adjust times to remove the 3s offset
            speech_regions[file] = [(float(start)-3, float(end)-3) for (start,end) in speech_regions[file]]

            # Append results to detection_project DataFrame
            from os.path import dirname, basename
            file_path = dirname(file)
            file_name = basename(file)

            for (start_time, end_time) in speech_regions[file]:
                new_row = {
                    'file_path': file_path,
                    'file_name': file_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'erase': 0,
                    'user_comment': '',
                    'review_datetime': pd.Timestamp('1899-12-31')
                }
                self.detection_project.df.loc[len(self.detection_project.df)] = new_row

            # Save partial results so we can resume later
            self.detection_project.save_detections()

            # Let UI know the file is completely done
            self.signals.fileDone.emit(file)

            # Update overall progress
            files_done += 1
            overall_progress = (files_done / total_files) * 100.0
            self.signals.overallProgressChanged.emit(overall_progress)

        # If we completed the loop normally or early-exited:
        self.signals.finished.emit()
