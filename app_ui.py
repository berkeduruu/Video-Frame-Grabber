import sys
import os
import cv2
import re 
import zipfile # Added for ZIP compression
import tarfile  # NEW: Added for .tar.gz compression
import io       # NEW: Added for in-memory file archiving
import time     # NEW: Added for tarfile timestamps
import subprocess # NEW: Added for opening the folder
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLineEdit, QSlider, QLabel, QFileDialog, QTextEdit,
    QGridLayout, QStatusBar, QFrame, QCheckBox, QComboBox, 
    QStackedWidget, QProgressBar 
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QUrl, QEvent
from PyQt6.QtGui import QIcon, QDropEvent
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from concurrent.futures import ThreadPoolExecutor, as_completed
from extractor import extract_specific_frame 

# --- Helper Functions (No Changes) ---
def format_ms_to_time(ms):
    """Converts milliseconds to 'mm:ss:zzz' format."""
    seconds = int(ms / 1000)
    milliseconds = int(ms % 1000)
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

def parse_time_to_ms(time_str):
    """
    Converts 'mm:ss:zzz', 'mm:ss' or 'ss' formats to milliseconds.
    Returns None on error.
    """
    time_str = time_str.strip()
    
    # Format: mm:ss:zzz
    match = re.fullmatch(r'(\d+):(\d{1,2}):(\d{1,3})', time_str)
    if match:
        m, s, zzz = map(int, match.groups())
        return (m * 60 * 1000) + (s * 1000) + zzz

    # Format: mm:ss
    match = re.fullmatch(r'(\d+):(\d{1,2})', time_str)
    if match:
        m, s = map(int, match.groups())
        return (m * 60 * 1000) + (s * 1000)

    # Format: ss (just seconds)
    match = re.fullmatch(r'(\d+)', time_str)
    if match:
        s = int(match.group(1))
        return s * 1000
        
    return None # Invalid format

# --- Custom VideoDropArea Widget (No Changes) ---
class VideoDropArea(QStackedWidget):
    """
    A custom QStackedWidget that handles drag-and-drop events.
    It emits a signal 'fileDropped' when a valid file is dropped.
    It also filters events from its child QVideoWidget to ensure
    drag-and-drop works even when the video is playing.
    """
    fileDropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True) 
        self.video_widget_ref = None # A reference to the QVideoWidget

    def setVideoWidget(self, video_widget: QVideoWidget):
        """
        Sets the QVideoWidget to be monitored for drop events.
        """
        self.video_widget_ref = video_widget
        # Install 'self' (this VideoDropArea) as an event filter
        # on the QVideoWidget.
        if self.video_widget_ref:
            self.video_widget_ref.installEventFilter(self)

    def eventFilter(self, obj, event: QEvent):
        """
        Intercepts events from the monitored object (self.video_widget_ref).
        """
        if obj is self.video_widget_ref:
            if event.type() == QEvent.Type.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                else:
                    event.ignore()
                return True 
            if event.type() == QEvent.Type.Drop:
                urls = event.mimeData().urls()
                if urls:
                    file_path = urls[0].toLocalFile()
                    if os.path.isfile(file_path):
                        self.fileDropped.emit(file_path) 
                        event.accept()
                else:
                    event.ignore()
                return True 
        return super().eventFilter(obj, event)

    def dragEnterEvent(self, event: QDropEvent):
        """Handles the drag enter event for the stack itself."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        """Handles the drop event for the stack itself."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isfile(file_path):
                self.fileDropped.emit(file_path) # Emit the signal
                event.accept()
        else:
            event.ignore()


# --- ExtractionWorker (Background Thread) ---
# Updated to handle archive formats (ZIP/TAR.GZ)
class ExtractionWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str) 
    progress_percent = pyqtSignal(int) # Progress percentage signal
    
    def __init__(self, video_path, output_dir, mode_config):
        super().__init__()
        self.video_path = video_path 
        self.output_dir = output_dir 
        self.mode_config = mode_config 
        self.video_base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.is_running = True
        
        # Get settings
        self.output_format = self.mode_config.get("output_format", "png")
        self.jpg_quality = self.mode_config.get("jpg_quality", 90)
        
        # Get archive format
        self.archive_format = self.mode_config.get("archive_format", "No Compression (Folder)")
        self.archive_file_name = ""
        self.archive_file_path = ""
        
        # Prepare archive file name if needed
        if self.archive_format == "ZIP (.zip)":
            self.archive_file_name = f"{self.video_base_name}_extracted_frames.zip"
            self.archive_file_path = os.path.join(self.output_dir, self.archive_file_name)
        elif self.archive_format == "TAR.GZ (.tar.gz)":
            self.archive_file_name = f"{self.video_base_name}_extracted_frames.tar.gz"
            self.archive_file_path = os.path.join(self.output_dir, self.archive_file_name)


    @pyqtSlot()
    def run(self):
        mode = self.mode_config.get("mode")
        try:
            self.progress_percent.emit(0) # Start at 0%
            if mode == "all_frames":
                self.progress.emit("Mode: 'Save All Frames' started.")
                self.run_all_frames()
            elif mode == "timestamp":
                self.progress.emit("Mode: 'Timestamp' started.")
                self.run_timestamp()
            else:
                self.progress.emit(f"Error: Invalid run mode '{mode}'.")
        except Exception as e:
            self.progress.emit(f"Critical Error: {e}")
            
        if self.is_running:
            self.progress_percent.emit(100) # End at 100%
            self.progress.emit(f"--- Extraction Finished ---")
        else:
            self.progress_percent.emit(0) # Reset on cancel
            self.progress.emit(f"--- Extraction Canceled ---")
        self.finished.emit()

    def run_all_frames(self):
        """Reads every frame in the range and saves to folder or archive."""
        start_ms = self.mode_config.get("start_ms")
        end_ms = self.mode_config.get("end_ms")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.progress.emit(f"Error: Could not open video {self.video_base_name}")
            return
        
        if end_ms == float('inf'):
            video_duration_ms = cap.get(cv2.CAP_PROP_FRAME_COUNT) * (1000 / cap.get(cv2.CAP_PROP_FPS))
            end_ms = video_duration_ms
            self.progress.emit(f"End time auto-detected: {format_ms_to_time(end_ms)}")

        total_duration = end_ms - start_ms
        if total_duration <= 0: total_duration = 1 
            
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        saved_count = 0
        
        try:
            # Block for writing to ZIP
            if self.archive_format == "ZIP (.zip)":
                self.progress.emit(f"Creating ZIP file: {self.archive_file_name}")
                with zipfile.ZipFile(self.archive_file_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive_f:
                    while self.is_running:
                        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        if current_ms > end_ms:
                            self.progress.emit("End time reached.")
                            break
                        ret, frame = cap.read()
                        if not ret:
                            self.progress.emit("End of video or read error.")
                            break
                        saved_count += 1
                        
                        encode_params = []
                        if self.output_format == "jpg":
                            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                        
                        ret, img_buffer = cv2.imencode(f'.{self.output_format}', frame, encode_params)
                        if not ret:
                            self.progress.emit(f"Error: Failed to encode frame {saved_count}.")
                            continue
                        
                        output_filename_in_zip = f"{self.video_base_name}_{saved_count:05d}.{self.output_format}"
                        archive_f.writestr(output_filename_in_zip, img_buffer.tobytes())
                        
                        percent = int(((current_ms - start_ms) / total_duration) * 100)
                        self.progress_percent.emit(percent)
                        
                        if saved_count % 100 == 0:
                            self.progress.emit(f"{saved_count} frames added to ZIP... ({format_ms_to_time(current_ms)})")
            
            # Block for writing to TAR.GZ
            elif self.archive_format == "TAR.GZ (.tar.gz)":
                self.progress.emit(f"Creating TAR.GZ file: {self.archive_file_name}")
                with tarfile.open(self.archive_file_path, 'w:gz') as archive_f:
                    while self.is_running:
                        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        if current_ms > end_ms:
                            self.progress.emit("End time reached.")
                            break
                        ret, frame = cap.read()
                        if not ret:
                            self.progress.emit("End of video or read error.")
                            break
                        saved_count += 1

                        encode_params = []
                        if self.output_format == "jpg":
                            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                        
                        ret, img_buffer = cv2.imencode(f'.{self.output_format}', frame, encode_params)
                        if not ret:
                            self.progress.emit(f"Error: Failed to encode frame {saved_count}.")
                            continue

                        output_filename_in_archive = f"{self.video_base_name}_{saved_count:05d}.{self.output_format}"
                        
                        img_bytes = img_buffer.tobytes()
                        img_stream = io.BytesIO(img_bytes)
                        
                        tar_info = tarfile.TarInfo(name=output_filename_in_archive)
                        tar_info.size = len(img_bytes)
                        tar_info.mtime = int(time.time())
                        
                        archive_f.addfile(tarinfo=tar_info, fileobj=img_stream)
                        
                        percent = int(((current_ms - start_ms) / total_duration) * 100)
                        self.progress_percent.emit(percent)
                        
                        if saved_count % 100 == 0:
                            self.progress.emit(f"{saved_count} frames added to TAR.GZ... ({format_ms_to_time(current_ms)})")

            # Block for writing to Folder ("No Compression (Folder)")
            else:
                while self.is_running:
                    current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if current_ms > end_ms:
                        self.progress.emit("End time reached.")
                        break
                    ret, frame = cap.read()
                    if not ret:
                        self.progress.emit("End of video or read error.")
                        break
                    
                    saved_count += 1
                    output_filename = f"{self.video_base_name}_{saved_count:05d}.{self.output_format}"
                    output_path = os.path.join(self.output_dir, output_filename)
                    
                    encode_params = []
                    if self.output_format == "jpg":
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                    cv2.imwrite(output_path, frame, encode_params)
                    
                    percent = int(((current_ms - start_ms) / total_duration) * 100)
                    self.progress_percent.emit(percent)
                    
                    if saved_count % 100 == 0:
                        self.progress.emit(f"{saved_count} frames saved to folder... ({format_ms_to_time(current_ms)})")
        
        except Exception as e:
            self.progress.emit(f"Frame save error: {e}")
        finally:
            cap.release()
            
        if not self.is_running:
            self.progress.emit("Operation canceled by user.")
        else:
            if self.archive_format != "No Compression (Folder)":
                self.progress.emit(f"Total {saved_count} frames saved to '{self.archive_file_name}'.")
            else:
                self.progress.emit(f"Total {saved_count} frames saved to folder.")


    def run_timestamp(self):
        """Processes timestamp list in parallel, filters, and saves to folder or archive."""
        timestamps = self.mode_config.get("timestamps")
        check_duplicates = self.mode_config.get("check_duplicates", False)
        total_tasks = len(timestamps)
        if total_tasks == 0: total_tasks = 1 
        
        self.progress.emit(f"Processing {total_tasks} timestamps.")
        if check_duplicates:
            self.progress.emit("Skipping duplicate frames.")

        results = [] 
        
        # Step 1: Capture frames in parallel and store in memory
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            future_to_index = {
                executor.submit(extract_specific_frame, self.video_path, ts): i 
                for i, ts in enumerate(timestamps)
            }
            
            tasks_done = 0
            for future in as_completed(future_to_index):
                if not self.is_running:
                    self.progress.emit("Canceling... Stopping worker threads.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                index = future_to_index[future]
                try:
                    success, msg, frame, frame_num = future.result()
                    results.append((index, success, msg, frame, frame_num))
                except Exception as e:
                    results.append((index, False, f"Thread error: {e}", None, -1))
                
                tasks_done += 1
                percent = int((tasks_done / total_tasks) * 50) # Capture is 50% of the job
                self.progress_percent.emit(percent)

        if not self.is_running:
            self.progress.emit("Operation canceled.")
            return

        self.progress.emit("All frames captured. Sorting and filtering...")
        
        # Step 2: Sort, filter, and save (to ZIP, TAR.GZ, or folder)
        results.sort(key=lambda x: x[0])
        
        last_saved_frame_num = -1
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        try:
            # Block for writing to ZIP
            if self.archive_format == "ZIP (.zip)":
                self.progress.emit(f"Creating ZIP file: {self.archive_file_name}")
                with zipfile.ZipFile(self.archive_file_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive_f:
                    for i, (index, success, msg, frame, frame_num) in enumerate(results):
                        if not self.is_running: break
                        if not success:
                            fail_count += 1
                            self.progress.emit(f"Error (Frame {index+1}): {msg}")
                            continue
                        if check_duplicates and frame_num == last_saved_frame_num:
                            skip_count += 1
                            continue 
                        
                        success_count += 1
                        
                        encode_params = []
                        if self.output_format == "jpg":
                            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                        
                        ret, img_buffer = cv2.imencode(f'.{self.output_format}', frame, encode_params)
                        if not ret:
                            self.progress.emit(f"Error: Failed to encode frame {success_count}.")
                            fail_count += 1
                            continue
                        
                        output_filename_in_zip = f"{self.video_base_name}_{success_count:05d}.{self.output_format}" 
                        archive_f.writestr(output_filename_in_zip, img_buffer.tobytes())
                        
                        last_saved_frame_num = frame_num 
                        
                        percent = 50 + int((i / total_tasks) * 50) # Save is other 50%
                        self.progress_percent.emit(percent)
                        
                        if success_count % 100 == 0:
                            self.progress.emit(f"{success_count} frames added to ZIP...")

            # Block for writing to TAR.GZ
            elif self.archive_format == "TAR.GZ (.tar.gz)":
                self.progress.emit(f"Creating TAR.GZ file: {self.archive_file_name}")
                with tarfile.open(self.archive_file_path, 'w:gz') as archive_f:
                    for i, (index, success, msg, frame, frame_num) in enumerate(results):
                        if not self.is_running: break
                        if not success:
                            fail_count += 1
                            self.progress.emit(f"Error (Frame {index+1}): {msg}")
                            continue
                        if check_duplicates and frame_num == last_saved_frame_num:
                            skip_count += 1
                            continue 
                        
                        success_count += 1
                        
                        encode_params = []
                        if self.output_format == "jpg":
                            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]

                        ret, img_buffer = cv2.imencode(f'.{self.output_format}', frame, encode_params)
                        if not ret:
                            self.progress.emit(f"Error: Failed to encode frame {success_count}.")
                            fail_count += 1
                            continue

                        output_filename_in_archive = f"{self.video_base_name}_{success_count:05d}.{self.output_format}"
                        
                        img_bytes = img_buffer.tobytes()
                        img_stream = io.BytesIO(img_bytes)
                        
                        tar_info = tarfile.TarInfo(name=output_filename_in_archive)
                        tar_info.size = len(img_bytes)
                        tar_info.mtime = int(time.time())
                        
                        archive_f.addfile(tarinfo=tar_info, fileobj=img_stream)
                        
                        last_saved_frame_num = frame_num 
                        
                        percent = 50 + int((i / total_tasks) * 50) # Save is other 50%
                        self.progress_percent.emit(percent)
                        
                        if success_count % 100 == 0:
                            self.progress.emit(f"{success_count} frames added to TAR.GZ...")
            
            # Block for writing to Folder ("No Compression (Folder)")
            else:
                for i, (index, success, msg, frame, frame_num) in enumerate(results):
                    if not self.is_running: break
                    if not success:
                        fail_count += 1
                        self.progress.emit(f"Error (Frame {index+1}): {msg}")
                        continue
                    if check_duplicates and frame_num == last_saved_frame_num:
                        skip_count += 1
                        continue 
                    
                    success_count += 1
                    output_filename = f"{self.video_base_name}_{success_count:05d}.{self.output_format}" 
                    output_path = os.path.join(self.output_dir, output_filename)
                    
                    encode_params = []
                    if self.output_format == "jpg":
                        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                    cv2.imwrite(output_path, frame, encode_params)
                    
                    last_saved_frame_num = frame_num 
                    
                    percent = 50 + int((i / total_tasks) * 50) # Save is other 50%
                    self.progress_percent.emit(percent)
                    
                    if success_count % 100 == 0:
                        self.progress.emit(f"{success_count} frames saved to folder...")
        
        except Exception as e:
            self.progress.emit(f"Frame save error: {e}")

        self.progress.emit(f"--- Timestamp Mode Finished ---")
        self.progress.emit(f"Successful: {success_count}")
        self.progress.emit(f"Failed: {fail_count}")
        self.progress.emit(f"Skipped (Duplicate): {skip_count}")
        if self.archive_format != "No Compression (Folder)" and success_count > 0:
             self.progress.emit(f"All frames saved to '{self.archive_file_name}'.")


    def stop(self):
        self.progress.emit("Stop signal received.")
        self.is_running = False

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Frame Grabber")
        self.setGeometry(100, 100, 900, 800)
        
        self.video_path = None 
        self.output_dir = None
        self.video_duration_ms = 0
        self.is_updating_ui_from_code = False
        
        self.worker_thread = None
        self.extraction_worker = None
        
        self.player = QMediaPlayer() 
        
        self.player.positionChanged.connect(self.sync_position_to_ui)
        self.player.durationChanged.connect(self.sync_duration_to_ui)
        self.player.playbackStateChanged.connect(self.sync_play_button_icon)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # --- 1. File Selection ---
        file_layout = QGridLayout()
        
        self.load_video_btn = QPushButton("1. Load Video")
        self.video_path_label = QLabel("No video loaded.")
        
        self.load_folder_btn = QPushButton("2. Select Output Folder")
        self.output_dir_label = QLabel("No folder selected.")
        
        self.open_output_btn = QPushButton("Open Folder")
        self.open_output_btn.setEnabled(False) 
        
        file_layout.addWidget(self.load_video_btn, 0, 0)
        file_layout.addWidget(self.video_path_label, 0, 1, 1, 2)
        file_layout.addWidget(self.load_folder_btn, 1, 0)
        file_layout.addWidget(self.output_dir_label, 1, 1)
        file_layout.addWidget(self.open_output_btn, 1, 2)
        
        main_layout.addLayout(file_layout)
        
        # --- 2. Video Preview Area ---
        
        self.video_stack = VideoDropArea()
        
        self.video_placeholder = QLabel("Drag and drop a video file here\n\nor\n\nuse the 'Load Video' button")
        self.video_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_placeholder.setStyleSheet("""
            QLabel {
                border: 2px dashed #888;
                border-radius: 5px;
                background-color: #2E2E2E;
                color: #AAA;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        
        self.video_widget = QVideoWidget()
        
        self.player.setVideoOutput(self.video_widget) 
        
        self.video_stack.addWidget(self.video_placeholder) # Index 0
        self.video_stack.addWidget(self.video_widget)      # Index 1
        
        self.video_stack.setVideoWidget(self.video_widget)
        
        self.video_stack.setCurrentIndex(0) 
        
        main_layout.addWidget(self.video_stack, 2) # Stretch factor = 2
        self.video_stack.setMinimumHeight(300)

        
        # --- 3. Playback Controls ---
        player_controls_layout = QHBoxLayout()
        self.play_pause_btn = QPushButton("Play (▶)")
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_label = QLabel("00:00:000")
        
        player_controls_layout.addWidget(self.play_pause_btn)
        player_controls_layout.addWidget(self.position_slider)
        player_controls_layout.addWidget(self.position_label)
        main_layout.addLayout(player_controls_layout)

        # Separator Line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        # --- 4. Time Range Selection ---
        time_select_layout = QGridLayout()
        
        self.start_slider_label = QLabel("Start:")
        self.start_slider = QSlider(Qt.Orientation.Horizontal)
        self.start_time_input = QLineEdit("00:00:000") 
        self.set_start_btn = QPushButton("Set Current") 
        
        time_select_layout.addWidget(self.start_slider_label, 0, 0)
        time_select_layout.addWidget(self.start_slider, 0, 1)
        time_select_layout.addWidget(self.start_time_input, 0, 2)
        time_select_layout.addWidget(self.set_start_btn, 0, 3)
        
        self.end_slider_label = QLabel("End:")
        self.end_slider = QSlider(Qt.Orientation.Horizontal)
        self.end_time_input = QLineEdit("00:00:000") 
        self.set_end_btn = QPushButton("Set Current") 
        
        time_select_layout.addWidget(self.end_slider_label, 1, 0)
        time_select_layout.addWidget(self.end_slider, 1, 1)
        time_select_layout.addWidget(self.end_time_input, 1, 2)
        time_select_layout.addWidget(self.set_end_btn, 1, 3)
        
        main_layout.addLayout(time_select_layout)
        
        # --- 5. Extraction Mode ---
        mode_layout = QGridLayout() 
        
        self.frames_input = QLineEdit()
        self.frames_input.setPlaceholderText("Total Frames (Mode A)...")
        self.interval_input = QLineEdit()
        self.interval_input.setPlaceholderText("Or Interval (ms) (Mode B)...")
        
        mode_layout.addWidget(self.frames_input, 0, 0)
        mode_layout.addWidget(self.interval_input, 0, 1)
        
        self.all_frames_check = QCheckBox("Save All Frames (Ignore A/B)")
        self.all_frames_check.setToolTip("If checked, ignores Mode A/B and saves ALL frames in the selected range.")
        mode_layout.addWidget(self.all_frames_check, 1, 0)
        
        self.no_duplicates_check = QCheckBox("Skip Duplicate Frames")
        self.no_duplicates_check.setToolTip("Prevents saving the same frame multiple times if the interval is too small.\nNot needed for 'Save All Frames' mode.")
        self.no_duplicates_check.setChecked(True) 
        mode_layout.addWidget(self.no_duplicates_check, 1, 1)
        
        self.output_format_label = QLabel("Image Format:")
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["png", "jpg", "webp", "bmp"])
        
        mode_layout.addWidget(self.output_format_label, 2, 0)
        mode_layout.addWidget(self.output_format_combo, 2, 1)
        
        self.jpg_quality_label = QLabel("JPG Quality (%):")
        self.jpg_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.jpg_quality_slider.setRange(1, 100)
        self.jpg_quality_slider.setValue(90)
        self.jpg_quality_value_label = QLabel("90%")
        
        mode_layout.addWidget(self.jpg_quality_label, 3, 0)
        mode_layout.addWidget(self.jpg_quality_slider, 3, 1)
        mode_layout.addWidget(self.jpg_quality_value_label, 3, 2)
        
        # Hide initially
        self.jpg_quality_label.setVisible(False)
        self.jpg_quality_slider.setVisible(False)
        self.jpg_quality_value_label.setVisible(False)
        
        self.archive_format_label = QLabel("Output Archive:")
        self.archive_format_combo = QComboBox()
        self.archive_format_combo.addItems([
            "No Compression (Folder)",
            "ZIP (.zip)",
            "TAR.GZ (.tar.gz)"
        ])
        self.archive_format_combo.setToolTip("Saves frames into a single compressed archive.\n'Folder' option saves each frame as a separate file.")
        mode_layout.addWidget(self.archive_format_label, 4, 0)
        mode_layout.addWidget(self.archive_format_combo, 4, 1) 
        
        main_layout.addLayout(mode_layout)
        
        # --- 6. Start/Cancel Buttons and Log Area ---
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("3. START EXTRACTION") 
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.start_btn.setEnabled(False) 
        button_layout.addWidget(self.start_btn, 2) 
        
        self.cancel_btn = QPushButton("CANCEL")
        self.cancel_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.cancel_btn.setEnabled(False) 
        button_layout.addWidget(self.cancel_btn, 1) 
        
        main_layout.addLayout(button_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Processing... %p%")
        main_layout.addWidget(self.progress_bar)
        
        
        log_header_layout = QHBoxLayout()
        log_header_layout.addWidget(QLabel("Process Log:"))
        log_header_layout.addStretch(1)
        self.clear_log_btn = QPushButton("Clear Log")
        log_header_layout.addWidget(self.clear_log_btn)
        main_layout.addLayout(log_header_layout)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        main_layout.addWidget(self.log_area, 1) 
        
        # Footer layout
        footer_layout = QHBoxLayout()
        credit_label = QLabel("Developed by Berke Duru (berkeduruu@gmail.com)")
        credit_label.setStyleSheet("color: #888; font-size: 9pt;")
        
        feature_label = QLabel("For feature requests, contact the developer.")
        feature_label.setStyleSheet("color: #888; font-size: 9pt;")
        
        footer_layout.addWidget(credit_label)
        footer_layout.addStretch(1)
        footer_layout.addWidget(feature_label)
        
        main_layout.addLayout(footer_layout)
        
        self.setStatusBar(QStatusBar(self))
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # --- Signal Connections ---
        self.load_video_btn.clicked.connect(self.load_video_dialog) 
        self.video_stack.fileDropped.connect(self.process_video_path_from_drop) 
        
        self.load_folder_btn.clicked.connect(self.select_output_dir)
        self.start_btn.clicked.connect(self.start_extraction)
        self.cancel_btn.clicked.connect(self.cancel_extraction) 
        
        self.play_pause_btn.clicked.connect(self.toggle_play_pause) 
        self.position_slider.sliderMoved.connect(self.player.setPosition) 
        
        self.set_start_btn.clicked.connect(self.set_start_from_player) 
        self.set_end_btn.clicked.connect(self.set_end_from_player) 
        
        self.start_slider.valueChanged.connect(self.sync_start_text_from_slider) 
        self.end_slider.valueChanged.connect(self.sync_end_text_from_slider) 
        
        self.start_time_input.editingFinished.connect(self.sync_start_slider_from_text) 
        self.end_time_input.editingFinished.connect(self.sync_end_slider_from_text) 
        
        self.all_frames_check.toggled.connect(self.toggle_mode_inputs)
        
        self.output_format_combo.currentTextChanged.connect(self.toggle_jpg_quality_slider)
        self.jpg_quality_slider.valueChanged.connect(self.update_jpg_quality_label)
        
        self.open_output_btn.clicked.connect(self.open_output_directory)
        self.clear_log_btn.clicked.connect(self.log_area.clear)
        
    def toggle_mode_inputs(self, checked):
        """Enables/disables inputs based on 'All Frames' checkbox."""
        if checked:
            self.frames_input.setEnabled(False)
            self.interval_input.setEnabled(False)
            self.no_duplicates_check.setEnabled(False)
            self.frames_input.setText("")
            self.interval_input.setText("")
        else:
            self.frames_input.setEnabled(True)
            self.interval_input.setEnabled(True)
            self.no_duplicates_check.setEnabled(True)

    def log(self, message):
        """Appends a message to the log area."""
        self.log_area.append(message)
        
    def check_if_ready(self):
        """Enables the start button if video and output dir are set."""
        if self.video_path and self.output_dir:
            self.start_btn.setEnabled(True)
            self.statusBar().showMessage("Ready to start.", 5000)
        else:
            self.start_btn.setEnabled(False)
            
    # --- Event Handlers and Slots ---
    
    @pyqtSlot(str)
    def process_video_path_from_drop(self, path):
        """Slot to handle the 'fileDropped' signal."""
        self.log(f"File received via Drag-Drop: {path}")
        if not self.process_video_path(path):
            self.log(f"Drag-Drop failed. Unsupported file: {path}")
            
    def load_video_dialog(self):
        """Opens a file dialog to select a single video."""
        video_filter = "Video Files (*.mp4 *.avi *.mkv *.mov *.webm *.flv *.wmv *.mpeg);;All Files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", video_filter)
        if path:
            self.log(f"Video loaded from dialog: {path}")
            self.process_video_path(path) 

    def process_video_path(self, path):
        """Loads and validates the video file from the given path."""
        if not path or not os.path.isfile(path):
            return False
            
        self.player.stop()
            
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.log(f"Error: {path} is not a valid video file (OpenCV read failed).")
            cap.release()
            return False
        
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.video_duration_ms = (frame_count * 1000) / fps
            else:
                self.video_duration_ms = 0
        except Exception:
            self.video_duration_ms = 0
            
        cap.release()
        
        self.video_path = path
        self.video_path_label.setText(os.path.basename(path))
        
        self.player.setSource(QUrl.fromLocalFile(self.video_path))
        
        self.video_stack.setCurrentIndex(1) 
        
        self.check_if_ready()
        return True # Success

    def select_output_dir(self):
        """Opens a dialog to select an output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_dir = path
            self.output_dir_label.setText(os.path.basename(path))
            self.log(f"Output folder set: {path}")
            self.check_if_ready()
            self.open_output_btn.setEnabled(True) 
            
    # --- Playback Control Slots ---
            
    def toggle_play_pause(self):
        """Toggles the media player state between playing and paused."""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()
            
    def sync_play_button_icon(self, state):
        """Updates the play/pause button text based on player state."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_btn.setText("Pause (❚❚)")
        else:
            self.play_pause_btn.setText("Play (▶)")

    def sync_position_to_ui(self, position_ms):
        """Updates the position slider and label as the video plays."""
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position_ms)
        self.position_label.setText(format_ms_to_time(position_ms))

    def sync_duration_to_ui(self, duration_ms):
        """Sets the range of all sliders when video duration is known."""
        self.video_duration_ms = duration_ms
        self.log(f"Video duration: {format_ms_to_time(duration_ms)}")
        
        self.position_slider.setRange(0, duration_ms)
        self.start_slider.setRange(0, duration_ms)
        self.end_slider.setRange(0, duration_ms)
        
        self.start_slider.setValue(0)
        self.start_time_input.setText(format_ms_to_time(0))
        self.end_slider.setValue(duration_ms)
        self.end_time_input.setText(format_ms_to_time(duration_ms))

    def set_start_from_player(self):
        """Sets the start time slider to the current player position."""
        current_pos_ms = self.player.position()
        self.start_slider.setValue(current_pos_ms)

    def set_end_from_player(self):
        """Sets the end time slider to the current player position."""
        current_pos_ms = self.player.position()
        self.end_slider.setValue(current_pos_ms)

    # --- Time Sync Slots ---

    def sync_start_text_from_slider(self, ms_value):
        """Updates the start time text input when the slider moves."""
        self.is_updating_ui_from_code = True
        self.start_time_input.setText(format_ms_to_time(ms_value))
        self.is_updating_ui_from_code = False
        if ms_value > self.end_slider.value():
            self.end_slider.setValue(ms_value)

    def sync_end_text_from_slider(self, ms_value):
        """Updates the end time text input when the slider moves."""
        self.is_updating_ui_from_code = True
        self.end_time_input.setText(format_ms_to_time(ms_value))
        self.is_updating_ui_from_code = False
        if ms_value < self.start_slider.value():
            self.start_slider.setValue(ms_value)

    def sync_start_slider_from_text(self):
        """Updates the start time slider when the text input is edited."""
        if self.is_updating_ui_from_code:
            return
        time_str = self.start_time_input.text()
        ms_value = parse_time_to_ms(time_str)
        if ms_value is None:
            self.log(f"Invalid start time format: {time_str}")
            self.sync_start_text_from_slider(self.start_slider.value())
        else:
            ms_value = max(0, min(ms_value, self.video_duration_ms))
            self.start_slider.setValue(ms_value)

    def sync_end_slider_from_text(self):
        """Updates the end time slider when the text input is edited."""
        if self.is_updating_ui_from_code:
            return
        time_str = self.end_time_input.text()
        ms_value = parse_time_to_ms(time_str)
        
        if time_str.strip() == "":
            ms_value = self.video_duration_ms
        elif ms_value is None:
            self.log(f"Invalid end time format: {time_str}")
            ms_value = self.end_slider.value() # Revert to slider
        
        ms_value = max(0, min(ms_value, self.video_duration_ms))
        if ms_value < self.start_slider.value():
            ms_value = self.start_slider.value() # Cannot be before start
            
        self.end_slider.setValue(int(ms_value))
        self.sync_end_text_from_slider(int(ms_value)) # Update text to match

    # --- Extraction Logic ---

    def start_extraction(self):
        """Validates inputs and starts the ExtractionWorker thread for ONE video."""
        
        start_ms = self.start_slider.value()
        end_ms = self.end_slider.value()
        
        if self.end_time_input.text().strip() == "":
            end_ms = float('inf')
        
        if end_ms != float('inf') and end_ms <= start_ms:
            self.log("Error: End time must be after start time.")
            return

        save_all_frames = self.all_frames_check.isChecked()
        check_duplicates = self.no_duplicates_check.isChecked()
        
        mode_config = {} 
        
        output_format = self.output_format_combo.currentText()
        mode_config["output_format"] = output_format
        
        if output_format == "jpg":
            mode_config["jpg_quality"] = self.jpg_quality_slider.value()
        
        archive_format = self.archive_format_combo.currentText()
        mode_config["archive_format"] = archive_format

        if not self.video_path:
            self.log("Error: No video loaded.")
            return

        mode_config["video_path"] = self.video_path


        if save_all_frames:
            self.log("Ignoring Mode A/B. 'Save All Frames' mode started.")
            mode_config["mode"] = "all_frames"
            mode_config["start_ms"] = start_ms
            mode_config["end_ms"] = end_ms
            
        else:
            timestamps_to_extract = []
            total_frames_str = self.frames_input.text().strip()
            interval_str = self.interval_input.text().strip()
            
            try:
                if total_frames_str:
                    if end_ms == float('inf'):
                         self.log("Error: Mode A (Total Frames) requires a specific End time.")
                         return
                    count = int(total_frames_str)
                    self.log(f"Mode A: Extracting {count} frames.")
                    if count == 1:
                        timestamps_to_extract.append(start_ms)
                    else:
                        interval = (end_ms - start_ms) / (count - 1)
                        for i in range(count):
                            timestamps_to_extract.append(start_ms + (i * interval))
                elif interval_str:
                    interval_ms = int(interval_str)
                    self.log(f"Mode B: Extracting one frame every {interval_ms}ms.")
                    if interval_ms <= 0:
                        self.log("Error: Interval must be greater than 0.")
                        return
                    current_time_ms = start_ms
                    while current_time_ms <= end_ms:
                        timestamps_to_extract.append(current_time_ms)
                        current_time_ms += interval_ms
                        if end_ms == float('inf') and current_time_ms > 3600 * 1000 * 10: 
                            self.log("Safety Limit: Mode B without an End time exceeded 10 hours.")
                            break
                else:
                    self.log("Error: Please fill in either 'Total Frames' or 'Interval'.")
                    return
            except ValueError:
                self.log("Error: Please enter valid numbers.")
                return
                
            if not timestamps_to_extract:
                self.log("Error: No frames to extract based on current settings.")
                return
                
            mode_config["mode"] = "timestamp"
            mode_config["timestamps"] = timestamps_to_extract
            mode_config["check_duplicates"] = check_duplicates

        # Start the background thread
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True) 
        self.start_btn.setText("PROCESSING...")
        self.log_area.clear()
        
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        self.worker_thread = QThread()
        self.extraction_worker = ExtractionWorker(
            video_path=self.video_path, 
            output_dir=self.output_dir,
            mode_config=mode_config 
        )
        self.extraction_worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.extraction_worker.run)
        self.extraction_worker.finished.connect(self.on_extraction_finished)
        self.extraction_worker.progress.connect(self.log)
        self.extraction_worker.progress_percent.connect(self.update_progress_bar) 
        
        self.worker_thread.start()

    # --- New Slots ---
    @pyqtSlot(str)
    def toggle_jpg_quality_slider(self, text):
        """Shows/hides the JPG quality slider based on selection."""
        is_jpg = (text.lower() == "jpg")
        self.jpg_quality_label.setVisible(is_jpg)
        self.jpg_quality_slider.setVisible(is_jpg)
        self.jpg_quality_value_label.setVisible(is_jpg)

    @pyqtSlot(int)
    def update_jpg_quality_label(self, value):
        """Updates the percentage label for the JPG slider."""
        self.jpg_quality_value_label.setText(f"{value}%")
        
    @pyqtSlot(int)
    def update_progress_bar(self, percent):
        """Slot to update the progress bar."""
        self.progress_bar.setValue(percent)

    def open_output_directory(self):
        """Opens the output folder in the native file explorer."""
        if not self.output_dir or not os.path.exists(self.output_dir):
            self.log(f"Error: Output folder not found: {self.output_dir}")
            return
        
        try:
            if sys.platform == "win32":
                os.startfile(self.output_dir)
            elif sys.platform == "darwin": # macOS
                subprocess.Popen(["open", self.output_dir])
            else: # Linux
                subprocess.Popen(["xdg-open", self.output_dir])
            self.log(f"Opened folder: {self.output_dir}")
        except Exception as e:
            self.log(f"Failed to open folder: {e}")

    # --- Updated Slots ---
    @pyqtSlot()
    def cancel_extraction(self):
        """Sends a stop signal to the running worker."""
        self.log("Cancel request sent...")
        if self.extraction_worker:
            self.extraction_worker.stop() 
        self.cancel_btn.setEnabled(False) 
        self.start_btn.setText("CANCELING...")
        self.progress_bar.setVisible(False) 

    def on_extraction_finished(self):
        """Cleans up the UI and thread after extraction is done."""
        self.statusBar().showMessage("Operation complete.", 5000)
        self.start_btn.setEnabled(True)
        self.start_btn.setText("3. START EXTRACTION")
        self.cancel_btn.setEnabled(False) 
        self.open_output_btn.setEnabled(True) 
        
        self.progress_bar.setVisible(False)
        
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.extraction_worker.deleteLater()
            self.worker_thread.deleteLater()
            self.worker_thread = None
            self.extraction_worker = None

    def closeEvent(self, event):
        """Ensures the worker thread is stopped when closing the window."""
        self.player.stop() 
        self.cancel_extraction() 
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.log("Waiting for shutdown...")
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()

# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

