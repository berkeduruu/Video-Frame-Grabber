#Video Frame Grabber    

<img width="901" height="844" alt="image" src="https://github.com/user-attachments/assets/a05d9f3c-1117-4607-b318-5037246fc812" />

A desktop application (built with PyQt6 and OpenCV) for extracting frames from video files. It allows selecting a precise time range and offers multiple extraction modes.

(You can replace this placeholder image with a real screenshot of your app)

Features

This tool is designed for both quick grabs and detailed sequence extractions, with a focus on precision and flexibility.

#1. Flexible Video Input

Load via Dialog: Use the "Load Video" button to select any common video file.

Drag-and-Drop: Simply drag your video file directly onto the preview window (even while another video is playing) to load it instantly.

#2. Time Range Selection

Select the time range for extraction:

Sliders: Drag the 'Start' and 'End' sliders.

Text Input: Type in exact timestamps in mm:ss:zzz format.

"Set Current" Buttons: Use the player to find the exact time and click "Set Current".

#3. Extraction Modes

Choose how frames are extracted from the selected range:

Total Frames (Mode A): Specify a fixed number of frames (e.g., 100). Frames will be spaced out evenly.

Interval (Mode B): Specify a time interval (e.g., 500ms). Extracts one frame every 500 milliseconds.

Save All Frames (Mode C): Captures every single frame in the range. Ignores Mode A and B.

#4. Output Options

Control the output format:

Image Format: Save frames as png, jpg, webp, or bmp.

JPG Quality: A slider (1-100%) appears when 'jpg' is selected to control image quality.

Archive Format:

Folder: Saves all frames as individual files.

ZIP (.zip): Compresses all frames into a single .zip file.

TAR.GZ (.tar.gz): Compresses all frames into a .tar.gz archive.

#5. Other Features

Skip Duplicates: An option to avoid saving identical consecutive frames.

Progress Bar: Shows the status of the extraction process.

Cancel Button: Stop an ongoing extraction.

Open Folder Button: Open the output folder directly from the app.

Process Log: Shows detailed logs, errors, and success messages.

#How to Use

Load Video: Click "Load Video" or drag-and-drop a file.

Select Output: Click "Select Output Folder" to choose a save location.

Set Range: Use the sliders or text boxes to define the time range.

Set Mode: Choose an extraction mode and your output/archive format.

Start: Click "START EXTRACTION".

#Installation & Running

You can either download the pre-compiled application or run the script from the source code.

