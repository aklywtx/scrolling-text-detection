# Rolling Text Detection and Recognition in Videos

## Overview

This is a course project for High Level Computer Vision in Saarland University. This project introduces a pipeline capable of detecting and reading moving horizontal text in videos, inspired by rolling-text seen on german train station information boards. The pipeline combines Optical Flow methods, pre-trained Text-Detection and OCR models, and Large Language Models to detect and recognize rolling-text in video.


## Scripts

- **`extract_frames.py`**: Extracts individual frames from the video input for further processing.
- **`detect_with_opflow.py`**: Utilizes Optical Flow techniques to assist in text detection in video sequences.
- **`detect_with_east.py`**: Detects text regions in video frames using the EAST text detector.
- **`combine_opflow_and_east.py`**: Integrates Optical Flow and EAST methods to get the best rolling text region.
- **`ocr.py`**: Performs OCR on detected text regions in the video frames.
- **`concat.py`**: Concatenates text detected across multiple video frames to generate the final output.
- **`main.py`**: The main script that orchestrates the entire pipeline from frame extraction to final text output.

## Requirements

numpy>=1.24
opencv-python>=4.7
pillow>=9.0
pytesseract>=0.3.10
tqdm>=4.60
scikit-image>=0.19
imutils>=0.5
matplotlib>=3.5
openai>=0.27.0
llama-cpp-python>=0.1.0

## Usage

To execute the pipeline, run the `main.py` script with the appropriate video input. The final detected and concatenated text string will be outputted to the user.

```bash
python main.py -v <path_to_video_file>
