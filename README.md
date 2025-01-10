# SAFETYWHAT Assignment - Object Detection System

## Setup Instructions

1. Clone the repository and navigate to the project folder:
   ```bash
   git clone <repo-url>
   cd SAFETYWHAT-Assignment


# Object Detection and Cropping Tool

This project utilizes a pre-trained deep learning model to perform object detection on images and videos. It dynamically generates cropped images of all detected objects and saves metadata (bounding boxes, labels) in a JSON file.

---

## Features

- **Object Detection**: Uses a pre-trained Faster R-CNN model for detecting objects in images and videos.
- **Dynamic Cropping**: Extracts cropped images of all detected objects with varying sizes.
- **JSON Output**: Saves detection results (bounding boxes, labels, confidence scores) in a structured JSON format.
- **Image and Video Support**: Processes both static images and video files.
- **Customizable Confidence Threshold**: Filters detections based on a confidence score threshold.

---

## Requirements

### Dependencies
- Python 3.8 or higher
- Libraries:
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `numpy`
  - `Pillow`


2. Install required Python packages:
- **pip install** torch torchvision opencv-python numpy Pillow

3. Usage
## Input Files
 - Place your input image or video file in the data/ folder.
## Supported formats:
 - Images: .jpg, .jpeg, .png
 - Videos: .mp4, .avi, .mov
## Running the Detection
- Update the file paths in the script:
**input_path**: Path to the input image or video file.
**output_json**: Path for the output JSON file.
**output_cropped**: Directory for saving cropped images.

- Run the script:
 - python object_detection.py

## Outputs
- Cropped Images:
 - Saved in the output_cropped directory with names like image_cropped_1.jpg,    frame_2_cropped_3.jpg, etc.
Dynamic sizes based on detected bounding boxes.







## Project Structure

project/
├── data/
│   ├── video/
│   │   └── test_video.mp4  # Input video file
│   └── images/
│   |    └── test_image.jpg  # Input image file
|   |__sub_objects/          # Directory for cropped images
|   |    |__frame_cropped.jpg #Getting cropped the images from the frames
|   |
|   |__outputs/
│        ├── detections.json     # JSON file with detection metadata
│          
├──scripts/
|       |__ object_detection.py     # Main Python script
├── README.md               # Project documentation
└── requirements.txt        # Python package dependencies
