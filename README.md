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

### Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
