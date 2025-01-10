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
**OpenCV plays a crucial role in handling image/video input/output, drawing, and saving cropped regions of detected objects.**
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
│   │   └── test_video.mp4  # Input video file (You can replace this with your actual video)
│   └── images/
│   │   └── test_image.jpg  # Input image file (You can replace this with your actual image)
│   └── sub_objects/        # Directory for cropped images
│   │   └── frame_cropped.jpg # Cropped images saved from the frames
│   └── outputs/
│       └── detections.json  # JSON file with detection metadata
├── scripts/
│   └── object_detection.py  # Main Python script
├── utils/
│   └── json_utils.py  # Contains save_json and load_json
├── requirements.txt     # Python package dependencies
└── run_detection.sh     # Bash script for automating the process







## STEPS TO RUN THIS PROJECT

# Step 1: Set up the environment

# Create a virtual environment if not already created
if [ ! -d "env" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv env
fi

# Activate the virtual environment
source env/bin/activate

# Step 2: Install dependencies (if not already installed)
echo "Installing required Python packages..."
pip install -r requirements.txt

# Step 3: Set up input/output directories if not already created
echo "Setting up necessary directories..."
mkdir -p data/outputs data/sub_objects scripts utils

# Step 4: Download input files if necessary (You can adjust this part based on where the input files are stored)
if [ ! -f "data/video/test_video.mp4" ]; then
    echo "Downloading sample video file..."
    # Replace this with your actual file download process
    curl -o data/video/test_video.mp4 <your_video_file_link>
fi

if [ ! -f "data/images/test_image.jpg" ]; then
    echo "Downloading sample image file..."
    # Replace this with your actual file download process
    curl -o data/images/test_image.jpg <your_image_file_link>
fi

# Step 5: Run the object detection script
echo "Running object detection on the video or image..."
python scripts/object_detection.py

# Step 6: Deactivate the virtual environment
deactivate

echo "Object detection completed. JSON and cropped images are saved."








### Handling Edge Cases: Occlusion and Overlapping Objects

#### **1. Occlusion Handling**
- **Partial Detection**: Faster R-CNN can detect partially occluded objects by identifying visible portions, though accuracy decreases with heavy occlusion.
- **Bounding Box Adjustment**: The system draws boxes around visible portions, using confidence scores to filter low-confidence detections.

#### **2. Overlapping Objects**
- **Non-Maximum Suppression (NMS)**: NMS helps to handle overlapping objects by removing redundant boxes, keeping only the most confident detection.
- **IoU Threshold**: NMS uses an IoU threshold (e.g., 0.5) to suppress boxes that overlap significantly.

#### **3. System Robustness**
- **High Confidence Threshold**: By using a high-confidence threshold (e.g., 0.7), the system reduces false positives.
- **Postprocessing and Fine-Tuning**: Custom postprocessing and fine-tuning on occlusion-heavy datasets improve detection in challenging scenarios.

In summary, Faster R-CNN is effective at handling occlusion and overlapping objects with NMS, confidence thresholds, and potential model fine-tuning for edge cases.