import torch
import torchvision
from torchvision import transforms
import cv2
import os
import json

# Load pre-trained model (Faster R-CNN for example)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

# Define transformations for input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Function to save JSON output
def save_json(data, output_path):
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Log the data before saving
        print("Saving JSON data:")
        print(data)

        # Open the file in write mode and save JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"JSON saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

# Function to crop and save images of all possible bounding boxes
def save_cropped_images(image, bboxes, output_dir, prefix):
    try:
        os.makedirs(output_dir, exist_ok=True)
        for idx, bbox in enumerate(bboxes):
            # Ensure bounding box coordinates are within image dimensions
            height, width, _ = image.shape
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Crop the image using the adjusted bounding box
            cropped = image[y1:y2, x1:x2]

            # Check if the cropped area is valid
            if cropped.size > 0:
                output_path = os.path.join(output_dir, f"{prefix}_cropped_{idx + 1}.jpg")
                cv2.imwrite(output_path, cropped)
                print(f"Cropped image saved to {output_path}")
            else:
                print(f"Invalid cropped area for bbox {bbox}. Skipping.")
    except Exception as e:
        print(f"Error saving cropped images: {e}")

# Function for object detection in an image
def detect_objects_image(image_path, output_json_path, output_cropped_dir, visualize=True):
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found at {image_path}. Please check the path.")
    
    try:
        # Read and process the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(image_rgb).unsqueeze(0)

        # Move model and image tensor to the appropriate device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)

        objects = outputs[0]
        boxes = objects['boxes'].cpu().numpy()  # Move to CPU for further processing
        labels = objects['labels'].cpu().numpy()
        scores = objects['scores'].cpu().numpy()

        json_output = []
        object_counter = 0

        for i in range(len(scores)):
            if scores[i] < 0.5:  # Confidence threshold
                continue

            object_counter += 1
            main_bbox = boxes[i]
            main_label = labels[i]

            # Here, we treat sub-object as a new detection for simplicity
            # In reality, you'd use more advanced methods (e.g., multi-task learning) to detect sub-objects
            subobject_data = {
                "object": f"Object_{main_label}",
                "id": object_counter,
                "bbox": main_bbox.tolist()
            }

            # Save JSON output for object and sub-object
            json_output.append({
                "object": f"Object_{main_label}",
                "id": object_counter,
                "bbox": main_bbox.tolist(),
                "subobject": subobject_data  # Including sub-object data as per your requirement
            })

        # Save cropped images for all detected objects
        save_cropped_images(image, boxes, output_cropped_dir, prefix="image")

        # Save the output JSON file
        save_json(json_output, output_json_path)

        # Visualize image with bounding boxes
        if visualize:
            for bbox in boxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Detected Objects", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("Detection completed. JSON and cropped images saved.")

    except Exception as e:
        print(f"Error in object detection: {e}")

# Function for object detection in a video
def detect_objects_video(video_path, output_json_path, output_cropped_dir, visualize=True):
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_counter = 0
        json_output = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(image_rgb).unsqueeze(0)

            # Move model and frame tensor to the appropriate device (GPU if available)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            tensor = tensor.to(device)

            with torch.no_grad():
                outputs = model(tensor)

            objects = outputs[0]
            boxes = objects['boxes'].cpu().numpy()  # Move to CPU for further processing
            labels = objects['labels'].cpu().numpy()
            scores = objects['scores'].cpu().numpy()

            frame_json_output = []

            for i in range(len(scores)):
                if scores[i] < 0.5:  # Confidence threshold
                    continue

                main_bbox = boxes[i]
                main_label = labels[i]

                # Here, we treat sub-object as a new detection for simplicity
                # In reality, you'd use more advanced methods (e.g., multi-task learning) to detect sub-objects
                subobject_data = {
                    "object": f"Object_{main_label}",
                    "id": frame_counter,
                    "bbox": main_bbox.tolist()
                }

                frame_json_output.append({
                    "object": f"Object_{main_label}",
                    "frame": frame_counter,
                    "bbox": main_bbox.tolist(),
                    "subobject": subobject_data
                })

            json_output.append(frame_json_output)

            # Save cropped images for all detected objects in the frame
            save_cropped_images(frame, boxes, output_cropped_dir, prefix=f"frame_{frame_counter}")

            # Visualize frame with bounding boxes
            if visualize:
                for bbox in boxes:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow(f"Frame {frame_counter}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()  # Release the video capture object

        # Save the output JSON file
        save_json(json_output, output_json_path)

        print("Detection completed. JSON and cropped images saved.")

    except Exception as e:
        print(f"Error in video detection: {e}")

# Main function to process either image or video
def detect_objects(input_path, output_json_path, output_cropped_dir, visualize=True):
    if input_path.endswith(('.mp4', '.avi', '.mov')):
        detect_objects_video(input_path, output_json_path, output_cropped_dir, visualize)
    elif input_path.endswith(('.jpg', '.jpeg', '.png')):
        detect_objects_image(input_path, output_json_path, output_cropped_dir, visualize)
    else:
        print("Unsupported file format. Please provide an image or video file.")

# Paths for input image/video, output JSON, and cropped sub-objects directory
input_path = "data/video/test_video.mp4"  # Change to an image or video path
output_json = "data/outputs/detections.json"
output_cropped = "data/sub_objects"

# Run object detection
detect_objects(input_path, output_json, output_cropped)
