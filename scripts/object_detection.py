import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import os
import json

# Load pre-trained model (Faster R-CNN for example)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')  # Using default pre-trained weights
model.eval()

# Define transformations for input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),  # Resize to a fixed size for consistency
    transforms.ToTensor()
])

# Function to save JSON output
def save_json(data, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"JSON saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

# Function to crop and save sub-object images with unique filenames
def save_cropped_image(image, bbox, object_id, sub_id, output_path):
    try:
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image[y1:y2, x1:x2]
        output_filename = f"subobject_{object_id}_{sub_id}.jpg"  # Unique name for each sub-object
        output_full_path = os.path.join(output_path, output_filename)
        cv2.imwrite(output_full_path, cropped)
        print(f"Cropped image saved to {output_full_path}")
    except Exception as e:
        print(f"Error saving cropped image: {e}")

# Function to generate sub-objects (for demonstration purposes)
def generate_sub_objects(main_bbox):
    x1, y1, x2, y2 = main_bbox
    sub_objects = []

    # Generate sub-objects as smaller regions within the main bounding box
    sub_width = (x2 - x1) // 2
    sub_height = (y2 - y1) // 2

    # Example logic for creating sub-objects (cutting the main bbox into 4 sub-bboxes)
    sub_objects.append({"object": "SubObject", "id": 1, "bbox": [x1, y1, x1 + sub_width, y1 + sub_height]})
    sub_objects.append({"object": "SubObject", "id": 2, "bbox": [x1 + sub_width, y1, x2, y1 + sub_height]})
    sub_objects.append({"object": "SubObject", "id": 3, "bbox": [x1, y1 + sub_height, x1 + sub_width, y2]})
    sub_objects.append({"object": "SubObject", "id": 4, "bbox": [x1 + sub_width, y1 + sub_height, x2, y2]})

    return sub_objects

# Main detection function
def detect_objects(image_path, output_json_path, output_cropped_dir):
    # Check if the image exists at the given path
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

            # Generate sub-objects dynamically
            sub_objects = generate_sub_objects(main_bbox)

            json_output.append({
                "object": f"Object_{main_label}",
                "id": object_counter,
                "bbox": main_bbox.tolist(),
                "subobject": sub_objects
            })

            # Save cropped images for sub-objects with unique names
            for sub in sub_objects:
                sub_bbox = sub["bbox"]
                sub_id = sub["id"]
                save_cropped_image(image, sub_bbox, object_counter, sub_id, output_cropped_dir)

        # Save the output JSON file
        save_json(json_output, output_json_path)
        print("Detection completed. JSON and cropped images saved.")

    except Exception as e:
        print(f"Error in object detection: {e}")

# Paths for input image, output JSON, and cropped sub-objects directory
input_image = "data\\images\\test_image.jpeg"  # Updated image path
output_json = "data\\outputs\\detections.json"  # Updated JSON output path
output_cropped = "data\\sub_objects"  # Updated sub-objects output directory

# Run object detection
detect_objects(input_image, output_json, output_cropped)
