import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import os
import json

# Load pre-trained model (Faster R-CNN for example)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')  # Updated model loading
model.eval()

# Transformations for input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),  # Resize to a fixed size for consistency
    transforms.ToTensor()
])

# Function to save JSON
def save_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

# Function to crop and save sub-object images
def save_cropped_image(image, bbox, output_path):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped)

# Main detection function
def detect_objects(image_path, output_json_path, output_cropped_dir):
    # Check if image exists
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found at {image_path}. Please check the path.")
    
    # Read and process the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(image_rgb).unsqueeze(0)

    # Move to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)

    objects = outputs[0]
    boxes = objects['boxes'].numpy()
    labels = objects['labels'].numpy()
    scores = objects['scores'].numpy()

    json_output = []
    object_counter = 0

    for i in range(len(scores)):
        if scores[i] < 0.5:  # Confidence threshold
            continue

        object_counter += 1
        main_bbox = boxes[i]
        main_label = labels[i]

        # Dummy example of sub-objects; Replace with logic for detecting sub-objects
        sub_objects = [{"object": "SubObject", "id": 1, "bbox": [50, 50, 100, 100]}]

        json_output.append({
            "object": f"Object_{main_label}",
            "id": object_counter,
            "bbox": main_bbox.tolist(),
            "subobject": sub_objects
        })

        # Save cropped images for sub-objects
        for sub in sub_objects:
            sub_bbox = sub["bbox"]
            sub_id = sub["id"]
            save_cropped_image(image, sub_bbox, f"{output_cropped_dir}/subobject_{sub_id}.jpg")

    save_json(json_output, output_json_path)
    print("Detection completed. JSON and cropped images saved.")

# Paths
input_image = "object_detection_system\\data\\images\\test_image.jpeg"  # Update this to your image path
output_json = "object_detection_system\\data\\outputs\\detections.json"
output_cropped = "object_detection_system\\data\\outputs\\sub_objects"

# Run detection
detect_objects(input_image, output_json, output_cropped)
