import cv2
import json

# Load the image
image_path = "object_detection_system/data/images/test_image.jpeg"  # Correct image path
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Image not found at {image_path}. Please check the path.")

# Load the detections from the JSON file
json_path = "object_detection_system/data/outputs/detections.json"  # Correct JSON path
with open(json_path, 'r') as f:
    detections = json.load(f)

# Check if there are any detections
if not detections:
    print("No detections found.")
else:
    # Loop through the detections and draw bounding boxes
    for detection in detections:
        main_bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, main_bbox)
        
        # Draw the main bounding box on the image (red color for main object)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Loop through the sub-objects and draw their bounding boxes (blue color for sub-objects)
        for sub in detection['subobject']:
            sub_bbox = sub['bbox']
            sx1, sy1, sx2, sy2 = map(int, sub_bbox)
            cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the image with bounding boxes
output_image_path = "object_detection_system/data/outputs/output_image_with_bboxes.jpg"  # Specify the output path
cv2.imwrite(output_image_path, image)
