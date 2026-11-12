
import cv2
from pathlib import Path

from annolab.formats import load_yolo_annotations
from annolab.preprocessing import resize_boxes


# Image and annotation paths
image_path = Path("task1_data/dataset/images/000000000139.jpg")
label_path = Path("task1_data/dataset/labels/000000000139.txt")

# Read original image
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Could not read image")

old_height, old_width = image.shape[:2]

# Load real YOLO annotations
boxes = load_yolo_annotations(label_path, old_width, old_height)

# Convert Box objects into dictionaries
box_dicts = [
    {
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2
    }
    for box in boxes
]

# Draw original bounding boxes
original_with_boxes = image.copy()

for box in box_dicts:
    cv2.rectangle(
        original_with_boxes,
        (int(box["x1"]), int(box["y1"])),
        (int(box["x2"]), int(box["y2"])),
        (0, 255, 0),
        2
    )

# Resize image
new_width, new_height = 640, 640

resized_image = cv2.resize(image, (new_width, new_height))

# Resize bounding boxes
new_boxes = resize_boxes(
    box_dicts,
    old_width,
    old_height,
    new_width,
    new_height
)

# Draw resized bounding boxes
resized_with_boxes = resized_image.copy()

for box in new_boxes:
    cv2.rectangle(
        resized_with_boxes,
        (int(box["x1"]), int(box["y1"])),
        (int(box["x2"]), int(box["y2"])),
        (0, 255, 0),
        2
    )

# Save results
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

cv2.imwrite(str(output_dir / "before_resize_boxes.jpg"), original_with_boxes)
cv2.imwrite(str(output_dir / "after_resize_boxes.jpg"), resized_with_boxes)

print("Original size:", old_width, "x", old_height)
print("Resized size:", new_width, "x", new_height)
print("Number of bounding boxes:", len(boxes))
print("Images saved successfully!")
