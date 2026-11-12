
import cv2
from pathlib import Path

from annolab.formats import load_yolo_annotations
from annolab.preprocessing import letterbox_image, letterbox_boxes


# Paths
image_path = Path("task1_data/dataset/images/000000000139.jpg")
label_path = Path("task1_data/dataset/labels/000000000139.txt")

# Read original image
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Could not read image")

old_height, old_width = image.shape[:2]

# Load YOLO annotations
boxes = load_yolo_annotations(label_path, old_width, old_height)

# Convert Box objects to dictionaries
box_dicts = [
    {
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2
    }
    for box in boxes
]

# Apply Letterbox
result, scale_x, scale_y, pad_left, pad_top = letterbox_image(
    image, 640, 640
)

# Transform bounding boxes
new_boxes = letterbox_boxes(
    box_dicts,
    scale_x,
    scale_y,
    pad_left,
    pad_top
)

# Draw transformed boxes
for box in new_boxes:
    cv2.rectangle(
        result,
        (round(box["x1"]), round(box["y1"])),
        (round(box["x2"]), round(box["y2"])),
        (0, 255, 0),
        2
    )

# Save output
output_path = Path("output/letterbox_with_boxes.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(output_path), result):
    raise IOError("Could not save image")

print("Original size:", old_width, "x", old_height)
print("Letterbox size:", result.shape[1], "x", result.shape[0])
print("Bounding boxes:", len(new_boxes))
print("Padding top:", pad_top)
print("Saved:", output_path)

