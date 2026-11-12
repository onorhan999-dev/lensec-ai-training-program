
# ============================================================
# STEP 13: VISUAL ANNOTATION VERIFICATION
# ============================================================
# Load the processed image and updated YOLO labels.
# Draw bounding boxes to verify their alignment.
# ============================================================

import cv2
from pathlib import Path
from annolab.formats import load_yolo_annotations

image_path = Path(
    "output/pipeline_with_labels/images/000000000139.jpg"
)

label_path = Path(
    "output/pipeline_with_labels/labels/000000000139.txt"
)

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Could not read processed image")

height, width = image.shape[:2]

boxes = load_yolo_annotations(
    label_path,
    width,
    height
)

for box in boxes:
    x1 = int(round(box.x1))
    y1 = int(round(box.y1))
    x2 = int(round(box.x2))
    y2 = int(round(box.y2))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        str(box.class_id),
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

output_path = Path(
    "output/pipeline_with_labels/verified_boxes.jpg"
)

if not cv2.imwrite(str(output_path), image):
    raise IOError("Could not save verification image")

print("Image size:", width, "x", height)
print("Bounding boxes drawn:", len(boxes))
print("Saved:", output_path)
