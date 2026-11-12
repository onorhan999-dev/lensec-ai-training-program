
from pathlib import Path
import cv2

from annolab.formats import load_yolo_annotations

images_dir = Path("output/batch_processed/images")
labels_dir = Path("output/batch_processed/labels")

images = sorted(images_dir.glob("*.jpg"))

total_boxes = 0

for image_path in images:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read {image_path}")

    height, width = image.shape[:2]
    label_path = labels_dir / f"{image_path.stem}.txt"

    boxes = load_yolo_annotations(label_path, width, height)

    for box in boxes:
        assert 0 <= box.x1 < box.x2 <= width
        assert 0 <= box.y1 < box.y2 <= height

    total_boxes += len(boxes)

print("Verified images:", len(images))
print("Total bounding boxes:", total_boxes)
print("All bounding boxes are valid!")
