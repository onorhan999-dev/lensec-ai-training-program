
import json
from pathlib import Path
from PIL import Image

from annolab.formats import load_yolo_annotations, calculate_dataset_statistics

images_dir = Path("task1_data/coco128/images/train2017")
labels_dir = Path("task1_data/coco128/labels/train2017")

all_boxes = []
images_without_annotations = 0
image_count = 0

for image_path in sorted(images_dir.glob("*.jpg")):
    image_count += 1

    with Image.open(image_path) as img:
        width, height = img.size

    label_path = labels_dir / (image_path.stem + ".txt")

    if not label_path.exists():
        images_without_annotations += 1
        continue

    boxes = load_yolo_annotations(label_path, width, height)

    if not boxes:
        images_without_annotations += 1

    all_boxes.extend(boxes)

stats = calculate_dataset_statistics(all_boxes)

report = {
    "dataset": "COCO128",
    "total_images": image_count,
    "images_without_annotations": images_without_annotations,
    **stats
}

output_path = Path("output/coco128_dataset_report.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print(json.dumps(report, indent=2))
print(f"\nReport saved: {output_path}")
