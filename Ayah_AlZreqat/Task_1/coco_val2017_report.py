
import json
from pathlib import Path
from collections import Counter

annotation_path = Path(
    r"C:\Lensec_Datasets\annotations\instances_val2017.json"
)

with open(annotation_path, "r", encoding="utf-8") as file:
    data = json.load(file)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

class_counts = Counter(
    ann["category_id"] for ann in annotations
)

image_ids_with_annotations = {
    ann["image_id"] for ann in annotations
}

areas = [
    ann["bbox"][2] * ann["bbox"][3]
    for ann in annotations
]

report = {
    "dataset": "COCO val2017",
    "total_images": len(images),
    "total_annotations": len(annotations),
    "total_categories": len(categories),
    "images_without_annotations": (
        len(images) - len(image_ids_with_annotations)
    ),
    "class_distribution": dict(class_counts),
    "average_box_area": sum(areas) / len(areas) if areas else 0,
    "min_box_area": min(areas) if areas else 0,
    "max_box_area": max(areas) if areas else 0,
}

output_path = Path("output/coco_val2017_report.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print(json.dumps(report, indent=2))
print(f"\nReport saved: {output_path}")
