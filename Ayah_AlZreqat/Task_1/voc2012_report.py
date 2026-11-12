
import json
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

from annolab.formats import (
    load_voc_annotations,
    calculate_dataset_statistics,
)

dataset_dir = Path(r"C:\Lensec_Datasets\VOCdevkit\VOC2012")
annotations_dir = dataset_dir / "Annotations"
images_dir = dataset_dir / "JPEGImages"

xml_files = sorted(annotations_dir.glob("*.xml"))
class_names = sorted({
    obj.findtext("name")
    for xml_path in xml_files
    for obj in ET.parse(xml_path).getroot().findall("object")
})

all_boxes = []
images_without_annotations = 0
invalid_boxes = 0

for xml_path in xml_files:
    boxes = load_voc_annotations(xml_path, class_names)
    all_boxes.extend(boxes)

    if not boxes:
        images_without_annotations += 1

    for box in boxes:
        if box.width <= 0 or box.height <= 0:
            invalid_boxes += 1

stats = calculate_dataset_statistics(all_boxes)

report = {
    "dataset": "VOC2012",
    "total_images": len(list(images_dir.glob("*.jpg"))),
    "total_annotation_files": len(xml_files),
    "images_without_annotations": images_without_annotations,
    "class_names": class_names,
    "invalid_boxes": invalid_boxes,
    **stats,
}

output_path = Path("output/voc2012_dataset_report.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print(json.dumps(report, indent=2))
print(f"\nReport saved: {output_path}")
