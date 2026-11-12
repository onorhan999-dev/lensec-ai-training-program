
from dataclasses import dataclass


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return max(0, self.width) * max(0, self.height)

from pathlib import Path


def load_yolo_annotations(label_path, image_width, image_height):
    boxes = []
    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(f"File not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.strip().split()

            if not values:
                continue

            if len(values) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation at line {line_number}"
                )

            class_id = int(values[0])
            x_center, y_center, w, h = map(float, values[1:])

            x1 = (x_center - w / 2) * image_width
            y1 = (y_center - h / 2) * image_height
            x2 = (x_center + w / 2) * image_width
            y2 = (y_center + h / 2) * image_height

            boxes.append(Box(class_id, x1, y1, x2, y2))

    return boxes

import xml.etree.ElementTree as ET


def load_voc_annotations(xml_path, class_names):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name")

        if class_name not in class_names:
            raise ValueError(
                f"Unknown class: {class_name}"
            )

        class_id = class_names.index(class_name)

        bbox = obj.find("bndbox")

        x1 = float(bbox.findtext("xmin"))
        y1 = float(bbox.findtext("ymin"))
        x2 = float(bbox.findtext("xmax"))
        y2 = float(bbox.findtext("ymax"))

        boxes.append(
            Box(class_id, x1, y1, x2, y2)
        )

    return boxes

import json

def load_coco_annotations(json_path, image_id):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    boxes = []

    category_ids = sorted(
        category["id"] for category in data["categories"]
    )
    category_mapping = {
        category_id: index
        for index, category_id in enumerate(category_ids)
    }

    for ann in data["annotations"]:
        if ann["image_id"] != image_id:
            continue

        x, y, w, h = ann["bbox"]
        class_id = category_mapping[ann["category_id"]]

        boxes.append(
            Box(
                class_id=class_id,
                x1=x,
                y1=y,
                x2=x + w,
                y2=y + h
            )
        )

    return boxes


def validate_boxes(boxes, image_width, image_height, num_classes=None):
    import math

    errors = []

    if not all(
        isinstance(v, (int, float)) and
        math.isfinite(v) and v > 0
        for v in (image_width, image_height)
    ):
        raise ValueError("Image dimensions must be positive and finite")

    for i, box in enumerate(boxes):
        coords = (box.x1, box.y1, box.x2, box.y2)

        if not all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in coords
        ):
            errors.append(f"Box {i}: Invalid coordinates")
            continue

        if box.x1 >= box.x2 or box.y1 >= box.y2:
            errors.append(f"Box {i}: Invalid dimensions")

        if (box.x1 < 0 or box.y1 < 0 or
                box.x2 > image_width or box.y2 > image_height):
            errors.append(f"Box {i}: Outside image boundaries")

        if (not isinstance(box.class_id, int) or
                box.class_id < 0 or
                (num_classes is not None and box.class_id >= num_classes)):
            errors.append(f"Box {i}: Invalid class ID")

    return errors


def save_yolo_annotations(boxes, output_path, image_width, image_height):
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    with open(output_path, "w", encoding="utf-8") as file:
        for box in boxes:
            x_center = ((box.x1 + box.x2) / 2) / image_width
            y_center = ((box.y1 + box.y2) / 2) / image_height
            width = box.width / image_width
            height = box.height / image_height

            file.write(
                f"{box.class_id} {x_center:.8f} {y_center:.8f} "
                f"{width:.8f} {height:.8f}\n"
            )


def boxes_to_coco(boxes, image_id, starting_id=1, category_ids=None):
    annotations = []

    for i, box in enumerate(boxes):
        if category_ids is not None:
            category_id = category_ids[box.class_id]
        else:
            category_id = box.class_id + 1

        annotations.append({
            "id": starting_id + i,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [
                box.x1,
                box.y1,
                box.width,
                box.height
            ],
            "area": box.area,
            "iscrowd": 0
        })

    return annotations



def save_coco_json(boxes, output_path, image_id,
                   image_width, image_height, class_names,
                   category_ids=None):

    if category_ids is None:
        category_ids = list(range(1, len(class_names) + 1))

    if len(category_ids) != len(class_names):
        raise ValueError("category_ids and class_names must have the same length")

    annotations = boxes_to_coco(
        boxes, image_id, category_ids=category_ids
    )

    data = {
        "images": [{
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": "image.jpg"
        }],
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name}
            for category_id, name in zip(category_ids, class_names)
        ]
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def detect_annotation_format(file_path):
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".txt":
        return "YOLO"

    elif extension == ".xml":
        return "VOC"

    elif extension == ".json":
        return "COCO"

    else:
        raise ValueError(
            f"Unsupported annotation format: {extension}"
        )

from collections import Counter

def calculate_dataset_statistics(boxes):
    class_counts = Counter(box.class_id for box in boxes)

    areas = [box.area for box in boxes]

    return {
        "total_boxes": len(boxes),
        "class_distribution": dict(class_counts),
        "average_box_area": sum(areas) / len(areas) if areas else 0,
        "min_box_area": min(areas) if areas else 0,
        "max_box_area": max(areas) if areas else 0
    }
