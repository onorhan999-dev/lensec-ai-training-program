
from annolab.formats import Box, validate_boxes


def test_box_dimensions():
    box = Box(0, 10, 20, 110, 220)

    assert box.width == 100
    assert box.height == 200
    assert box.area == 20000


def test_valid_box():
    boxes = [Box(0, 10, 20, 100, 200)]
    errors = validate_boxes(boxes, 640, 426)

    assert errors == []


def test_invalid_box():
    boxes = [Box(0, -5, 20, 100, 200)]
    errors = validate_boxes(boxes, 640, 426)

    assert len(errors) == 1

from annolab.formats import (
    load_yolo_annotations,
    save_yolo_annotations,
    detect_annotation_format
)


def test_yolo_roundtrip(tmp_path):
    original = [Box(0, 100, 50, 300, 250)]
    path = tmp_path / "labels.txt"

    save_yolo_annotations(original, path, 640, 426)
    restored = load_yolo_annotations(path, 640, 426)

    assert len(restored) == 1

    for coord in ("x1", "y1", "x2", "y2"):
        assert abs(
            getattr(original[0], coord)
            - getattr(restored[0], coord)
        ) <= 1


def test_format_detection():
    assert detect_annotation_format("labels.txt") == "YOLO"
    assert detect_annotation_format("labels.xml") == "VOC"
    assert detect_annotation_format("labels.json") == "COCO"


def test_empty_yolo_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    boxes = load_yolo_annotations(path, 640, 426)

    assert boxes == []

import json
from annolab.formats import (
    load_voc_annotations,
    load_coco_annotations
)


def test_voc_loader(tmp_path):
    xml_path = tmp_path / "test.xml"
    xml_path.write_text(
        "<annotation><object><name>person</name>"
        "<bndbox><xmin>10</xmin><ymin>20</ymin>"
        "<xmax>110</xmax><ymax>220</ymax></bndbox>"
        "</object></annotation>",
        encoding="utf-8"
    )

    boxes = load_voc_annotations(xml_path, ["person"])

    assert len(boxes) == 1
    assert boxes[0] == Box(0, 10, 20, 110, 220)


def test_coco_loader(tmp_path):
    json_path = tmp_path / "test.json"

    data = {
        "images": [{"id": 1, "width": 640, "height": 426}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10, 20, 100, 200],
            "area": 20000,
            "iscrowd": 0
        }]
    }

    json_path.write_text(json.dumps(data), encoding="utf-8")

    boxes = load_coco_annotations(json_path, 1)

    assert len(boxes) == 1
    assert boxes[0] == Box(0, 10, 20, 110, 220)

import math
import pytest

from annolab.formats import Box, validate_boxes


def test_nan_coordinates():
    box = Box(0, math.nan, 10, 100, 100)
    errors = validate_boxes([box], 640, 480)
    assert any("Invalid coordinates" in e for e in errors)


def test_infinite_coordinates():
    box = Box(0, 10, 10, math.inf, 100)
    errors = validate_boxes([box], 640, 480)
    assert any("Invalid coordinates" in e for e in errors)


def test_negative_class_id():
    box = Box(-1, 10, 10, 100, 100)
    errors = validate_boxes([box], 640, 480)
    assert any("Invalid class ID" in e for e in errors)


def test_class_id_out_of_range():
    box = Box(5, 10, 10, 100, 100)
    errors = validate_boxes([box], 640, 480, num_classes=3)
    assert any("Invalid class ID" in e for e in errors)


def test_invalid_image_dimensions():
    box = Box(0, 10, 10, 100, 100)
    with pytest.raises(ValueError):
        validate_boxes([box], 0, 480)

def test_coco_category_mapping():
    from annolab.formats import boxes_to_coco

    category_ids = list(range(1, 12)) + list(range(13, 26)) + [
        27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 84, 85, 86, 87, 88, 89, 90
    ]

    assert len(category_ids) == 80

    box = Box(79, 10, 20, 100, 200)

    annotations = boxes_to_coco(
        [box],
        image_id=1,
        category_ids=category_ids
    )

    assert annotations[0]["category_id"] == 90

def test_save_coco_json_mapping(tmp_path):
    import json
    from annolab.formats import save_coco_json

    output_file = tmp_path / "test_mapping.json"

    boxes = [Box(1, 10, 20, 100, 200)]

    save_coco_json(
        boxes=boxes,
        output_path=output_file,
        image_id=1,
        image_width=640,
        image_height=480,
        class_names=["person", "car"],
        category_ids=[1, 90]
    )

    with open(output_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["annotations"][0]["category_id"] == 90
    assert data["categories"][1]["id"] == 90
    assert data["categories"][1]["name"] == "car"

def test_coco_yolo_coco_roundtrip(tmp_path):
    from annolab.formats import (
        save_yolo_annotations,
        load_yolo_annotations,
        boxes_to_coco
    )

    original = [Box(1, 10, 20, 100, 200)]
    category_ids = [1, 90]

    yolo_file = tmp_path / "labels.txt"

    save_yolo_annotations(original, yolo_file, 640, 480)

    restored = load_yolo_annotations(yolo_file, 640, 480)

    coco = boxes_to_coco(
        restored,
        image_id=1,
        category_ids=category_ids
    )

    assert coco[0]["category_id"] == 90

    for a, b in zip(original[0].__dict__.values(),
                    restored[0].__dict__.values()):
        assert abs(a - b) <= 1
