
import numpy as np


# Calculate Intersection over Union (IoU) between two bounding boxes.
def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# Evaluate bounding boxes using IoU and calculate detection metrics.
def evaluate_detections(predictions, ground_truth, iou_threshold=0.5):
    """Evaluate predictions against ground truth boxes."""

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in predictions:
        best_iou = 0.0
        best_idx = -1

        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue

            iou = calculate_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold and best_idx != -1:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall
    }


# Filter predictions based on the confidence threshold.
def filter_predictions(predictions, confidence_threshold=0.5):
    """Keep predictions above the confidence threshold."""

    return [
        pred
        for pred in predictions
        if pred["confidence"] >= confidence_threshold
    ]


# Evaluate detections using confidence, class ID, and IoU to calculate TP, FP, FN, Precision, and Recall.
def evaluate_predictions(
    predictions,
    ground_truth,
    confidence_threshold=0.5,
    iou_threshold=0.5
):
    """Evaluate detections using confidence, class ID, and IoU."""

    filtered = filter_predictions(
        predictions,
        confidence_threshold
    )

    filtered = sorted(
        filtered,
        key=lambda p: p["confidence"],
        reverse=True
    )

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in filtered:
        best_iou = 0.0
        best_idx = -1

        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue

            if pred["class_id"] != gt["class_id"]:
                continue

            iou = calculate_iou(
                pred["bbox"],
                gt["bbox"]
            )

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx != -1 and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall
    }


# Draw detection boxes on images for FP/FN visualization.
def draw_detection_boxes(image, boxes, color=(0, 0, 255)):
    """Draw bounding boxes on an image."""
    import cv2

    result = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box["bbox"])

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

    return result



# Save annotated detection images for the FP/FN gallery.
def save_detection_image(image, boxes, output_path, color=(0, 0, 255)):
    import cv2
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = draw_detection_boxes(image, boxes, color)

    if not cv2.imwrite(str(output_path), annotated):
        raise IOError(f"Could not save image: {output_path}")

    return str(output_path)



# Identify false positive and false negative boxes for visualization.
def find_detection_errors(
    predictions,
    ground_truth,
    confidence_threshold=0.5,
    iou_threshold=0.5
):
    filtered = filter_predictions(predictions, confidence_threshold)
    filtered = sorted(
        filtered,
        key=lambda p: p["confidence"],
        reverse=True
    )

    matched_gt = set()
    false_positives = []

    for pred in filtered:
        best_iou = 0.0
        best_idx = -1

        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt:
                continue

            if pred["class_id"] != gt["class_id"]:
                continue

            iou = calculate_iou(pred["bbox"], gt["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx != -1 and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
        else:
            false_positives.append(pred)

    false_negatives = [
        gt
        for idx, gt in enumerate(ground_truth)
        if idx not in matched_gt
    ]

    return false_positives, false_negatives


# Generate FP and FN visualization images.
def generate_error_gallery(
    image,
    predictions,
    ground_truth,
    output_dir
):
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp, fn = find_detection_errors(
        predictions,
        ground_truth
    )

    saved_files = []

    if fp:
        path = save_detection_image(
            image,
            fp,
            output_dir / "false_positives.jpg",
            color=(0, 0, 255)
        )
        saved_files.append(path)

    if fn:
        path = save_detection_image(
            image,
            fn,
            output_dir / "false_negatives.jpg",
            color=(255, 0, 0)
        )
        saved_files.append(path)

    return saved_files