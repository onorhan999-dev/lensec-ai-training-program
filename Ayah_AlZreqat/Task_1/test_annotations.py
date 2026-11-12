
import cv2
import pandas as pd
from pathlib import Path
from annolab.annotations import draw_bbox, calculate_iou


# =====================================
# 1. Read Image
# =====================================

image_path = Path("task1_data/dataset/images/000000000139.jpg")
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Could not read image")

height, width = image.shape[:2]

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. Draw One Bounding Box
# =====================================

single_image = image.copy()

x1, y1 = 409.1, 157.4
x2, y2 = 464.3, 296.2

draw_bbox(
    single_image,
    x1,
    y1,
    x2 - x1,
    y2 - y1
)

cv2.imwrite("output/person_bbox.jpg", single_image)

print("Bounding box saved successfully!")


# =====================================
# 3. Read Predictions CSV
# =====================================

predictions = pd.read_csv("task1_data/preds.csv")

image_name = "000000000139.jpg"

image_predictions = predictions[
    predictions["image"] == image_name
]

print("Number of detections:", len(image_predictions))


# =====================================
# 4. Filter Predictions
# =====================================

confidence_threshold = 0.5

filtered_predictions = image_predictions[
    image_predictions["confidence"] >= confidence_threshold
]

print("Detections after filtering:", len(filtered_predictions))


# =====================================
# 5. Draw Predictions
# =====================================

pred_image = image.copy()

for _, row in filtered_predictions.iterrows():

    x1 = row["x1"]
    y1 = row["y1"]
    x2 = row["x2"]
    y2 = row["y2"]

    draw_bbox(
        pred_image,
        x1,
        y1,
        x2 - x1,
        y2 - y1,
        color=(0, 255, 0)
    )

    label = f"{row['class_name']} {row['confidence']:.2f}"

    cv2.putText(
        pred_image,
        label,
        (int(x1), max(15, int(y1) - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2
    )

cv2.imwrite("output/all_bboxes.jpg", pred_image)

print("All bounding boxes saved successfully!")


# =====================================
# 6. Read YOLO Ground Truth
# =====================================

yolo_path = Path("task1_data/dataset/labels/000000000139.txt")

with open(yolo_path, "r") as file:
    lines = file.readlines()

print("YOLO Ground Truth objects:", len(lines))


# =====================================
# 7. Convert YOLO to Pixel Coordinates
# =====================================

ground_truth_boxes = []

for line in lines:

    values = line.strip().split()

    class_id = int(values[0])

    xc, yc, bw, bh = map(float, values[1:5])

    x1 = (xc - bw / 2) * width
    y1 = (yc - bh / 2) * height
    x2 = (xc + bw / 2) * width
    y2 = (yc + bh / 2) * height

    ground_truth_boxes.append({
        "class_id": class_id,
        "box": (x1, y1, x2, y2)
    })


# =====================================
# 8. Draw Ground Truth
# =====================================

gt_image = image.copy()

for gt in ground_truth_boxes:

    x1, y1, x2, y2 = gt["box"]

    draw_bbox(
        gt_image,
        x1,
        y1,
        x2 - x1,
        y2 - y1,
        color=(255, 0, 0)
    )

cv2.imwrite("output/ground_truth.jpg", gt_image)

print("Ground Truth saved successfully!")


# =====================================
# 9. Compare Predictions and Ground Truth
# =====================================

comparison_image = gt_image.copy()

for _, row in filtered_predictions.iterrows():

    x1 = row["x1"]
    y1 = row["y1"]
    x2 = row["x2"]
    y2 = row["y2"]

    draw_bbox(
        comparison_image,
        x1,
        y1,
        x2 - x1,
        y2 - y1,
        color=(0, 255, 0)
    )

cv2.imwrite("output/comparison.jpg", comparison_image)

print("Comparison image saved successfully!")


# =====================================
# 10. Test IoU Function
# =====================================

box1 = (10, 10, 100, 100)
box2 = (50, 50, 150, 150)

iou = calculate_iou(box1, box2)

print("Test IoU:", round(iou, 4))


# =====================================
# 11. Calculate IoU for First TV
# =====================================

tv_predictions = filtered_predictions[
    filtered_predictions["class_id"] == 62
]

tv_ground_truth = [
    gt for gt in ground_truth_boxes
    if gt["class_id"] == 62
]

if not tv_predictions.empty and tv_ground_truth:

    tv_prediction = tv_predictions.iloc[0]

    pred_box = (
        tv_prediction["x1"],
        tv_prediction["y1"],
        tv_prediction["x2"],
        tv_prediction["y2"]
    )

    gt_box = tv_ground_truth[0]["box"]

    tv_iou = calculate_iou(pred_box, gt_box)

    print("Actual TV IoU:", round(tv_iou, 4))


# =====================================
# 12. Calculate Best IoU for All Predictions
# =====================================

print("\n===== Best IoU Results =====")

for _, pred in filtered_predictions.iterrows():

    pred_box = (
        pred["x1"],
        pred["y1"],
        pred["x2"],
        pred["y2"]
    )

    best_iou = 0.0

    for gt in ground_truth_boxes:

        if gt["class_id"] != pred["class_id"]:
            continue

        gt_box = gt["box"]

        current_iou = calculate_iou(pred_box, gt_box)

        best_iou = max(best_iou, current_iou)

    print(
        pred["class_name"],
        "Best IoU:",
        round(best_iou, 4)
    )

print("\nProcessing completed successfully!")
# Evaluate predictions using IoU threshold
iou_threshold = 0.5

tp = 0
fp = 0
matched_gt = set()

sorted_predictions = filtered_predictions.sort_values(
    by="confidence",
    ascending=False
)

for _, pred in sorted_predictions.iterrows():

    pred_box = (
        pred["x1"],
        pred["y1"],
        pred["x2"],
        pred["y2"]
    )

    best_iou = 0
    best_gt_index = -1

    for i, gt in enumerate(ground_truth_boxes):

        if i in matched_gt:
            continue

        if gt["class_id"] != pred["class_id"]:
            continue

        iou = calculate_iou(pred_box, gt["box"])

        if iou > best_iou:
            best_iou = iou
            best_gt_index = i

    if best_iou >= iou_threshold:
        tp += 1
        matched_gt.add(best_gt_index)
    else:
        fp += 1

fn = len(ground_truth_boxes) - len(matched_gt)

print("\n===== Detection Evaluation =====")
print("True Positives:", tp)
print("False Positives:", fp)
print("False Negatives:", fn)
# Calculate Precision, Recall, and F1-score

precision = tp / (tp + fp) if (tp + fp) > 0 else 0

recall = tp / (tp + fn) if (tp + fn) > 0 else 0

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

print("\n===== Evaluation Metrics =====")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-score: {f1_score:.2%}")
# Save evaluation results to CSV

results = {
    "image": [image_name],
    "confidence_threshold": [confidence_threshold],
    "iou_threshold": [iou_threshold],
    "TP": [tp],
    "FP": [fp],
    "FN": [fn],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1_score]
}

results_df = pd.DataFrame(results)

results_df.to_csv(
    "output/evaluation_results.csv",
    index=False
)

print("Evaluation results saved successfully!")
# =====================================
# 13. Prepare Full Dataset Evaluation
# =====================================

images_dir = Path("task1_data/dataset/images")
labels_dir = Path("task1_data/dataset/labels")

all_images = sorted(images_dir.glob("*.jpg"))

print("\n===== Full Dataset Information =====")
print("Total Images:", len(all_images))

images_with_predictions = set(predictions["image"])

images_without_predictions = [
    img.name
    for img in all_images
    if img.name not in images_with_predictions
]

print("Images without Predictions:", len(images_without_predictions))
print("Missing Prediction Images:", images_without_predictions)
# =====================================
# 14. Read Ground Truth for All Images
# =====================================

all_ground_truth = {}

for img_path in all_images:

    label_path = labels_dir / (img_path.stem + ".txt")

    img = cv2.imread(str(img_path))

    if img is None:
        print("Could not read:", img_path.name)
        continue

    img_height, img_width = img.shape[:2]

    gt_boxes = []

    if label_path.exists():

        with open(label_path, "r") as file:
            for line in file:

                values = line.strip().split()

                if len(values) < 5:
                    continue

                class_id = int(values[0])
                xc, yc, bw, bh = map(float, values[1:5])

                x1 = (xc - bw / 2) * img_width
                y1 = (yc - bh / 2) * img_height
                x2 = (xc + bw / 2) * img_width
                y2 = (yc + bh / 2) * img_height

                gt_boxes.append({
                    "class_id": class_id,
                    "box": (x1, y1, x2, y2)
                })

    all_ground_truth[img_path.name] = gt_boxes

print("\n===== Full Ground Truth =====")
print("Images processed:", len(all_ground_truth))
print(
    "Total Ground Truth Objects:",
    sum(len(boxes) for boxes in all_ground_truth.values())
)

# =====================================
# 15. Evaluate Full Dataset
# =====================================

total_tp = 0
total_fp = 0
total_fn = 0

evaluation_rows = []

for img_path in all_images:

    image_name = img_path.name

    gt_boxes = all_ground_truth[image_name]

    image_preds = predictions[
        (predictions["image"] == image_name) &
        (predictions["confidence"] >= confidence_threshold)
    ].sort_values("confidence", ascending=False)

    matched_gt = set()
    image_tp = 0
    image_fp = 0

    for _, pred in image_preds.iterrows():

        pred_box = (
            pred["x1"],
            pred["y1"],
            pred["x2"],
            pred["y2"]
        )

        best_iou = 0
        best_gt_index = -1

        for i, gt in enumerate(gt_boxes):

            if i in matched_gt:
                continue

            if gt["class_id"] != pred["class_id"]:
                continue

            current_iou = calculate_iou(pred_box, gt["box"])

            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = i

        if best_iou >= iou_threshold:
            image_tp += 1
            matched_gt.add(best_gt_index)
        else:
            image_fp += 1

    image_fn = len(gt_boxes) - len(matched_gt)

    total_tp += image_tp
    total_fp += image_fp
    total_fn += image_fn

    evaluation_rows.append({
        "image": image_name,
        "TP": image_tp,
        "FP": image_fp,
        "FN": image_fn
    })

print("\n===== Full Dataset Evaluation =====")
print("Total TP:", total_tp)
print("Total FP:", total_fp)
print("Total FN:", total_fn)
# =====================================
# 16. Save Full Dataset Evaluation
# =====================================

full_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

full_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

full_f1 = (
    2 * full_precision * full_recall / (full_precision + full_recall)
    if (full_precision + full_recall) > 0 else 0
)

results_df = pd.DataFrame(evaluation_rows)

results_df.to_csv(
    "output/full_evaluation_results.csv",
    index=False
)

print("\n===== Full Dataset Metrics =====")
print(f"Precision: {full_precision:.2%}")
print(f"Recall: {full_recall:.2%}")
print(f"F1-score: {full_f1:.2%}")
print("Full evaluation results saved successfully!")
# =====================================
# 17. Visualize Evaluation Results
# =====================================

import matplotlib.pyplot as plt

metrics = ["TP", "FP", "FN"]
values = [total_tp, total_fp, total_fn]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    metrics,
    values,
    color=["green", "red", "orange"]
)

plt.title("Object Detection Evaluation - 1000 Images")
plt.xlabel("Detection Results")
plt.ylabel("Number of Objects")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(int(bar.get_height())),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("output/evaluation_chart.png")
plt.close()

print("Evaluation chart saved successfully!")
# =====================================
# 18. Visualize Precision, Recall, F1
# =====================================

metric_names = ["Precision", "Recall", "F1-score"]

metric_values = [
    full_precision * 100,
    full_recall * 100,
    full_f1 * 100
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    metric_names,
    metric_values,
    color=["green", "orange", "blue"]
)

plt.title("Object Detection Performance - 1000 Images")
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{bar.get_height():.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("output/performance_metrics.png")
plt.close()

print("Performance metrics chart saved successfully!")
from annolab.annotations import load_mot_annotations, draw_mot_annotations
import cv2

mot_path = r"C:\Lensec_Datasets\MOT17\MOT17\train\MOT17-02-DPM"

annotations = load_mot_annotations(mot_path + r"\gt\gt.txt")

image = cv2.imread(mot_path + r"\img1\000001.jpg")

image = draw_mot_annotations(image, annotations[1])

cv2.imwrite("output/mot_first_frame.jpg", image)

print("Done! First frame annotated successfully.")