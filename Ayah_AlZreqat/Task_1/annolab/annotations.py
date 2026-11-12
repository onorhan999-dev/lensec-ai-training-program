
import cv2


# 1. Draw Bounding Box
def draw_bbox(image, x, y, w, h, color=(0, 255, 0), thickness=2):
    x1 = int(x)
    y1 = int(y)

    x2 = int(x + w)
    y2 = int(y + h)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    return image


# 2. Calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# 3. Load MOT17 Ground Truth Annotations
def load_mot_annotations(file_path):
    annotations = {}

    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split(",")

            frame_id = int(values[0])
            track_id = int(values[1])

            x = float(values[2])
            y = float(values[3])
            w = float(values[4])
            h = float(values[5])

            flag = int(values[6])
            class_id = int(values[7])

            # Keep valid pedestrians only
            if flag != 1 or class_id != 1:
                continue

            if frame_id not in annotations:
                annotations[frame_id] = []

            annotations[frame_id].append({
                "track_id": track_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

    return annotations


# 4. Draw MOT17 Bounding Boxes, Track IDs, and Trails
def draw_mot_annotations(image, objects, track_history=None):

    if track_history is None:
        track_history = {}

    for obj in objects:
        track_id = obj["track_id"]

        x = int(obj["x"])
        y = int(obj["y"])
        w = int(obj["w"])
        h = int(obj["h"])

        # Generate a stable color for each Track ID
        color = (
            (track_id * 37) % 256,
            (track_id * 67) % 256,
            (track_id * 97) % 256
        )

        # Draw Bounding Box
        draw_bbox(image, x, y, w, h, color=color)

        # Draw Track ID
        cv2.putText(
            image,
            f"ID {track_id}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # Calculate bottom-center of the person
        center_x = x + w // 2
        center_y = y + h

        # Create history for new Track IDs
        if track_id not in track_history:
            track_history[track_id] = []

        # Save current position
        track_history[track_id].append(
            (center_x, center_y)
        )

        # Keep only the last 30 positions
        track_history[track_id] = track_history[track_id][-30:]

        # Draw movement trail
        points = track_history[track_id]

        for i in range(1, len(points)):
            cv2.line(
                image,
                points[i - 1],
                points[i],
                color,
                2
            )

    return image
