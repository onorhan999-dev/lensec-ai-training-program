
# ============================================================
# STEP 16.3: VERIFY PARALLEL YOLO LABELS
# ============================================================
# Purpose:
# - Verify processed image dimensions.
# - Check that YOLO label files exist.
# - Validate bounding box coordinates.
# - Print details of invalid bounding boxes.
# ============================================================

from pathlib import Path
import cv2

from annolab.formats import load_yolo_annotations


# Verify processed images and annotations
def main():

    images_dir = Path("output/parallel_with_labels/images")
    labels_dir = Path("output/parallel_with_labels/labels")

    images = sorted(images_dir.glob("*.jpg"))

    total_boxes = 0
    invalid_boxes = 0

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot read: {image_path}")

        height, width = image.shape[:2]

        # Verify image dimensions
        assert (width, height) == (640, 640), (
            f"Invalid image size: {image_path.name}"
        )

        # Find corresponding YOLO label
        label_path = labels_dir / f"{image_path.stem}.txt"

        assert label_path.exists(), (
            f"Missing label: {label_path}"
        )

        # Load bounding boxes
        boxes = load_yolo_annotations(
            label_path,
            image_width=width,
            image_height=height
        )

        # Validate each bounding box
        for box in boxes:

            total_boxes += 1

            if not (
                0 <= box.x1 < box.x2 <= width
                and 0 <= box.y1 < box.y2 <= height
            ):
                invalid_boxes += 1

                # Print invalid bounding box details
                print("\nINVALID BOUNDING BOX FOUND")
                print("Image:", image_path.name)
                print("Class ID:", box.class_id)
                print("x1:", box.x1)
                print("y1:", box.y1)
                print("x2:", box.x2)
                print("y2:", box.y2)
                print("Image width:", width)
                print("Image height:", height)

    # Print verification results
    print("\n========== VERIFICATION RESULTS ==========")
    print("Images checked:", len(images))
    print("Bounding boxes checked:", total_boxes)
    print("Invalid bounding boxes:", invalid_boxes)

    assert len(images) == 10, "Unexpected image count"
    assert invalid_boxes == 0, "Invalid bounding boxes found"

    print("Parallel YOLO verification passed!")


if __name__ == "__main__":
    main()
