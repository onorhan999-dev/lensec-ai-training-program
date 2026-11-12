
from pathlib import Path
import cv2

from annolab.formats import load_yolo_annotations


def main():

    # Define input and output directories
    images_dir = Path("output/parallel_with_labels/images")
    labels_dir = Path("output/parallel_with_labels/labels")
    output_dir = Path("output/visual_verification")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select processed images
    images = sorted(images_dir.glob("*.jpg"))

    for image_path in images:

        # Read processed image
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        height, width = image.shape[:2]

        # Load corresponding YOLO annotations
        label_path = labels_dir / f"{image_path.stem}.txt"

        boxes = load_yolo_annotations(
            label_path,
            image_width=width,
            image_height=height
        )

        # Draw bounding boxes on the image
        for box in boxes:

            x1, y1 = int(box.x1), int(box.y1)
            x2, y2 = int(box.x2), int(box.y2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display class ID
            cv2.putText(
                image,
                f"Class {box.class_id}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # Save annotated image
        output_path = output_dir / image_path.name

        if not cv2.imwrite(str(output_path), image):
            raise IOError(f"Cannot save: {output_path}")

        print("Saved:", output_path)

    print("\nVisual verification images generated!")
    print("Total images:", len(images))


if __name__ == "__main__":
    main()
