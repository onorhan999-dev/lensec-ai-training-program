
import numpy as np

from annolab.preprocessing import letterbox_image


# Test letterbox output dimensions
def test_letterbox_dimensions():

    # Create a sample image
    image = np.zeros((426, 640, 3), dtype=np.uint8)

    # Apply letterbox resizing
    result, scale_x, scale_y, pad_left, pad_top = (
        letterbox_image(image, 640, 640)
    )

    # Verify output dimensions
    assert result.shape == (640, 640, 3)

    # Verify scaling factors
    assert scale_x == 1.0
    assert scale_y == 1.0

    # Verify padding
    assert pad_left == 0
    assert pad_top == 107

# Test bounding box transformation after letterbox
def test_letterbox_boxes():

    from annolab.preprocessing import letterbox_boxes

    # Original bounding box
    boxes = [
        {
            "x1": 100,
            "y1": 50,
            "x2": 300,
            "y2": 200
        }
    ]

    # Apply letterbox transformation
    transformed = letterbox_boxes(
        boxes,
        scale_x=1.0,
        scale_y=1.0,
        pad_left=0,
        pad_top=107
    )

    # Verify transformed coordinates
    assert transformed[0]["x1"] == 100
    assert transformed[0]["y1"] == 157
    assert transformed[0]["x2"] == 300
    assert transformed[0]["y2"] == 307

# Test saving and loading YOLO annotations
def test_yolo_save_and_load(tmp_path):

    from annolab.formats import (
        Box,
        save_yolo_annotations,
        load_yolo_annotations
    )

    # Create a sample bounding box
    boxes = [
        Box(
            class_id=2,
            x1=100,
            y1=150,
            x2=300,
            y2=350
        )
    ]

    # Create a temporary YOLO label file
    label_path = tmp_path / "test.txt"

    # Save annotations in YOLO format
    save_yolo_annotations(
        boxes,
        label_path,
        640,
        640
    )

    # Load saved annotations
    loaded_boxes = load_yolo_annotations(
        label_path,
        image_width=640,
        image_height=640
    )

    # Verify number of bounding boxes
    assert len(loaded_boxes) == 1

    # Verify class ID
    assert loaded_boxes[0].class_id == 2

    # Verify bounding box coordinates
    assert abs(loaded_boxes[0].x1 - 100) < 0.01
    assert abs(loaded_boxes[0].y1 - 150) < 0.01
    assert abs(loaded_boxes[0].x2 - 300) < 0.01
    assert abs(loaded_boxes[0].y2 - 350) < 0.01

# Test the complete preprocessing pipeline with YOLO labels
def test_preprocessing_pipeline_with_labels(tmp_path):

    import cv2
    import numpy as np

    from annolab.preprocessing import preprocess_image_with_labels
    from annolab.formats import (
        Box,
        save_yolo_annotations,
        load_yolo_annotations
    )

    # Create a sample image (640x426)
    image = np.zeros((426, 640, 3), dtype=np.uint8)

    image_path = tmp_path / "input.jpg"
    label_path = tmp_path / "input.txt"

    output_image = tmp_path / "output" / "image.jpg"
    output_label = tmp_path / "output" / "image.txt"

    # Save sample image
    assert cv2.imwrite(str(image_path), image)

    # Create sample bounding box
    boxes = [
        Box(
            class_id=2,
            x1=100,
            y1=50,
            x2=300,
            y2=200
        )
    ]

    # Save original YOLO annotations
    save_yolo_annotations(
        boxes,
        label_path,
        640,
        426
    )

    # Run complete preprocessing pipeline
    report = preprocess_image_with_labels(
        image_path=image_path,
        label_path=label_path,
        output_image_path=output_image,
        output_label_path=output_label,
        target_width=640,
        target_height=640
    )

    # Verify output files exist
    assert output_image.exists()
    assert output_label.exists()

    # Verify output image dimensions
    processed_image = cv2.imread(str(output_image))
    assert processed_image.shape[:2] == (640, 640)

    # Load transformed YOLO annotations
    transformed_boxes = load_yolo_annotations(
        output_label,
        image_width=640,
        image_height=640
    )

    # Verify transformed bounding box
    assert len(transformed_boxes) == 1

    box = transformed_boxes[0]

    assert box.class_id == 2
    assert abs(box.x1 - 100) < 0.01
    assert abs(box.y1 - 157) < 0.01
    assert abs(box.x2 - 300) < 0.01
    assert abs(box.y2 - 307) < 0.01

    # Verify processing report
    assert report["bounding_boxes"] == 1


# Test bounding boxes at image boundaries
def test_bounding_box_clamping(tmp_path):

    import cv2
    import numpy as np

    from annolab.preprocessing import preprocess_image_with_labels
    from annolab.formats import load_yolo_annotations

    # Create a sample image
    image = np.zeros((640, 640, 3), dtype=np.uint8)

    image_path = tmp_path / "input.jpg"
    label_path = tmp_path / "input.txt"

    output_image = tmp_path / "output.jpg"
    output_label = tmp_path / "output.txt"

    # Save sample image
    assert cv2.imwrite(str(image_path), image)

    # Create a YOLO box slightly beyond the right boundary
    label_path.write_text(
        "2 0.98 0.55 0.05 0.02\n",
        encoding="utf-8"
    )

    # Run preprocessing pipeline
    preprocess_image_with_labels(
        image_path=image_path,
        label_path=label_path,
        output_image_path=output_image,
        output_label_path=output_label,
        target_width=640,
        target_height=640
    )

    # Load corrected annotations
    boxes = load_yolo_annotations(
        output_label,
        image_width=640,
        image_height=640
    )

    # Verify bounding box is inside image boundaries
    assert len(boxes) == 1

    box = boxes[0]

    assert box.class_id == 2
    assert 0 <= box.x1 < box.x2 <= 640
    assert 0 <= box.y1 < box.y2 <= 640

# Test bounding boxes at image boundaries
def test_bounding_box_clamping(tmp_path):

    import cv2
    import numpy as np

    from annolab.preprocessing import preprocess_image_with_labels
    from annolab.formats import load_yolo_annotations

    # Create a sample image
    image = np.zeros((640, 640, 3), dtype=np.uint8)

    image_path = tmp_path / "input.jpg"
    label_path = tmp_path / "input.txt"

    output_image = tmp_path / "output.jpg"
    output_label = tmp_path / "output.txt"

    # Save sample image
    assert cv2.imwrite(str(image_path), image)

    # Create a YOLO box slightly beyond the right boundary
    label_path.write_text(
        "2 0.98 0.55 0.05 0.02\n",
        encoding="utf-8"
    )

    # Run preprocessing pipeline
    preprocess_image_with_labels(
        image_path=image_path,
        label_path=label_path,
        output_image_path=output_image,
        output_label_path=output_label,
        target_width=640,
        target_height=640
    )

    # Load corrected annotations
    boxes = load_yolo_annotations(
        output_label,
        image_width=640,
        image_height=640
    )

    # Verify bounding box is inside image boundaries
    assert len(boxes) == 1

    box = boxes[0]

    assert box.class_id == 2
    assert 0 <= box.x1 < box.x2 <= 640
    assert 0 <= box.y1 < box.y2 <= 640

# Test removal of bounding boxes outside image boundaries
def test_invalid_bounding_box_removal(tmp_path):

    import cv2
    import numpy as np

    from annolab.preprocessing import preprocess_image_with_labels
    from annolab.formats import load_yolo_annotations

    # Create a sample image
    image = np.zeros((640, 640, 3), dtype=np.uint8)

    image_path = tmp_path / "input.jpg"
    label_path = tmp_path / "input.txt"

    output_image = tmp_path / "output.jpg"
    output_label = tmp_path / "output.txt"

    # Save sample image
    assert cv2.imwrite(str(image_path), image)

    # Create a YOLO box completely outside the image
    label_path.write_text(
        "2 1.20 0.50 0.10 0.10\n",
        encoding="utf-8"
    )

    # Run preprocessing pipeline
    report = preprocess_image_with_labels(
        image_path=image_path,
        label_path=label_path,
        output_image_path=output_image,
        output_label_path=output_label,
        target_width=640,
        target_height=640
    )

    # Load output annotations
    boxes = load_yolo_annotations(
        output_label,
        image_width=640,
        image_height=640
    )

    # Verify invalid box was removed
    assert len(boxes) == 0
    assert report["bounding_boxes"] == 0
def test_horizontal_flip_boxes():
    from annolab.preprocessing import horizontal_flip_boxes
    from annolab.formats import Box

    # Create an original bounding box
    boxes = [
        Box(class_id=0, x1=100, y1=50, x2=200, y2=150)
    ]

    # Flip the bounding box horizontally
    flipped = horizontal_flip_boxes(boxes, image_width=640)

    # Verify the new coordinates
    assert flipped[0].x1 == 440
    assert flipped[0].x2 == 540
    assert flipped[0].y1 == 50
    assert flipped[0].y2 == 150
    assert flipped[0].class_id == 0

def test_flip_with_letterbox_and_labels(tmp_path):
    import cv2
    import numpy as np

    from annolab.preprocessing import preprocess_image_with_labels
    from annolab.formats import Box, save_yolo_annotations, load_yolo_annotations

    # Create a test image: width=200, height=100
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    image_path = tmp_path / "original.jpg"
    label_path = tmp_path / "original.txt"
    output_image = tmp_path / "processed.jpg"
    output_label = tmp_path / "processed.txt"

    cv2.imwrite(str(image_path), image)

    # Original bounding box
    boxes = [
        Box(class_id=0, x1=20, y1=10, x2=60, y2=40)
    ]

    save_yolo_annotations(boxes, label_path, 200, 100)

    # Apply horizontal flip followed by letterbox
    preprocess_image_with_labels(
        image_path=image_path,
        label_path=label_path,
        output_image_path=output_image,
        output_label_path=output_label,
        target_width=640,
        target_height=640,
        filters=["flip"]
    )

    # Read the transformed annotation
    result = load_yolo_annotations(output_label, 640, 640)

    assert len(result) == 1

    # Flip: x1=140, x2=180
    # Scale: 640/200 = 3.2
    # Vertical padding: 160
    assert abs(result[0].x1 - 448) < 1
    assert abs(result[0].x2 - 576) < 1
    assert abs(result[0].y1 - 192) < 1
    assert abs(result[0].y2 - 288) < 1




def test_grayscale_pipeline(tmp_path):
    import cv2
    import numpy as np
    from annolab.preprocessing import preprocess_image

    # Create a sample color image
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :] = (50, 100, 150)

    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "grayscale.jpg"

    cv2.imwrite(str(input_path), image)

    # Apply grayscale and CLAHE
    report = preprocess_image(
        image_path=input_path,
        output_path=output_path,
        filters=["grayscale", "clahe"]
    )

    # Read the saved image in grayscale mode
    result = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)

    # Verify image dimensions and single channel
    assert result is not None
    assert result.shape == (640, 640)
    assert report["output_size"] == [640, 640]



def test_batch_processing_with_flip(tmp_path):
    import cv2
    import numpy as np
    from annolab.preprocessing import process_dataset
    from annolab.formats import load_yolo_annotations

    # Create temporary dataset folders
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "output"

    images_dir.mkdir()
    labels_dir.mkdir()

    # Create a sample image
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "test.jpg"), image)

    # Create a YOLO bounding box on the left side
    (labels_dir / "test.txt").write_text(
        "0 0.25 0.5 0.2 0.4\n"
    )

    # Process image and labels with horizontal flip
    processed, failed = process_dataset(
        images_dir,
        labels_dir,
        output_dir,
        limit=1,
        filters=["flip"]
    )

    assert processed == 1
    assert failed == 0

    # Verify flipped bounding box position
    boxes = load_yolo_annotations(
        output_dir / "labels" / "test.txt",
        640,
        640
    )

    assert len(boxes) == 1

    # Original center x = 0.25, flipped center x = 0.75
    # Letterbox adds vertical padding only
    center_x = (boxes[0].x1 + boxes[0].x2) / 2

    assert abs(center_x - 480) < 1

def test_batch_resume(tmp_path):
    import cv2
    import numpy as np
    from annolab.preprocessing import process_dataset

    # Create temporary dataset
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "output"

    images_dir.mkdir()
    labels_dir.mkdir()

    # Create sample image and YOLO label
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "test.jpg"), image)

    (labels_dir / "test.txt").write_text(
        "0 0.5 0.5 0.2 0.4\n"
    )

    # First run: process the image
    processed, failed = process_dataset(
        images_dir,
        labels_dir,
        output_dir,
        resume=False
    )

    assert processed == 1
    assert failed == 0

    # Second run: skip completed image
    processed, failed = process_dataset(
        images_dir,
        labels_dir,
        output_dir,
        resume=True
    )

    assert processed == 0
    assert failed == 0



def test_batch_resume_corrupted_image(tmp_path):
    import cv2
    import numpy as np
    from annolab.preprocessing import process_dataset

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "output"

    images_dir.mkdir()
    labels_dir.mkdir()

    # Create sample image and label
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "test.jpg"), image)
    (labels_dir / "test.txt").write_text(
        "0 0.5 0.5 0.2 0.4\n"
    )

    # Process image for the first time
    processed, failed = process_dataset(
        images_dir, labels_dir, output_dir
    )
    assert processed == 1
    assert failed == 0

    # Corrupt the saved image
    output_image = output_dir / "images" / "test.jpg"
    output_image.write_bytes(b"invalid image data")

    # Resume should reprocess the corrupted image
    processed, failed = process_dataset(
        images_dir, labels_dir, output_dir, resume=True
    )

    assert processed == 1
    assert failed == 0
    assert cv2.imread(str(output_image)) is not None

def test_batch_resume_corrupted_label(tmp_path):
    import cv2
    import numpy as np
    from annolab.preprocessing import process_dataset

    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    output_dir = tmp_path / "output"

    images_dir.mkdir()
    labels_dir.mkdir()

    # Create a sample image and valid YOLO label
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "test.jpg"), image)

    (labels_dir / "test.txt").write_text(
        "0 0.5 0.5 0.2 0.4\n"
    )

    # First run
    processed, failed = process_dataset(
        images_dir, labels_dir, output_dir
    )

    assert processed == 1
    assert failed == 0

    # Corrupt the output label with invalid coordinates
    output_label = output_dir / "labels" / "test.txt"
    output_label.write_text("0 2.0 0.5 0.2 0.4\n")

    # Resume should reprocess the image and label
    processed, failed = process_dataset(
        images_dir, labels_dir, output_dir, resume=True
    )

    assert processed == 1
    assert failed == 0