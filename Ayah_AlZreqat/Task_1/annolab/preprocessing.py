
import cv2
from pathlib import Path
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
# ============================================================
# STEP 1: IMAGE RESIZING
# ============================================================
# Resize an image to a fixed width and height.
#
# Purpose:
# - Standardize image dimensions.
# - Save the resized image.
#
# Note:
# Direct resizing may distort the original aspect ratio.
# ============================================================

def resize_image(input_path, output_path, width=640, height=640):

    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    original_height, original_width = image.shape[:2]

    print("Original size:", original_width, "x", original_height)

    resized_image = cv2.resize(image, (width, height))

    new_height, new_width = resized_image.shape[:2]

    print("Resized size:", new_width, "x", new_height)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), resized_image)

    if not success:
        raise IOError(f"Could not save image: {output_path}")

    print("Image resized successfully!")
    print("Saved to:", output_path)

    return resized_image


# ============================================================
# STEP 2: RESIZE BOUNDING BOXES
# ============================================================
# Update bounding box coordinates after image resizing.
#
# Purpose:
# - Calculate horizontal and vertical scaling factors.
# - Keep bounding boxes aligned with their objects.
# ============================================================

def resize_boxes(boxes, old_width, old_height, new_width, new_height):

    scale_x = new_width / old_width
    scale_y = new_height / old_height

    resized_boxes = []

    for box in boxes:
        resized_boxes.append({
            "x1": box["x1"] * scale_x,
            "y1": box["y1"] * scale_y,
            "x2": box["x2"] * scale_x,
            "y2": box["y2"] * scale_y
        })

    return resized_boxes


# ============================================================
# STEP 3: LETTERBOX RESIZING
# ============================================================
# Resize an image while preserving its aspect ratio.
#
# Purpose:
# - Avoid image distortion.
# - Add padding to reach the target dimensions.
# - Return scaling factors and padding offsets.
#
# Example:
# Original: 640x426
# Target:   640x640
# Padding:  107 pixels at the top and bottom.
# ============================================================

def letterbox_image(image, target_width=640, target_height=640):

    if image is None:
        raise ValueError("Invalid image")

    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive")

    old_height, old_width = image.shape[:2]

    scale = min(
        target_width / old_width,
        target_height / old_height
    )

    new_width = min(target_width, max(1, round(old_width * scale)))
    new_height = min(target_height, max(1, round(old_height * scale)))

    resized = cv2.resize(image, (new_width, new_height))

    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left

    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top

    result = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return (
        result,
        new_width / old_width,
        new_height / old_height,
        pad_left,
        pad_top
    )


# ============================================================
# STEP 4: LETTERBOX BOUNDING BOX TRANSFORMATION
# ============================================================
# Update bounding box coordinates after letterbox resizing.
#
# Purpose:
# - Apply image scaling factors.
# - Add padding offsets.
# - Preserve correct object locations.
# ============================================================

def letterbox_boxes(boxes, scale_x, scale_y, pad_left, pad_top):

    new_boxes = []

    for box in boxes:
        new_boxes.append({
            "x1": box["x1"] * scale_x + pad_left,
            "y1": box["y1"] * scale_y + pad_top,
            "x2": box["x2"] * scale_x + pad_left,
            "y2": box["y2"] * scale_y + pad_top
        })

    return new_boxes


# ============================================================
# STEP 5: DATASET BATCH PROCESSING
# ============================================================
# Process multiple images and YOLO annotations.
#
# Purpose:
# - Load images and labels.
# - Apply letterbox resizing.
# - Transform bounding boxes.
# - Save processed images and updated YOLO labels.
# - Measure processing time and count failures.
#
# Previous test:
# 10 images processed successfully, 0 failures.
# ============================================================

   
def _process_single_image(
    image_path,
    labels_dir,
    output_images,
    output_labels,
    target_width,
    target_height,
    filters,
    resume
):
    """Process one image and its YOLO annotations."""

    # Convert paths to Path objects
    image_path = Path(image_path)
    labels_dir = Path(labels_dir)
    output_images = Path(output_images)
    output_labels = Path(output_labels)

    # Read the input image
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    old_height, old_width = image.shape[:2]

    return old_width, old_height
 

def process_dataset(
    images_dir,
    labels_dir,
    output_dir,
    limit=10,
    target_width=640,
    target_height=640,
    filters=None,
    resume=False,
    workers=1
):
    from annolab.formats import (
        load_yolo_annotations,
        Box,
        save_yolo_annotations
    )

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    output_images = output_dir / "images"
    output_labels = output_dir / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg"))

    if limit is not None:
        image_files = image_files[:limit]

    if not image_files:
        raise ValueError("No JPG images found")

    processed = 0
    failed = 0

    start_time = time.perf_counter()

    for image_path in image_files:

        try:
                    # Define output paths
            image_output = output_images / image_path.name
            label_output = output_labels / f"{image_path.stem}.txt"

            # Resume only when output files are valid
            if resume and image_output.exists() and label_output.exists():
                saved_image = cv2.imread(
                    str(image_output),
                    cv2.IMREAD_UNCHANGED
                )

                if (
                    saved_image is not None
                    and saved_image.shape[:2] == (target_height, target_width)
                ):
                    # Validate saved YOLO labels
                    label_valid = True

                    try:
                        with open(label_output, "r", encoding="utf-8") as file:
                            for line in file:
                                if not line.strip():
                                    continue

                                parts = line.split()

                                if len(parts) != 5:
                                    label_valid = False
                                    break

                                class_id, x, y, w, h = map(float, parts)

                                if (
                                    not all(np.isfinite([class_id, x, y, w, h]))
                                    or not class_id.is_integer()
                                    or class_id < 0
                                    or not (0 <= x <= 1)
                                    or not (0 <= y <= 1)
                                    or not (0 < w <= 1)
                                    or not (0 < h <= 1)
                                    or x - w / 2 < -1e-6
                                    or x + w / 2 > 1 + 1e-6
                                    or y - h / 2 < -1e-6
                                    or y + h / 2 > 1 + 1e-6
                                ):
                                    label_valid = False
                                    break

                    except (ValueError, OSError):
                        label_valid = False

                    if label_valid:
                        print(f"Skipping completed: {image_path.name}")
                        continue

            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError("Could not read image")

            old_height, old_width = image.shape[:2]

            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing label: {label_path}"
                )

            boxes = load_yolo_annotations(
                label_path,
                old_width,
                old_height
            )

            box_dicts = [
                {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "class_id": box.class_id
                }
                for box in boxes
            ]

            # Apply selected preprocessing filters
            selected_filters = filters if filters is not None else []

            processed_image = apply_image_filters(
                image,
                selected_filters
            )


            # Apply letterbox to the filtered image
            result, scale_x, scale_y, pad_left, pad_top = (
                letterbox_image(
                    processed_image,
                    target_width,
                    target_height
                )
            )
                        # Flip bounding boxes when horizontal flip is applied
            flip_count = selected_filters.count("flip")

            if flip_count % 2 == 1:
                box_dicts = [
                    {
                        **box,
                        "x1": old_width - box["x2"],
                        "x2": old_width - box["x1"]
                    }
                    for box in box_dicts
                ]


            # Apply letterbox transformation to bounding boxes
            transformed = letterbox_boxes(
                box_dicts,
                scale_x,
                scale_y,
                pad_left,
                pad_top
            ) 

            new_boxes = [
                Box(
                    class_id=old_box["class_id"],
                    x1=new_box["x1"],
                    y1=new_box["y1"],
                    x2=new_box["x2"],
                    y2=new_box["y2"]
                )
                for old_box, new_box in zip(
                    box_dicts,
                    transformed
                )
            ]

            image_output = output_images / image_path.name
            label_output = output_labels / f"{image_path.stem}.txt"

            if not cv2.imwrite(str(image_output), result):
                raise IOError("Could not save image")

            save_yolo_annotations(
                new_boxes,
                label_output,
                target_width,
                target_height
            )

            processed += 1

        except Exception as error:
            failed += 1
            print(f"Failed: {image_path.name} - {error}")

    elapsed = time.perf_counter() - start_time

    print("\nBatch Processing Results")
    print("Total images selected:", len(image_files))
    print("Successfully processed:", processed)
    print("Failed:", failed)
    print("Processing time:", round(elapsed, 2), "seconds")

    return processed, failed


# ============================================================
# STEP 6: BLUR DETECTION
# ============================================================
# Measure image sharpness using Laplacian variance.
#
# Purpose:
# - Detect potentially blurry images.
# - Compare sharpness against a threshold.
#
# Previous test:
# Original image: 854.18 (Not blurry)
# Blurred image: 3.63 (Blurry)
#
# Note:
# The threshold is experimental.
# ============================================================

def detect_blur(image, threshold=100):

    if image is None:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    is_blurry = blur_score < threshold

    return blur_score, is_blurry


# ============================================================
# STEP 7: BRIGHTNESS AND CONTRAST ASSESSMENT
# ============================================================
# Measure brightness and contrast using grayscale pixels.
#
# Purpose:
# - Calculate average grayscale intensity.
# - Calculate grayscale standard deviation.
# - Identify potential image quality issues.
#
# Previous test:
# Original: Brightness=110.77, Contrast=49.50
# Dark:     Brightness=55.38, Contrast=24.75
# Bright:   Brightness=189.85, Contrast=48.31
# ============================================================

def assess_brightness_contrast(image):

    if image is None:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()
    contrast = gray.std()

    return brightness, contrast


# ============================================================
# STEP 8: IMAGE ENHANCEMENT USING CLAHE
# ============================================================
# CLAHE = Contrast Limited Adaptive Histogram Equalization.
#
# Purpose:
# - Improve local image contrast.
# - Enhance details in low-contrast regions.
# - Limit excessive contrast amplification.
#
# Previous test:
# Before: Brightness=110.77, Contrast=49.50
# After:  Brightness=121.40, Contrast=57.20
# ============================================================

def enhance_image_clahe(image):

    if image is None:
        raise ValueError("Invalid image")
        # Apply CLAHE directly to grayscale images
    if image.ndim == 2:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge(
        (enhanced_l, a, b)
    )

    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image


# ============================================================
# STEP 9: DIFFERENCE HASH (dHash)
# ============================================================
# Generate a perceptual fingerprint for an image.
#
# Purpose:
# - Represent an image using a compact hash.
# - Help detect visually similar or duplicate images.
#
# Previous test:
# Original vs Copy:     0
# Original vs Modified: 0
# Original vs Flipped: 35
#
# Note:
# Equal hashes do not guarantee pixel-level equality.
# ============================================================


def calculate_dhash(image, hash_size=8):

    if image is None:
        raise ValueError("Invalid image")

    if hash_size <= 0:
        raise ValueError("hash_size must be positive")

    # Support both color and grayscale images
    gray = image if image.ndim == 2 else cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    # Resize image for hash calculation
    resized = cv2.resize(
        gray,
        (hash_size + 1, hash_size)
    )

    # Compare neighboring pixels
    differences = resized[:, 1:] > resized[:, :-1]

    hash_value = 0

    for value in differences.flatten():
        hash_value = (hash_value << 1) | int(value)

    return hash_value



# ============================================================
# STEP 10: HAMMING DISTANCE
# ============================================================
# Compare two image hashes.
#
# Purpose:
# - Count the number of differing bits.
# - Estimate similarity between image fingerprints.
#
# Smaller distance = more similar hashes.
# Larger distance  = more different hashes.
# ============================================================

def hamming_distance(hash1, hash2):

    return (hash1 ^ hash2).bit_count()


# ============================================================
# STEP 11: INTEGRATED PREPROCESSING PIPELINE
# ============================================================
# Combine image preprocessing functions into one workflow.
#
# Workflow:
# 1. Read the input image.
# 2. Assess blur, brightness, and contrast.
# 3. Optionally apply CLAHE enhancement.
# 4. Apply letterbox resizing.
# 5. Generate a dHash fingerprint.
# 6. Save the processed image.
# 7. Return a quality and processing report.
#
# Note:
# Bounding box integration will be added later.
# ============================================================

def preprocess_image(
    image_path,
    output_path,
    target_width=640,
    target_height=640,
    apply_clahe=False,
    filters=None
):
    
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Assess original image quality
    blur_score, is_blurry = detect_blur(image)

    brightness, contrast = assess_brightness_contrast(image)

   
    # Apply selected image filters before letterbox resizing
    selected_filters = list(filters) if filters is not None else []

    # Preserve compatibility with the existing CLAHE option
    if apply_clahe and "clahe" not in selected_filters:
        selected_filters.insert(0, "clahe")

   

  


    #Apply selected filters before letterbox
    processed = apply_image_filters(image, selected_filters)
    # Letterbox resizing
    processed, scale_x, scale_y, pad_left, pad_top = (
        letterbox_image(
            processed,
            target_width,
            target_height
        )
    )

    # Generate fingerprint of processed image
    image_hash = calculate_dhash(processed)

    # Save processed image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), processed):
        raise IOError(f"Could not save image: {output_path}")

    # Generate report
    report = {
        "image": str(image_path),
        "output": str(output_path),
        "blur_score": round(float(blur_score), 2),
        "is_blurry": bool(is_blurry),
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
        "clahe_applied": bool(apply_clahe),
        "output_size": [target_width, target_height],
        "scale_x": scale_x,
        "scale_y": scale_y,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "dhash": format(image_hash, "016x")
    }

    return report


# ============================================================
# STEP 12: YOLO ANNOTATION INTEGRATION
# ============================================================
# Process an image and its YOLO bounding boxes together.
#
# Purpose:
# - Load original YOLO annotations.
# - Apply image preprocessing.
# - Transform bounding boxes using letterbox parameters.
# - Save updated YOLO annotations.
# ============================================================

def preprocess_image_with_labels(
    image_path,
    label_path,
    output_image_path,
    output_label_path,
    target_width=640,
    target_height=640,
    apply_clahe=False,
    filters=None
):
    from annolab.formats import (
        load_yolo_annotations,
        save_yolo_annotations,
        Box
    )

    # Read original image
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    old_height, old_width = image.shape[:2]

    # Load YOLO annotations
    boxes = load_yolo_annotations(
        label_path,
        old_width,
        old_height
    )

    # Process image and get transformation report
        # Process image and get transformation report
    report = preprocess_image(
        image_path=image_path,
        output_path=output_image_path,
        target_width=target_width,
        target_height=target_height,
        apply_clahe=apply_clahe,
        filters=filters
    )
        # Flip bounding boxes if the image was flipped
    if filters is not None and filters.count("flip") % 2 == 1:
        boxes = horizontal_flip_boxes(boxes, old_width) 
    # Transform bounding boxes
    new_boxes = []

    for box in boxes:

        # Apply letterbox scaling and padding
        x1 = box.x1 * report["scale_x"] + report["pad_left"]
        y1 = box.y1 * report["scale_y"] + report["pad_top"]
        x2 = box.x2 * report["scale_x"] + report["pad_left"]
        y2 = box.y2 * report["scale_y"] + report["pad_top"]

        # Clamp coordinates to image boundaries
        x1 = max(0.0, min(x1, target_width))
        y1 = max(0.0, min(y1, target_height))
        x2 = max(0.0, min(x2, target_width))
        y2 = max(0.0, min(y2, target_height))

        # Skip invalid bounding boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # Create corrected bounding box
        new_boxes.append(
            Box(
                class_id=box.class_id,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2
            )
        )

   
    # Create output directory
    output_label_path = Path(output_label_path)
    output_label_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save transformed YOLO annotations
    save_yolo_annotations(
        new_boxes,
        output_label_path,
        target_width,
        target_height
    )

    report["bounding_boxes"] = len(new_boxes)
    report["output_label"] = str(output_label_path)

    return report

# ============================================================
# MAIN: DIRECT FILE EXECUTION
# ============================================================
# This block runs only when preprocessing.py is executed
# directly. It does not run when functions are imported.
# ============================================================

# ============================================================
# STEP 13: IMAGE DENOISING
# ============================================================
# Reduce image noise using a Gaussian filter.
#
# Purpose:
# - Smooth small pixel variations.
# - Reduce unwanted image noise.
# - Preserve the original image dimensions.
# ============================================================

def denoise_image(image, kernel_size=5):
    """
    Apply Gaussian blur to reduce image noise.

    Args:
        image: Input image as a NumPy array.
        kernel_size: Odd kernel size (e.g., 3, 5, 7).

    Returns:
        Denoised image with the same dimensions.
    """

    if image is None:
        raise ValueError("Invalid image")

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number")

    denoised = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=0
    )

    return denoised



# ============================================================
# STEP 14: IMAGE SHARPENING
# ============================================================
# Enhance image details using an unsharp mask.
#
# Purpose:
# - Improve edge sharpness.
# - Highlight fine image details.
# - Preserve the original image dimensions.
# ============================================================

def sharpen_image(image, amount=1.0, kernel_size=5):
    """
    Sharpen an image using an unsharp mask.

    Args:
        image: Input image as a NumPy array.
        amount: Sharpening strength.
        kernel_size: Gaussian blur kernel size.

    Returns:
        Sharpened image with the same dimensions.
    """

    if image is None:
        raise ValueError("Invalid image")

    if amount < 0:
        raise ValueError("Amount must be non-negative")

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number")

    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=0
    )

    # Enhance details by subtracting the blurred image
    sharpened = cv2.addWeighted(
        image,
        1.0 + amount,
        blurred,
        -amount,
        0
    )

    return sharpened



# ============================================================
# STEP 15: GAMMA CORRECTION
# ============================================================
# Adjust image brightness using gamma correction.
#
# Purpose:
# - Brighten dark images.
# - Darken overly bright images.
# - Preserve image dimensions.
# ============================================================

def adjust_gamma(image, gamma=1.0):
    """
    Apply gamma correction to an image.

    Args:
        image: Input image as a NumPy array.
        gamma: Correction value (must be positive).

    Returns:
        Gamma-corrected image.
    """

    if image is None:
        raise ValueError("Invalid image")

    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero")

    # Create a lookup table for gamma correction
    inv_gamma = 1.0 / gamma

    table = [
        int(((i / 255.0) ** inv_gamma) * 255)
        for i in range(256)
    ]

    # Convert the lookup table to uint8
    
    table = np.array(table, dtype=np.uint8)

    # Apply gamma correction
    corrected = cv2.LUT(image, table)

    return corrected










# ============================================================
# STEP 16: GRAYSCALE CONVERSION
# ============================================================
# Convert a color image to grayscale.
#
# Purpose:
# - Remove color information.
# - Reduce the number of image channels.
# - Preserve image width and height.
# ============================================================

def convert_grayscale(image):
    """
    Convert a BGR image to grayscale.

    Args:
        image: Input BGR image as a NumPy array.

    Returns:
        Grayscale image with one channel.
    """

    if image is None:
        raise ValueError("Invalid image")

    # Convert BGR image to grayscale
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return grayscale


# ============================================================
# STEP 17: HORIZONTAL IMAGE FLIP
# ============================================================
# Flip an image horizontally.
#
# Purpose:
# - Create a mirrored version of the image.
# - Support data augmentation.
# - Preserve image dimensions.
# ============================================================

def horizontal_flip(image):
    """
    Flip an image horizontally.

    Args:
        image: Input image as a NumPy array.

    Returns:
        Horizontally flipped image.
    """

    if image is None:
        raise ValueError("Invalid image")

    # Flip image from left to right
    flipped = cv2.flip(image, 1)

    return flipped




# ============================================================
# STEP 18: HORIZONTAL BOUNDING BOX FLIP
# ============================================================
# Update bounding boxes after horizontal image flipping.
#
# Purpose:
# - Mirror bounding box coordinates horizontally.
# - Keep annotations aligned with flipped objects.
# - Preserve class IDs and vertical coordinates.
# ============================================================

def horizontal_flip_boxes(boxes, image_width):
    """
    Flip bounding boxes horizontally.

    Args:
        boxes: List of Box objects.
        image_width: Width of the original image.

    Returns:
        List of horizontally flipped Box objects.
    """

    from annolab.formats import Box

    if image_width <= 0:
        raise ValueError("Image width must be positive")

    flipped_boxes = []

    for box in boxes:
        # Mirror horizontal coordinates
        new_x1 = image_width - box.x2
        new_x2 = image_width - box.x1

        # Keep vertical coordinates unchanged
        flipped_boxes.append(
            Box(
                class_id=box.class_id,
                x1=new_x1,
                y1=box.y1,
                x2=new_x2,
                y2=box.y2
            )
        )

    return flipped_boxes





# ============================================================
# STEP 19: CHAINABLE IMAGE FILTERS
# ============================================================
# Apply multiple preprocessing filters in a chosen order.
#
# Purpose:
# - Combine image enhancement operations.
# - Allow users to choose which filters to apply.
# - Keep the original image unchanged.
# ============================================================

def apply_image_filters(image, filters=None):
    """
    Apply a sequence of image filters.

    Args:
        image: Input BGR image.
        filters: List of filter names in execution order.

    Returns:
        Processed image.
    """

    if image is None:
        raise ValueError("Invalid image")

    if filters is None:
        filters = []

    processed = image.copy()

    for filter_name in filters:

        if filter_name == "denoise":
            processed = denoise_image(processed)

        elif filter_name == "clahe":
            processed = enhance_image_clahe(processed)

        elif filter_name == "sharpen":
            processed = sharpen_image(processed)

        elif filter_name == "gamma":
            processed = adjust_gamma(processed, gamma=1.5)

        elif filter_name == "grayscale":
            processed = convert_grayscale(processed)

        elif filter_name == "flip":
            processed = horizontal_flip(processed)

        else:
            raise ValueError(f"Unknown filter: {filter_name}")

    return processed



if __name__ == "__main__":
    resize_image(
        input_path="task1_data/dataset/images/000000000139.jpg",
        output_path="output/resized_image.jpg",
        width=640,
        height=640
    )
