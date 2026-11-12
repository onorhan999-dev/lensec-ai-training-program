
# ============================================================
# STEP 14: MULTIPROCESSING
# ============================================================
# Process multiple images using separate processes.
#
# Purpose:
# - Run image preprocessing tasks in parallel.
# - Reuse the existing preprocessing pipeline.
# - Measure total processing time.
# - Count successful and failed images.
# ============================================================

import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from annolab.preprocessing import preprocess_image


# Process one image
def process_single_image(task):
    image_path, output_dir = task

    output_path = Path(output_dir) / image_path.name

    try:
        report = preprocess_image(
            image_path=image_path,
            output_path=output_path,
            target_width=640,
            target_height=640,
            apply_clahe=True
        )

        return {
            "success": True,
            "image": image_path.name,
            "report": report
        }

    except Exception as error:
        return {
            "success": False,
            "image": image_path.name,
            "error": str(error)
        }


# Process a dataset in parallel
def process_images_parallel(
    images_dir,
    output_dir,
    limit=10,
    workers=2
):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg"))

    if limit is not None:
        image_files = image_files[:limit]

    if not image_files:
        raise ValueError("No JPG images found")

    if workers < 1:
        raise ValueError("workers must be at least 1")

    tasks = [
        (image_path, output_dir)
        for image_path in image_files
    ]

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(process_single_image, tasks)
        )

    elapsed = time.perf_counter() - start_time

    processed = sum(
        result["success"] for result in results
    )

    failed = len(results) - processed

    print("\nParallel Processing Results")
    print("Total images:", len(image_files))
    print("Workers:", workers)
    print("Successfully processed:", processed)
    print("Failed:", failed)
    print("Processing time:", round(elapsed, 2), "seconds")

    for result in results:
        if not result["success"]:
            print(
                "Failed:",
                result["image"],
                "-",
                result["error"]
            )

    return results

# ============================================================
# STEP 16: PARALLEL PROCESSING WITH YOLO LABELS
# ============================================================
# Process images and their bounding box annotations together.
# ============================================================


# Process one image and its YOLO label file
def process_single_image_with_labels(task):
    image_path, labels_dir, output_dir = task

    image_path = Path(image_path)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    label_path = labels_dir / (image_path.stem + ".txt")

    output_image = output_dir / "images" / image_path.name
    output_label = output_dir / "labels" / label_path.name

    try:
        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label: {label_path}"
            )

        from annolab.preprocessing import preprocess_image_with_labels

        report = preprocess_image_with_labels(
            image_path=image_path,
            label_path=label_path,
            output_image_path=output_image,
            output_label_path=output_label,
            target_width=640,
            target_height=640,
            apply_clahe=True
        )

        return {
            "success": True,
            "image": image_path.name,
            "report": report
        }

    except Exception as error:
        return {
            "success": False,
            "image": image_path.name,
            "error": str(error)
        }


# Process multiple images and YOLO labels in parallel
def process_images_with_labels_parallel(
    images_dir,
    labels_dir,
    output_dir,
    limit=10,
    workers=2
):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg"))

    if limit is not None:
        image_files = image_files[:limit]

    if not image_files:
        raise ValueError("No JPG images found")

    if workers < 1:
        raise ValueError("workers must be at least 1")

    tasks = [
        (image_path, labels_dir, output_dir)
        for image_path in image_files
    ]

    start_time = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                process_single_image_with_labels,
                tasks
            )
        )

    elapsed = time.perf_counter() - start_time

    processed = sum(
        result["success"] for result in results
    )
    failed = len(results) - processed

    print("\nParallel Processing with YOLO Labels")
    print("Total images:", len(image_files))
    print("Workers:", workers)
    print("Successfully processed:", processed)
    print("Failed:", failed)
    print("Processing time:", round(elapsed, 2), "seconds")

    for result in results:
        if not result["success"]:
            print(
                "Failed:",
                result["image"],
                "-",
                result["error"]
            )

    return results

