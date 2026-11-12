
# ============================================================
# STEP 15: SEQUENTIAL VS PARALLEL BENCHMARK
# ============================================================
# Compare image preprocessing performance.
#
# Purpose:
# - Measure sequential processing time.
# - Measure parallel processing time.
# - Calculate images per second and speedup.
# ============================================================

import time
from pathlib import Path

from annolab.preprocessing import preprocess_image
from annolab.parallel import process_images_parallel


def benchmark():
    images_dir = Path("task1_data/dataset/images")
    images = sorted(images_dir.glob("*.jpg"))[:1000]
    if not images:
        raise ValueError("No images found")

    sequential_dir = Path("output/benchmark_sequential")
    sequential_dir.mkdir(parents=True, exist_ok=True)

    # Sequential processing
    start = time.perf_counter()

    for image_path in images:
        preprocess_image(
            image_path=image_path,
            output_path=sequential_dir / image_path.name,
            target_width=640,
            target_height=640,
            apply_clahe=True
        )

    sequential_time = time.perf_counter() - start

    # Parallel processing
    start = time.perf_counter()

    results = process_images_parallel(
        images_dir=images_dir,
        output_dir="output/benchmark_parallel",
        limit=len(images),
        workers=2
    )

    parallel_time = time.perf_counter() - start

    if not all(result["success"] for result in results):
        raise RuntimeError("Parallel processing failed")

    count = len(images)

    print("\n========== BENCHMARK RESULTS ==========")
    print("Images:", count)
    print("Sequential time:", round(sequential_time, 3), "seconds")
    print("Parallel time:", round(parallel_time, 3), "seconds")

    print(
        "Sequential throughput:",
        round(count / sequential_time, 2),
        "images/sec"
    )

    print(
        "Parallel throughput:",
        round(count / parallel_time, 2),
        "images/sec"
    )

    print(
        "Speedup:",
        round(sequential_time / parallel_time, 2),
        "x"
    )


if __name__ == "__main__":
    benchmark()
