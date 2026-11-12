
# ============================================================
# TEST: PARALLEL IMAGE PROCESSING
# ============================================================
# Test multiprocessing on 10 images using 2 workers.
# ============================================================

from annolab.parallel import process_images_parallel


def main():
    results = process_images_parallel(
        images_dir="task1_data/dataset/images",
        output_dir="output/parallel_images",
        limit=10,
        workers=2
    )

    assert len(results) == 10, "Unexpected number of results"
    assert all(
        result["success"] for result in results
    ), "Some images failed"

    print("Parallel processing test passed!")


if __name__ == "__main__":
    main()
