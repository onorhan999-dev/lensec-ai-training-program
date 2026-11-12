
# ============================================================
# STEP 16.2: TEST PARALLEL PROCESSING WITH YOLO LABELS
# ============================================================
# Test parallel preprocessing of images and annotations.
# ============================================================

from annolab.parallel import process_images_with_labels_parallel


def main():
    results = process_images_with_labels_parallel(
        images_dir="task1_data/coco128/images/train2017",
        labels_dir="task1_data/coco128/labels/train2017",
        output_dir="output/parallel_with_labels",
        limit=10,
        workers=2
    )

    successful = sum(
        result["success"] for result in results
    )

    print("\nSuccessful images:", successful)

    assert successful == 10, "Some images failed"

    print("Parallel YOLO processing test passed!")


if __name__ == "__main__":
    main()
