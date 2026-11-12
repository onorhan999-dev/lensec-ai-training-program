# AnnoLab – Video and Dataset Toolkit

## 1. Project Overview

AnnoLab is a Python-based toolkit developed for video processing, dataset annotation management, image preprocessing, and performance optimization in computer vision workflows.

The toolkit uses OpenCV and Python to prepare image and video data for object detection and tracking applications.

## 2. Technologies Used

- Python 3.12
- OpenCV
- NumPy
- Pytest
- Concurrent Futures (ProcessPoolExecutor)

## 3. Main Features

### Part A – Video Processing and Annotation

- Video loading and frame processing.
- Bounding box visualization.
- Object tracking annotation visualization.
- Support for tracking IDs and trajectories using existing ground-truth annotations.

### Part B – Dataset Annotation Formats

- YOLO annotation loading and saving.
- COCO JSON annotation processing.
- Pascal VOC XML annotation processing.
- Annotation format conversion.
- Bounding box validation and coordinate transformations.

### Part C – Image Preprocessing

- Image resizing.
- Aspect-ratio-preserving letterbox resizing.
- Bounding box coordinate adjustment.
- Blur detection using Laplacian variance.
- Brightness and contrast assessment.
- CLAHE-based image enhancement.
- Difference Hash (dHash) generation.
- Integrated image and annotation preprocessing.

### Part D – Parallel Processing

- Multiprocessing using ProcessPoolExecutor.
- Parallel image preprocessing.
- Parallel processing of images and YOLO labels.
- Bounding box boundary correction.
- Batch processing and performance measurement.

## 4. Datasets Used

The project was developed and tested using:

- COCO128
- COCO val2017
- Pascal VOC 2012
- MOT17
- Sample video datasets

Datasets are used for annotation handling, preprocessing, and validation.

## 5. Automated Testing

Automated tests were implemented using Pytest.

**Results:**
- Total tests: 22
- Passed: 22
- Failed: 0

Tests cover annotation formats, coordinate validation, letterbox resizing, YOLO file handling, and preprocessing pipelines.

Run the tests:

```bash
python -m pytest -v
```

## 6. Performance Benchmark

Sequential and parallel image preprocessing were compared using 1,000 images and two parallel workers.

| Run | Sequential (s) | Parallel (s) | Speedup |
|---|---:|---:|---:|
| 1 | 7.795 | 3.914 | 1.99x |
| 2 | 6.869 | 4.229 | 1.62x |
| 3 | 8.997 | 5.072 | 1.77x |
| Average | 7.887 | 4.405 | 1.79x |

**Results:**
- Average speedup: approximately 1.79x
- Processing time reduction: approximately 44.1%
- Images processed per run: 1,000
- Parallel workers: 2
- Failed images: 0

Run the benchmark:

```bash
python benchmark.py
```

## 7. Running the Toolkit

Activate the Python virtual environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Run parallel processing with YOLO labels:

```bash
python test_parallel_labels.py
```

Verify processed annotations:

```bash
python verify_parallel_labels.py
```

Generate bounding box visualizations:

```bash
python visualize_parallel_labels.py
```

Run all automated tests:

```bash
python -m pytest -v
```

## 8. Output Files

The toolkit generates:

- Preprocessed images.
- Updated YOLO annotation files.
- Images with bounding box visualizations.
- Image quality reports.
- Processing and benchmark results.

Example output directories:

```text
output/
├── parallel_with_labels/
│   ├── images/
│   └── labels/
└── visual_verification/
```

## 9. Conclusion

AnnoLab provides reusable tools for preparing and validating image and video datasets for computer vision applications.

The implemented preprocessing pipeline preserves image aspect ratios, adjusts bounding box annotations, validates coordinate boundaries, and supports multiprocessing to improve processing efficiency.

The project achieved 22 passing automated tests and an average processing speedup of approximately 1.79x on the tested workload.