# Task 1 report: annolab

Every number below comes from a run whose output is in `outputs/` (commands in `scripts/run_all.sh`).
Machine: Intel i7-14700HX (20 cores / 28 threads), Windows 11, Python 3.13, OpenCV 5.0, local NVMe disk.

## What I built

One package, `annolab`, with one `Box` dataclass shared by all parts, a format loader per annotation type,
and one CLI (`annolab video | ann | images | preds`). 29 pytest tests on tiny hand-made fixtures
(coordinates of each format, conversions, every validation rule, letterbox/flip label math, dHash,
thresholds and edge cases, IoU, matching, P/R counts, video probing, extraction naming and MOT filtering).

## Part A: video

| file | fps | frames | size | duration | codec |
|---|---|---|---|---|---|
| LPD.mp4 (real clip from mentor) | 25 | 1,756 (counted = reported) | 1920x1080 | 70.24 s | h264 |
| SYN-01 built from frames + seqinfo.ini | 25 | 375 | 1280x720 | 15.00 s | mp4v |
| SYN-01_annotated.mp4 | 25 | 375 | 1280x720 | 15.00 s | mp4v |
| SYN-01_annotated_h264.mp4 (bonus, ffmpeg) | 25 | 375 | 1280x720 | 15.00 s | h264 |

* **MOT17 was not available** in this environment, so the annotated video uses a synthetic sequence written in the exact MOT17
  layout (`img1/`, `seqinfo.ini`, `gt/gt.txt` with 9 columns). It deliberately contains rows that must be ignored
  (flag 0 and class 3): 750 of 2,561 rows were dropped, 1,811 drawn. The same command runs unchanged on a real MOT17 folder.
* Correctness check (`annotate_check.json`): 375 frames in and out, 15.00 s in and out, same size and fps.
* Speed: annotation runs at **~200 frames/s** (1280x720, read + draw + encode). Extraction from the 1080p LPD clip:
  every 30th frame -> 59 frames in 0.96 s; `--fps 2` -> 141 frames in 1.45 s (skipped frames are `grab()`bed, not decoded).

## Part B: annotations

| dataset (format) | images | boxes | empty images | boxes/img | small / medium / large | problems found |
|---|---|---|---|---|---|---|
| COCO128 (YOLO) | 128 | 929 | 2 | 7.26 | 337 / 281 / 311 | 4: 2 images without label file, 2 label files without image |
| COCO val2017 (JSON) | 5,000 | 36,335 (+446 crowd skipped) | 48 | 7.27 | 11,454 / 12,775 / 12,106 | 0 at 1 px; 19 near-duplicate boxes at 3 px |
| VOC 2012 trainval (XML) | 11,540 | 31,561 (4,111 difficult) | 0 | 2.74 | 3,347 / 9,204 / 19,010 | 0 (also 0 over all 17,125 XMLs) |
| 1,000-image subset (YOLO) | 1,000 | 7,117 | 8 | 7.12 | 2,157 / 2,494 / 2,466 | 8 empty label files |

* Cross-checks: VOC non-difficult boxes = 31,561 - 4,111 = 27,450, the official VOC2012 trainval count. COCO 446 crowd
  and 48 empty images match the dataset README.
* Conversion proof: COCO -> YOLO -> COCO on all 5,000 images returns all 36,335 boxes, **max corner error 0.0 px**
  (< 0.0001). My COCO -> YOLO labels are identical to the provided `dataset/labels` for all 1,000 files.
* Converted datasets: COCO val2017 -> YOLO (5,000 files) and VOC2012 -> YOLO (11,540 files); a sample is in `outputs/partB/voc2012_to_yolo_sample`.

## Part C: image processing (5,000 COCO val2017 images)

Steps: `clahe:2, denoise:gaussian:3, sharpen:1, letterbox:640`, plus metrics (blur, brightness, size, dHash) for every image.

| workers | seconds | images/s | speed-up | efficiency |
|---|---|---|---|---|
| 1 | 62.24 | 80.3 | 1.00 | 1.00 |
| 2 | 30.55 | 163.7 | 2.04 | 1.02 |
| 4 | 17.04 | 293.5 | 3.65 | 0.91 |
| 8 | 13.52 | 369.7 | 4.60 | 0.58 |
| 16 | 10.90 | 458.8 | 5.71 | 0.36 |
| 28 (all) | 9.12 | 548.5 | 6.82 | 0.24 |

**Why the speed-up is less than the number of workers.** The CPU has 8 performance cores and 12 slower efficiency
cores; the 28 "workers" share 20 physical cores, and hyper-threads share a core's arithmetic units, so beyond ~8
workers each new worker is slower than the first ones. Every image is also read from disk, JPEG-decoded and
re-encoded, and those bytes pass through one disk and the memory bus that all workers share. Finally the
main process does serial work (listing files, sending jobs, collecting and writing results), and starting
28 Python processes costs about a second on Windows (spawn), which is a large part of a 9-second run (Amdahl's law).

Peak memory: **~55-60 MB per worker** and 53 MB for the main process, independent of the number of images (one image at a time).
Safety (`safety_demo.json`): a text file named `.jpg` and a truncated JPEG were logged and skipped; a run stopped
after 20 images then `--resume`d processed only the remaining 20; sub-folders kept; originals' SHA-256 unchanged; no temp files left.
Labels: YOLO labels are scaled and offset with the image; `letterbox_check_grid.jpg` shows 20 random processed images with boxes on the objects.
Near-duplicates: 4 pairs at Hamming <= 5, **all false alarms** (low-texture sky/fog images give almost the same dHash).

## Part D: predictions (YOLO26s on 1,000 images, threshold rule: confidence >= t)

| threshold | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| 0.3 | 4,219 | 1,339 | 2,898 | 0.759 | 0.593 |
| 0.5 | 3,445 | 493 | 3,672 | 0.875 | 0.484 |
| 0.7 | 2,533 | 145 | 4,584 | 0.946 | 0.356 |

At 0.5, 3,938 of 15,273 detections are kept on 967 images (CSV and YOLO-text inputs give the same 3,938). Mean AP@0.5 = **0.649** (bonus).

**Finding.** The model is weak on small, cluttered household objects: spoon (AP 0.21), handbag (0.22), book (0.30),
backpack (0.30), banana (0.32), broccoli (0.34), apple (0.35), and on truck (0.36, often confused with car).
For these, precision at 0.5 is acceptable (0.55-0.87) but recall is very low (0.06-0.24), so a single 0.5 threshold
throws away most of what the model finds. Large distinctive animals are near-perfect (elephant 0.99, cat 0.97, zebra 0.96).
Recommended thresholds (max F1 on this set, see `recommended_thresholds.csv`): **person 0.30** (P 0.82 / R 0.73),
car 0.35, chair 0.30, cup 0.25, bottle 0.25, weak small classes 0.15-0.20, and keep **0.45-0.65 for
dog, cat, zebra, toilet, laptop**, where precision is already ~1.0. Two caveats: 1,000 images is small (many classes have
< 50 GT boxes), and some "false positives" are missing labels in COCO (e.g. a 0.96 truck on an unlabelled truck);
crowd regions were removed from GT, so detections on crowds count as FP.

## What went wrong and how I fixed it

1. The first benchmark showed a super-linear 2.4x speed-up on 2 workers: the 1-worker run had paid for a cold disk cache, and
   OpenCV's own threads competed with the pool. Fix: a warm-up read before timing and `cv2.setNumThreads(1)` per worker.
2. With letterbox first, CLAHE and sharpen changed the gray padding. Fix: letterbox last, plus a warning if pixel steps follow it.
3. OpenCV decodes a truncated JPEG without error (the rest comes out gray). Fix: check the end-of-image marker; such files are now skipped.
4. Track trails drew lines across the frame when a track jumped (wrap-around in the synthetic data, or an ID re-entry in real data). Fix: break trails on jumps > 25% of the width.

## What I would do next

Run Part A on real MOT17 sequences; use COCO's crowd-aware evaluation (ignore detections on `iscrowd` regions);
replace dHash with a hash that is robust on low-texture images (or add a pixel check on hash matches);
pick thresholds on a separate validation split rather than the set they are scored on.
