
import cv2
import time
from pathlib import Path

from annolab.video_writer import images_to_video
from annolab.annotations import (
    load_mot_annotations,
    draw_mot_annotations
)


# 1. Reconstruct video from extracted frames
images_to_video(
    image_dir="output/fps_frames",
    output_path="output/reconstructed_video.mp4",
    fps=2
)


# 2. MOT17 Dataset Path
mot_path = Path(
    r"C:\Lensec_Datasets\MOT17\MOT17\train\MOT17-02-DPM"
)

annotations = load_mot_annotations(
    mot_path / "gt" / "gt.txt"
)


# 3. Video settings
output_path = "output/mot_annotated_video.mp4"

fps = 30
width = 1920
height = 1080
total_frames = 600

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not video.isOpened():
    raise RuntimeError("Could not create output video")


# 4. Tracking history
track_history = {}

start_time = time.perf_counter()


# 5. Process all 600 frames
try:
    for frame_id in range(1, total_frames + 1):

        image_path = mot_path / "img1" / f"{frame_id:06d}.jpg"

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        # Get pedestrians
        objects = annotations.get(frame_id, [])

        # Draw Bounding Boxes, IDs and Trails
        image = draw_mot_annotations(
            image,
            objects,
            track_history
        )

        # Calculate timestamp
        timestamp = (frame_id - 1) / fps

        info = (
            f"Frame: {frame_id} | "
            f"Time: {timestamp:.2f}s | "
            f"People: {len(objects)}"
        )

        # Black background
        cv2.rectangle(
            image,
            (20, 20),
            (850, 85),
            (0, 0, 0),
            -1
        )

        # Frame information
        cv2.putText(
            image,
            info,
            (35, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # IMPORTANT: Write EVERY frame inside the loop
        video.write(image)

finally:
    video.release()


# 6. Measure processing speed
elapsed_time = time.perf_counter() - start_time
processing_fps = total_frames / elapsed_time

print(f"Processing Time: {elapsed_time:.2f} seconds")
print(f"Processing FPS: {processing_fps:.2f}")
print("MOT17 annotated video created successfully!")
