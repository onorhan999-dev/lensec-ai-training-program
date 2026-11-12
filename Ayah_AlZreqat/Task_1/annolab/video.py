
import cv2
import time
import csv
from pathlib import Path


# Function 1: Get video information
def get_video_info(video_path):

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        video.release()
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0

        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(
            chr((fourcc >> (8 * i)) & 255)
            for i in range(4)
        )

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "codec": codec
        }

    finally:
        video.release()


# Function 2: Extract frames from video
def extract_frames(
    video_path,
    output_dir,
    frame_interval=5,
    start_time=0,
    end_time=None,
    target_fps=None
):

    # Validate input values
    if frame_interval <= 0:
        raise ValueError("frame_interval must be greater than zero")

    if start_time < 0:
        raise ValueError("start_time cannot be negative")

    if end_time is not None and end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    # Create output folder
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Open video
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        video.release()
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = video.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            raise ValueError("Invalid video FPS")

        # Validate target FPS
        if target_fps is not None:
            if target_fps <= 0 or target_fps > fps:
                raise ValueError(
                    "target_fps must be between 0 and video FPS"
                )

        # Convert seconds to frame numbers
        start_frame = int(start_time * fps)

        end_frame = (
            int(end_time * fps)
            if end_time is not None
            else None
        )

        frame_number = 0
        saved_frames = 0
        read_frames = 0
        start_processing = time.perf_counter()
        # Read video frame by frame
        while True:
            ret, frame = video.read()
            read_frames += 1
            if not ret:
                break

            # Stop at the end of selected time range
            if end_frame is not None and frame_number >= end_frame:
                break

            # Process frames within selected time range
            if frame_number >= start_frame:

                relative_frame = frame_number - start_frame

                if target_fps is not None:
                    # Choose frames based on target FPS
                    current_sample = int(
                        relative_frame * target_fps / fps
                    )

                    previous_sample = int(
                        (relative_frame - 1) * target_fps / fps
                    ) if relative_frame > 0 else -1

                    should_save = current_sample > previous_sample

                else:
                    # Choose every Nth frame
                    should_save = (
                        relative_frame % frame_interval == 0
                    )

                if should_save:
                    timestamp = frame_number / fps

                    filename = (
                        output_folder
                        / f"frame_{frame_number:06d}_t{timestamp:.2f}s.jpg"
                    )

                    success = cv2.imwrite(str(filename), frame)

                    if not success:
                        raise IOError(
                            f"Could not save frame: {filename}"
                        )

                    saved_frames += 1

            frame_number += 1

      

        processing_time = time.perf_counter() - start_processing
        processing_fps = read_frames / processing_time if processing_time > 0 else 0
        print("Total frames read:", read_frames)
        print("Total frames saved:", saved_frames)
        print("Processing time:", round(processing_time, 2), "seconds")
        print("Processing FPS:", round(processing_fps, 2))
        csv_path = output_folder / "processing_stats.csv"

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(["Metric", "Value"])
            writer.writerow(["Frames Read", read_frames])
            writer.writerow(["Frames Saved", saved_frames])
            writer.writerow(["Processing Time", round(processing_time, 2)])
            writer.writerow(["Processing FPS", round(processing_fps, 2)])
        return saved_frames

    finally:
        video.release()