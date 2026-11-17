
"""AnnoLab – Task 1, Part A: video frames and MOT17 videos."""

import argparse
import configparser
import time
from pathlib import Path

import cv2

from annolab.annotations import load_mot_annotations, draw_mot_annotations


# ============================================================
# POINT 1 — READ VIDEO AND PRINT ITS PROPERTIES
# FPS, frame count, width, height, duration, codec
# ============================================================

def get_video_info(video_path):
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = reported_frames / fps if fps > 0 else 0

        codec_number = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(
            chr((codec_number >> (8 * i)) & 255)
            for i in range(4)
        )

    finally:
        cap.release()

    info = {
        "FPS": fps,
        "Frame Count": reported_frames,
        "Width": width,
        "Height": height,
        "Duration": duration,
        "Codec": codec,
    }

    print("\nVideo Information:")

    for key, value in info.items():
        print(f"{key}: {value}")

    return info


# ============================================================
# POINT 2 — SELECT FRAMES
#
# Save every Nth frame: --every-n 5
# Save at chosen rate: --fps 2
# Save within time window: --start 10 --end 30
# ============================================================

def should_save_frame(
    frame_number,
    source_fps,
    every_n,
    target_fps,
    start_time
):
    timestamp = frame_number / source_fps

    # OPTION 1: Save every Nth frame
    if target_fps is None:
        return frame_number % every_n == 0

    # OPTION 2: Save at a chosen FPS
    relative_time = timestamp - start_time

    if relative_time < -1e-9:
        return False

    sample_now = int(
        max(0.0, relative_time) * target_fps + 1e-9
    )

    previous_relative_time = (
        (frame_number - 1) / source_fps - start_time
    )

    if previous_relative_time < -1e-9:
        return True

    sample_before = int(
        max(0.0, previous_relative_time) * target_fps + 1e-9
    )

    return sample_now > sample_before


# ============================================================
# POINT 3 — NAME EACH IMAGE
#
# Include frame index and timestamp
# Example: frame_000123_t4.10s.jpg
# ============================================================

def make_frame_filename(frame_number, timestamp):
    return f"frame_{frame_number:06d}_t{timestamp:.2f}s.jpg"


# ============================================================
# POINT 4 — STREAM VIDEO FRAME BY FRAME
#
# Never load the whole video into memory
#
# POINT 5 — HANDLE ERRORS
#
# Missing file, unreadable video, wrong frame count
# ============================================================

def extract_frames(
    video_path,
    output_dir,
    every_n=5,
    target_fps=None,
    start_time=0,
    end_time=None
):

    # POINT 5: Validate arguments
    if every_n <= 0:
        raise ValueError("--every-n must be greater than 0")

    if target_fps is not None and target_fps <= 0:
        raise ValueError("--fps must be greater than 0")

    if start_time < 0:
        raise ValueError("--start cannot be negative")

    if end_time is not None and end_time <= start_time:
        raise ValueError("--end must be greater than --start")

    # POINT 1: Read video information
    info = get_video_info(video_path)

    source_fps = info["FPS"]
    reported_frames = info["Frame Count"]

    if source_fps <= 0:
        raise ValueError("Invalid source video FPS")

    if target_fps is not None and target_fps > source_fps:
        raise ValueError(
            "--fps cannot exceed the source video FPS"
        )

    # Create output folder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # POINT 4: Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")

    actual_frames = 0
    saved_frames = 0

    started = time.perf_counter()

    try:
        # POINT 4: Read one frame at a time
        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp = actual_frames / source_fps

            # POINT 2: Time window
            # Example: --start 10 --end 30
            inside_window = (
                timestamp >= start_time - 1e-9
                and (
                    end_time is None
                    or timestamp < end_time - 1e-9
                )
            )

            # POINT 2: Every Nth frame or target FPS
            if inside_window and should_save_frame(
                actual_frames,
                source_fps,
                every_n,
                target_fps,
                start_time
            ):

                # POINT 3: Name the frame
                filename = make_frame_filename(
                    actual_frames,
                    timestamp
                )

                image_path = output_dir / filename

                # Save the frame
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(
                        f"Could not save image: {image_path}"
                    )

                saved_frames += 1

            actual_frames += 1

    finally:
        cap.release()

    elapsed = time.perf_counter() - started

    # POINT 5: Check actual frame count
    if actual_frames == 0:
        raise ValueError(
            "No readable frames found in the input video"
        )

    print(f"\nReported Frames: {reported_frames}")
    print(f"Actual Frames Read: {actual_frames}")

    if actual_frames != reported_frames:
        print(
            "WARNING: Reported and actual frame counts differ!"
        )
    else:
        print("Frame count matches.")

    print(f"Saved Frames: {saved_frames}")
    print(f"Processing Time: {elapsed:.2f}s")

    if elapsed > 0:
        print(
            f"Processing FPS: {actual_frames / elapsed:.2f}"
        )

    return saved_frames


# ============================================================
# POINT 6 — BUILD VIDEO FROM MOT17 FRAMES
#
# Read img1/ and seqinfo.ini
# Use correct frameRate, width, height
# Save as MP4
# ============================================================

def read_mot_sequence(sequence_dir):
    sequence_dir = Path(sequence_dir)

    ini_path = sequence_dir / "seqinfo.ini"

    if not ini_path.is_file():
        raise FileNotFoundError(
            f"Missing seqinfo.ini: {ini_path}"
        )

    config = configparser.ConfigParser()

    if not config.read(ini_path) or "Sequence" not in config:
        raise ValueError(
            f"Invalid seqinfo.ini: {ini_path}"
        )

    sequence = config["Sequence"]

    fps = int(sequence["frameRate"])
    total_frames = int(sequence["seqLength"])
    width = int(sequence["imWidth"])
    height = int(sequence["imHeight"])

    image_dir = sequence_dir / sequence["imDir"]
    extension = sequence["imExt"]

    if (
        fps <= 0
        or total_frames <= 0
        or width <= 0
        or height <= 0
    ):
        raise ValueError(
            "Invalid sequence FPS, frame count, or dimensions"
        )

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Missing MOT17 image directory: {image_dir}"
        )

    return (
        image_dir,
        extension,
        fps,
        total_frames,
        width,
        height
    )


def create_video_writer(output_path, fps, width, height):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Cannot create output video: {output_path}"
        )

    return writer


def images_to_video(sequence_dir, output_path):

    (
        image_dir,
        extension,
        fps,
        total_frames,
        width,
        height
    ) = read_mot_sequence(sequence_dir)

    writer = create_video_writer(
        output_path,
        fps,
        width,
        height
    )

    written_frames = 0
    started = time.perf_counter()

    try:
        for frame_id in range(1, total_frames + 1):

            image_path = (
                image_dir / f"{frame_id:06d}{extension}"
            )

            frame = cv2.imread(str(image_path))

            if frame is None:
                raise FileNotFoundError(
                    f"Cannot read MOT17 frame: {image_path}"
                )

            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Wrong frame dimensions: {image_path}"
                )

            writer.write(frame)
            written_frames += 1

    finally:
        writer.release()

    elapsed = time.perf_counter() - started

    print(f"\nVideo saved: {output_path}")
    print(f"Frames: {written_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {written_frames / fps:.2f}s")

    if elapsed > 0:
        print(
            f"Processing FPS: {written_frames / elapsed:.2f}"
        )


# ============================================================
# POINT 7 — ANNOTATE MOT17 VIDEO
#
# Read gt.txt
# Draw bounding boxes, IDs and trails
# Add frame number, timestamp and people count
# Save annotated MP4
# ============================================================

def annotate_mot_video(sequence_dir, output_path):

    sequence_dir = Path(sequence_dir)

    gt_path = sequence_dir / "gt" / "gt.txt"

    if not gt_path.is_file():
        raise FileNotFoundError(
            f"Missing ground truth: {gt_path}"
        )

    annotations = load_mot_annotations(gt_path)

    (
        image_dir,
        extension,
        fps,
        total_frames,
        width,
        height
    ) = read_mot_sequence(sequence_dir)

    writer = create_video_writer(
        output_path,
        fps,
        width,
        height
    )

    track_history = {}
    written_frames = 0

    started = time.perf_counter()

    try:
        for frame_id in range(1, total_frames + 1):

            image_path = (
                image_dir / f"{frame_id:06d}{extension}"
            )

            frame = cv2.imread(str(image_path))

            if frame is None:
                raise FileNotFoundError(
                    f"Cannot read MOT17 frame: {image_path}"
                )

            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Wrong frame dimensions: {image_path}"
                )

            objects = annotations.get(frame_id, [])

            frame = draw_mot_annotations(
                frame,
                objects,
                track_history
            )

            timestamp = (frame_id - 1) / fps

            overlay = (
                f"Frame: {frame_id} | "
                f"Time: {timestamp:.2f}s | "
                f"People: {len(objects)}"
            )

            cv2.rectangle(
                frame,
                (20, 20),
                (min(width - 1, 850), 85),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                overlay,
                (35, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)
            written_frames += 1

    finally:
        writer.release()

    elapsed = time.perf_counter() - started

    print(f"\nAnnotated video saved: {output_path}")
    print(f"Frames: {written_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {written_frames / fps:.2f}s")
    print(f"Processing Time: {elapsed:.2f}s")

    if elapsed > 0:
        print(
            f"Processing FPS: {written_frames / elapsed:.2f}"
        )


# ============================================================
# COMMAND LINE INTERFACE (CLI)
#
# Receive commands from the Terminal
# Call the functions defined above
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        prog="annolab",
        description="AnnoLab Video Toolkit"
    )

    commands = parser.add_subparsers(
        dest="command"
    )

    video_parser = commands.add_parser("video")

    video_commands = video_parser.add_subparsers(
        dest="video_command"
    )

    # POINTS 1–5: Extract video frames
    extract_parser = video_commands.add_parser("extract")

    extract_parser.add_argument(
        "--input",
        required=True
    )

    extract_parser.add_argument(
        "--output",
        required=True
    )

    # POINT 2: --every-n 5
    extract_parser.add_argument(
        "--every-n",
        "--interval",
        dest="every_n",
        type=int,
        default=5
    )

    # POINT 2: --fps 2
    extract_parser.add_argument(
        "--fps",
        "--target-fps",
        dest="target_fps",
        type=float
    )

    # POINT 2: --start 10 --end 30
    extract_parser.add_argument(
        "--start",
        type=float,
        default=0
    )

    extract_parser.add_argument(
        "--end",
        type=float
    )

    # POINT 6: Build video
    build_parser = video_commands.add_parser("build")

    build_parser.add_argument(
        "--sequence",
        required=True
    )

    build_parser.add_argument(
        "--output",
        required=True
    )

    # POINT 7: Annotate video
    annotate_parser = video_commands.add_parser("annotate")

    annotate_parser.add_argument(
        "--sequence",
        required=True
    )

    annotate_parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    try:
        if (
            args.command == "video"
            and args.video_command == "extract"
        ):
            extract_frames(
                args.input,
                args.output,
                args.every_n,
                args.target_fps,
                args.start,
                args.end
            )

        elif (
            args.command == "video"
            and args.video_command == "build"
        ):
            images_to_video(
                args.sequence,
                args.output
            )

        elif (
            args.command == "video"
            and args.video_command == "annotate"
        ):
            annotate_mot_video(
                args.sequence,
                args.output
            )

        elif args.command == "video":
            video_parser.print_help()

        else:
            parser.print_help()

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        KeyError,
        OSError
    ) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
