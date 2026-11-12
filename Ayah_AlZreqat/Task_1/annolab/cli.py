
import argparse
import configparser
import time
from pathlib import Path

import cv2

from annolab.video import get_video_info, extract_frames
from annolab.annotations import (
    load_mot_annotations,
    draw_mot_annotations
)


def annotate_mot_video(sequence_dir, output_path):
    sequence_dir = Path(sequence_dir)
    output_path = Path(output_path)

    config_path = sequence_dir / "seqinfo.ini"
    gt_path = sequence_dir / "gt" / "gt.txt"

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing file: {config_path}")

    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing file: {gt_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

    sequence = config["Sequence"]

    fps = int(sequence["frameRate"])
    total_frames = int(sequence["seqLength"])
    width = int(sequence["imWidth"])
    height = int(sequence["imHeight"])
    image_dir = sequence_dir / sequence["imDir"]
    extension = sequence["imExt"]

    annotations = load_mot_annotations(gt_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not video.isOpened():
        raise RuntimeError("Could not create output video")

    track_history = {}
    start_time = time.perf_counter()

    try:
        for frame_id in range(1, total_frames + 1):
            image_path = image_dir / f"{frame_id:06d}{extension}"
            image = cv2.imread(str(image_path))

            if image is None:
                raise FileNotFoundError(
                    f"Could not read image: {image_path}"
                )

            if image.shape[:2] != (height, width):
                raise ValueError(
                    f"Unexpected image dimensions: {image_path}"
                )

            objects = annotations.get(frame_id, [])

            image = draw_mot_annotations(
                image,
                objects,
                track_history
            )

            timestamp = (frame_id - 1) / fps

            info = (
                f"Frame: {frame_id} | "
                f"Time: {timestamp:.2f}s | "
                f"People: {len(objects)}"
            )

            cv2.rectangle(
                image,
                (20, 20),
                (850, 85),
                (0, 0, 0),
                -1
            )

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

            video.write(image)

    finally:
        video.release()

    elapsed = time.perf_counter() - start_time
    processing_fps = total_frames / elapsed if elapsed > 0 else 0

    print(f"Video saved: {output_path}")
    print(f"Total Frames: {total_frames}")
    print(f"Video FPS: {fps}")
    print(f"Duration: {total_frames / fps:.2f} seconds")
    print(f"Processing Time: {elapsed:.2f} seconds")
    print(f"Processing FPS: {processing_fps:.2f}")


def main():
    parser = argparse.ArgumentParser(
        prog="annolab",
        description="AnnoLab Video and Dataset Toolkit"
    )

    subparsers = parser.add_subparsers(dest="command")

    video_parser = subparsers.add_parser(
        "video",
        help="Video processing tools"
    )

    video_commands = video_parser.add_subparsers(
        dest="video_command"
    )

    # Extract command
    extract_parser = video_commands.add_parser(
        "extract",
        help="Extract frames from a video"
    )

    extract_parser.add_argument("--input", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--interval", type=int, default=5)
    extract_parser.add_argument("--start", type=float, default=0)
    extract_parser.add_argument("--end", type=float, default=None)
    extract_parser.add_argument("--target-fps", type=float, default=None)

    # Annotate command
    annotate_parser = video_commands.add_parser(
        "annotate",
        help="Annotate a MOT17 sequence"
    )

    annotate_parser.add_argument(
        "--sequence",
        required=True,
        help="MOT17 sequence folder"
    )

    annotate_parser.add_argument(
        "--output",
        required=True,
        help="Output MP4 video path"
    )

    args = parser.parse_args()

    if args.command == "video" and args.video_command == "extract":
        info = get_video_info(args.input)

        print("\nVideo Information:")
        for key, value in info.items():
            print(f"{key}: {value}")

        saved = extract_frames(
            video_path=args.input,
            output_dir=args.output,
            frame_interval=args.interval,
            start_time=args.start,
            end_time=args.end,
            target_fps=args.target_fps
        )

        print(f"\nExtraction completed: {saved} frames saved.")

    elif args.command == "video" and args.video_command == "annotate":
        annotate_mot_video(
            sequence_dir=args.sequence,
            output_path=args.output
        )

    elif args.command == "video":
        video_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
