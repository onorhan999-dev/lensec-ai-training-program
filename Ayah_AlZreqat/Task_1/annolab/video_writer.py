import cv2
from pathlib import Path


def images_to_video(image_dir, output_path, fps=25):

    image_folder = Path(image_dir)

    image_files = sorted(image_folder.glob("*.jpg"))

    if not image_files:
        raise ValueError("No JPG images found")

    first_frame = cv2.imread(str(image_files[0]))

    if first_frame is None:
        raise ValueError("Could not read the first image")

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not video_writer.isOpened():
        raise ValueError("Could not create output video")

    try:
        for image_path in image_files:

            frame = cv2.imread(str(image_path))

            if frame is None:
                raise ValueError(f"Could not read image: {image_path}")

            if frame.shape[:2] != (height, width):
                raise ValueError(f"Image size mismatch: {image_path}")

            video_writer.write(frame)

    finally:
        video_writer.release()

    print("Video saved successfully:", output_path)