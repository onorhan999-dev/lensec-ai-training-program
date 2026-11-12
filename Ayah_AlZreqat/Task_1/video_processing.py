
from annolab.video import extract_frames

saved_frames = extract_frames(
    video_path="LPD.mp4",
    output_dir="output/fps_frames",
    start_time=10,
    end_time=20,
    target_fps=2
)

print("Saved frames:", saved_frames)