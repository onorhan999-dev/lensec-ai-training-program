
import json

from annolab.preprocessing import preprocess_image_with_labels

report = preprocess_image_with_labels(
    image_path="task1_data/dataset/images/000000000139.jpg",
    label_path="task1_data/dataset/labels/000000000139.txt",
    output_image_path="output/pipeline_with_labels/images/000000000139.jpg",
    output_label_path="output/pipeline_with_labels/labels/000000000139.txt",
    target_width=640,
    target_height=640,
    apply_clahe=True
)

print(json.dumps(report, indent=4))
