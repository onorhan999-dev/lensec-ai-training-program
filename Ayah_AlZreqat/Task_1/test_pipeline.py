
import json
from annolab.preprocessing import preprocess_image

# Test the integrated preprocessing pipeline
report = preprocess_image(
    image_path="task1_data/dataset/images/000000000139.jpg",
    output_path="output/pipeline_image.jpg",
    target_width=640,
    target_height=640,
    apply_clahe=True
)

# Display the processing report
print(json.dumps(report, indent=4))
