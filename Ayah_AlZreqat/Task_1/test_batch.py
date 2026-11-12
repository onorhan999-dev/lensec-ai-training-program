
from annolab.preprocessing import process_dataset

process_dataset(
    images_dir="task1_data/dataset/images",
    labels_dir="task1_data/dataset/labels",
    output_dir="output/batch_processed",
    limit=10,
    target_width=640,
    target_height=640
)
