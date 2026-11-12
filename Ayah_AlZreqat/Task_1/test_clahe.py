
import cv2
from pathlib import Path

from annolab.preprocessing import (
    enhance_image_clahe,
    assess_brightness_contrast
)

# Read image
image_path = "task1_data/dataset/images/000000000139.jpg"
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Could not read image")

# Apply CLAHE
enhanced = enhance_image_clahe(image)

# Compare before and after
before_brightness, before_contrast = assess_brightness_contrast(image)
after_brightness, after_contrast = assess_brightness_contrast(enhanced)

# Save result
output_path = Path("output/clahe_enhanced.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(output_path), enhanced):
    raise IOError("Could not save enhanced image")

print("Before CLAHE")
print("Brightness:", round(before_brightness, 2))
print("Contrast:", round(before_contrast, 2))

print("\nAfter CLAHE")
print("Brightness:", round(after_brightness, 2))
print("Contrast:", round(after_contrast, 2))

print("\nImage saved:", output_path)



def calculate_dhash(image, hash_size=8):
    if image is None:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray, (hash_size + 1, hash_size)
    )

    differences = resized[:, 1:] > resized[:, :-1]

    hash_value = 0

    for value in differences.flatten():
        hash_value = (hash_value << 1) | int(value)

    return hash_value


def hamming_distance(hash1, hash2):
    return (hash1 ^ hash2).bit_count()
