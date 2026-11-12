
import cv2

from annolab.preprocessing import (
    calculate_dhash,
    hamming_distance
)

image_path = "task1_data/dataset/images/000000000139.jpg"

original = cv2.imread(image_path)

if original is None:
    raise ValueError("Could not read image")

# Identical copy
copy_image = original.copy()

# Slightly modified image
modified = cv2.convertScaleAbs(
    original, alpha=1.0, beta=20
)

# Different image
different = cv2.flip(original, 1)

# Calculate hashes
hash_original = calculate_dhash(original)
hash_copy = calculate_dhash(copy_image)
hash_modified = calculate_dhash(modified)
hash_different = calculate_dhash(different)

print("Original vs Copy:",
      hamming_distance(hash_original, hash_copy))

print("Original vs Modified:",
      hamming_distance(hash_original, hash_modified))

print("Original vs Flipped:",
      hamming_distance(hash_original, hash_different))
