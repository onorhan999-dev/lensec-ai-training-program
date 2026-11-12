
import cv2
from annolab.preprocessing import detect_blur

image_path = "task1_data/dataset/images/000000000139.jpg"

image = cv2.imread(image_path)

if image is None:
    raise ValueError("Could not read image")

# Original image
score_original, blurry_original = detect_blur(image)

# Artificially blurred image
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

score_blurred, blurry_blurred = detect_blur(blurred_image)

cv2.imwrite("output/blurred_image.jpg", blurred_image)

print("Original image score:", round(score_original, 2))
print("Original image blurry:", blurry_original)

print("Blurred image score:", round(score_blurred, 2))
print("Blurred image blurry:", blurry_blurred)
