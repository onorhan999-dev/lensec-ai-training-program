
import cv2

from annolab.preprocessing import assess_brightness_contrast

image_path = "task1_data/dataset/images/000000000139.jpg"

image = cv2.imread(image_path)

if image is None:
    raise ValueError("Could not read image")

# Original image
brightness, contrast = assess_brightness_contrast(image)

# Dark image
dark_image = cv2.convertScaleAbs(
    image, alpha=0.5, beta=0
)

# Bright image
bright_image = cv2.convertScaleAbs(
    image, alpha=1.0, beta=80
)

dark_brightness, dark_contrast = assess_brightness_contrast(
    dark_image
)

bright_brightness, bright_contrast = assess_brightness_contrast(
    bright_image
)

print("Original Image")
print("Brightness:", round(brightness, 2))
print("Contrast:", round(contrast, 2))

print("\nDark Image")
print("Brightness:", round(dark_brightness, 2))
print("Contrast:", round(dark_contrast, 2))

print("\nBright Image")
print("Brightness:", round(bright_brightness, 2))
print("Contrast:", round(bright_contrast, 2))

cv2.imwrite("output/dark_image.jpg", dark_image)
cv2.imwrite("output/bright_image.jpg", bright_image)
