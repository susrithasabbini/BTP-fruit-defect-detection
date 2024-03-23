import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
image = cv2.imread(
    "../data/RGB data/train/defective/IMG20230519180945_jpg.rf.5ca9a11519bae6b5bf1af3017b426ca8 (2).jpg"
)

# convert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply the Otsu thresholding method to segment the image
threshold_value, threshold_image = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# display the images
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.axis("off")
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(132)
plt.axis("off")
plt.title("Grayscale Image")
plt.imshow(gray, cmap="gray")

plt.subplot(133)
plt.axis("off")
plt.title("Otsu Thresholding (threshold = {})".format(threshold_value))
plt.imshow(threshold_image, cmap="gray")

plt.tight_layout()
plt.show()
