import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma**2)),
        (size, size),
    )
    return kernel / np.sum(kernel)


def gaussian_blur(image, kernel_size, sigma):
    """Apply Gaussian blur to the image."""
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(image, kernel)


image_path = "data/RGB data/Defective/IMG20230519180913.jpg"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    print("Error: Image not loaded. Please check the file path.")
else:
    denoised_image = gaussian_blur(original_image, kernel_size=5, sigma=1.5)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap="gray")
    plt.title("Denoised Image (Gaussian Filter)")

    plt.show()
