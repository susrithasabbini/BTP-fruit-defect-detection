import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img

image_path = 'data/RGB data/Defective/IMG20230519180913.jpg'


img = load_image(image_path)

if img is None:
    print("Error: Image not loaded. Please check the file path.")
else:

    denoised_img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_img, cmap='gray')
    plt.title('NLM Denoised Image')

    plt.show()
