import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img

image_path = 'data/Thermal imaging/test/defective/Mango_5_jpg.rf.73ebfaf73a2e0a8ee6411ae9c549083a.jpg'


img = load_image(image_path)

if img is None:
    print("Error: Image not loaded. Please check the file path.")
else:

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    adaptive_equalized_img = clahe.apply(img)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(adaptive_equalized_img, cmap='gray')
    plt.title('Adaptive Equalized Image')

    plt.show()
