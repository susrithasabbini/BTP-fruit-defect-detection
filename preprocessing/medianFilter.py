import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img

image_path = 'data/RGB data/Defective/IMG20230519180913.jpg'
# image_path = 'data/Thermal imaging/test/defective/Mango_5_jpg.rf.73ebfaf73a2e0a8ee6411ae9c549083a.jpg'


img = load_image(image_path)

if img is None:
    print("Error: Image not loaded. Please check the file path.")
else:
    median_filtered_img = cv2.medianBlur(img, ksize=5)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(median_filtered_img, cmap='gray')
    plt.title('Median Filtered Image')

    plt.show()
