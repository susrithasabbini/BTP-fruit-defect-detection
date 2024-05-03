import cv2
import matplotlib.pyplot as plt


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img


image_path = "../data/RGB data/test/defective/IMG20230519180913_jpg.rf.3e5798971b05532ed162e96e4166a5aa (2).jpg"
# image_path = "data/RGB data/Raw/IMG20230703125745.jpg"
# image_path ='data/RGB data/Ripened/IMG20230703130447.jpg'
# image_path = 'data/Thermal imaging/test/defective/Mango_5_jpg.rf.73ebfaf73a2e0a8ee6411ae9c549083a.jpg'


img = load_image(image_path)

if img is None:
    print("Error: Image not loaded. Please check the file path.")
else:
    equalized_img = cv2.equalizeHist(img)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(equalized_img, cmap="gray")
    plt.title("Equalized Image")

    plt.show()
