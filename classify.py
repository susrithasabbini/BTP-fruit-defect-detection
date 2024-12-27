import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("mango_classification_model_RGB.h5")

# Load the defective image
defective_image_path = (
    "data/Thermal imaging/test/Raw/Mango_9_jpg.rf.e993adba2cdfff1033bc0d87847ffc27.jpg"  # Update with the path to your defective image
)
defective_img = image.load_img(
    defective_image_path, target_size=(64, 64)
)  # Resize as needed
defective_img_array = image.img_to_array(defective_img)
defective_img_array = np.expand_dims(defective_img_array, axis=0)
defective_img_array /= 255.0  # Normalize the image

# Predict the class label
predicted_class = model.predict(defective_img_array)
class_label = np.argmax(predicted_class)

# Define class labels
class_labels = {0: "Ripened", 1: "Ripened", 2: "Ripened"}  # Update with your class labels

# Display the image and predicted class label
plt.imshow(defective_img)
plt.title(f"Predicted Class: {class_labels[class_label]}", loc='left', color='red')
plt.axis("off")

# Example bounding box coordinates (update with actual coordinates)
x, y, width, height = 10, 10, 44, 44  # Example values

# Draw bounding box
rect = plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
plt.gca().add_patch(rect)

plt.show()
