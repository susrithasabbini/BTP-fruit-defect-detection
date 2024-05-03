import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tensorflow.keras.models.load_model("mango_classification_model_RGB.h5")

# Load the defective image
defective_image_path = (
    "data/RGB data/test/defective/IMG20230519180913_jpg.rf.3e5798971b05532ed162e96e4166a5aa (2).jpg"  # Update with the path to your defective image
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
class_labels = {0: "Defective", 1: "Raw", 2: "Ripened"}  # Update with your class labels

# Display the image and predicted class label
plt.imshow(defective_img)
plt.title(f"Predicted Class: {class_labels[class_label]}")
plt.axis("off")
plt.show()
