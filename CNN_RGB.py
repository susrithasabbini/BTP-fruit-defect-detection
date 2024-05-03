import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define constants
IMAGE_SIZE = (64, 64)  # You can adjust this based on your image size
BATCH_SIZE = 16
NUM_CLASSES = 3  # Defective, Raw, Ripened
EPOCHS = 6

# Data generators
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    "./data/RGB data/train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# validation_generator = test_datagen.flow_from_directory(
#     "./data/RGB data/val",
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
# )

test_generator = test_datagen.flow_from_directory(
    "./data/RGB data/test",
    target_size=IMAGE_SIZE,
    batch_size=1,  # Set batch size to 1 for testing to avoid shuffling
    class_mode="categorical",
    shuffle=False,
)

# Model
model = Sequential(
    [
        Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        ),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
)

# Testing
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples)
print(f"Test Accuracy: {accuracy}")

# save accuracy to csv

import csv

with open("./results/rgb/resultsWithCNN.csv", mode="w") as file:
    writer = csv.writer(file)
    writer.writerow(["Accuracy"])
    writer.writerow([accuracy])


# Save the model
model.save("mango_classification_model_RGB.h5")
