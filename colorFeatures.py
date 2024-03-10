import os
import numpy as np
import pandas as pd
from skimage import io


# Function to calculate average color (RGB) values
def calculate_average_color(image):
    # Calculate average RGB values
    avg_color = np.mean(image, axis=(0, 1))
    return avg_color


# Function to load images and extract color features
def load_images_and_extract_color_features(folder):
    features = []  # Features
    labels = []  # Labels
    for class_folder in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, class_folder)):
            continue  # Skip if not a directory
        for filename in os.listdir(os.path.join(folder, class_folder)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = io.imread(os.path.join(folder, class_folder, filename))
                color_features = calculate_average_color(image)
                features.append(color_features)
                labels.append(class_folder)  # Use folder name as label

    return features, labels


# Modify the RGB data folder path
rgb_folder = "data/Thermal imaging"

# Load images and extract color features for train and test folders
train_features, train_labels = load_images_and_extract_color_features(
    os.path.join(rgb_folder, "train")
)
test_features, test_labels = load_images_and_extract_color_features(
    os.path.join(rgb_folder, "test")
)
val_features, val_labels = load_images_and_extract_color_features(
    os.path.join(rgb_folder, "val")
)

# Create DataFrames for train and test sets
train_df = pd.DataFrame(
    train_features,
    columns=["Avg_Red", "Avg_Green", "Avg_Blue"],
)
train_df["Label"] = train_labels

test_df = pd.DataFrame(
    test_features,
    columns=["Avg_Red", "Avg_Green", "Avg_Blue"],
)
test_df["Label"] = test_labels

val_df = pd.DataFrame(
    val_features,
    columns=["Avg_Red", "Avg_Green", "Avg_Blue"],
)

# Save the DataFrames to CSV files
train_csv = "train.csv"
test_csv = "test.csv"
val_csv = "val.csv"

train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("Train CSV saved to:", train_csv)
print("Test CSV saved to:", test_csv)
