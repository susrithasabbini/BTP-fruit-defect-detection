import os
import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# Function to calculate GLCM features
def calculate_glcm(image):
    # Convert image to grayscale
    gray_image = color.rgb2gray(image)

    # Convert to uint8 for GLCM computation
    gray_image = img_as_ubyte(gray_image)

    # Define GLCM properties
    distances = [1, 2, 3]  # Distances for co-occurrence matrix
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Angles for co-occurrence matrix

    # Calculate GLCM
    glcm = graycomatrix(
        gray_image, distances=distances, angles=angles, symmetric=True, normed=True
    )

    # Calculate GLCM properties
    contrast = graycoprops(glcm, "contrast")
    correlation = graycoprops(glcm, "correlation")
    dissimilarity = graycoprops(glcm, "dissimilarity")
    homogeneity = graycoprops(glcm, "homogeneity")
    energy = graycoprops(glcm, "energy")

    # Return GLCM features as a 1D array
    return np.array(
        [
            contrast.mean(),
            correlation.mean(),
            dissimilarity.mean(),
            homogeneity.mean(),
            energy.mean(),
        ]
    )

# Function to load images and extract features
def load_images_and_extract_features(folder):
    features = []  # Features
    labels = []  # Labels
    for class_folder in os.listdir(folder):
        if not os.path.isdir(os.path.join(folder, class_folder)):
            continue  # Skip if not a directory
        for filename in tqdm(os.listdir(os.path.join(folder, class_folder))):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image = io.imread(os.path.join(folder, class_folder, filename))
                features.append(calculate_glcm(image))
                labels.append(class_folder)  # Use folder name as label

    return features, labels

# Modify the RGB data folder path
rgb_folder = "data/RGB data"

# Load images and extract features for Defective, Raw, and Ripened folders
dataset_features, dataset_labels = load_images_and_extract_features(
    os.path.join(rgb_folder)
)


dataset_df = pd.DataFrame(
    dataset_features,
        columns=["Contrast", "Dissimilarity", "Homogeneity", "Energy", "Correlation"],
)
dataset_df["Label"] = dataset_labels

dataset_csv = "dataset_RGB.csv"

dataset_df.to_csv(dataset_csv, index=False)

print("Dataset CSV(RGB) saved to:", dataset_csv)

