import numpy as np
from skimage import io

# Function to calculate average color (RGB) values
def calculate_average_color(image):
    # Calculate average RGB values
    avg_color = np.mean(image, axis=(0, 1))
    return avg_color

# Load the image (replace 'image_path.jpg' with the path to your image)
image_path = "background.jpg"
image = io.imread(image_path)

# Calculate average RGB values using the function
avg_color = calculate_average_color(image)

# Print the average RGB values
print("Average Red:", avg_color[0])
print("Average Green:", avg_color[1])
print("Average Blue:", avg_color[2])
