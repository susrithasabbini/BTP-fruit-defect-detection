# Fruit Defect Detection

This project focuses on detecting defects in fruits using image processing and machine learning techniques. By automating the quality assessment process, this system identifies and classifies defects in fruit images. 

## Features

- **Data Collection:** 
   - Utilizes fruit images sourced from local mango orchards for training and testing.
   - The dataset is organized in the `data` folder.

- **Image Preprocessing:** 
   - Enhances image quality and prepares data for analysis.
   - Preprocessing scripts are located in the `preprocessing` folder.

- **Feature Extraction:** 
   - Extracts color and texture features from RGB and thermal images.

- **Classification Models:** 
   - Implements traditional machine learning models, such as KNN, Random Forest (RF), Support Vector Machines (SVM), and Linear Discriminant Analysis (LDA).
   - Incorporates Convolutional Neural Networks (CNN) for advanced defect detection.

- **Evaluation:** 
   - Evaluates models using metrics such as accuracy, precision, recall, and F1-score.

## Technologies Used

- **Languages & Libraries:**
  - Python
  - OpenCV
  - TensorFlow/Keras
  - NumPy
  - Pandas

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- pip (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/susrithasabbini/fruit-defect-detection.git
   cd fruit-defect-detection
   ```
---

## Machine Learning Classification Workflow

### 1. Feature Extraction

Use the following commands to extract features and save them to respective CSV files for RGB and thermal images:

1. **GLCM Features Only**
   - RGB: `python glcm_RGB.py`
   - Thermal: `python glcm.py`

2. **Color Features Only**
   - RGB: `python colorFeatures_RGB.py`
   - Thermal: `python colorFeatures.py`

3. **Combined GLCM and Color Features**
   - RGB: `python glcmAndColor_RGB.py`
   - Thermal: `python glcmAndColor.py`

4. **Combined GLCM and Color Features with Histogram Equalization**
   - RGB: `python histogramGLCMAndColor_RGB.py`
   - Thermal: `python histogramGLCMAndColor.py`

5. **Combined GLCM and Color Features with OTSU Thresholding**
   - RGB: `python glcmAndColorOTSU_RGB.py`
   - Thermal: `python glcmAndColorOTSU.py`

---

### 2. Classification

Once features are extracted and saved to CSV files (`train_RGB.csv`, `test_RGB.csv`, `train.csv`, `test.csv`), use the following commands to classify the images and save accuracy metrics to the `results` folder:

1. **Classification**
   - RGB: `python classification_RGB.py`
   - Thermal: `python classification.py`

2. **Confusion Matrix Display**
   - RGB: `python classificationConfusion_RGB.py`
   - Thermal: `python classificationConfusion.py`

---

## CNN Classification Workflow

1. **Classification using CNN**

   - RGB Images: `python CNN_RGB.py`
   - Thermal Images: `python CNN.py`

   Results, including accuracies, will be saved to the `results` folder.

---

## Results

- Model performance metrics such as accuracy, precision, recall, and F1-score are saved in the `results` folder.
- Confusion matrices for each classification run can be generated as needed.

---