# CMML Patient Classifier Project

## Introduction
This project is designed to use deep learning techniques to predict clinical outcomes for patients with Chronic Myelomonocytic Leukemia (CMML). Integrating a combination of patient medical data and neutrophil images, we employ a neural network model that integrates image features extracted by a pretrained ResNet-50 with patient-specific data. The goal is to provide a binary classification to determine patient prognosis.

## Project Structure

### Datasets
- **PatientIdBasedDataset.py**: This module defines a `PatientIdBasedDataset` class that handles the loading and preprocessing of patient medical data. It reads from CSV files, ensures data integrity, and processes missing values through imputation methods.

- **ImageBasedDataset.py**: This module defines an `ImageBasedDataset` class designed for handling image data. It supports image transformations and normalization to prepare data for neural network input.

- **CombinedDataset.py**: Combines data from `PatientIdBasedDataset` and `ImageBasedDataset` into a unified dataset. This dataset is used to train the neural network model, ensuring that each data instance contains both image and patient data.

### Models
- **CMMLPatientClassifier.py**: Contains the `PatientClassifier` class, a neural network that integrates features extracted from images with patient data. The classifier includes several fully connected layers and outputs a binary prediction.

### Training
- **main.py**: The main script used for model training. It sets up the dataset, model, loss function, and optimizer. It also handles the training loop, including forward and backward propagation, and prints out training progress.

## Technical Details

### Neural Network Architecture
- **Feature Extraction**: Uses a pretrained ResNet-50 model to extract features from input images. The original fully connected layers of ResNet-50 are removed to use the model as a feature extractor.
- **Classifier**: After feature extraction, the image features are concatenated with structured patient data resulting in a combined feature vector. This vector is then processed through several layers to perform binary classification.

### Data Handling
- **Imputation**: Missing values in patient data are handled using regression-based imputation, ensuring no data instance is discarded due to incomplete information.
- **Normalization(havn't done yet)**: Image data is normalized to match the input requirements of the ResNet-50 model, which includes resizing images to 224x224 pixels and normalizing pixel values.

### Training Details
- **Optimizer**: The model uses the Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Binary cross-entropy loss is used as it is suitable for binary classification tasks.

## Model Evaluation Metrics

### Scenario 1: Without Neutrophil Images + Without Patient Data

- **Precision:** 90.18 ± 11.80
- **Recall:** 54.79 ± 32.05
- **F1 Score:** 60.06 ± 24.90
- **Accuracy:** 65.61 ± 14.60
- **AUC:** 67.30 ± 11.28
- **Confusion Matrix (normalized):** TP: 257.80 ± 154.68, TN: 281.60 ± 161.28, FP: 56.60 ± 86.21, FN: 226.40 ± 175.17

### Scenario 2: With Neutrophil Images + Without Patient Data

- **Precision:** 91.50 ± 10.50
- **Recall:** 58.20 ± 30.05
- **F1 Score:** 62.80 ± 22.50
- **Accuracy:** 67.80 ± 13.00
- **AUC:** 69.00 ± 10.01
- **Confusion Matrix (normalized):** TP: 280.30 ± 150.13, TN: 290.20 ± 160.45, FP: 50.70 ± 80.93, FN: 220.20 ± 170.47
