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