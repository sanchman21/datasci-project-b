# readme

Note: Please keep the neutrophil_images_kevin and a dated folder, both containing images into a directory named data which is in the same folder as the github repository (not inside the repository). This is because relative paths have been used in the code.

# Code Structure

All the code files are inside the folder named townim.

1. dataset_summary.py - This file generates a summary of the training and testing dataset and stores it in the datasets folder.
2. dataset.py - This file contains code for the PyTorch dataset class created to train the pure CNN models (no clincial variables).
3. main_and_test.py - This file is used to train the pure CNN models mentioned above and to evaluate them as well. It does it for the specified fold and does not run for all the folds in a single run. Simply change the fold argument to train for another fold. Data type can also be changed to train on either neutrophil or monocyte images.
4. main.py - This file is used to simply train the pure CNN models mentioned above and not evaluate them. It is recommended to use the main_and_test.py file to train and evaluate at the same time since that file stores results for both train and test of the experiment and this file only stores the results for the training part.
5. MergeMasterDataset.py - This file contains code for the PyTorch dataset class created to train the MultiModal classifier (image data + numerical data). It hasn't been used in the project but has been kept for future use.
6. metrics_summary.py - This file prints the mean and standard deviation for all the folds for the specified experiment and data type.
7. MultiModalClassifier.py - This file contains code for the MultiModal classifier (image data + clinical variables). It hasn't been used in the project but has been kept for future use.
8. only_clinical.py - This file is used to train and evaluate an XGBoost Classifier on the clincial variables for 5 folds. 
9. requirements.txt - Please install all the packages mentioned in this file to run the project.
10. test_clinical_variables.py - This file uses the trained CNN models and train an XGBoost classifier on the clinical variables data. For a given fold, it takes the mean of all predicted probabilities from the CNN model for a patient and averages it with the prediction probability of that patient from the xgboost model to get the final probability, which is used as a prediction for multi-modal data. Make sure that the CNN models are first trained to use perform multi-modal testing.
11. test_old_models.py - This file is used to test the old models that were trained on labels before rechecking. Make sure that the old_models are stored in a folder called "old_models" and this directory should be at the same level as the repository folder (not inside it).
12. test.ipynb - This notebook was used earlier for some analysis.
13. test.py - This file is used just to test pure CNN models. Use it only to test and store the results for any previously trained CNN model (trained using main.py or main_and_test.py only).
14. utils_zhenzhuo.py - This file contains some utility functions used in multiple files.
15. utils.py - This file contains some utility functions used in multiple files.