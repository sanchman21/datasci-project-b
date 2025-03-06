'''
This script is used to test and get predictions using a Multimodal approach (averaging a patient's image probabilities and clinical variables probabilities).
'''

# Import libraries
import numpy as np
import subprocess
import pandas as pd
import os
import argparse
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchmetrics
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import utils
from utils import FixedRotation, plot_confusion_matrix, plot_save_roc_curve, save_metrics_csv, assign_patient_folds_splits
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH

# Create a cache directory to store the downloaded models
cache_dir = "../cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir  # Set cache directory
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

torch.cuda.empty_cache()  # Empty cache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device
print(f"Using device: {device}")

# Argument parser for data type
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='neutrophil', choices=('monocyte', 'neutrophil'), help='data type')
args = parser.parse_args()

# Define the dataset path based on the data type
data_type = args.data_type  # neutrophil or monocyte
CSV_PATH = NEUTROPHIL_CSV_PATH if data_type == 'neutrophil' else MONOCYTE_CSV_PATH
assign_patient_folds_splits(CSV_PATH)
batch_size = 32  # Batch size
IMAGE_SIZE = 352  # Image size
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]  # Std of ImageNet dataset (used for normalization)

# Define the test transforms
test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Define the test-time augmentations (TTA)
TTAs = [
    test_transform,
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomVerticalFlip(p=1.0), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([FixedRotation(angle=45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomHorizontalFlip(p=1.0), FixedRotation(angle=45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomVerticalFlip(p=1.0), FixedRotation(angle=45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0), FixedRotation(angle=45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
]

experiment_name = f"{data_type}_clinical" # experiment name
existing_experiment = mlflow.get_experiment_by_name(experiment_name)
if existing_experiment: # experiment handling if it already exists
    experiment_id = existing_experiment.experiment_id  # Reuse existing ID
    overwrite_exp = input(f"DO YOU WANT TO OVERWRITE EXISTING {experiment_name} EXPERIMENT? [Y/N]")
    if overwrite_exp.lower() == "y":
        mlflow.delete_experiment(experiment_id)
        subprocess.run(["mlflow", "gc", "--experiment-ids", experiment_id], check=True)
    else:
        print("To run the code further, you need to overwrite existing experiment. Please modify code otherwise.")
        exit()
        
# set experiment
mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# xgboost params
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'nthread': 4,
    'booster': 'gbtree',
}

new_dir = f"./experiments/{data_type}_clinical"
new_dir_train = new_dir + "/train"
new_dir_test = new_dir + "/test"
new_dir_test_figures = new_dir_test + "/figures"
os.makedirs(new_dir, exist_ok=True)
os.makedirs(new_dir_train, exist_ok=True)
os.makedirs(new_dir_test, exist_ok=True)
os.makedirs(new_dir_test_figures, exist_ok=True)

with mlflow.start_run(run_name="train-val") as parent_run: # parent run
    mlflow.log_params(params) # log xgboost params

    for fold_id in range(5): # for each fold
        with mlflow.start_run(run_name=f"fold_{fold_id}", nested=True): # initialise child run
            print(f"Fold {fold_id}")
            # Define directories
            model_dir = f"./experiments/{data_type}/train/{data_type}_fold_{fold_id}_with_TTA/model"
            new_dir_with_fold = os.path.join(new_dir_train, f"fold_{fold_id}")
            new_dir_train_figures = new_dir_with_fold + "/figures"
            os.makedirs(new_dir_with_fold, exist_ok=True)
            os.makedirs(new_dir_train_figures, exist_ok=True)

            num_classes = 2  # binary classification
            model = models.resnet50(weights='IMAGENET1K_V1') # resnet
            hidden_layer_size = 512
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, hidden_layer_size),
                nn.ReLU(),
                nn.Linear(hidden_layer_size, num_classes)
            )
            model = model.to(device)
            model.load_state_dict(torch.load(os.path.join(model_dir, f'model_fold_{fold_id}.pth'))) # load model weights
            model.eval() # set to eval

            val_dataset = CustomDataset('val', CSV_PATH, fold_id, transform=test_transform) # val set
            val_loaders = [
                DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
                for _ in TTAs
            ] # val loader

            with torch.no_grad(): # get patient predictions
                # initialise empty variables to store data
                preds = []
                logits = []
                labels = []
                correct = 0
                for i, data in enumerate(zip(*val_loaders)): # for each batch
                    inputs = torch.cat([img for img, _ in data], dim=0).to(device).float() # get input
                    targets = data[0][1].to(device).long() # get labels
                    outputs = model(inputs) # get output
                    outputs = F.softmax(outputs, dim=-1) # get output softmax
                    outputs = outputs.reshape(len(TTAs), int(inputs.shape[0]/len(TTAs)), -1).mean(dim=0) # scores for each tta batch
                    _, predicted = torch.max(outputs, 1) # predictions for class 1
                    # append preds and other data
                    preds.append(predicted.detach().cpu().numpy())
                    labels.append(targets.detach().cpu().numpy())
                    logits.append(outputs.detach().cpu().numpy().astype(np.float32))
                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0)
                print(f"Fold {fold_id} - Image-level labels: {np.unique(labels, return_counts=True)}")
                logits = np.concatenate(logits, axis=0)

            # Patient-level evaluation with clinical variables
            clinical_variable_df = pd.read_csv('../datasets/patients_fold.csv') # patient data
            feature_columns = ['Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count', 
                               'Neutrophil count', 'Monocyte count', 'Platelet count', 
                               'Blast percentage (PB)', 'LDH'] # patient features
            target_column = 'morphology' # target
            rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125] # rechecked ids
            clinical_variable_df = clinical_variable_df.loc[clinical_variable_df["patient_id"] != 2209801848] # remove this patient
            clinical_variable_df.loc[clinical_variable_df["patient_id"].isin(rechecked_patient_ids), "morphology"] = 0 # correct label for rechecked patient

            set_col = f'set{fold_id}' # fold
            X_train = clinical_variable_df[clinical_variable_df[set_col] == 'train'][feature_columns] # get train features
            y_train = clinical_variable_df[clinical_variable_df[set_col] == 'train'][target_column] # get train label
            X_val = clinical_variable_df[clinical_variable_df[set_col] == 'val'][feature_columns] # get val features
            y_val = clinical_variable_df[clinical_variable_df[set_col] == 'val'][target_column] # get val label

            model_xgb = xgb.XGBClassifier(**params, importance_type='gain', validate_parameters=True) # define xgb
            model_xgb.fit(X_train, y_train) # train xgb
            preds_prob_clinical_variables = model_xgb.predict_proba(X_val) # predict using xgb
            patient_ids_clinical_variables = clinical_variable_df[clinical_variable_df[set_col] == 'val']['patient_id'].values # get patient ids

            # intialise lists to store data
            id_patients = []
            id_patient_logits = []
            id_patient_preds = []
            id_patient_labels = []
            patient_ids = val_dataset.df['patient_id'].to_numpy()
            correct = 0
            
            for id in np.unique(patient_ids): # for each unique patient
                indices = np.where(patient_ids == id)[0] # get the indices
                id_patients.append(id) # append patient id
                mean_cnn_logit = np.mean(logits[indices, :], axis=0) # get the cnn prediction (mean logit)
                ind = np.where(patient_ids_clinical_variables == id)[0] # get patient ind
                if len(ind) > 0:
                    xgb_logit = preds_prob_clinical_variables[ind, :][0]
                    combined_logit = (mean_cnn_logit + xgb_logit) / 2
                else:
                    combined_logit = mean_cnn_logit  # Use CNN logit only if no clinical data
                    print(f"Fold {fold_id} - Patient {id}: No clinical data, using CNN logit only")
                id_patient_logits.append(combined_logit) # append the logit
                id_patient_labels.append(np.mean(labels[indices], axis=0).round()) # append label
                id_patient_preds.append(combined_logit.argmax()) # append prediction
                correct += int(id_patient_preds[-1] == id_patient_labels[-1]) # see if prediction is correct or not

            # convert to array
            id_patient_logits = np.array(id_patient_logits)
            id_patient_labels = np.array(id_patient_labels)
            id_patient_preds = np.array(id_patient_preds)
            # get metrics
            accuracy = accuracy_score(id_patient_labels, id_patient_preds) if len(id_patients) > 0 else 0
            precision = precision_score(id_patient_labels, id_patient_preds, zero_division=0) if len(id_patients) > 0 else 0
            recall = recall_score(id_patient_labels, id_patient_preds, zero_division=0) if len(id_patients) > 0 else 0
            f1 = f1_score(id_patient_labels, id_patient_preds, zero_division=0) if len(id_patients) > 0 else 0
            auc = roc_auc_score(id_patient_labels, id_patient_logits[:, 1]) if len(id_patients) > 0 else 0
            # print metrics
            print(f'[Fold {fold_id}] Patient level with clinical variable => Accuracy: {accuracy*100:.2f}% AUC: {auc*100:.2f}%')
            # log metrics
            mlflow.log_metric("patient_level_accuracy", accuracy * 100)
            mlflow.log_metric("patient_level_precision", precision * 100)
            mlflow.log_metric("patient_level_recall", recall * 100)
            mlflow.log_metric("patient_level_f1", f1 * 100)
            mlflow.log_metric("patient_level_auc", auc * 100)

            metrics_path = os.path.join(new_dir_train, "metrics_patient.csv") # metrics path
            save_metrics_csv(fold_id, accuracy, precision, recall, f1, auc, metrics_path, train=True)
            mlflow.log_artifact(metrics_path)

            # Save predictions and logits
            np.savez_compressed(os.path.join(new_dir_with_fold, 'patient_level_with_clinical_variables_results'),
                                labels=id_patient_labels,
                                preds=id_patient_preds,
                                logits=id_patient_logits,
                                patient_ids=np.array(id_patients))
            # mlflow.log_artifact(os.path.join(new_dir_with_fold, 'patient_level_with_clinical_variables_results.npz'))

            confmat_vals = np.zeros((num_classes, num_classes))
            for true, pred in zip(id_patient_labels, id_patient_preds):
                confmat_vals[int(true), int(pred)] += 1
            plot_confusion_matrix(confmat_vals, num_classes,
                                  os.path.join(new_dir_train_figures, "patient_level_with_clinical_variables_conf_mat.png"),
                                  f"Confusion Matrix [Patient level] Fold {fold_id}")
            mlflow.log_artifact(os.path.join(new_dir_train_figures, "patient_level_with_clinical_variables_conf_mat.png"))

            # ROC curve using utils.plot_save_roc_curve
            plot_save_roc_curve(id_patient_labels, id_patient_logits,
                                os.path.join(new_dir_train_figures, "patient_level_roc_curve_fold_{}.png".format(fold_id)))
            mlflow.log_artifact(os.path.join(new_dir_train_figures, "patient_level_roc_curve_fold_{}.png".format(fold_id)))

if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="test"):
    print("Final Model")
    final_model_dir = f"./experiments/{data_type}/train/model" # final cnn model path

    final_model = models.resnet50(weights='IMAGENET1K_V1') # load cnn
    num_ftrs = final_model.fc.in_features
    final_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, num_classes)
    )
    final_model = final_model.to(device)
    final_model.load_state_dict(torch.load(os.path.join(final_model_dir, 'final.pth'))) # load weights
    final_model.eval() # set to eval mode

    test_dataset = CustomDataset('test', CSV_PATH, 0, transform=test_transform)  # test dataset
    test_loaders = [
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
        for _ in TTAs
    ] # test loader

    with torch.no_grad(): # no gradient updates
        # initialise empty variables to store data
        preds = []
        logits = []
        labels = []
        correct = 0
        for i, data in enumerate(zip(*test_loaders)): # for each batch
            inputs = torch.cat([img for img, _ in data], dim=0).to(device).float() # get inputs
            targets = data[0][1].to(device).long() # get labels
            outputs = final_model(inputs) # get outputs
            outputs = F.softmax(outputs, dim=-1) # get output softmax
            outputs = outputs.reshape(len(TTAs), int(inputs.shape[0]/len(TTAs)), -1).mean(dim=0) # get preds for each tta and batch
            _, predicted = torch.max(outputs, 1) # get preds of class 1
            correct += (predicted == targets).sum().item()
            # append data
            preds.append(predicted.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
            logits.append(outputs.detach().cpu().numpy().astype(np.float32))
        # concatenate everything
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)

    clinical_variable_df = pd.read_csv('../datasets/patients_fold.csv') # clinical data
    feature_columns = ['Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count', 
                        'Neutrophil count', 'Monocyte count', 'Platelet count', 
                        'Blast percentage (PB)', 'LDH'] # clinical features
    target_column = 'morphology' # target column
    rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125] # rechecked patient ids
    clinical_variable_df = clinical_variable_df.loc[clinical_variable_df["patient_id"] != 2209801848] # remove this id
    clinical_variable_df.loc[clinical_variable_df["patient_id"].isin(rechecked_patient_ids), "morphology"] = 0 # assign right label for rechecked patients

    X_train_final = clinical_variable_df[clinical_variable_df['set0'] != 'test'][feature_columns] # get train features
    y_train_final = clinical_variable_df[clinical_variable_df['set0'] != 'test'][target_column] # get train labels
    X_test_final = clinical_variable_df[clinical_variable_df['set0'] == 'test'][feature_columns] # get test features
    y_test_final = clinical_variable_df[clinical_variable_df['set0'] == 'test'][target_column] # get test labels

    model_xgb_final = xgb.XGBClassifier(**params, importance_type='gain', validate_parameters=True) # initialise xgb
    model_xgb_final.fit(X_train_final, y_train_final) # train xgb
    preds_prob_clinical_variables = model_xgb_final.predict_proba(X_test_final) # get predicted probabilities
    patient_ids_clinical_variables = clinical_variable_df[clinical_variable_df['set0'] == 'test']['patient_id'].values # get the values

    # initialise empty variables to store data
    id_patients = []
    id_patient_logits = []
    id_patient_preds = []
    id_patient_labels = []
    patient_ids = test_dataset.df['patient_id'].to_numpy() # get patient ids
    correct = 0

    for id in np.unique(patient_ids): # for each patient id
        indices = np.where(patient_ids == id)[0] # get index
        id_patients.append(id) # append id
        mean_cnn_logit = np.mean(logits[indices, :], axis=0) # get mean of cnn logit
        ind = np.where(patient_ids_clinical_variables == id)[0] # get index for xgb
        if len(ind) == 0: # if no index, continue since no patient
            continue
        xgb_logit = preds_prob_clinical_variables[ind, :][0] # get xgb logit
        combined_logit = (mean_cnn_logit + xgb_logit) / 2 # combine logits
        id_patient_logits.append(combined_logit) # append
        id_patient_labels.append(np.mean(labels[indices], axis=0).round()) # append label
        id_patient_preds.append(combined_logit.argmax()) # append pred
        correct += int(id_patient_preds[-1] == id_patient_labels[-1]) # see if pred is correct or not
    
    # convert to arrays
    id_patient_logits = np.array(id_patient_logits)
    id_patient_labels = np.array(id_patient_labels)
    id_patient_preds = np.array(id_patient_preds)
    # get metrics
    accuracy = accuracy_score(id_patient_labels, id_patient_preds) if len(id_patients) > 0 else 0
    precision = precision_score(id_patient_labels, id_patient_preds, zero_division=0) if len(id_patients) > 0 else 0
    recall = recall_score(id_patient_labels, id_patient_preds, zero_division=0) if len(id_patients) > 0 else 0
    f1 = f1_score(id_patient_labels, id_patient_preds, zero_division=0) if len(id_patients) > 0 else 0
    auc = roc_auc_score(id_patient_labels, id_patient_logits[:, 1]) if len(id_patients) > 0 else 0
    print(f'[Final Model] Patient level with clinical variable => Accuracy: {accuracy*100:.2f}% AUC: {auc*100:.2f}%')
    # log metrics
    mlflow.log_metric("final_patient_level_accuracy", accuracy * 100)
    mlflow.log_metric("final_patient_level_precision", precision * 100)
    mlflow.log_metric("final_patient_level_recall", recall * 100)
    mlflow.log_metric("final_patient_level_f1", f1 * 100)
    mlflow.log_metric("final_patient_level_auc", auc * 100)

    metrics_path = os.path.join(new_dir_test, "metrics.csv") # save metrics
    save_metrics_csv("Patient", accuracy, precision, recall, f1, auc, metrics_path, train=False) # save metrics
    mlflow.log_artifact(metrics_path) # log metrics

    np.savez_compressed(os.path.join(new_dir_test, 'patient_level_with_clinical_variables_results'),
                        labels=id_patient_labels,
                        preds=id_patient_preds,
                        logits=id_patient_logits,
                        patient_ids=np.array(id_patients)) # save preds and logits

    # plot and save confusion matrix
    confmat_vals = np.zeros((num_classes, num_classes))
    for true, pred in zip(id_patient_labels, id_patient_preds):
        confmat_vals[int(true), int(pred)] += 1
    plot_confusion_matrix(confmat_vals, num_classes,
                            os.path.join(new_dir_test_figures, "patient_level_with_clinical_variables_conf_mat.png"),
                            "Confusion Matrix [Patient level] Final Model")
    mlflow.log_artifact(os.path.join(new_dir_test_figures, "patient_level_with_clinical_variables_conf_mat.png"))

    # plot and save roc curve
    plot_save_roc_curve(id_patient_labels, id_patient_logits,
                        os.path.join(new_dir_test_figures, "patient_level_roc_curve_final.png"))
    mlflow.log_artifact(os.path.join(new_dir_test_figures, "patient_level_roc_curve_final.png"))