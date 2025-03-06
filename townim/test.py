'''
This script is used to test the models trained after rechecking the patient IDs.
'''

# import the libraries
import numpy as np
import pandas as pd
import os, random, sys, argparse, torchvision, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from torchvision import models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import subprocess

# sys.path.append('/home/tchowdhury/data/code/CMML-v2/townim')
sys.path.append('./towmin')
import utils
from utils import FixedRotation, plot_confusion_matrix, plot_save_roc_curve, save_metrics_csv
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH, MONOCYTE_NEW_NORMALS_CSV_PATH

# creating a cache directory since running docker using specific user doesn't allow to use the home cache directory
cache_dir = "../cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir # set cache directory

rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125] # patient ids that were rechecked

torch.cuda.empty_cache() # clear the cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set the device

# create an argument parser, with arguments for fold, data type, and tta
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='monocyte', choices=('monocyte', 'neutrophil'), help='data type')
args = parser.parse_args()

# set the data type and csv path
data_type = args.data_type # neutrophil, monocyte
if data_type == "neutrophil":
    CSV_PATH = NEUTROPHIL_CSV_PATH
elif data_type == "monocyte":
    CSV_PATH = MONOCYTE_CSV_PATH
elif data_type == "monocyte_new_normals":
    CSV_PATH = MONOCYTE_NEW_NORMALS_CSV_PATH
else:
    raise ValueError("Invalid data type")
batch_size = 32 # batch size 
IMAGE_SIZE = 352 # image size
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225] # Std of ImageNet dataset (used for normalization)

# define the test transforms
test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# create multiple test time augmentations
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

# check if experiment exists, create if not
existing_experiment = mlflow.get_experiment_by_name(data_type)
if existing_experiment is None:
    mlflow.create_experiment(data_type)  # Create new experiment with name
    mlflow.set_experiment(experiment_name=data_type)
    print(f"Created new experiment for {data_type} (Experiment won't include train-val metrics).")
else:
    experiment_id = existing_experiment.experiment_id  # Reuse existing ID
    overwrite_exp = input(f"DO YOU WANT TO OVERWRITE EXISTING {data_type} EXPERIMENT? [Y/N]")
    if overwrite_exp.lower() == "y":
        runs = mlflow.search_runs(experiment_ids = [experiment_id], filter_string="run_name='test'")
        if runs:
            run_id = runs["run_id"]
            mlflow.delete_run(run_id)
            subprocess.run(["mlflow", "gc", "--run-ids", run_id], check=True)
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        print("Cannot run if existing experiment test run is not allowed to be overwritten! Please modify code to change behaviour.")
        exit()
    print(f"Using existing experiment for {data_type}")

# clear test directory
test_dir = f"./experiments/{data_type}/test"
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
    print(f"Cleared test directory: {test_dir}")

with mlflow.start_run("test"):
    root = f"./experiments/{data_type}"
    test_root = root + "/test"
    os.makedirs(test_root, exist_ok=True)
    test_figure_dir = os.path.join(test_root, "figures")
    os.makedirs(test_figure_dir, exist_ok=True)

    # Define test dataset and loaders
    test_dataset = CustomDataset('test', CSV_PATH, 0, transform=test_transform)
    test_loaders = [
        DataLoader(CustomDataset('test', CSV_PATH, 0, transform=transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
        for transform in TTAs
    ]
    print("Test dataset stats [Normal, CMML]:", test_dataset.disease_count)
    
    final_model_dir = os.path.join(root, "model")
    final_model = models.resnet50(weights='IMAGENET1K_V1')  # define final model
    num_ftrs = final_model.fc.in_features
    hidden_layer_size = 512
    num_classes = len(set(test_dataset.labels))
    final_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, num_classes)
    )
    final_model = final_model.to(device)
    try:
        final_model.load_state_dict(torch.load(os.path.join(final_model_dir, f'final.pth')))
    except FileNotFoundError as e:
        err_message = "Model file not found. Please train the model first or check if the directory exists."
        e.add_note(err_message)
        raise
    final_model.eval()

    # No-TTA Evaluation on Test Set
    with torch.no_grad():
        preds, labels, logits = [], [], []
        for i, (inputs, targets) in enumerate(test_loaders[0]):  # Use first loader (no TTA)
            inputs, targets = inputs.to(device).float(), targets.to(device).long()
            outputs = final_model(inputs)
            outputs = F.softmax(outputs, dim=-1)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
            logits.append(outputs.detach().cpu().numpy().astype(np.float32))

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")
        f1 = f1_score(labels, preds, average="binary")
        auc = roc_auc_score(labels, logits[:, 1])

        # Save metrics
        save_metrics_csv("Image W/O TTA", accuracy, precision, recall, f1, auc, os.path.join(test_root, "metrics.csv"), train=False)

        # Confusion matrix
        confmat_vals = confusion_matrix(labels, preds)
        plot_confusion_matrix(confmat_vals, num_classes, os.path.join(test_figure_dir, "image_level_conf_mat_test.png"), "Confusion Matrix on Test [Image level] Final Model")
        mlflow.log_artifact(os.path.join(test_figure_dir, "image_level_conf_mat_test.png"))

        # ROC curve
        plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(test_figure_dir, "image_level_roc_curve_test.png"))
        mlflow.log_artifact(os.path.join(test_figure_dir, "image_level_roc_curve_test.png"))
        print(f'[W/O TTA] Final Model Image-level Test Accuracy: {accuracy} AUC: {auc}')

    # TTA Evaluation on Test Set
    with torch.no_grad():
        preds, labels, logits = [], [], []
        for i, data in enumerate(zip(*test_loaders)):
            inputs, targets = torch.cat([img for img, _ in data], dim=0).to(device).float(), data[0][1].to(device).long()
            outputs = final_model(inputs)
            outputs = F.softmax(outputs, dim=-1)
            outputs = outputs.reshape(len(data), int(inputs.shape[0]/len(data)), -1).mean(dim=0)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
            logits.append(outputs.detach().cpu().numpy().astype(np.float32))

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")
        f1 = f1_score(labels, preds, average="binary")
        auc = roc_auc_score(labels, logits[:, 1])

        # Save metrics
        save_metrics_csv("Image TTA", accuracy, precision, recall, f1, auc, os.path.join(test_root, "metrics.csv"), train=False)

        # Confusion matrix
        confmat_vals = confusion_matrix(labels, preds)
        plot_confusion_matrix(confmat_vals, num_classes, os.path.join(test_figure_dir, "tta_image_level_conf_mat_test.png"), "Confusion Matrix on Test [Image level with TTA] Final Model")
        mlflow.log_artifact(os.path.join(test_figure_dir, "tta_image_level_conf_mat_test.png"))

        # ROC curve
        plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(test_figure_dir, "tta_image_level_roc_curve_test.png"))
        mlflow.log_artifact(os.path.join(test_figure_dir, "tta_image_level_roc_curve_test.png"))
        print(f'[TTA] Final Model Image-level Test Accuracy: {accuracy} AUC: {auc}')

    # Patient-Level Evaluation on Test Set
    id_patients, id_patient_logits, id_patient_preds, id_patient_labels = [], [], [], []
    patient_ids = test_dataset.df['patient_id'].to_numpy()
    incorrect_ids = []

    for id in np.unique(patient_ids):
        indices = np.where(patient_ids == id)[0]
        id_patient_logits.append(np.mean(logits[indices, :], axis=0))
        id_patient_labels.append(np.mean(labels[indices], axis=0))
        id_patient_preds.append(id_patient_logits[-1].argmax())
        if id in rechecked_patient_ids:
            print(f"Patient Id (Rechecked): {id}, Ground Truth: {id_patient_labels[-1]}, Predicted: {id_patient_preds[-1]}")
        if id_patient_labels[-1] != id_patient_preds[-1]:
            incorrect_ids.append((id, id_patient_labels[-1], id_patient_preds[-1]))
        id_patients.append(id)

    for id_data in incorrect_ids:
        print(f"Patient Id: {id_data[0]}, Ground Truth: {id_data[1]}, Predicted: {id_data[2]}")

    preds = np.array(id_patient_preds)
    labels = np.array(id_patient_labels)
    logits = np.array(id_patient_logits)

    # Calculate patient-level metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:, 1])

    # Save metrics
    save_metrics_csv("Patient", accuracy, precision, recall, f1, auc, os.path.join(test_root, "metrics.csv"), train=False)

    # Confusion matrix
    confmat_vals = confusion_matrix(labels, preds)
    plot_confusion_matrix(confmat_vals, num_classes, os.path.join(test_figure_dir, "patient_level_conf_mat_test.png"), "Confusion Matrix on Test [Patient level] Final Model")
    mlflow.log_artifact(os.path.join(test_figure_dir, "patient_level_conf_mat_test.png"))

    # ROC curve
    plot_save_roc_curve(np.eye(num_classes)[np.round(labels).astype(int)], logits, os.path.join(test_figure_dir, "patient_level_roc_curve_test.png"))
    mlflow.log_artifact(os.path.join(test_figure_dir, "patient_level_roc_curve_test.png"))
    print(f'[TTA] Final Model Patient-level Test Accuracy: {accuracy} AUC: {auc}')

    # Log all test artifacts to MLflow
    mlflow.log_artifact(os.path.join(test_root, "metrics.csv"))

print("Training and testing complete!")