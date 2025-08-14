import numpy as np
import pandas as pd
import os, sys, random, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
from PIL import Image

sys.path.append('./towmin')
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH, MONOCYTE_NEW_NORMALS_CSV_PATH, SEGMENTED_MONOCYTE_CSV_PATH, SEGMENTED_MONOCYTE_NEW_NORMALS_CSV_PATH

cache_dir = "../cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='monocyte', choices=('monocyte', 'neutrophil', 'monocyte_new_normals', 'segmented_monocyte', 'segmented_monocyte_new_normals'), help='data type')
parser.add_argument('--fold', type=int, default=0, choices=range(5), help='fold number (0-4)')
args = parser.parse_args()

data_type = args.data_type
if data_type == "neutrophil":
    CSV_PATH = NEUTROPHIL_CSV_PATH
elif data_type == "monocyte":
    CSV_PATH = MONOCYTE_CSV_PATH
elif data_type == "monocyte_new_normals":
    CSV_PATH = MONOCYTE_NEW_NORMALS_CSV_PATH
elif data_type == "segmented_monocyte":
    CSV_PATH = SEGMENTED_MONOCYTE_CSV_PATH
elif data_type == "segmented_monocyte_new_normals":
    CSV_PATH = SEGMENTED_MONOCYTE_NEW_NORMALS_CSV_PATH
else:
    raise ValueError("Invalid data type")

batch_size = 32
IMAGE_SIZE = 352
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

TTAs = [
    test_transform,
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomVerticalFlip(p=1.0), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomRotation(45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomRotation(45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomVerticalFlip(p=1.0), T.RandomRotation(45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0), T.RandomRotation(45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
]

test_dataset = CustomDataset('test', CSV_PATH, 0, transform=test_transform)
test_loaders = [
    DataLoader(CustomDataset('test', CSV_PATH, 0, transform=transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for transform in TTAs
]
print("Test dataset stats [Normal, CMML]:", test_dataset.disease_count)

model_dir = f"./experiments/{data_type}/train/{data_type}_fold_{args.fold}_with_TTA/model"
model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
hidden_layer_size = 512
num_classes = len(set(test_dataset.labels))
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, num_classes)
)
model = model.to(device)
try:
    model.load_state_dict(torch.load(os.path.join(model_dir, f'model_fold_{args.fold}.pth')))
except FileNotFoundError as e:
    print(f"Model file for fold {args.fold} not found. Please check if the directory exists.")
    raise
model.eval()

with torch.no_grad():
    preds, labels, logits = [], [], []
    for inputs, targets in test_loaders[0]:
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=-1)
        _, predicted = torch.max(outputs, 1)
        preds.append(predicted.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())
        logits.append(outputs.detach().cpu().numpy().astype(np.float32))

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    logits = np.concatenate(logits, axis=0)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:, 1])
    print(f'[No TTA] Fold {args.fold} Image-level Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')

with torch.no_grad():
    preds, labels, logits = [], [], []
    for data in zip(*test_loaders):
        inputs, targets = torch.cat([img for img, _ in data], dim=0).to(device).float(), data[0][1].to(device).long()
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=-1)
        outputs = outputs.reshape(len(data), int(inputs.shape[0]/len(data)), -1).mean(dim=0)
        _, predicted = torch.max(outputs, 1)
        preds.append(predicted.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())
        logits.append(outputs.detach().cpu().numpy().astype(np.float32))

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    logits = np.concatenate(logits, axis=0)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:, 1])
    print(f'[TTA] Fold {args.fold} Image-level Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')

rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]
patient_ids = test_dataset.df['patient_id'].to_numpy()
id_patients, id_patient_logits, id_patient_preds, id_patient_labels = [], [], [], []
incorrect_ids = []

for id in np.unique(patient_ids):
    indices = np.where(patient_ids == id)[0]
    id_patient_logits.append(np.mean(logits[indices, :], axis=0))
    id_patient_labels.append(np.mean(labels[indices], axis=0))
    id_patient_preds.append(id_patient_logits[-1].argmax())
    id_patients.append(id)
    if id in rechecked_patient_ids:
        print(f"Patient Id (Rechecked): {id}, Ground Truth: {id_patient_labels[-1]}, Predicted: {id_patient_preds[-1]}")
    if id_patient_labels[-1] != id_patient_preds[-1]:
        incorrect_ids.append((id, id_patient_labels[-1], id_patient_preds[-1]))

for id_data in incorrect_ids:
    print(f"Patient Id: {id_data[0]}, Ground Truth: {id_data[1]}, Predicted: {id_data[2]}")

preds = np.array(id_patient_preds)
labels = np.array(id_patient_labels)
logits = np.array(id_patient_logits)

accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average="binary")
recall = recall_score(labels, preds, average="binary")
f1 = f1_score(labels, preds, average="binary")
auc = roc_auc_score(labels, logits[:, 1])
print(f'[TTA] Fold {args.fold} Patient-level Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')

print("Testing complete!")