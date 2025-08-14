import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd 
from sklearn.metrics import roc_curve, roc_auc_score

def set_random_seed(seed: int) -> None:
    print(f"Setting seeds: {seed} ...... ")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True
    
def worker_init_fn(worker_id):     
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def make_weights_for_balanced_classes(labels, device):
    count = torch.bincount(torch.tensor(labels)).to(device)
    print('Count:', count.cpu().detach().numpy())
    
    weight = 1. / count.cpu().detach().numpy()
    print('Data sampling weight:', weight)
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight

def convert_path_to_os_specific(path: str) -> str:
    normalized_path = os.path.normpath(path)
    if os.sep == '/':
        return normalized_path.replace('\\', os.sep)
    else:
        return normalized_path.replace('/', os.sep)

def denormalize(image, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    return image * std + mean

class FixedRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return T.functional.rotate(x, self.angle)
        
def save_metrics_csv(fold, accuracy, precision, recall, f1, auroc, metrics_path, train=True):
    columns = []
    if train:
        columns = ["fold", "accuracy", "precision", "recall", "f1", "auroc"]
    else:
        columns = ["test type", "accuracy", "precision", "recall", "f1", "auroc"]
    new_metrics = pd.DataFrame([[fold, round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), round(auroc, 3)]], 
                                columns=columns)

    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        if train and fold in df['fold'].values:
            df.loc[df['fold'] == fold, 'accuracy'] = round(accuracy, 4)
            df.loc[df['fold'] == fold, 'precision'] = round(precision, 4)
            df.loc[df['fold'] == fold, 'recall'] = round(recall, 4)
            df.loc[df['fold'] == fold, 'f1'] = round(f1, 4)
            df.loc[df['fold'] == fold, 'auroc'] = round(auroc, 4)
        elif not train and fold in df["test type"].values:
            df.loc[df['test type'] == fold, 'accuracy'] = round(accuracy, 4)
            df.loc[df['test type'] == fold, 'precision'] = round(precision, 4)
            df.loc[df['test type'] == fold, 'recall'] = round(recall, 4)
            df.loc[df['test type'] == fold, 'f1'] = round(f1, 4)
            df.loc[df['test type'] == fold, 'auroc'] = round(auroc, 4)
        else:
            df = pd.concat([df, new_metrics], ignore_index=True)
    else:
        df = new_metrics
    
    df.to_csv(metrics_path, index=False)

def plot_save_roc_curve(labels, logits, figure_path):
    if labels.ndim > 1:
        labels = labels[:, 1]
        
    if logits.ndim > 1:
        logits = logits[:, 1]
        
    fpr, tpr, _ = roc_curve(labels, logits)
    roc_auc = roc_auc_score(labels, logits)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.yticks(np.arange(0.0, 1.1, step=0.1))

    plt.savefig(figure_path)
    plt.close()

def plot_confusion_matrix(confmat_vals, num_classes, figure_path, title):
    fig, ax = plt.subplots()
    im = ax.imshow(confmat_vals)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, confmat_vals[i, j], ha="center", va="center", color="black", fontsize=12)

    ax.set_title(title)
    plt.savefig(figure_path)
    plt.close()
    
def assign_patient_folds_splits(data_type_path, patient_folds_path="../datasets/patients_fold.csv"):
    monocyte_path = data_type_path
    patient_folds_path = patient_folds_path

    monocyte_reassigned = pd.read_csv(monocyte_path)
    patient_folds = pd.read_csv(patient_folds_path)

    set_columns = ['set0', 'set1', 'set2', 'set3', 'set4']
    monocyte_reassigned.drop_duplicates(subset=["patient_id"], inplace=True)

    unique_patients_csv1 = list(patient_folds["patient_id"].unique())
    unique_patients_csv2 = list(monocyte_reassigned["patient_id"].unique())

    for patient_id in unique_patients_csv1:
        if patient_id in unique_patients_csv2:
            for col in set_columns:
                patient_folds.loc[patient_folds['patient_id'] == patient_id, col] = monocyte_reassigned.loc[monocyte_reassigned["patient_id"] == patient_id, col].values[0]

    patient_folds.to_csv(patient_folds_path, index=False)
    print("Updated patient_folds.csv")
    
class Identity(torch.nn.Module):
    def forward(self, x):
        return x
    
def extract_embeddings(model, loader, dataset, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(device).float()
            feats = model(inputs)
            embeddings.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)
    patient_ids = dataset.df['patient_id'].to_numpy()
    image_paths = dataset.df['image_path'].to_numpy()
    return embeddings, labels, patient_ids, image_paths

def extract_logits(model, loader, dataset, device):
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=-1)
            logits.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())
    logits = np.concatenate(logits)
    labels = np.array(labels)
    patient_ids = dataset.df['patient_id'].to_numpy()
    patient_ids = patient_ids.astype(str)
    return logits, labels, patient_ids