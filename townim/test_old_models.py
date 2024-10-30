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

# sys.path.append('/home/tchowdhury/data/code/CMML-v2/townim')
sys.path.append('./towmin')
import utils
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH

# creating a cache directory since running docker using specific user doesn't allow to use the home cache directory
cache_dir = "../cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir # set cache directory

rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]

# Function to save or update metrics CSV
def save_metrics_csv(fold, accuracy, precision, recall, f1, auroc, metrics_path):
    new_metrics = pd.DataFrame([[fold, round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), round(auroc, 3)]], 
                                columns=["fold", "accuracy", "precision", "recall", "f1", "auroc"])

    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        if fold in df['fold'].values:
            df.loc[df['fold'] == fold, 'accuracy'] = round(accuracy, 4)
            df.loc[df['fold'] == fold, 'precision'] = round(precision, 4)
            df.loc[df['fold'] == fold, 'recall'] = round(recall, 4)
            df.loc[df['fold'] == fold, 'f1'] = round(f1, 4)
            df.loc[df['fold'] == fold, 'auroc'] = round(auroc, 4)
        else:
            df = pd.concat([df, new_metrics], ignore_index=True)
    else:
        df = new_metrics
    
    df.to_csv(metrics_path, index=False)

def plot_save_roc_curve(labels, logits, figure_path):
    # Compute ROC curve and ROC area
    if labels.ndim > 1:
        labels = labels[:, 1]
        
    if logits.ndim > 1:
        logits = logits[:, 1]
    fpr, tpr, _ = roc_curve(labels, logits)
    roc_auc = roc_auc_score(labels, logits)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')

    # Plot the diagonal line (no discrimination)
    plt.plot([0, 1], [0, 1], 'k--')

    # Set plot limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Set x and y ticks
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.yticks(np.arange(0.0, 1.1, step=0.1))

    # Save figure to path
    plt.savefig(figure_path)
    plt.close()

# Confusion matrix plot function
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

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=1, help='fold_id')
parser.add_argument('--data_type', type=str, default='monocyte', choices=('monocyte', 'neutrophil'), help='data type')
parser.add_argument('--tta', type=bool, default=False, choices=(False, True), help="Test Time Augmentations")
args = parser.parse_args()

set_id = int(args.fold)
is_tta = args.tta

# Create data loaders
data_type = args.data_type # neutrophil, monocyte
CSV_PATH = NEUTROPHIL_CSV_PATH if data_type == 'neutrophil' else MONOCYTE_CSV_PATH
batch_size = 32
IMAGE_SIZE = 352
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)


test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

class FixedRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return T.functional.rotate(x, self.angle)


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

test_dataset = CustomDataset('test', CSV_PATH, set_id, transform=test_transform)
test_loaders =[
    DataLoader(CustomDataset('test', CSV_PATH, set_id, transform=transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for transform in TTAs
]

num_classes = len(set(test_dataset.labels))
# Define the model architecture (ResNet-50 as an example)
print("Model: Resnet50")
model = models.resnet50(weights='IMAGENET1K_V1')
hidden_layer_size = 512
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_layer_size),
    nn.ReLU(),
    # nn.BatchNorm1d(hidden_layer_size),
    # nn.Dropout(0.2),
    nn.Linear(hidden_layer_size, num_classes)
)

model = model.to(device)
# model_dir = f'/home/tchowdhury/data/code/CMML-v2/townim/models/{data_type}_fold_{args.fold}_without_TTA'
exp_dir = f'./old_models'
exp_subdir = exp_dir + f'/{data_type}_fold_{args.fold}_'
exp_subdir += "with_TTA" if is_tta else "without_TTA"
model_dir = exp_subdir
figure_dir = exp_subdir + "/figures/test"
os.makedirs(figure_dir, exist_ok=True)
model.load_state_dict(torch.load(os.path.join(model_dir, f'last.pth')))
model.eval()
print(model_dir)

with torch.no_grad():
    correct = 0
    preds = []
    labels = []
    logits = []
    # for i, (inputs, targets) in tqdm(enumerate(test_loaders[0]), total=len(test_loaders[0]), smoothing=0.9, position=0, leave=True,):
    for i, (inputs, targets) in enumerate(test_loaders[0]):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=-1)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        preds.append(predicted.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())
        logits.append(outputs.detach().cpu().numpy().astype(np.float32))

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    logits = np.concatenate(logits, axis=0)

    # Calculate metrics at image level
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:, 1])

    # save_metrics_csv(args.fold, accuracy, precision, recall, f1, auc, os.path.join(exp_dir, "metrics_image.csv"))

    # # Confusion matrix
    confmat_vals = confusion_matrix(labels, preds)
    # plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, "image_level_conf_mat.png"), "Confusion Matrix on Test [Image level]")

    # # ROC curve
    # plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, "image_level_roc_curve.png"))
    
    print(f'[W/O TTA] Image level Accuracy: {accuracy} AUC: {auc}')
    
    preds = []
    logits = []
    labels = []
    correct = 0
    # for i, data in tqdm(enumerate(zip(*test_loaders)), total=len(test_loaders[0]), smoothing=0.9, position=0, leave=True,):
    for i, data in enumerate(zip(*test_loaders)):
        inputs, targets = torch.cat([img for img,_ in data], dim=0).to(device).float(), data[0][1].to(device).long()#torch.stack([l.squeeze(0) for _,l in data], dim=0).to(device).long()
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=-1)
        outputs = outputs.reshape(len(data), int(inputs.shape[0]/len(data)), -1).mean(dim=0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        preds.append(predicted.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())
        logits.append(outputs.detach().cpu().numpy().astype(np.float32))

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    logits = np.concatenate(logits, axis=0)

    # Calculate metrics with TTA
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:, 1])

    # Save metrics to CSV
    # save_metrics_csv(args.fold, accuracy, precision, recall, f1, auc, os.path.join(exp_dir, "metrics_image_tta.csv"))

    # # Confusion matrix
    confmat_vals = confusion_matrix(labels, preds)
    # plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, "tta_image_level_conf_mat.png"), "Confusion Matrix on Test [Image level with TTA]")

    # # ROC curve
    # plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, "tta_image_level_roc_curve.png"))
    print(f'[TTA] Image level Accuracy: {accuracy} AUC: {auc}')
    
    # np.savez_compressed(os.path.join(model_dir, 'image_level_results'),
    #         labels=labels, 
    #         preds=preds,
    #         logits=logits,
    # )

##### patient level
id_patients = []
id_patient_logits = []
id_patient_preds = []
id_patient_labels = []
patient_ids = test_dataset.df['patient_id'].to_numpy()
correct = 0
incorrect_ids = []
for id in np.unique(patient_ids):
    indices = np.where(patient_ids==id)[0]
    id_patient_logits.append(np.mean(logits[indices,:], axis=0))
    id_patient_labels.append(np.mean(labels[indices], axis=0))
    id_patient_preds.append(id_patient_logits[-1].argmax())
    if id in rechecked_patient_ids:
        print(f"Patient Id: {id}, Ground Truth: {id_patient_labels[-1]}, Predicted: {id_patient_preds[-1]}")
    if id_patient_labels[-1] != id_patient_preds[-1]:
        incorrect_ids.append((id, id_patient_labels[-1], id_patient_preds[-1]))
    correct += int(id_patient_preds[-1]==id_patient_labels[-1])
    id_patients.append(id)
    
# for id_data in incorrect_ids:
#     print(f"Patient Id: {id_data[0]}, Ground Truth: {id_data[1]}, Predicted: {id_data[2]}")

preds = np.array(id_patient_preds)
labels = np.array(id_patient_labels)
logits = np.array(id_patient_logits)

# Calculate patient-level metrics
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average="binary")
recall = recall_score(labels, preds, average="binary")
f1 = f1_score(labels, preds, average="binary")
auc = roc_auc_score(labels, logits[:, 1])

# Save patient-level metrics to CSV
# save_metrics_csv(args.fold, accuracy, precision, recall, f1, auc, os.path.join(exp_dir, "metrics_patient.csv"))

# # Confusion matrix for patient-level
confmat_vals = confusion_matrix(labels, preds)
# plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, "patient_level_conf_mat.png"), "Confusion Matrix on Test [Patient level]")

print(f'[TTA] Patient level Accuracy: {accuracy} AUC: {auc}')
