'''
This script is used to test the old models that were trained (before the labels were rechecked).
Make sure that the old_models directory is present in the github repo directory (CMML-v2).
'''

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

rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125] # patient ids that were rechecked

# Function to save or update metrics CSV
def save_metrics_csv(fold, accuracy, precision, recall, f1, auroc, metrics_path):
    '''
    function: save metrics file
    paramters:
        fold: int, fold number
        accuracy: float, accuracy
        precision: float, precision
        recall: float, recall
        f1: float, f1 score
        auroc: float, auroc score
        metrics_path: str, path to save the metrics
    returns: None
    '''
    new_metrics = pd.DataFrame([[fold, round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), round(auroc, 3)]], 
                                columns=["fold", "accuracy", "precision", "recall", "f1", "auroc"]) # create new metrics dataframe

    if os.path.exists(metrics_path): # if metrics file exists
        df = pd.read_csv(metrics_path) # read the metrics file
        if fold in df['fold'].values: # if the fold exists in the metrics file
            df.loc[df['fold'] == fold, 'accuracy'] = round(accuracy, 4) # update the accuracy
            df.loc[df['fold'] == fold, 'precision'] = round(precision, 4) # update the precision
            df.loc[df['fold'] == fold, 'recall'] = round(recall, 4) # update the recall
            df.loc[df['fold'] == fold, 'f1'] = round(f1, 4) # update the f1 score
            df.loc[df['fold'] == fold, 'auroc'] = round(auroc, 4) # update the auroc score
        else: # if the fold does not exist in the metrics file
            df = pd.concat([df, new_metrics], ignore_index=True) # add the new metrics to the existing metrics
    else: # if the metrics file does not exist
        df = new_metrics # create a new metrics file
    
    df.to_csv(metrics_path, index=False) # save the metrics file

def plot_save_roc_curve(labels, logits, figure_path):
    '''
    function: plot and save the ROC curve
    parameters:
        labels: numpy array, true labels
        logits: numpy array, predicted logits
        figure_path: str, path to save the figure
    returns: None
    '''
    # Compute ROC curve and ROC area
    if labels.ndim > 1: # if labels has more than 1 dimension
        labels = labels[:, 1] # get the second column
        
    if logits.ndim > 1: # if logits has more than 1 dimension
        logits = logits[:, 1] # get the second column
    fpr, tpr, _ = roc_curve(labels, logits) # compute the ROC curve
    roc_auc = roc_auc_score(labels, logits) # compute the ROC area

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
    '''
    function: plot and save the confusion matrix
    paramters:
        confmat_vals: numpy array, confusion matrix values
        num_classes: int, number of classes
        figure_path: str, path to save the figure
        title: str, title of the figure
    returns: None
    '''
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

torch.cuda.empty_cache() # clear the cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set the device

# create an argument parser, to get the fold number and data type
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=1, help='fold_id')
parser.add_argument('--data_type', type=str, default='monocyte', choices=('monocyte', 'neutrophil'), help='data type')
parser.add_argument('--tta', type=bool, default=False, choices=(False, True), help="Test Time Augmentations")
args = parser.parse_args()

set_id = int(args.fold)
is_tta = args.tta

# define the data type and csv path
data_type = args.data_type # neutrophil, monocyte
CSV_PATH = NEUTROPHIL_CSV_PATH if data_type == 'neutrophil' else MONOCYTE_CSV_PATH
batch_size = 32 # batch size
IMAGE_SIZE = 352 # image size
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225] # Std of ImageNet dataset (used for normalization)

# Define the test transforms
test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
class FixedRotation:
    '''
    This class is used to create a fixed rotation transformation
    '''
    def __init__(self, angle):
        '''
        function: initializes the FixedRotation class
        parameters:
            angle: int, angle of rotation
        returns: None
        '''
        self.angle = angle # set the angle of rotation

    def __call__(self, x):
        '''
        function: applies the rotation transformation
        parameters:
            x: image
        returns: image
        '''
        return T.functional.rotate(x, self.angle) # apply and return the rotated image

# Define the Test Time Augmentations (TTAs)
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

# Define the test dataset and loaders
test_dataset = CustomDataset('test', CSV_PATH, set_id, transform=test_transform)
test_loaders =[
    DataLoader(CustomDataset('test', CSV_PATH, set_id, transform=transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for transform in TTAs
]

num_classes = len(set(test_dataset.labels)) # number of classes (2)
# define the model and load the weights
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

model = model.to(device) # set the model to the device
# define the experiment directory and model directory (relative path)
exp_dir = f'./old_models'
exp_subdir = exp_dir + f'/{data_type}_fold_{args.fold}_'
exp_subdir += "with_TTA" if is_tta else "without_TTA"
model_dir = exp_subdir
figure_dir = exp_subdir + "/figures/test"
os.makedirs(figure_dir, exist_ok=True)

model.load_state_dict(torch.load(os.path.join(model_dir, f'last.pth'))) # load the model weights
model.eval() # set the model to evaluation mode
print(model_dir)

with torch.no_grad(): # turn off the gradients
    correct = 0 # initialize the correct count
    preds = [] # initialize the predictions
    labels = [] # initialize the labels
    logits = [] # initialize the logits
    for i, (inputs, targets) in enumerate(test_loaders[0]): # iterate over the test loader
        inputs, targets = inputs.to(device).float(), targets.to(device).long() # set the inputs and targets to the device
        outputs = model(inputs) # get the model outputs
        outputs = F.softmax(outputs, dim=-1) # get the probabilities
        _, predicted = torch.max(outputs, 1) # get the predicted class
        correct += (predicted == targets).sum().item() # update the correct count
        preds.append(predicted.detach().cpu().numpy()) # append the predictions
        labels.append(targets.detach().cpu().numpy()) # append the labels
        logits.append(outputs.detach().cpu().numpy().astype(np.float32)) # append the logits

    preds = np.concatenate(preds, axis=0) # concatenate the predictions
    labels = np.concatenate(labels, axis=0) # concatenate the labels
    logits = np.concatenate(logits, axis=0) # concatenate the logits

    # Calculate metrics at image level
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:, 1])

    save_metrics_csv(args.fold, accuracy, precision, recall, f1, auc, os.path.join(exp_dir, "metrics_image.csv"))

    # Confusion matrix
    confmat_vals = confusion_matrix(labels, preds)
    plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, "image_level_conf_mat.png"), "Confusion Matrix on Test [Image level]")

    # ROC curve
    plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, "image_level_roc_curve.png"))
    
    print(f'[W/O TTA] Image level Accuracy: {accuracy} AUC: {auc}')
    
    preds = [] # initialize the predictions
    logits = [] # initialize the logits
    labels = [] # initialize the labels
    correct = 0 # initialize the correct count

    for i, data in enumerate(zip(*test_loaders)): # iterate over the test loaders (zipped)
        inputs, targets = torch.cat([img for img,_ in data], dim=0).to(device).float(), data[0][1].to(device).long() # set the inputs and targets to the device
        outputs = model(inputs) # get the model outputs
        outputs = F.softmax(outputs, dim=-1) # get the probabilities
        outputs = outputs.reshape(len(data), int(inputs.shape[0]/len(data)), -1).mean(dim=0) # get the mean of the outputs

        _, predicted = torch.max(outputs, 1) # get the predicted class
        correct += (predicted == targets).sum().item() # update the correct count
        preds.append(predicted.detach().cpu().numpy()) # append the predictions
        labels.append(targets.detach().cpu().numpy()) # append the labels
        logits.append(outputs.detach().cpu().numpy().astype(np.float32)) # append the logits

    preds = np.concatenate(preds, axis=0) # concatenate the predictions
    labels = np.concatenate(labels, axis=0) # concatenate the labels
    logits = np.concatenate(logits, axis=0) # concatenate the logits

    # Calculate metrics with TTA
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:, 1])

    # Save metrics to CSV
    save_metrics_csv(args.fold, accuracy, precision, recall, f1, auc, os.path.join(exp_dir, "metrics_image_tta.csv"))

    # # Confusion matrix
    confmat_vals = confusion_matrix(labels, preds)
    plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, "tta_image_level_conf_mat.png"), "Confusion Matrix on Test [Image level with TTA]")

    # ROC curve
    plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, "tta_image_level_roc_curve.png"))
    print(f'[TTA] Image level Accuracy: {accuracy} AUC: {auc}')
    
    np.savez_compressed(os.path.join(model_dir, 'image_level_results'),
            labels=labels, 
            preds=preds,
            logits=logits,
    )

# patient level
id_patients = [] # initialize the patient ids
id_patient_logits = [] # initialize the patient logits
id_patient_preds = [] # initialize the patient predictions
id_patient_labels = [] # initialize the patient labels
patient_ids = test_dataset.df['patient_id'].to_numpy() # get the patient ids
correct = 0 # initialize the correct count
incorrect_ids = [] # initialize the incorrect ids

for id in np.unique(patient_ids): # iterate over the unique patient ids
    indices = np.where(patient_ids==id)[0] # get the indices of the patient id
    id_patient_logits.append(np.mean(logits[indices,:], axis=0)) # get the mean of the logits
    id_patient_labels.append(np.mean(labels[indices], axis=0)) # get the mean of the labels
    id_patient_preds.append(id_patient_logits[-1].argmax()) # get the predicted class
    if id in rechecked_patient_ids: # if the patient id is in the rechecked patient ids
        print(f"Patient Id: {id}, Ground Truth: {id_patient_labels[-1]}, Predicted: {id_patient_preds[-1]}")
    if id_patient_labels[-1] != id_patient_preds[-1]: # if the label is not equal to the prediction
        incorrect_ids.append((id, id_patient_labels[-1], id_patient_preds[-1])) # append the incorrect ids
    correct += int(id_patient_preds[-1]==id_patient_labels[-1]) # update the correct count
    id_patients.append(id) # append the patient id
    
for id_data in incorrect_ids: # iterate over the incorrect ids and print their information
    print(f"Patient Id: {id_data[0]}, Ground Truth: {id_data[1]}, Predicted: {id_data[2]}")

preds = np.array(id_patient_preds) # convert the predictions to numpy array
labels = np.array(id_patient_labels) # convert the labels to numpy array
logits = np.array(id_patient_logits) # convert the logits to numpy array

# Calculate patient-level metrics
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average="binary")
recall = recall_score(labels, preds, average="binary")
f1 = f1_score(labels, preds, average="binary")
auc = roc_auc_score(labels, logits[:, 1])

# Save patient-level metrics to CSV
save_metrics_csv(args.fold, accuracy, precision, recall, f1, auc, os.path.join(exp_dir, "metrics_patient.csv"))

# Confusion matrix for patient-level
confmat_vals = confusion_matrix(labels, preds)
plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, "patient_level_conf_mat.png"), "Confusion Matrix on Test [Patient level]")

print(f'[TTA] Patient level Accuracy: {accuracy} AUC: {auc}')
