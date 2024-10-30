import numpy as np
import pandas as pd
import os, random, sys, argparse, torchvision, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve, confusion_matrix
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# sys.path.append('/home/tchowdhury/data/code/CMML-v2/townim')
sys.path.append('../townim') # running python main.py from the directory the file is located in
import utils
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH

from MergeMasterDataset import MergeMasterDataset # import the MergeMasterDataset class
from MultimodalClassifier import MultimodalClassifier # import the MultimodalClassifier class

# creating a cache directory since running docker using specific user doesn't allow to use the home cache directory
cache_dir = "../cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir # set cache directory

if torch.cuda.is_available(): # if cuda is available
    torch.cuda.empty_cache() # empty the cache
    device = "cuda" # set the device to cuda
elif torch.backends.mps.is_available(): # if mps is available
    torch.mps.empty_cache() # empty the cache
    device = "mps" # set the device to mps
else: # otherwise
    device = "cpu" # set the device to cpu
print(f"Using device: {device}") # print the device being used

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=2, help='fold_id')
args = parser.parse_args()

set_id = int(args.fold)

# Training loop
data_type = 'monocyte_clinical' # neutrophil_clinical, monocyte_clinical
num_epochs = 50 if data_type == "neutrophil_clincial" else 100
best_test_acc = 0
best_epoch = 0
train_losses = []  # To store training losses
val_losses = []    # To store validation losses
train_accuracies = []  # To store training accuracies
val_accuracies = []    # To store validation accuracies
logs = ''
num_patient_features = 10
CSV_PATH = NEUTROPHIL_CSV_PATH if data_type == 'neutrophil_clinical' else MONOCYTE_CSV_PATH

# output_dir = f'./models/{data_type}_fold_{args.fold}'
os.makedirs(f"./experiments/{data_type}", exist_ok=True)
output_dir = f'./experiments/{data_type}/{data_type}_fold_{args.fold}_with_TTA'
model_dir = output_dir + "/model"
figure_dir = output_dir + "/figures/train"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)
    
shutil.copyfile('./main_and_test_clinical.py', os.path.join(output_dir, 'main.py')) # copying code file used to train the model
utils.set_random_seed(123)

# Create data loaders
batch_size = 32
IMAGE_SIZE = 352
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)
# IMAGENET_MEAN = [0.5, 0.5, 0.5]
# IMAGENET_STD = [0.5, 0.5, 0.5]

class FixedRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return T.functional.rotate(x, self.angle)

def preprocess_patient_data(batch, device: str) -> torch.Tensor:
    '''
    Function: Preprocessing function for patient data
    Parameters:
        batch: The batch of data
        device (str): The device to use
    Returns: torch.Tensor
    '''
    # List of required keys for patient data
    patient_keys = ['Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count', 'Neutrophil count', 'Monocyte count', 'Platelet count', 'Blast percentage (PB)', 'LDH']
    # Create a list of tensors, each of shape [batch_size, 1], then concatenate them along dim=1
    patient_data = [batch[key].unsqueeze(1) for key in patient_keys] # for each key in patient_keys, get the value from the batch and unsqueeze it along dim=1
    patient_data = torch.cat(patient_data, dim=1).float().to(device) # concatenate the list of tensors along dim=1, convert to float, and move to the device
    return patient_data # return the patient data

transform = T.Compose([
        # transforms.Resize((image_size, image_size)),
        T.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = MergeMasterDataset(CSV_PATH, fold=set_id, train=True, use_patient_data=True, transform=transform)
val_dataset = MergeMasterDataset(CSV_PATH, fold=set_id, train=False, use_patient_data=True, transform=transform)
# For unbalanced dataset we create a weighted sampler                       

count = torch.bincount(torch.tensor(list(train_dataset.frame["morphology"]))).to(device)
class_weight = len(list(train_dataset.frame["morphology"])) / count
weights = utils.make_weights_for_balanced_classes(list(train_dataset.frame["morphology"]), device)

# weights = utils.make_weights_for_balanced_classes(train_dataset.frame['morphology'], device)
weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=weighted_sampler) # create a DataLoader for the training dataset
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False) # create a DataLoader for the validation dataset

# Define the model architecture (ResNet-50 as an example)
num_classes = 2
print("Model: Resnet50")
model = MultimodalClassifier(num_patient_features) # load the MultimodalClassifier model
model = model.to(device)

use_scheduler = True
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.5)
end_factor = 1e-5/1e-3
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=end_factor, total_iters=num_epochs)

# CSV file to store metrics
metrics_data = []

# Early stopping variables
patience = 100  # Number of epochs to wait for improvement
best_val_loss = float('inf')
best_test_acc = 0.0
best_epoch = 0
epochs_no_improve = 0
early_stop = False

for epoch in range(num_epochs):
    if early_stop:
        break

    # Training loop
    t = tqdm(enumerate(train_loader), total=len(train_loader),
            smoothing=0.9, position=0, leave=True,
            desc="Train: Epoch: " + str(epoch + 1) + "/" + str(num_epochs))
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for i, batch in t:
        inputs = batch['image'].to(device).float()
        labels = batch['morphology'].to(device).long()
        patient_data = preprocess_patient_data(batch, device)
        optimizer.zero_grad()
        outputs = model(inputs, patient_data)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        outputs = F.softmax(outputs, dim=-1)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds, average='binary')
    train_recall = recall_score(all_labels, all_preds, average='binary')
    train_f1 = f1_score(all_labels, all_preds, average='binary')
    train_auroc = roc_auc_score(all_labels, all_preds)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_preds, val_labels, val_probs = [], [], []
    
    num_augmentations = 5 # Number of test-time augmentations
    # Test Time Augmentations (TTA)
    augmentations = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.RandomResizedCrop(224, scale=(0.8, 1.0))
    ])

    with torch.no_grad():
        t = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                smoothing=0.9, position=0, leave=True,
                desc="Val: Epoch: " + str(epoch + 1) + "/" + str(num_epochs))
        for i, batch in t:
            inputs = batch['image'].to(device).float()
            labels = batch['morphology'].to(device).long()
            outputs_list = [] # initialize a list to store the outputs
            for _ in range(num_augmentations): # for each augmentation
                augmented_images = augmentations(inputs)# apply TTA to the images
                patient_data = preprocess_patient_data(batch, device) # preprocess the patient data
                outputs = model(augmented_images, patient_data) # get the outputs from the model
                outputs_list.append(outputs) # append the outputs to the list
            outputs = torch.stack(outputs_list).mean(0) # calculate the mean of the outputs
            loss = F.cross_entropy(outputs, labels)
            val_loss += loss.item()
            outputs = F.softmax(outputs, dim=-1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = outputs[:, 1].cpu().numpy()  # Probabilities for class 1
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())
            val_probs.extend(probs)

    val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision = precision_score(val_labels, val_preds, average='binary')
    val_recall = recall_score(val_labels, val_preds, average='binary')
    val_f1 = f1_score(val_labels, val_preds, average='binary')
    val_auroc = roc_auc_score(val_labels, val_preds)
    
    if use_scheduler:
        scheduler.step()

    print(f"Epoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}")
    # Store metrics in a CSV file
    metrics_data.append([epoch+1, train_loss, round(train_accuracy, 4), round(train_precision, 4), round(train_recall, 4), 
                        round(train_f1, 4), round(train_auroc, 4), val_loss, round(val_accuracy, 4), round(val_precision, 4), 
                        round(val_recall, 4), round(val_f1, 4), round(val_auroc, 4)])

    df = pd.DataFrame(metrics_data, columns=['Epoch', 'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUROC',
                                            'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUROC'])
    df.to_csv(os.path.join(output_dir, 'train_time_metrics.csv'), index=False)

    # Early stopping check
    if val_loss < best_val_loss: # if current loss is less than best loss
        best_val_loss = val_loss # update best loss
        epochs_no_improve = 0 # set early stopping epochs to 0
        torch.save(model.state_dict(), os.path.join(model_dir, 'best.pth')) # save the best model
    else:
        epochs_no_improve += 1 # increment early stopping epochs
        if epochs_no_improve == patience: # if early stopping epochs is equal to the patience
            early_stop = True # early stop
            break  # Stop training

    # Save confusion matrix and model when validation accuracy improves
    if best_test_acc <= val_accuracy and epoch != 0:
        best_epoch = epoch + 1
        best_test_acc = val_accuracy

        # Confusion matrix
        conf_matrix = confusion_matrix(val_labels, val_preds, normalize='true')
        fig, ax = plt.subplots()
        im = ax.imshow(conf_matrix)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')

        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, np.around(conf_matrix[i, j], 2), ha="center", va="center", color="black", fontsize=12)

        ax.set_title("Confusion Matrix on Validation Set (Best Accuracy)")
        fig.savefig(os.path.join(figure_dir, "conf_mat_best.png"))
        plt.close()

torch.save(model.state_dict(), os.path.join(model_dir, 'last.pth')) # save the last model

# Plot ROC curve after training
fpr, tpr, _ = roc_curve(val_labels, val_probs, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set xticks and yticks with a step size of 0.1
plt.xticks(np.arange(0.0, 1.1, 0.1))
plt.yticks(np.arange(0.0, 1.1, 0.1))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(figure_dir, 'roc_curve.png'))
plt.close()

# Save the loss and accuracy graphs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
plt.plot(df['Epoch'], df['Val Loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy')
plt.plot(df['Epoch'], df['Val Accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validation Accuracy')

plt.savefig(os.path.join(figure_dir, 'loss_accuracy_graph.png'))
plt.close()

# Function to save or update metrics CSV
def save_metrics_csv(fold, accuracy, precision, recall, f1, auroc, metrics_path):
    new_metrics = pd.DataFrame([[fold, round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4), round(auroc, 4)]], 
                                columns=["fold", "accuracy", "precision", "recall", "f1", "auroc"])

    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        if fold in df['fold'].values:
            df.loc[df['fold'] == fold] = new_metrics
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

# Create data loaders
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

test_dataset = MergeMasterDataset(CSV_PATH, fold=set_id, train=False, use_patient_data=True, transform=None)
test_loaders = [
    DataLoader(MergeMasterDataset(CSV_PATH, fold=set_id, train=False, use_patient_data=True, transform=transform), batch_size=batch_size, num_workers=8, shuffle=False) 
    for transform in TTAs
    ]# create a DataLoader for the validation dataset

# model_dir = f'/home/tchowdhury/data/code/CMML-v2/townim/models/{data_type}_fold_{args.fold}_without_TTA'
exp_dir = f'./experiments/{data_type}'
exp_subdir = exp_dir + f'/{data_type}_fold_{args.fold}_with_TTA'
figure_dir = exp_subdir + "/figures/test"
os.makedirs(figure_dir, exist_ok=True)
model.eval()

with torch.no_grad():
    correct = 0
    preds = []
    labels = []
    logits = []
    # for i, (inputs, targets) in tqdm(enumerate(test_loaders[0]), total=len(test_loaders[0]), smoothing=0.9, position=0, leave=True,):
    for i, batch in enumerate(test_loaders[0]):
        inputs = batch['image'].to(device).float()
        targets = batch['morphology'].to(device).long()
        patient_data = preprocess_patient_data(batch, device)
        outputs = model(inputs, patient_data)
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

    save_metrics_csv(args.fold, accuracy, precision, recall, f1, auc, os.path.join(exp_dir, "metrics_image.csv"))

    # Confusion matrix
    confmat_vals = confusion_matrix(labels, preds)
    plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, "image_level_conf_mat.png"), "Confusion Matrix on Test [Image level]")

    # ROC curve
    plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, "image_level_roc_curve.png"))
    
    print(f'[W/O TTA] Image level Accuracy: {accuracy} AUC: {auc}')
    
    # preds = []
    # logits = []
    # labels = []
    # correct = 0
    # # for i, data in tqdm(enumerate(zip(*test_loaders)), total=len(test_loaders[0]), smoothing=0.9, position=0, leave=True,):
    # for i, batch in enumerate(zip(*test_loaders)):
    #     inputs = batch['image'].to(device).float()
    #     targets = batch['morphology'].to(device).long()
    #     patient_data = preprocess_patient_data(batch, device)
    #     outputs = model(inputs, patient_data)
    #     outputs = F.softmax(outputs, dim=-1)
    #     outputs = outputs.reshape(len(batch), int(inputs.shape[0]/len(batch)), -1).mean(dim=0)

    #     _, predicted = torch.max(outputs, 1)
    #     correct += (predicted == labels).sum().item()
    #     preds.append(predicted.detach().cpu().numpy())
    #     labels.append(targets.detach().cpu().numpy())
    #     logits.append(outputs.detach().cpu().numpy().astype(np.float32))

    # preds = np.concatenate(preds, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # logits = np.concatenate(logits, axis=0)

    # # Calculate metrics with TTA
    # accuracy = accuracy_score(labels, preds)
    # precision = precision_score(labels, preds, average="binary")
    # recall = recall_score(labels, preds, average="binary")
    # f1 = f1_score(labels, preds, average="binary")
    # auc = roc_auc_score(labels, logits[:, 1])

    # # Save metrics to CSV
    # save_metrics_csv(args.fold, accuracy, precision, recall, f1, auc, os.path.join(exp_dir, "metrics_image_tta.csv"))

    # # Confusion matrix
    # confmat_vals = confusion_matrix(labels, preds)
    # plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, "tta_image_level_conf_mat.png"), "Confusion Matrix on Test [Image level with TTA]")

    # # ROC curve
    # plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, "tta_image_level_roc_curve.png"))
    # print(f'[TTA] Image level Accuracy: {accuracy} AUC: {auc}')
    
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
patient_ids = test_dataset.frame['patient_id'].to_numpy()
correct = 0
# confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true').to(device)
for id in np.unique(patient_ids):
    indices = np.where(patient_ids==id)[0]
    id_patient_logits.append(np.mean(logits[indices,:], axis=0))
    id_patient_labels.append(np.mean(labels[indices], axis=0))
    id_patient_preds.append(id_patient_logits[-1].argmax())
    correct += int(id_patient_preds[-1]==id_patient_labels[-1])
    # print(id_patient_labels[-1], id_patient_logits[-1])
    id_patients.append(id)
    
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
save_metrics_csv(args.fold, accuracy, precision, recall, f1, auc, os.path.join(exp_dir, "metrics_patient.csv"))

# Confusion matrix for patient-level
confmat_vals = confusion_matrix(labels, preds)
plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, "patient_level_conf_mat.png"), "Confusion Matrix on Test [Patient level]")

print(f'[W/O TTA] Patient level Accuracy: {accuracy} AUC: {auc}')