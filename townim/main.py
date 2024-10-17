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
parser.add_argument('--fold', type=int, default=4, help='fold_id')
args = parser.parse_args()

set_id = int(args.fold)

# Training loop
data_type = 'monocyte' # neutrophil, monocyte
is_tta = True
num_epochs = 50 if data_type == 'neutrophil' else 100
best_test_acc = 0
best_epoch = 0
train_losses = []  # To store training losses
val_losses = []    # To store validation losses
train_accuracies = []  # To store training accuracies
val_accuracies = []    # To store validation accuracies
logs = ''
CSV_PATH = NEUTROPHIL_CSV_PATH if data_type == 'neutrophil' else MONOCYTE_CSV_PATH


# output_dir = f'./models/{data_type}_fold_{args.fold}'
os.makedirs(f"./experiments/{data_type}")
output_dir = f'./experiments/{data_type}/{data_type}_fold_{args.fold}'
output_dir += '_with_TTA' if is_tta else '_without_TTA'
model_dir = output_dir + "/model"
figure_dir = output_dir + "/figures/train"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)
    
shutil.copyfile('./main.py', os.path.join(output_dir, 'main.py')) # copying code file used to train the model
utils.set_random_seed(123)

# Create data loaders
batch_size = 32
IMAGE_SIZE = 352
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)
# IMAGENET_MEAN = [0.5, 0.5, 0.5]
# IMAGENET_STD = [0.5, 0.5, 0.5]

train_transform = T.Compose([
    # T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(90),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

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
    DataLoader(CustomDataset('test', CSV_PATH, set_id, transform=transform), batch_size=8, shuffle=False, pin_memory=True, num_workers=4)
    for transform in TTAs
]

# dataset
train_dataset = CustomDataset('train', CSV_PATH, set_id, transform=train_transform)
print("Training dataset stats [Normal, CMML]:", train_dataset.disease_count)
val_dataset = CustomDataset('test', CSV_PATH, set_id, transform=test_transform)
print("Test dataset stats [Normal, CMML]:", val_dataset.disease_count)

# For unbalanced dataset we create a weighted sampler                       
weights = utils.make_weights_for_balanced_classes(train_dataset.labels, device)
weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                        sampler=weighted_sampler,
                        num_workers=8, worker_init_fn=utils.worker_init_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
print("Dataset loaded")

# Define the model architecture (ResNet-50 as an example)
num_classes = len(set(train_dataset.labels))
print("Model: Resnet50")
model = models.resnet50(weights='IMAGENET1K_V1')
hidden_layer_size = 512
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, num_classes)
)

model = model.to(device)

# Define loss function and optimizer
count = torch.bincount(torch.tensor(train_dataset.labels)).to(device)
class_weight = len(train_dataset.labels) / count

print('Loss class weight:', class_weight)
class_weight = None
optimizer = optim.SGD(model.parameters(), lr=5e-5, weight_decay=1e-4, momentum=0.9)

# CSV file to store metrics
metrics_data = []

# Early stopping variables
patience = 20  # Number of epochs to wait for improvement
best_val_loss = float('inf')
best_test_acc = 0.0
best_epoch = 0
epochs_no_improve = 0
early_stop = False

for epoch in range(num_epochs):
    if early_stop:
        break

    # Training loop
    t = tqdm(enumerate(train_loader, 0), total=len(train_loader),
            smoothing=0.9, position=0, leave=True,
            desc="Train: Epoch: " + str(epoch + 1) + "/" + str(num_epochs))
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for i, (inputs, labels) in t:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
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

    with torch.no_grad():
        t = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                smoothing=0.9, position=0, leave=True,
                desc="Val: Epoch: " + str(epoch + 1) + "/" + str(num_epochs))
        for i, (inputs, labels) in t:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
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

    print(f"Epoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}")
    # Store metrics in a CSV file
    metrics_data.append([epoch+1, train_loss, train_accuracy, train_precision, train_recall, train_f1, train_auroc,
                        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auroc])

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