# import libraries
import numpy as np
import pandas as pd
import os, random, sys, argparse, torchvision, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler, ConcatDataset
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve, confusion_matrix
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import yaml

# sys.path.append('/home/tchowdhury/data/code/CMML-v2/townim')
sys.path.append('../townim') # running python main.py from the directory the file is located in
import utils
from utils import FixedRotation
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH, MONOCYTE_NEW_NORMALS_CSV_PATH

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

# argument parser with argument for fold
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default="monocyte_new_normals", help="image type to train the model on")
args = parser.parse_args()
data_type = str(args.data_type)
final_epochs = 0
if data_type == 'neutrophil':
    CSV_PATH = NEUTROPHIL_CSV_PATH
elif data_type == 'monocyte':
    CSV_PATH = MONOCYTE_CSV_PATH
elif data_type == 'monocyte_new_normals':
    CSV_PATH = MONOCYTE_NEW_NORMALS_CSV_PATH
else:
    raise ValueError("Invalid data type")

with open(f'./experiments/{data_type}/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

mlflow.set_experiment(data_type)

root = f"./experiments/{data_type}/train"
os.makedirs(root, exist_ok=True) # output directory (main level)

# Create data loaders
num_epochs = config['num_epochs']
batch_size = config['batch_size']
IMAGE_SIZE = config['image_size']
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225] # Std of ImageNet dataset (used for normalization)
use_scheduler = config['use_scheduler']
lr = config['lr']
weight_decay = config['weight_decay']
momentum = config['momentum']
start_factor = config['start_factor']
end_factor = config['end_factor']
patience = config['patience']  # Number of epochs to wait for improvement (100 for no early stopping)

# Define the transformations for training and testing
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

# Define the transformations for test time augmentation
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

for set_id in range(5):
    print(f"Fold: {set_id}, Data type: {data_type}")
    
    best_test_acc = 0
    best_epoch = 0
    train_losses = []  # To store training losses
    val_losses = []    # To store validation losses
    train_accuracies = []  # To store training accuracies
    val_accuracies = []    # To store validation accuracies
    logs = ''

    # define output directories using relative paths
    output_dir = root + f'/{data_type}_fold_{set_id}_with_TTA'
    model_dir = output_dir + "/model"
    figure_dir = output_dir + "/figures"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
        
    shutil.copyfile('./main_and_test.py', os.path.join(output_dir, 'main.py')) # copying code file used to train the model
    utils.set_random_seed(123)

    # Create the test dataset and data loaders
    test_dataset = CustomDataset('val', CSV_PATH, set_id, transform=test_transform)
    test_loaders = [
        DataLoader(CustomDataset('val', CSV_PATH, set_id, transform=transform), batch_size=8, shuffle=False, pin_memory=True, num_workers=4)
        for transform in TTAs
    ]

    # create the train and validation datasets
    train_dataset = CustomDataset('train', CSV_PATH, set_id, transform=train_transform)
    print("Training dataset stats [Normal, CMML]:", train_dataset.disease_count)
    val_dataset = CustomDataset('val', CSV_PATH, set_id, transform=test_transform)
    print("Test dataset stats [Normal, CMML]:", val_dataset.disease_count)

    # For unbalanced dataset we create a weighted sampler to balance the classes                  
    weights = utils.make_weights_for_balanced_classes(train_dataset.labels, device)
    weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))
    # create train and validation data loaders (trainloader using the weighted sampler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                            sampler=weighted_sampler,
                            num_workers=8, worker_init_fn=utils.worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    # Define the model architecture (ResNet-50 as an example)
    num_classes = len(set(train_dataset.labels)) # number of classes (2)
    model = models.resnet50(weights='IMAGENET1K_V1')
    hidden_layer_size = 512
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, num_classes)
    )

    model = model.to(device) # move the model to the device

    # Define loss function and optimizer
    count = torch.bincount(torch.tensor(train_dataset.labels)).to(device)
    class_weight = len(train_dataset.labels) / count

    print('Loss class weight:', class_weight)
    # define the SGD optimizer and the learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=num_epochs)

    # CSV file to store metrics
    metrics_data = []

    # Early stopping variables
    best_val_loss = float('inf')
    best_test_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    early_stop = False

    with mlflow.start_run(run_name=f"fold_{set_id}"):
        mlflow.log_artifact('config.yaml')
        mlflow.log_params(config)
        mlflow.set_tag("train_normal_count", train_dataset.disease_count[0])
        mlflow.set_tag("train_cmml_count", train_dataset.disease_count[1])
        mlflow.set_tag("val_normal_count", val_dataset.disease_count[0])
        mlflow.set_tag("val_cmml_count", val_dataset.disease_count[1])

        for epoch in range(num_epochs): # for each epoch
            if early_stop: # if early stopping is true, break
                break

            # Training loop
            t = tqdm(enumerate(train_loader, 0), total=len(train_loader),
                    smoothing=0.9, position=0, leave=True,
                    desc="Train: Epoch: " + str(epoch + 1) + "/" + str(num_epochs)) # progress bar for train loader
            model.train() # set the model to training mode
            running_loss = 0.0 # initialize running loss
            all_preds, all_labels = [], [] # initialize lists to store predictions and labels

            for i, (inputs, labels) in t: # for each batch
                inputs = inputs.to(device).float() # move inputs to device
                labels = labels.to(device).long() # move labels to device
                optimizer.zero_grad() # zero the gradients
                outputs = model(inputs) # forward pass
                loss = F.cross_entropy(outputs, labels) # calculate the loss
                loss.backward() # backward pass
                optimizer.step() # update the weights

                running_loss += loss.item() # update the running loss
                outputs = F.softmax(outputs, dim=-1) # get the probabilities
                preds = outputs.argmax(dim=1).cpu().numpy() # get the predictions
                all_preds.extend(preds) # append the predictions to the list
                all_labels.extend(labels.cpu().numpy()) # append the labels to the list

            # Calculate metrics
            train_loss = running_loss / len(train_loader)
            train_accuracy = accuracy_score(all_labels, all_preds)
            train_precision = precision_score(all_labels, all_preds, average='binary')
            train_recall = recall_score(all_labels, all_preds, average='binary')
            train_f1 = f1_score(all_labels, all_preds, average='binary')
            train_auroc = roc_auc_score(all_labels, all_preds)

            # Validation loop
            model.eval() # set the model to evaluation mode
            val_loss = 0.0 # initialize validation loss
            val_preds, val_labels, val_probs = [], [], [] # initialize lists to store predictions, labels and probabilities

            with torch.no_grad(): # no gradients
                
                t = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                        smoothing=0.9, position=0, leave=True,
                        desc="Val: Epoch: " + str(epoch + 1) + "/" + str(num_epochs)) # progress bar for validation loader
                
                for i, (inputs, labels) in t: # for each batch
                    inputs, labels = inputs.to(device).float(), labels.to(device).long() # move inputs and labels to device
                    outputs = model(inputs) # forward pass
                    loss = F.cross_entropy(outputs, labels) # calculate the loss
                    val_loss += loss.item() # update the validation loss
                    outputs = F.softmax(outputs, dim=-1) # get the probabilities
                    preds = outputs.argmax(dim=1).cpu().numpy() # get the predictions
                    probs = outputs[:, 1].cpu().numpy()  # probabilities for class 1
                    val_preds.extend(preds) # append the predictions to the list
                    val_labels.extend(labels.cpu().numpy()) # append the labels to the list
                    val_probs.extend(probs) # append the probabilities to the list

            # Calculate metrics
            val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, average='binary')
            val_recall = recall_score(val_labels, val_preds, average='binary')
            val_f1 = f1_score(val_labels, val_preds, average='binary')
            val_auroc = roc_auc_score(val_labels, val_preds)

            if use_scheduler: # if using scheduler, step the scheduler
                scheduler.step()
                
            print(f"Epoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}")
            
            # Store metrics in a CSV file
            metrics_data.append([epoch+1, round(optimizer.param_groups[0]['lr'], 4), train_loss, round(train_accuracy, 4), round(train_precision, 4), round(train_recall, 4), 
                                round(train_f1, 4), round(train_auroc, 4), val_loss, round(val_accuracy, 4), round(val_precision, 4), 
                                round(val_recall, 4), round(val_f1, 4), round(val_auroc, 4)])

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "train_auroc": train_auroc,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_auroc": val_auroc,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)

            # Early stopping check
            if val_loss < best_val_loss: # if current loss is less than best loss
                best_val_loss = val_loss # update best loss
                best_epoch = epoch + 1 # update best epoch
                epochs_no_improve = 0 # set early stopping epochs to 0
            else:
                epochs_no_improve += 1 # increment early stopping epochs
                if epochs_no_improve == patience: # if early stopping epochs is equal to the patience
                    early_stop = True # early stop
                    print("Early Stopping....")
                    break  # Stop training

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)

        torch.save(model.state_dict(), os.path.join(model_dir, f'{data_type}_fold_{set_id}.pth'))
        final_epochs += best_epoch
    
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
        fig.savefig(os.path.join(figure_dir, "conf_mat.png"))
        plt.close()

        # Save the metrics and the last model
        df = pd.DataFrame(metrics_data, columns=['Epoch', 'LR', 'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUROC', 
                                                    'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUROC'])
        df.to_csv(os.path.join(output_dir, 'train_time_metrics.csv'), index=False)

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

        mlflow.log_artifact(os.path.join(model_dir, f'{data_type}_fold_{set_id}.pth'))
        mlflow.log_artifact(os.path.join(figure_dir, "conf_mat.png"))
        mlflow.log_artifact(os.path.join(figure_dir, "roc_curve.png"))
        mlflow.log_artifact(os.path.join(figure_dir, "loss_accuracy_graph.png"))
        mlflow.log_artifact(os.path.join(output_dir, 'train_time_metrics.csv'))

# combine train and validation datasets for final training
trainval_dataset = ConcatDataset([
    CustomDataset('train', CSV_PATH, set_id, transform=train_transform),
    CustomDataset('val', CSV_PATH, set_id, transform=train_transform)
])
all_labels = train_dataset.labels + val_dataset.labels # combine labels
weights = utils.make_weights_for_balanced_classes(all_labels, device) # get new weights for loss
weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights)) # create sampler
trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                                sampler=weighted_sampler, num_workers=8, worker_init_fn=utils.worker_init_fn) # create data laoder

final_model = models.resnet50(weights='IMAGENET1K_V1') # define final model
num_ftrs = final_model.fc.in_features
final_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, num_classes)
)
final_model = final_model.to(device)
final_model_dir = root + "/model"
final_optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
final_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=num_epochs)
final_model.train()
final_epochs = final_epochs // 5
print(f"Final training for {final_epochs} epochs")

with mlflow.start_run(run_name="final_model"):
    params = {
        "learning_rate": lr,
        "batch_size": batch_size,
        "num_epochs": final_epochs,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "image_size": IMAGE_SIZE,
        "use_scheduler": use_scheduler,
        "start_factor": start_factor,
        "end_factor": end_factor,
    }
    mlflow.log_params(params)

    for epoch in range(final_epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for i, (inputs, labels) in enumerate(trainval_loader):
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            final_optimizer.step()
            running_loss += loss.item()
            outputs = F.softmax(outputs, dim=-1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
        if use_scheduler:
            final_scheduler.step()
        epoch_loss = running_loss / len(trainval_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        print(f"Final Training - Epoch {epoch+1}/{best_epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        mlflow.log_metrics({
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc
        }, step=epoch)

    # Save final model trained on train+val
    torch.save(final_model.state_dict(), os.path.join(final_model_dir, 'final.pth'))
    mlflow.log_artifact(os.path.join(final_model_dir, 'final.pth'))

print("Training complete!")