import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights

from torch import nn, optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from MergeMasterDataset import MergeMasterDataset
import os

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def train_model(config):
    """
    Train a model using a specified CSV file to create datasets.
    
    Parameters:
    csv_file (str): Path to the CSV file used to create the datasets.
    num_epochs (int): Number of epochs for training.
    batch_size (int): Batch size for DataLoader.
    learning_rate (float): Learning rate for optimizer.
    weight_decay (float): Weight decay for optimizer.
    momentum (float): Momentum for optimizer.
    
    Returns:
    all_fold_metrics (list): List of dictionaries containing metrics for each fold.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training configurations
    csv_file = config['data']['csv_file']
    
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    image_size = config['model']['image_size']

    # TODO
    use_neutrophil_images = config['model']['use_neutrophil_images']


    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_folds = 5
    all_fold_metrics = []  # Stores metrics for each fold

    for fold in tqdm(range(num_folds), desc="Folds"):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        train_dataset = MergeMasterDataset(csv_file, fold=fold, train=True, transform=transform)
        val_dataset = MergeMasterDataset(csv_file, fold=fold, train=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        # Stores metrics for each epoch
        epoch_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_time': []
        }

        # Epoch loop
        for epoch in tqdm(range(num_epochs), desc=f"Epochs (Fold {fold+1})"):
            start_time = time.time()
            
            # Training phase
            train_loss, train_correct, train_total = train_one_epoch(model, device, train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_correct, val_total = validate(model, device, val_loader, criterion)

            end_time = time.time()
            
            # Calculating metrics
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            epoch_time = end_time - start_time
            
            # Append epoch metrics
            epoch_metrics['train_loss'].append(epoch_train_loss)
            epoch_metrics['val_loss'].append(epoch_val_loss)
            epoch_metrics['train_accuracy'].append(epoch_train_acc)
            epoch_metrics['val_accuracy'].append(epoch_val_acc)
            epoch_metrics['epoch_time'].append(epoch_time)

            plot_metrics(epoch_metrics, fold, epoch)


            tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        all_fold_metrics.append(epoch_metrics)
        
        # Plotting the metrics after each fold
        # plot_metrics(epoch_metrics)

    return all_fold_metrics

def train_one_epoch(model, device, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients for the next train
        outputs = model(inputs)  # Forward pass: compute the output class given a image
        loss = criterion(outputs, labels)  # Calculate loss: difference between the predicted value and the actual label
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)  # Get the predicted classes
        total_correct += (predicted == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)
    
    return total_loss, total_correct, total_samples

def validate(model, device, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():  # For validation, we don't need to compute gradients (for memory efficiency)
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    return total_loss, total_correct, total_samples

def plot_metrics(metrics, fold, epoch, save_dir="plots"):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Plot training and validation loss
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accuracy'], label='Train Accuracy')
    plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Construct the file path for the plot
    plot_filename = f"fold_{fold}_epoch_{epoch+1}.png"  # epoch+1 because epoch starts at 0
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"Plot saved: {plot_path}")
    plt.close()  # Close the plot to free up memory