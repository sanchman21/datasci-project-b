import torch
import numpy as np
import pandas as pd
import os
import datetime
import utils
import time


from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from MergeMasterDataset import MergeMasterDataset
from MultimodalClassifier import MultimodalClassifier

import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_patient_data(batch, device):
    patient_keys = ['Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count', 'Neutrophil count', 'Monocyte count', 'Platelet count', 'Blast percentage (PB)', 'LDH']
    # Create a list of tensors, each of shape [batch_size, 1], then concatenate them along dim=1
    patient_data = [batch[key].unsqueeze(1) for key in patient_keys]
    patient_data = torch.cat(patient_data, dim=1).float().to(device)
    return patient_data


#TODO: untested code here
def load_model(model_path, model_type='resnet50', num_patient_features=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_type == 'MultimodalClassifier':
        model = MultimodalClassifier(num_patient_features)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # timestamp for the reference of creating folders
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    config_details = f"{config['model']['name']}_neutrophils{config['model']['use_neutrophil_images']}"
    base_dir = os.path.join('saved_models', f"{timestamp}_{config_details}")
    os.makedirs(base_dir, exist_ok=True)

    # Create subdirectories for different types of data
    logs_dir = os.path.join(base_dir, 'logs')
    confusion_matrices_dir = os.path.join(base_dir, 'confusion_matrices')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(confusion_matrices_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load training configurations
    csv_file = config['data']['csv_file']
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    learning_rate_end = config['training']['learning_rate_end']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']

    image_size = config['model']['image_size']
    model_name = config['model']['name']
    use_neutrophil_images = config['model']['use_neutrophil_images']
    freeze_backbone = config['model']['freeze_backbone']
    use_scheduler = config['model']['use_scheduler']

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomRotation(90),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_folds = 5
    num_patient_features = 10

    all_metrics = []
    for fold in range(num_folds):
        writer = SummaryWriter(log_dir=os.path.join(logs_dir, f"fold_{fold}"))

        print(f"Training fold {fold+1}/{num_folds}")

        #load model by model name
        if model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, 2)

            if freeze_backbone:
                for name, param in model.named_parameters():
                    if 'fc' not in name:
                        param.requires_grad = False
                    else:
                        print(f"{name} is not frozen.")

        elif model_name == 'MultimodalClassifier':
            model = MultimodalClassifier(num_patient_features)
            
            if freeze_backbone:
                for name, param in model.named_parameters():
                        if 'resnet' in name:
                            param.requires_grad = False
                        else:
                            print(f"{name} is not frozen.")
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        


        
        model.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        if use_scheduler:
            end_factor = learning_rate_end / learning_rate
            scheduler = LinearLR(optimizer, start_factor=1, end_factor=end_factor, total_iters=num_epochs)

        scaler = GradScaler()

        # Dataset and DataLoader setup
        train_dataset = MergeMasterDataset(csv_file, fold=fold, train=True, use_neutrophil_images= use_neutrophil_images, transform=transform)
        val_dataset = MergeMasterDataset(csv_file, fold=fold, train=False, use_neutrophil_images= use_neutrophil_images, transform=transform)

        # load the patient data if choose to use multimodal
        if model_name == 'MultimodalClassifier':
            train_dataset = MergeMasterDataset(csv_file, fold=fold, train=True, use_patient_data= True, use_neutrophil_images= use_neutrophil_images, transform=transform)
            val_dataset = MergeMasterDataset(csv_file, fold=fold, train=False, use_patient_data= True, use_neutrophil_images= use_neutrophil_images, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

        # for the calculation of confusion matrix
        all_labels = []
        all_preds = []

        # Stores metrics for each epoch
        epoch_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_time': []
        }

        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            total_loss, total_correct, total_samples = 0, 0, 0

            for batch in train_loader:
                images = batch['image'].to(device)
                labels = batch['morphology'].to(device)
                
                with autocast():
                    if model_name == 'MultimodalClassifier':
                        patient_data = preprocess_patient_data(batch, device)
                        outputs = model(images, patient_data)
                    else:
                        outputs = model(images)

                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.float(), 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            if use_scheduler:
                scheduler.step()
                
            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples

            writer.add_scalar(f'Train/Loss_fold_{fold}', train_loss, epoch)
            writer.add_scalar(f'Train/Accuracy_fold_{fold}', train_acc, epoch)
    

            # val
            val_loss, val_correct, val_total = 0, 0, 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['morphology'].to(device)
                    
                    if model_name == 'MultimodalClassifier':
                        patient_data = preprocess_patient_data(batch, device)
                        outputs = model(images, patient_data)
                    else:
                        outputs = model(images)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.float(), 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

                    all_labels.extend(labels.tolist())
                    all_preds.extend(predicted.tolist())

            val_acc = val_correct / val_total
            val_loss = val_loss / val_total

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            writer.add_scalar(f'Validation/Loss_fold_{fold}', val_loss, epoch)
            writer.add_scalar(f'Validation/Accuracy_fold_{fold}', val_acc, epoch)

            # Append epoch metrics for plot
            epoch_metrics['train_loss'].append(train_loss)
            epoch_metrics['val_loss'].append(val_loss)
            epoch_metrics['train_accuracy'].append(train_acc)
            epoch_metrics['val_accuracy'].append(val_acc)
            epoch_metrics['epoch_time'].append(epoch_time)
        
        #end fold operations
        utils.plot_metrics(epoch_metrics, fold, plots_dir)
        
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
        utils.plot_and_save_confusion_matrix(fold= fold, cm=cm, dir= confusion_matrices_dir, classes= ['0', '1'])
        metrics = utils.compute_metrics(cm, all_labels= all_labels, all_preds= all_preds)
        all_metrics.append(metrics)

        model_save_path = os.path.join(models_dir, f'model_fold_{fold}.pt')
        torch.save(model.state_dict(), model_save_path)
        writer.close()
        print(f'Model saved to {model_save_path}')
    
    average_metrics = {}
    for key in all_metrics[0]:
        values = [metric[key] for metric in all_metrics]
        average_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    utils.save_metrics_to_yaml(average_metrics, confusion_matrices_dir)

