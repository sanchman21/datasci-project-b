import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import resnet50, ResNet50_Weights

from tqdm import tqdm
import os
from MergeMasterDataset import MergeMasterDataset
from MultimodalClassifier import MultimodalClassifier



def preprocess_patient_data(batch, device):
    patient_keys = ['Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count', 'Neutrophil count', 'Monocyte count', 'Platelet count', 'Blast percentage (PB)', 'LDH']
    return torch.tensor([list(batch[key]) for key in patient_keys]).float().to(device)


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    
    num_patient_features = 10

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    num_folds = 5
    all_fold_metrics = [] 

    for fold in range(num_folds):
        print(f"Training fold {fold+1}/{num_folds}")

        if model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif model_name == 'MultimodalClassifier':
            model = MultimodalClassifier(num_patient_features)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        model.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        end_factor = learning_rate_end / learning_rate
        scheduler = LinearLR(optimizer, start_factor=1, end_factor=end_factor, total_iters=num_epochs)

        scaler = GradScaler()

        # Dataset and DataLoader setup
        train_dataset = MergeMasterDataset(csv_file, fold=fold, train=True, transform=transform)
        val_dataset = MergeMasterDataset(csv_file, fold=fold, train=False, transform=transform)

        if model_name == 'MultimodalClassifier':
            train_dataset = MergeMasterDataset(csv_file, fold=fold, train=True, use_patient_data= True, transform=transform)
            val_dataset = MergeMasterDataset(csv_file, fold=fold, train=False, use_patient_data= True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        fold_metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(num_epochs):
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

            scheduler.step()
            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

            fold_metrics['train_loss'].append(train_loss)
            fold_metrics['train_acc'].append(train_acc)

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
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            print(f'Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            fold_metrics['val_loss'].append(val_loss)
            fold_metrics['val_acc'].append(val_acc)
        all_fold_metrics.append(fold_metrics)
    return all_fold_metrics
