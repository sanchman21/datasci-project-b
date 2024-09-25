import torch # import the PyTorch library
import numpy as np # import the NumPy library
import pandas as pd # import the Pandas library
import os # import the os module
import datetime # import the datetime module
import utils_zhenzhuo # import the utility functions
import time # import the time module

from torch.utils.data import DataLoader # import the DataLoader class
import torchvision.transforms as transforms # import the transforms module
from torchvision import models # import the models module
from torch import nn, optim # import the nn and optim modules
from torch.optim.lr_scheduler import LinearLR # import the LinearLR class
from torch.amp import GradScaler # import the GradScaler and autocast classes
from torchvision.models import resnet50, ResNet50_Weights # import the resnet50 model and ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter # import the SummaryWriter class
from sklearn.metrics import confusion_matrix # import the confusion_matrix function

from tqdm import tqdm # import the tqdm module
from MergeMasterDataset import MergeMasterDataset # import the MergeMasterDataset class
from MultimodalClassifier import MultimodalClassifier # import the MultimodalClassifier class

import matplotlib.pyplot as plt # import the matplotlib.pyplot module


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

#TODO: untested code here
def load_model(model_path: str, model_type: str='resnet50', num_patient_features: int=10) -> torch.nn.Module:
    '''
    Function: Load a model from a file
    Parameters:
        model_path (str): The path to the model file
        model_type (str): The type of model to load. Can be 'resnet50' or 'MultimodalClassifier'
        num_patient_features (int): The number of patient features
    Returns: torch.nn.Module
    '''
    if torch.cuda.is_available(): # if cuda is available
        device = "cuda" # set the device to cuda
    elif torch.backends.mps.is_available(): # if mps is available
        device = "mps" # set the device to mps
    else: # otherwise
        device = "cpu" # set the device to cpu
    print(f"Using device: {device}") # print the device being used
    
    if model_type == 'resnet50': # if the model type is resnet50
        model = resnet50(weights=ResNet50_Weights.DEFAULT) # load the pretrained resnet50 model
        model.fc = nn.Linear(model.fc.in_features, 2) # replace the final fully connected layer with a new one
    elif model_type == 'MultimodalClassifier': # if the model type is MultimodalClassifier
        model = MultimodalClassifier(num_patient_features) # load the MultimodalClassifier model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device)) # load the model state dictionary from the model file
    model.to(device) # move the model to the device
    return model # return the model


def train_model(config: dict) -> None:
    '''
    Function: Train a model
    Parameters:
        config (dict): The configuration dictionary
    Returns: None
    '''
    if torch.cuda.is_available(): # if cuda is available
        device = "cuda" # set the device to cuda
    elif torch.backends.mps.is_available(): # if mps is available
        device = "mps" # set the device to mps
    else: # otherwise
        device = "cpu" # set the device to cpu
    print(f"Using device: {device}") # print the device being used

    # timestamp for the reference of creating folders
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # Create a base directory for saving models
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

    # Load training configurations from the config file by first converting the path to the OS-specific format
    csv_file = utils_zhenzhuo.convert_path_to_os_specific(config['data']['csv_file'])
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
    num_workers = config['model']['num_workers']

    # Data augmentation for training
    transform = transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_folds = 5 # number of folds for cross-validation
    num_patient_features = 10 # number of patient features

    all_metrics = [] # list to store metrics for each fold
    for fold in range(num_folds): # for each fold
        writer = SummaryWriter(log_dir=os.path.join(logs_dir, f"fold_{fold}")) # create a SummaryWriter for logging
        print(f"Training fold {fold+1}/{num_folds}") # print the fold number

        if model_name == 'resnet50': # if the model name is resnet50
            model = resnet50(weights=ResNet50_Weights.DEFAULT) # load the pretrained resnet50 model
            model.fc = nn.Linear(model.fc.in_features, 2) # replace the final fully connected layer with a new one

            if freeze_backbone: # if the backbone is to be frozen
                for name, param in model.named_parameters(): # for each parameter in the model
                    if 'fc' not in name: # if the parameter is not in the final fully connected layer
                        param.requires_grad = False # set the parameter to not require gradients
                    else: # otherwise
                        print(f"{name} is not frozen.") # print that the parameter is not frozen

        elif model_name == 'MultimodalClassifier': # if the model name is MultimodalClassifier
            model = MultimodalClassifier(num_patient_features) # load the MultimodalClassifier model
            
            if freeze_backbone: # if the backbone is to be frozen
                for name, param in model.named_parameters(): # for each parameter in the model
                        if 'resnet' in name: # if the parameter is in the resnet backbone
                            param.requires_grad = False # set the parameter to not require gradients
                        else: # otherwise
                            print(f"{name} is not frozen.") # print that the parameter is not frozen
        else: # otherwise
            raise ValueError(f"Unsupported model type: {model_name}") # raise a ValueError for an unsupported model type
        
        model.to(device) # move the model to the device
        criterion = nn.CrossEntropyLoss().to(device) # define the loss function and move it to the device
        # define the optimizer
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        if use_scheduler: # if a scheduler is to be used
            end_factor = learning_rate_end / learning_rate # calculate the end factor for the scheduler
            scheduler = LinearLR(optimizer, start_factor=1, end_factor=end_factor, total_iters=num_epochs) # create a LinearLR scheduler

        scaler = GradScaler() # create a GradScaler for mixed precision training

        if model_name == 'MultimodalClassifier': # load the patient data if choose to use multimodal classifier
            train_dataset = MergeMasterDataset(csv_file, fold=fold, train=True, use_patient_data= True, use_neutrophil_images= use_neutrophil_images, transform=transform)
            val_dataset = MergeMasterDataset(csv_file, fold=fold, train=False, use_patient_data= True, use_neutrophil_images= use_neutrophil_images, transform=transform)
        else: # otherwise load only the images
            train_dataset = MergeMasterDataset(csv_file, fold=fold, train=True, use_neutrophil_images= use_neutrophil_images, transform=transform)
            val_dataset = MergeMasterDataset(csv_file, fold=fold, train=False, use_neutrophil_images= use_neutrophil_images, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers= num_workers, shuffle=True) # create a DataLoader for the training dataset
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers= num_workers, shuffle=False) # create a DataLoader for the validation dataset

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

        for epoch in range(num_epochs): # for each epoch
            start_time = time.time() # get the start time
            model.train() # set the model to training mode
            total_loss, total_correct, total_samples = 0, 0, 0 # initialize the total loss, total correct, and total samples to 0

            for batch in train_loader: # for each batch in the training loader
                images = batch['image'].to(device) # get the images from the batch and move them to the device
                labels = batch['morphology'].to(device) # get the labels from the batch and move them to the device
                
                with torch.autocast(): # use mixed precision training
                    if model_name == 'MultimodalClassifier': # if the model is a MultimodalClassifier
                        patient_data = preprocess_patient_data(batch, device) # preprocess the patient data
                        outputs = model(images, patient_data) # get the outputs from the model
                    else:
                        outputs = model(images) # get the outputs from the model

                    loss = criterion(outputs, labels) # calculate the loss
                
                optimizer.zero_grad() # reset the gradients
                scaler.scale(loss).backward() # scale and backpropagate the loss
                scaler.step(optimizer) # take a step with the optimizer
                scaler.update() # update the scaler
                
                total_loss += loss.item() * images.size(0) # update the total loss
                _, predicted = torch.max(outputs.float(), 1) # get the predicted labels
                total_correct += (predicted == labels).sum().item() # update the total correct
                total_samples += labels.size(0) # update the total samples

            if use_scheduler: # if a scheduler is to be used
                scheduler.step() # step the scheduler
                
            train_loss = total_loss / total_samples # calculate the training loss
            train_acc = total_correct / total_samples # calculate the training accuracy

            writer.add_scalar(f'Train/Loss_fold_{fold}', train_loss, epoch) # log the training loss
            writer.add_scalar(f'Train/Accuracy_fold_{fold}', train_acc, epoch) # log the training accuracy

            # Validation
            num_augmentations = 5 # Number of test-time augmentations
            # Test Time Augmentations (TTA)
            augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
            ])
            
            val_loss, val_correct, val_total = 0, 0, 0 # initialize the validation loss, validation correct, and validation total to 0
            model.eval() # set the model to evaluation mode
            with torch.no_grad(): # do not calculate gradients
                for batch in val_loader: # for each batch in the validation loader
                    images = batch['image'].to(device) # get the images from the batch and move them to the device
                    labels = batch['morphology'].to(device) # get the labels from the batch and move them to the device
                    
                    outputs_list = [] # initialize a list to store the outputs
                    for _ in range(num_augmentations): # for each augmentation
                        augmented_images = augmentations(images)# apply TTA to the images
                        if model_name == 'MultimodalClassifier': # if the model is a MultimodalClassifier
                            patient_data = preprocess_patient_data(batch, device) # preprocess the patient data
                            outputs = model(augmented_images, patient_data) # get the outputs from the model
                        else: # otherwise
                            outputs = model(augmented_images) # get the outputs from the model
                        outputs_list.append(outputs) # append the outputs to the list

                    outputs = torch.stack(outputs_list).mean(0) # calculate the mean of the outputs
                    # torch.mean(torch.stack(predictions), dim=0)

                    loss = criterion(outputs, labels) # calculate the loss
                    val_loss += loss.item() * images.size(0) # update the validation loss
                    _, predicted = torch.max(outputs.float(), 1) # get the predicted labels
                    val_correct += (predicted == labels).sum().item() # update the validation correct
                    val_total += labels.size(0) # update the validation total

                    all_labels.extend(labels.tolist()) # extend the list of all labels
                    all_preds.extend(predicted.tolist()) # extend the list of all predictions

            val_acc = val_correct / val_total # calculate the validation accuracy
            val_loss = val_loss / val_total # calculate the validation loss

            end_time = time.time() # get the end time
            epoch_time = end_time - start_time # calculate the epoch time
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            writer.add_scalar(f'Validation/Loss_fold_{fold}', val_loss, epoch) # log the validation loss
            writer.add_scalar(f'Validation/Accuracy_fold_{fold}', val_acc, epoch) # log the validation accuracy

            # Append epoch metrics for plot
            epoch_metrics['train_loss'].append(train_loss)
            epoch_metrics['val_loss'].append(val_loss)
            epoch_metrics['train_accuracy'].append(train_acc)
            epoch_metrics['val_accuracy'].append(val_acc)
            epoch_metrics['epoch_time'].append(epoch_time)
        
        # Plot metrics
        utils_zhenzhuo.plot_metrics(epoch_metrics, fold, plots_dir)
        
        cm = confusion_matrix(all_labels, all_preds) # calculate the confusion matrix
        print(cm)
        # Plot and save confusion matrix
        utils_zhenzhuo.plot_and_save_confusion_matrix(fold= fold, cm=cm, dir= confusion_matrices_dir, classes= ['0', '1'])
        metrics = utils_zhenzhuo.compute_metrics(cm, all_labels= all_labels, all_preds= all_preds) # compute the metrics
        all_metrics.append(metrics) # append the metrics to the list

        model_save_path = os.path.join(models_dir, f'model_fold_{fold}.pt') # get the model save path
        torch.save(model.state_dict(), model_save_path) # save the model state dictionary to the model save path
        writer.close() # close the SummaryWriter
        print(f'Model saved to {model_save_path}') # print that the model has been saved
    
    average_metrics = {} # dictionary to store the average metrics
    for key in all_metrics[0]: # for each key in the first metric
        values = [metric[key] for metric in all_metrics] # get the values for the key from each metric
        # calculate the mean and standard deviation of the values
        average_metrics[key] = { 
            'mean': np.mean(values),
            'std': np.std(values)
        }
    utils_zhenzhuo.save_metrics_to_yaml(average_metrics, confusion_matrices_dir) # save the average metrics to a YAML file

