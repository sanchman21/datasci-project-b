import torch # import the PyTorch library
import numpy as np # import the NumPy library
import pandas as pd # import the pandas library
import os # import the os library
import datetime # import the datetime library
import utils_zhenzhuo # import the utility functions
import time # import the time library

from torch.utils.data import DataLoader # import the PyTorch DataLoader class
import torchvision.transforms as transforms # import the PyTorch transforms module
from torchvision import models # import the PyTorch vision models
from torch import nn, optim # import the PyTorch neural network and optimization modules
from torch.optim.lr_scheduler import LinearLR # import the PyTorch learning rate scheduler
from torch.amp import GradScaler # import the PyTorch gradient scaler and autocast modules
from torchvision.models import resnet50, ResNet50_Weights # import the ResNet50 model and weights
from torch.utils.tensorboard import SummaryWriter # import the SummaryWriter class for TensorBoard logging
from sklearn.metrics import confusion_matrix # import the confusion matrix function

from tqdm import tqdm # import the tqdm library for progress bars
from MergeMasterDataset import MergeMasterDataset # import the MergeMasterDataset class
from MultimodalClassifier import MultimodalClassifier # import the MultimodalClassifier class

import matplotlib.pyplot as plt # import the matplotlib library for plotting
import torchvision.transforms.functional # import the functional module from torchvision.transforms
from resnet50_model import monocyte_dataset # import the monocyte_dataset module from the resnet50_model package

def train_model_pure(config: dict) -> None:
    '''
    Function: Train the model using the specified configuration
    Parameters:
        config (dict): Configuration dictionary
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
    
    base_dir = f"./towmim/experiments/{config['date']['type']}" # get the base directory for saving the models
    os.makedirs(base_dir, exist_ok=True) # create the base directory if it does not exist

    # Create subdirectories for different types of data
    logs_dir = os.path.join(base_dir, 'logs') # create a directory for logs
    confusion_matrices_dir = os.path.join(base_dir, 'confusion_matrices') # create a directory for confusion matrices
    models_dir = os.path.join(base_dir, 'models') # create a directory for models
    plots_dir = os.path.join(base_dir, '') # create a directory for plots
    
    os.makedirs(logs_dir, exist_ok=True) # create the logs directory
    os.makedirs(confusion_matrices_dir, exist_ok=True) # create the confusion matrices directory
    os.makedirs(models_dir, exist_ok=True) # create the models directory
    os.makedirs(plots_dir, exist_ok=True) # create the plots directory

    # Load training configurations
    csv_file = utils_zhenzhuo.convert_path_to_os_specific(config['data']['csv_file']) # get the path to the CSV file
    num_epochs = config['training']['num_epochs'] # get the number of epochs
    batch_size = config['training']['batch_size'] # get the batch size
    learning_rate = config['training']['learning_rate'] # get the learning rate
    learning_rate_end = config['training']['learning_rate_end'] # get the end learning rate
    weight_decay = config['training']['weight_decay'] # get the weight decay
    momentum = config['training']['momentum'] # get the momentum

    image_size = config['model']['image_size'] # get the image size
    use_scheduler = config['model']['use_scheduler'] # get the use scheduler flag
    num_workers = config['model']['num_workers'] # get the number of workers

    # Data augmentation for training
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data augmentation for testing
    test_transforms = transforms.Compose([
        transforms.Resize(size=(352, 352)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # test_augmented_transforms = [
    #     transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    #     transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  
    #     transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    #     transforms.Compose([transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  
    #     transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
    #     transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  
    #     transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
    #     transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 
    # ]

    # desired_height = 352 # get the desired height
    # desired_width = 352 # get the desired width

    # test_augmented_transforms = [
    #     transforms.Compose([transforms.Resize((desired_height, desired_width)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
    #     transforms.Compose([transforms.Resize((desired_height, desired_width)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
    #     transforms.Compose([transforms.Resize((desired_height, desired_width)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    #     transforms.Compose([transforms.Resize((desired_height, desired_width)), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  
    #     transforms.Compose([transforms.Resize((desired_height, desired_width)), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
    #     transforms.Compose([transforms.Resize((desired_height, desired_width)), transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]), 
    #     transforms.Compose([transforms.Resize((desired_height, desired_width)), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    #     transforms.Compose([transforms.Resize((desired_height, desired_width)), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # ]

    num_folds = 5 # get the number of folds

    all_metrics = [] # list to store all metrics
    
    for fold in range(num_folds): # for each fold
        writer = SummaryWriter(log_dir=os.path.join(logs_dir, f"fold_{fold}")) # create a SummaryWriter for logging

        print(f"Training fold {fold+1}/{num_folds}") # print the current fold

        model = resnet50(weights=ResNet50_Weights.DEFAULT) # load the ResNet50 model
        model.fc = nn.Linear(model.fc.in_features, 2) # change the fully connected layer to output 2 classes

        model.to(device) # move the model to the device
        criterion = nn.CrossEntropyLoss().to(device) # define the loss function
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum) # define the optimizer

        if use_scheduler: # if using a scheduler
            end_factor = learning_rate_end / learning_rate # calculate the end factor
            scheduler = LinearLR(optimizer, start_factor=1, end_factor=end_factor, total_iters=num_epochs) # create the scheduler

        scaler = GradScaler() # create the gradient scaler

        # Dataset and DataLoader setup
        train_dataset = monocyte_dataset.MonocyteDataset(csv_file, fold=fold, train=True, transform=transform)
        val_dataset = monocyte_dataset.MonocyteDataset(csv_file, fold=fold, train=False, transform=test_transforms)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers = num_workers, shuffle=False)

        # for the calculation of confusion matrix, store all labels, predictions and patient ids
        all_labels = []
        all_preds = []
        all_patient_ids = []


        # Stores metrics for each epoch
        epoch_metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_time': []
        }

        # Training
        for epoch in range(num_epochs): # for each epoch
            start_time = time.time() # get the start time
            model.train() # set the model to training mode
            total_loss, total_correct, total_samples = 0, 0, 0 # initialize the total loss, total correct, and total samples

            for batch in train_loader: # for each batch
                images = batch['image'].to(device) # get the images and move them to the device
                labels = batch['morphology'].to(device) # get the labels and move them to the device
                
                with torch.autocast(device_type=device): # use autocast
                    outputs = model(images) # get the outputs
                    loss = criterion(outputs, labels) # calculate the loss
                
                optimizer.zero_grad() # reset the gradients
                scaler.scale(loss).backward() # backward pass
                scaler.step(optimizer) # step the optimizer
                scaler.update() # update the scaler
                
                total_loss += loss.item() * images.size(0) # update the total loss
                _, predicted = torch.max(outputs.float(), 1) # get the predicted labels
                total_correct += (predicted == labels).sum().item() # update the total correct
                total_samples += labels.size(0) # update the total samples

            if use_scheduler: # if using a scheduler
                scheduler.step() # step the scheduler
                
            train_loss = total_loss / total_samples # calculate the average training loss
            train_acc = total_correct / total_samples # calculate the training accuracy

            writer.add_scalar(f'Train/Loss_fold_{fold}', train_loss, epoch) # log the training loss
            writer.add_scalar(f'Train/Accuracy_fold_{fold}', train_acc, epoch) # log the training accuracy

            # Validation
            val_loss, val_correct, val_total = 0, 0, 0 # initialize the validation loss, validation correct, and validation total
            model.eval() # set the model to evaluation mode
            with torch.no_grad(): # disable gradient calculation
                for batch in val_loader: # for each batch
                    images = batch['image'].to(device) # get the images and move them to the device
                    labels = batch['morphology'].to(device) # get the labels and move them to the device
                    patient_ids = batch['patient_id'].tolist() # get the patient IDs and convert them to a list


                    outputs = model(images) # get the outputs
                    # outputs_list = []
                    # for transform in test_augmented_transforms:
                    #     augmented_images = torch.stack([transform(torchvision.transforms.functional.to_pil_image(image)) for image in images])
                    #     augmented_images = augmented_images.to(device)

                    #     outputs = model(augmented_images)
                    #     outputs_list.append(outputs)
                    
                    # outputs = torch.stack(outputs_list).mean(0)
                    

                    loss = criterion(outputs, labels) # calculate the loss
                    val_loss += loss.item() * images.size(0) # update the validation loss
                    _, predicted = torch.max(outputs.float(), 1) # get the predicted labels
                    val_correct += (predicted == labels).sum().item() # update the validation correct
                    val_total += labels.size(0) # update the validation total

                    all_labels.extend(labels.tolist()) # extend the list of all labels
                    all_preds.extend(predicted.tolist()) # extend the list of all predictions
                    all_patient_ids.extend(patient_ids) # extend the list of all patient IDs

            val_acc = val_correct / val_total # calculate the validation accuracy
            val_loss = val_loss / val_total # calculate the average validation loss

            end_time = time.time() # get the end time
            epoch_time = end_time - start_time # calculate the epoch time
            # Print epoch metrics
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            writer.add_scalar(f'Validation/Loss_fold_{fold}', val_loss, epoch) # log the validation loss
            writer.add_scalar(f'Validation/Accuracy_fold_{fold}', val_acc, epoch) # log the validation accuracy

            # Append epoch metrics for plot
            epoch_metrics['train_loss'].append(train_loss)
            epoch_metrics['val_loss'].append(val_loss)
            epoch_metrics['train_accuracy'].append(train_acc)
            epoch_metrics['val_accuracy'].append(val_acc)
            epoch_metrics['epoch_time'].append(epoch_time)
        
        #end fold operations
        utils_zhenzhuo.plot_metrics(epoch_metrics, fold, plots_dir) # plot the metrics

        # Save the model and metrics as a pandas dataframe
        results_df = pd.DataFrame({
            'patient_id': all_patient_ids,
            'predicted': all_preds,
            'label': all_labels
        })
        results_df.to_csv(os.path.join(base_dir,'labelandpredic')) # save the results dataframe

        patient_predictions = results_df.groupby('patient_id')['predicted'].mean().round().astype(int) # get the patient predictions
        patient_labels = results_df.groupby('patient_id')['label'].first() # get the patient labels

        patient_correct = (patient_predictions == patient_labels).sum() # get the number of correct patient predictions
        patient_total = patient_labels.size # get the total number of patients

        patient_acc = patient_correct / patient_total # calculate the patient accuracy

        print(f'Patient-level accuracy: {patient_acc}') # print the patient accuracy

        cm = confusion_matrix(all_labels, all_preds) # calculate the confusion matrix
        print(cm) # print the confusion matrix
        
        # Plot and save the confusion matrix
        utils_zhenzhuo.plot_and_save_confusion_matrix(fold= fold, cm=cm, dir= confusion_matrices_dir, classes= ['0', '1'])
        metrics = utils_zhenzhuo.compute_metrics(cm, all_labels= all_labels, all_preds= all_preds) # compute the metrics
        all_metrics.append(metrics) # append the metrics to the list

        model_save_path = os.path.join(models_dir, f'model_fold_{fold}.pt') # get the model save path
        torch.save(model.state_dict(), model_save_path) # save the model
        writer.close() # close the writer
        print(f'Model saved to {model_save_path}') # print the model save path
    
    average_metrics = {} # dictionary to store the average metrics
    for key in all_metrics[0]: # for each key in the first metric
        values = [metric[key] for metric in all_metrics] # get the metric values
        # calculate the average and standard deviation
        average_metrics[key] = { 
            'mean': np.mean(values),
            'std': np.std(values)
        }
    utils_zhenzhuo.save_metrics_to_yaml(average_metrics, confusion_matrices_dir) # save the average metrics to a YAML file

