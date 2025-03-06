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
sys.path.append('../townim')  # running python main.py from the directory the file is located in
import utils
from utils import FixedRotation, plot_confusion_matrix, plot_save_roc_curve, save_metrics_csv
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH, MONOCYTE_NEW_NORMALS_CSV_PATH

# creating a cache directory since running docker using specific user doesn't allow to use the home cache directory
cache_dir = "../cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir  # set cache directory

if torch.cuda.is_available():  # if cuda is available
    torch.cuda.empty_cache()  # empty the cache
    device = "cuda"  # set the device to cuda
elif torch.backends.mps.is_available():  # if mps is available
    torch.mps.empty_cache()  # empty the cache
    device = "mps"  # set the device to mps
else:  # otherwise
    device = "cpu"  # set the device to cpu
print(f"Using device: {device}")  # print the device being used

# argument parser with argument for fold
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default="monocyte_new_normals", help="image type to train the model on", 
                    choices=('neutrophil', 'monocyte', 'monocyte_new_normals'))
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

yaml_dir = f'./experiments/{data_type}/config.yaml'
with open(yaml_dir, 'r') as file:
    config = yaml.safe_load(file) 

# check if experiment exists, create if not
existing_experiment = mlflow.get_experiment_by_name(data_type)
if existing_experiment is None:
    mlflow.create_experiment(data_type)  # Create new experiment with name
    mlflow.set_experiment(experiment_name=data_type)
    print(f"Created new experiment for {data_type}")
else:
    experiment_id = existing_experiment.experiment_id  # Reuse existing ID
    overwrite_exp = input(f"DO YOU WANT TO OVERWRITE EXISTING {data_type} EXPERIMENT? [Y/N]")
    if overwrite_exp.lower() == "y":
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        exit()
    print(f"Using existing experiment for {data_type}")

root = f"./experiments/{data_type}/train"
os.makedirs(root, exist_ok=True)  # output directory (main level)

# Create data loaders
num_epochs = config['num_epochs']
batch_size = config['batch_size']
IMAGE_SIZE = config['image_size']
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]  # Std of ImageNet dataset (used for normalization)
use_scheduler = config['use_scheduler']
lr = config['lr']
weight_decay = config['weight_decay']
momentum = config['momentum']
start_factor = config['start_factor']
end_factor = config['end_factor']
patience = config['patience']  # Number of epochs to wait for improvement (100 for no early stopping)

# Define the transformations for training and testing
train_transform = T.Compose([
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

# Start parent run for train-val
with mlflow.start_run(run_name="train-val") as parent_run:
    # Log config to parent run
    mlflow.log_artifact(yaml_dir)
    mlflow.log_params(config)

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
            
        shutil.copyfile('./trainer.py', os.path.join(output_dir, 'trainer.py'))  # copying code file used to train the model
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
        num_classes = len(set(train_dataset.labels))  # number of classes (2)
        model = models.resnet50(weights='IMAGENET1K_V1')
        hidden_layer_size = 512
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_classes)
        )

        model = model.to(device)  # move the model to the device

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

        # Start nested run for each fold
        with mlflow.start_run(run_name=f"fold_{set_id}", nested=True):
            mlflow.set_tag("train_normal_count", train_dataset.disease_count[0])
            mlflow.set_tag("train_cmml_count", train_dataset.disease_count[1])
            mlflow.set_tag("val_normal_count", val_dataset.disease_count[0])
            mlflow.set_tag("val_cmml_count", val_dataset.disease_count[1])

            for epoch in range(num_epochs):  # for each epoch
                if early_stop:  # if early stopping is true, break
                    break

                # Training loop
                t = tqdm(enumerate(train_loader, 0), total=len(train_loader),
                        smoothing=0.9, position=0, leave=True,
                        desc="Train: Epoch: " + str(epoch + 1) + "/" + str(num_epochs))  # progress bar for train loader
                model.train()  # set the model to training mode
                running_loss = 0.0  # initialize running loss
                all_preds, all_labels = [], []  # initialize lists to store predictions and labels

                for i, (inputs, labels) in t:  # for each batch
                    inputs = inputs.to(device).float()  # move inputs to device
                    labels = labels.to(device).long()  # move labels to device
                    optimizer.zero_grad()  # zero the gradients
                    outputs = model(inputs)  # forward pass
                    loss = F.cross_entropy(outputs, labels)  # calculate the loss
                    loss.backward()  # backward pass
                    optimizer.step()  # update the weights

                    running_loss += loss.item()  # update the running loss
                    outputs = F.softmax(outputs, dim=-1)  # get the probabilities
                    preds = outputs.argmax(dim=1).cpu().numpy()  # get the predictions
                    all_preds.extend(preds)  # append the predictions to the list
                    all_labels.extend(labels.cpu().numpy())  # append the labels to the list

                # Calculate metrics
                train_loss = running_loss / len(train_loader)
                train_accuracy = accuracy_score(all_labels, all_preds)
                train_precision = precision_score(all_labels, all_preds, average='binary')
                train_recall = recall_score(all_labels, all_preds, average='binary')
                train_f1 = f1_score(all_labels, all_preds, average='binary')
                train_auroc = roc_auc_score(all_labels, all_preds)

                # Validation loop
                model.eval()  # set the model to evaluation mode
                val_loss = 0.0  # initialize validation loss
                val_preds, val_labels, val_probs = [], [], []  # initialize lists to store predictions, labels and probabilities

                with torch.no_grad():  # no gradients
                    
                    t = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                            smoothing=0.9, position=0, leave=True,
                            desc="Val: Epoch: " + str(epoch + 1) + "/" + str(num_epochs))  # progress bar for validation loader
                    
                    for i, (inputs, labels) in t:  # for each batch
                        inputs, labels = inputs.to(device).float(), labels.to(device).long()  # move inputs and labels to device
                        outputs = model(inputs)  # forward pass
                        loss = F.cross_entropy(outputs, labels)  # calculate the loss
                        val_loss += loss.item()  # update the validation loss
                        outputs = F.softmax(outputs, dim=-1)  # get the probabilities
                        preds = outputs.argmax(dim=1).cpu().numpy()  # get the predictions
                        probs = outputs[:, 1].cpu().numpy()  # probabilities for class 1
                        val_preds.extend(preds)  # append the predictions to the list
                        val_labels.extend(labels.cpu().numpy())  # append the labels to the list
                        val_probs.extend(probs)  # append the probabilities to the list

                # Calculate metrics
                val_loss = val_loss / len(val_loader)
                val_accuracy = accuracy_score(val_labels, val_preds)
                val_precision = precision_score(val_labels, val_preds, average='binary')
                val_recall = recall_score(val_labels, val_preds, average='binary')
                val_f1 = f1_score(val_labels, val_preds, average='binary')
                val_auroc = roc_auc_score(val_labels, val_preds)

                if use_scheduler:  # if using scheduler, step the scheduler
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
                if val_loss < best_val_loss:  # if current loss is less than best loss
                    best_val_loss = val_loss  # update best loss
                    best_epoch = epoch + 1  # update best epoch
                    epochs_no_improve = 0  # set early stopping epochs to 0
                    torch.save(model.state_dict(), os.path.join(model_dir, f'model_fold_{set_id}.pth'))
                else:
                    epochs_no_improve += 1  # increment early stopping epochs
                    if epochs_no_improve == patience:  # if early stopping epochs is equal to the patience
                        early_stop = True  # early stop
                        print("Early Stopping....")
                        break  # Stop training

            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("best_epoch", best_epoch)

            final_epochs += best_epoch

            # Save the metrics and the last model
            df = pd.DataFrame(metrics_data, columns=['Epoch', 'LR', 'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUROC', 
                                                        'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUROC'])
            df.to_csv(os.path.join(output_dir, f'train_time_metrics_fold_{set_id}.csv'), index=False)

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

            plt.savefig(os.path.join(figure_dir, f'loss_curve_fold_{set_id}.png'))
            plt.close()

            mlflow.log_artifact(os.path.join(model_dir, f'model_fold_{set_id}.pth'))
            mlflow.log_artifact(os.path.join(figure_dir, f"loss_curve_fold_{set_id}.png"))
            mlflow.log_artifact(os.path.join(output_dir, f'train_time_metrics_fold_{set_id}.csv'))
            
            val_dataset = CustomDataset('val', CSV_PATH, set_id, transform=test_transform)
            val_loaders = [
                DataLoader(CustomDataset('val', CSV_PATH, set_id, transform=transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
                for transform in TTAs
            ]
            
            rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]  # patient ids that were rechecked
            
            # define the model architecture
            print("Model: Resnet50")
            model = models.resnet50(weights='IMAGENET1K_V1')
            hidden_layer_size = 512
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, hidden_layer_size),
                nn.ReLU(),
                nn.Linear(hidden_layer_size, num_classes)
            )

            model = model.to(device)  # set the model to the device
            model.load_state_dict(torch.load(os.path.join(model_dir, f'model_fold_{set_id}.pth')))
            model.eval()  # set the model to evaluation mode

            with torch.no_grad():  # turn off gradients
                correct = 0  # number of correct predictions
                preds = []  # predicted labels
                labels = []  # true labels
                logits = []  # predicted logits
                
                for i, (inputs, targets) in enumerate(val_loaders[0]):  # iterate through the test loaders
                    inputs, targets = inputs.to(device).float(), targets.to(device).long()  # set the inputs and targets to the device
                    outputs = model(inputs)  # get the model outputs
                    outputs = F.softmax(outputs, dim=-1)  # get the probabilities
                    _, predicted = torch.max(outputs, 1)  # get the predicted labels
                    correct += (predicted == targets).sum().item()  # update the number of correct predictions
                    preds.append(predicted.detach().cpu().numpy())  # append the predicted labels
                    labels.append(targets.detach().cpu().numpy())  # append the true labels
                    logits.append(outputs.detach().cpu().numpy().astype(np.float32))  # append the predicted logits

                preds = np.concatenate(preds, axis=0)  # concatenate the predicted labels
                labels = np.concatenate(labels, axis=0)  # concatenate the true labels
                logits = np.concatenate(logits, axis=0)  # concatenate the predicted logits

                # Calculate metrics at image level
                accuracy = accuracy_score(labels, preds)
                precision = precision_score(labels, preds, average="binary")
                recall = recall_score(labels, preds, average="binary")
                f1 = f1_score(labels, preds, average="binary")
                auc = roc_auc_score(labels, logits[:, 1])

                # Save metrics to CSV
                save_metrics_csv(set_id, accuracy, precision, recall, f1, auc, os.path.join(root, "metrics_image.csv"))

                # Confusion matrix
                confmat_vals = confusion_matrix(labels, preds)
                plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, f"image_level_conf_mat_fold_{set_id}.png"), f"Confusion Matrix on Test [Image level] Fold: {set_id}")
                mlflow.log_artifact(os.path.join(figure_dir, f"image_level_conf_mat_fold_{set_id}.png"))

                # ROC curve
                plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, f"image_level_roc_curve_fold_{set_id}.png"))
                mlflow.log_artifact(os.path.join(figure_dir, f"image_level_roc_curve_fold_{set_id}.png"))
                print(f'[W/O TTA] Image level Accuracy (Fold {set_id}): {accuracy} AUC: {auc}')
                
                preds = []  # predicted labels
                logits = []  # predicted logits
                labels = []  # true labels
                correct = 0  # number of correct predictions
                
                for i, data in enumerate(zip(*test_loaders)):  # iterate through the test loaders
                    inputs, targets = torch.cat([img for img,_ in data], dim=0).to(device).float(), data[0][1].to(device).long()  # set the inputs and targets to the device
                    outputs = model(inputs)  # get the model outputs
                    outputs = F.softmax(outputs, dim=-1)  # get the probabilities
                    outputs = outputs.reshape(len(data), int(inputs.shape[0]/len(data)), -1).mean(dim=0)  # average the probabilities

                    _, predicted = torch.max(outputs, 1)  # get the predicted labels
                    correct += (predicted == targets).sum().item()  # update the number of correct predictions
                    preds.append(predicted.detach().cpu().numpy())  # append the predicted labels
                    labels.append(targets.detach().cpu().numpy())  # append the true labels
                    logits.append(outputs.detach().cpu().numpy().astype(np.float32))  # append the predicted logits

                preds = np.concatenate(preds, axis=0)  # concatenate the predicted labels
                labels = np.concatenate(labels, axis=0)  # concatenate the true labels
                logits = np.concatenate(logits, axis=0)  # concatenate the predicted logits

                # Calculate metrics with TTA
                accuracy = accuracy_score(labels, preds)
                precision = precision_score(labels, preds, average="binary")
                recall = recall_score(labels, preds, average="binary")
                f1 = f1_score(labels, preds, average="binary")
                auc = roc_auc_score(labels, logits[:, 1])

                # Save metrics to CSV
                save_metrics_csv(set_id, accuracy, precision, recall, f1, auc, os.path.join(root, "metrics_image_tta.csv"))

                # Confusion matrix
                confmat_vals = confusion_matrix(labels, preds)
                plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, f"tta_image_level_conf_mat_fold_{set_id}.png"), f"Confusion Matrix on Test [Image level with TTA] Fold: {set_id}")
                mlflow.log_artifact(os.path.join(figure_dir, f"tta_image_level_conf_mat_fold_{set_id}.png"))

                # ROC curve
                plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, f"tta_image_level_roc_curve_fold_{set_id}.png"))
                mlflow.log_artifact(os.path.join(figure_dir, f"tta_image_level_roc_curve_fold_{set_id}.png"))
                print(f'[TTA] Image level Accuracy (Fold: {set_id}): {accuracy} AUC: {auc}')
                
                # Save results to npz
                np.savez_compressed(os.path.join(model_dir, f'image_level_results_fold_{set_id}'),
                        labels=labels, 
                        preds=preds,
                        logits=logits,
                )

            # patient level
            id_patients = []  # patient ids
            id_patient_logits = []  # patient logits
            id_patient_preds = []  # patient predictions
            id_patient_labels = []  # patient labels
            patient_ids = test_dataset.df['patient_id'].to_numpy()  # patient ids
            correct = 0  # number of correct predictions
            incorrect_ids = []  # incorrect patient ids

            for id in np.unique(patient_ids):  # iterate through the unique patient ids
                indices = np.where(patient_ids==id)[0]  # get the indices of the patient ids
                id_patient_logits.append(np.mean(logits[indices,:], axis=0))  # get the average logits
                id_patient_labels.append(np.mean(labels[indices], axis=0))  # get the average labels
                id_patient_preds.append(id_patient_logits[-1].argmax())  # get the predicted label
                if id in rechecked_patient_ids:  # if the patient id is in the rechecked patient ids, print info
                    print(f"Patient Id (Rechecked): {id}, Ground Truth: {id_patient_labels[-1]}, Predicted: {id_patient_preds[-1]}")
                if id_patient_labels[-1] != id_patient_preds[-1]:  # if the predicted label is incorrect, append the incorrect patient ids
                    incorrect_ids.append((id, id_patient_labels[-1], id_patient_preds[-1]))  # append the incorrect patient ids
                correct += int(id_patient_preds[-1]==id_patient_labels[-1])  # update the number of correct predictions
                id_patients.append(id)  # append the patient id
                
            for id_data in incorrect_ids:  # iterate through the incorrect patient ids
                print(f"Patient Id: {id_data[0]}, Ground Truth: {id_data[1]}, Predicted: {id_data[2]}")  # print info

            preds = np.array(id_patient_preds)  # predicted labels
            labels = np.array(id_patient_labels)  # true labels
            logits = np.array(id_patient_logits)  # predicted logits

            # Calculate patient-level metrics
            accuracy = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, average="binary") 
            recall = recall_score(labels, preds, average="binary")
            f1 = f1_score(labels, preds, average="binary")
            auc = roc_auc_score(labels, logits[:, 1])

            # Save patient-level metrics to CSV
            save_metrics_csv(set_id, accuracy, precision, recall, f1, auc, os.path.join(root, "metrics_patient.csv"))

            # Confusion matrix for patient-level
            confmat_vals = confusion_matrix(labels, preds)
            plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, f"patient_level_conf_mat_fold_{set_id}.png"), f"Confusion Matrix on Test [Patient level] Fold: {set_id}")
            mlflow.log_artifact(os.path.join(figure_dir, f"patient_level_conf_mat_fold_{set_id}.png"))
            
            # ROC curve
            plot_save_roc_curve(np.eye(num_classes)[np.round(labels).astype(int)], logits, os.path.join(figure_dir, f"patient_level_roc_curve_fold_{set_id}.png"))
            mlflow.log_artifact(os.path.join(figure_dir, f"patient_level_roc_curve_fold_{set_id}.png"))
            print(f'[TTA] Patient level Accuracy (Fold: {set_id}): {accuracy} AUC: {auc}')

    # Log validation metrics to parent run after all folds
    mlflow.log_artifact(os.path.join(root, "metrics_image.csv"))
    mlflow.log_artifact(os.path.join(root, "metrics_image_tta.csv"))
    mlflow.log_artifact(os.path.join(root, "metrics_patient.csv"))

# combine train and validation datasets for final training
trainval_dataset = ConcatDataset([
    CustomDataset('train', CSV_PATH, set_id, transform=train_transform),
    CustomDataset('val', CSV_PATH, set_id, transform=train_transform)
])
all_labels = train_dataset.labels + val_dataset.labels  # combine labels
weights = utils.make_weights_for_balanced_classes(all_labels, device)  # get new weights for loss
weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))  # create sampler
trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                             sampler=weighted_sampler, num_workers=8, worker_init_fn=utils.worker_init_fn)  # create data loader

final_model_dir = os.path.join(root, "model")
os.makedirs(final_model_dir, exist_ok=True)
final_model = models.resnet50(weights='IMAGENET1K_V1')  # define final model
num_ftrs = final_model.fc.in_features
final_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, num_classes)
)
final_model = final_model.to(device)
final_optimizer = optim.SGD(final_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
final_scheduler = torch.optim.lr_scheduler.LinearLR(final_optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=final_epochs)
final_model.train()
final_epochs = (final_epochs // 5) + 5
print(f"Final training for {final_epochs} epochs")

with mlflow.start_run(run_name="test"):
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
        print(f"Final Training - Epoch {epoch+1}/{final_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        mlflow.log_metrics({
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc
        }, step=epoch)

    # Save final model trained on train+val
    torch.save(final_model.state_dict(), os.path.join(final_model_dir, 'final.pth'))
    mlflow.log_artifact(os.path.join(final_model_dir, 'final.pth'))

    # Create test directory
    test_root = f"./experiments/{data_type}/test"
    os.makedirs(test_root, exist_ok=True)
    test_figure_dir = os.path.join(test_root, "figures")
    os.makedirs(test_figure_dir, exist_ok=True)

    # Define test dataset and loaders
    test_dataset = CustomDataset('test', CSV_PATH, set_id, transform=test_transform)
    test_loaders = [
        DataLoader(CustomDataset('test', CSV_PATH, set_id, transform=transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
        for transform in TTAs
    ]
    print("Test dataset stats [Normal, CMML]:", test_dataset.disease_count)

    final_model.eval()

    # No-TTA Evaluation on Test Set
    with torch.no_grad():
        preds, labels, logits = [], [], []
        for i, (inputs, targets) in enumerate(test_loaders[0]):  # Use first loader (no TTA)
            inputs, targets = inputs.to(device).float(), targets.to(device).long()
            outputs = final_model(inputs)
            outputs = F.softmax(outputs, dim=-1)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
            logits.append(outputs.detach().cpu().numpy().astype(np.float32))

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")
        f1 = f1_score(labels, preds, average="binary")
        auc = roc_auc_score(labels, logits[:, 1])

        # Save metrics
        save_metrics_csv("Image W/O TTA", accuracy, precision, recall, f1, auc, os.path.join(test_root, "metrics.csv"), train=False)

        # Confusion matrix
        confmat_vals = confusion_matrix(labels, preds)
        plot_confusion_matrix(confmat_vals, num_classes, os.path.join(test_figure_dir, "image_level_conf_mat_test.png"), "Confusion Matrix on Test [Image level] Final Model")
        mlflow.log_artifact(os.path.join(test_figure_dir, "image_level_conf_mat_test.png"))

        # ROC curve
        plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(test_figure_dir, "image_level_roc_curve_test.png"))
        mlflow.log_artifact(os.path.join(test_figure_dir, "image_level_roc_curve_test.png"))
        print(f'[W/O TTA] Final Model Image-level Test Accuracy: {accuracy} AUC: {auc}')

    # TTA Evaluation on Test Set
    with torch.no_grad():
        preds, labels, logits = [], [], []
        for i, data in enumerate(zip(*test_loaders)):
            inputs, targets = torch.cat([img for img, _ in data], dim=0).to(device).float(), data[0][1].to(device).long()
            outputs = final_model(inputs)
            outputs = F.softmax(outputs, dim=-1)
            outputs = outputs.reshape(len(data), int(inputs.shape[0]/len(data)), -1).mean(dim=0)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())
            logits.append(outputs.detach().cpu().numpy().astype(np.float32))

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)

        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")
        f1 = f1_score(labels, preds, average="binary")
        auc = roc_auc_score(labels, logits[:, 1])

        # Save metrics
        save_metrics_csv("Image TTA", accuracy, precision, recall, f1, auc, os.path.join(test_root, "metrics.csv"), train=False)

        # Confusion matrix
        confmat_vals = confusion_matrix(labels, preds)
        plot_confusion_matrix(confmat_vals, num_classes, os.path.join(test_figure_dir, "tta_image_level_conf_mat_test.png"), "Confusion Matrix on Test [Image level with TTA] Final Model")
        mlflow.log_artifact(os.path.join(test_figure_dir, "tta_image_level_conf_mat_test.png"))

        # ROC curve
        plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(test_figure_dir, "tta_image_level_roc_curve_test.png"))
        mlflow.log_artifact(os.path.join(test_figure_dir, "tta_image_level_roc_curve_test.png"))
        print(f'[TTA] Final Model Image-level Test Accuracy: {accuracy} AUC: {auc}')

    # Patient-Level Evaluation on Test Set
    id_patients, id_patient_logits, id_patient_preds, id_patient_labels = [], [], [], []
    patient_ids = test_dataset.df['patient_id'].to_numpy()
    incorrect_ids = []

    for id in np.unique(patient_ids):
        indices = np.where(patient_ids == id)[0]
        id_patient_logits.append(np.mean(logits[indices, :], axis=0))
        id_patient_labels.append(np.mean(labels[indices], axis=0))
        id_patient_preds.append(id_patient_logits[-1].argmax())
        if id in rechecked_patient_ids:
            print(f"Patient Id (Rechecked): {id}, Ground Truth: {id_patient_labels[-1]}, Predicted: {id_patient_preds[-1]}")
        if id_patient_labels[-1] != id_patient_preds[-1]:
            incorrect_ids.append((id, id_patient_labels[-1], id_patient_preds[-1]))
        id_patients.append(id)

    for id_data in incorrect_ids:
        print(f"Patient Id: {id_data[0]}, Ground Truth: {id_data[1]}, Predicted: {id_data[2]}")

    preds = np.array(id_patient_preds)
    labels = np.array(id_patient_labels)
    logits = np.array(id_patient_logits)

    # Calculate patient-level metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:, 1])

    # Save metrics
    save_metrics_csv("Patient", accuracy, precision, recall, f1, auc, os.path.join(test_root, "metrics.csv"), train=False)

    # Confusion matrix
    confmat_vals = confusion_matrix(labels, preds)
    plot_confusion_matrix(confmat_vals, num_classes, os.path.join(test_figure_dir, "patient_level_conf_mat_test.png"), "Confusion Matrix on Test [Patient level] Final Model")
    mlflow.log_artifact(os.path.join(test_figure_dir, "patient_level_conf_mat_test.png"))

    # ROC curve
    plot_save_roc_curve(np.eye(num_classes)[np.round(labels).astype(int)], logits, os.path.join(test_figure_dir, "patient_level_roc_curve_test.png"))
    mlflow.log_artifact(os.path.join(test_figure_dir, "patient_level_roc_curve_test.png"))
    print(f'[TTA] Final Model Patient-level Test Accuracy: {accuracy} AUC: {auc}')

    # Log all test artifacts to MLflow
    mlflow.log_artifact(os.path.join(test_root, "metrics.csv"))

print("Training and testing complete!")