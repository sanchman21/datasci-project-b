'''
This script consists of some utility functions that are used in the training and evaluation scripts.
'''

import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
import pandas as pd 
from sklearn.metrics import roc_curve, roc_auc_score

# set seed
def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    Also, need to add "worker_init_fn=np.random.seed(seed)" in dataloader
    # https://discuss.pytorch.org/t/determinism-in-pytorch-across-multiple-files/156269
    # https://stackoverflow.com/questions/65685060/unique-seed-acrossing-multiple-imported-files-with-random-module-python
    """
    print(f"Setting seeds: {seed} ...... ")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True
    
def worker_init_fn(worker_id):     
    '''
    this function is for dataloader's worker_init_fn
    '''                                                     
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
# Define a custom dataset class for loading and preprocessing data
def make_weights_for_balanced_classes(labels, device):
    count = torch.bincount(torch.tensor(labels)).to(device)
    print('Count:', count.cpu().detach().numpy())
    
    weight = 1. / count.cpu().detach().numpy()
    print('Data sampling weight:', weight)
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)

    return samples_weight

def convert_path_to_os_specific(path: str) -> str:
    normalized_path = os.path.normpath(path)
    if os.sep == '/':
        return normalized_path.replace('\\', os.sep)
    else:
        return normalized_path.replace('/', os.sep)

def denormalize(image, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    return image * std + mean

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
        
# Function to save or update metrics CSV
def save_metrics_csv(fold, accuracy, precision, recall, f1, auroc, metrics_path, train=True):
    '''
    function: save or update metrics csv
    parameters:
        fold: Union[int, str], fold id (or test type for test metrics)
        accuracy: float, accuracy value
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 score value
        auroc: float, auroc value
        metrics_path: str, path to save the metric
    return: None
    '''
    columns = []
    if train:
        columns = ["fold", "accuracy", "precision", "recall", "f1", "auroc"]
    else:
        columns = ["test type", "accuracy", "precision", "recall", "f1", "auroc"]
    new_metrics = pd.DataFrame([[fold, round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), round(auroc, 3)]], 
                                columns=columns) # create new metrics dataframe

    if os.path.exists(metrics_path): # if the metrics file exists
        df = pd.read_csv(metrics_path) # read the metrics file
        if train and fold in df['fold'].values: # if the fold is already in the metrics file
            df.loc[df['fold'] == fold, 'accuracy'] = round(accuracy, 4) # update the accuracy value
            df.loc[df['fold'] == fold, 'precision'] = round(precision, 4) # update the precision value
            df.loc[df['fold'] == fold, 'recall'] = round(recall, 4) # update the recall value
            df.loc[df['fold'] == fold, 'f1'] = round(f1, 4) # update the f1 score value
            df.loc[df['fold'] == fold, 'auroc'] = round(auroc, 4) # update the auroc value
        elif not train and fold in df["test type"].values:
            df.loc[df['test type'] == fold, 'accuracy'] = round(accuracy, 4) # update the accuracy value
            df.loc[df['test type'] == fold, 'precision'] = round(precision, 4) # update the precision value
            df.loc[df['test type'] == fold, 'recall'] = round(recall, 4) # update the recall value
            df.loc[df['test type'] == fold, 'f1'] = round(f1, 4) # update the f1 score value
            df.loc[df['test type'] == fold, 'auroc'] = round(auroc, 4) # update the auroc value
        else:
            df = pd.concat([df, new_metrics], ignore_index=True) # concatenate the new metrics dataframe with the existing metrics dataframe
    else:
        df = new_metrics # if the metrics file doesn't exist, set the new metrics dataframe as the metrics dataframe
    
    df.to_csv(metrics_path, index=False) # save the metrics dataframe to the metrics file

def plot_save_roc_curve(labels, logits, figure_path):
    '''
    function: plot and save the ROC curve
    parameters:
        labels: int: numpy array, true labels
        logits: float: numpy array, predicted logits
        figure_path: str, path to save the figure
    return: None
    '''
    # Compute ROC curve and ROC area
    if labels.ndim > 1: # if the labels have more than 1 dimension
        labels = labels[:, 1] # set the labels to the second column
        
    if logits.ndim > 1: # if the logits have more than 1 dimension
        logits = logits[:, 1] # set the logits to the second column
        
    fpr, tpr, _ = roc_curve(labels, logits) # get fpr, tpr
    roc_auc = roc_auc_score(labels, logits) # get roc auc score

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
    parameters:
        confmat_vals: numpy array, confusion matrix values
        num_classes: int, number of classes
        figure_path: str, path to save the figure
        title: str, title of the figure
    return: None
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
    
def assign_patient_folds_splits(data_type_path, patient_folds_path="../datasets/patients_fold.csv"):
    '''
    fn: Assigns clincial features splits based on splits made in the image dataset of the particular data type
    '''
    # initialise paths
    monocyte_path = data_type_path
    patient_folds_path = patient_folds_path

    # read csv files
    monocyte_reassigned = pd.read_csv(monocyte_path)
    patient_folds = pd.read_csv(patient_folds_path)

    # set columns
    set_columns = ['set0', 'set1', 'set2', 'set3', 'set4']
    monocyte_reassigned.drop_duplicates(subset=["patient_id"], inplace=True)

    unique_patients_csv1 = list(patient_folds["patient_id"].unique())
    unique_patients_csv2 = list(monocyte_reassigned["patient_id"].unique())

    for patient_id in unique_patients_csv1:
        if patient_id in unique_patients_csv2:
            for col in set_columns:
                patient_folds.loc[patient_folds['patient_id'] == patient_id, col] = monocyte_reassigned.loc[monocyte_reassigned["patient_id"] == patient_id, col].values[0]

    # Save the updated patient_folds.csv
    patient_folds.to_csv(patient_folds_path, index=False)
    print("Updated patient_folds.csv")
    
class Identity(torch.nn.Module):
    '''
    This class is used to extract the features from the model
    '''
    def forward(self, x):
        '''
        function: forward pass
        parameters:
            x: input
        returns: input
        '''
        return x
    
def extract_embeddings(model, loader, dataset, device):
    '''
    function: extracts embeddings from the model
    parameters:
        model: model
        loader: data loader
        dataset: dataset
        device: device
    returns: embeddings, labels, patient_ids, image_paths
    '''
    model.eval() # set the model to evaluation mode
    # initialise empty lists for embeddings and labels
    embeddings = []
    labels = []
    with torch.no_grad(): # turn off gradient computation
        for inputs, lbls in loader: # iterate over the data loader
            # get and move the inputs to the device
            inputs = inputs.to(device).float()
            feats = model(inputs) # get the features from the model
            embeddings.append(feats.cpu().numpy()) # append the features to the embeddings list
            labels.extend(lbls.numpy()) # extend the labels list with the labels
    # concatenate the embeddings and convert the labels to a numpy array
    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)
    # get the patient ids and image paths from the dataset
    patient_ids = dataset.df['patient_id'].to_numpy()
    image_paths = dataset.df['image_path'].to_numpy()
    return embeddings, labels, patient_ids, image_paths # return the embeddings, labels, patient_ids, and image_paths

def extract_logits(model, loader, dataset, device):
    '''
    function: extract logits from the model
    parameters:
        model: trained model
        loader: DataLoader
        dataset: CustomDataset
        device: device to run the model
    return:
        logits: numpy array of logits
        labels: numpy array of labels
        patient_ids: numpy array of patient_ids
    '''
    model.eval() # set the model to evaluation mode
    # initialise empty lists for logits and labels
    logits = []
    labels = []
    with torch.no_grad(): # disable gradient calculation
        for inputs, lbls in loader: # iterate over the loader
            inputs = inputs.to(device).float() # get the inputs
            outputs = model(inputs) # get the outputs
            outputs = torch.nn.functional.softmax(outputs, dim=-1) # apply softmax
            logits.append(outputs.cpu().numpy()) # append the outputs to logits
            labels.extend(lbls.numpy()) # extend the labels
    # concatenate the logits and convert labels to numpy array
    logits = np.concatenate(logits)
    labels = np.array(labels)
    patient_ids = dataset.df['patient_id'].to_numpy() # get the patient_ids
    patient_ids = patient_ids.astype(str) # convert to string
    return logits, labels, patient_ids # return logits, labels, patient_ids