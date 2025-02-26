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
def save_metrics_csv(fold, accuracy, precision, recall, f1, auroc, metrics_path):
    '''
    function: save or update metrics csv
    parameters:
        fold: int, fold id
        accuracy: float, accuracy value
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 score value
        auroc: float, auroc value
        metrics_path: str, path to save the metric
    return: None
    '''
    new_metrics = pd.DataFrame([[fold, round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), round(auroc, 3)]], 
                                columns=["fold", "accuracy", "precision", "recall", "f1", "auroc"]) # create new metrics dataframe

    if os.path.exists(metrics_path): # if the metrics file exists
        df = pd.read_csv(metrics_path) # read the metrics file
        if fold in df['fold'].values: # if the fold is already in the metrics file
            df.loc[df['fold'] == fold, 'accuracy'] = round(accuracy, 4) # update the accuracy value
            df.loc[df['fold'] == fold, 'precision'] = round(precision, 4) # update the precision value
            df.loc[df['fold'] == fold, 'recall'] = round(recall, 4) # update the recall value
            df.loc[df['fold'] == fold, 'f1'] = round(f1, 4) # update the f1 score value
            df.loc[df['fold'] == fold, 'auroc'] = round(auroc, 4) # update the auroc value
        else:
            df = pd.concat([df, new_metrics], ignore_index=True) # concatenate the new metrics dataframe with the existing metrics dataframe
    else:
        df = new_metrics # if the metrics file doesn't exist, set the new metrics dataframe as the metrics dataframe
    
    df.to_csv(metrics_path, index=False) # save the metrics dataframe to the metrics file

def plot_save_roc_curve(labels, logits, figure_path):
    '''
    function: plot and save the ROC curve
    parameters:
        labels: numpy array, true labels
        logits: numpy array, predicted logits
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