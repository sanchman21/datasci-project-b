'''
This script consists of some utility functions that are used in the training and evaluation scripts.
'''

import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pandas as pd

import numpy as np
import os

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import os

def convert_path_to_os_specific(path: str) -> str:
    """
    Function: Convert a file path string to the current OS's path format.
    Parameters:
        path (str): The original file path string, which might contain either forward slashes (/) or backslashes (\).

    Returns:
        str: The converted file path compatible with the current operating system.
    """
    # Normalize path to remove redundant separators and up-level references
    normalized_path = os.path.normpath(path)

    # Replace separators according to the current OS
    if os.sep == '/':
        # If the OS separator is '/', replace all '\\' with '/'
        return normalized_path.replace('\\', os.sep)
    else:
        # If the OS separator is '\', replace all '/' with '\\'
        return normalized_path.replace('/', os.sep)


def compute_metrics(cm, all_labels, all_preds) -> dict:
    '''
    Function: compute_metrics
    Description: Compute the metrics from the confusion matrix
    Parameters:
        cm (array): The confusion matrix
        all_labels (array): The true labels
        all_preds (array): The predicted labels
    Returns:
        dict: A dictionary containing the computed metrics
    '''
    TN, FP, FN, TP = cm.ravel() # Unravel the confusion matrix

    selectivity = TN / (TN + FP) # Compute the selectivity
    precision = precision_score(all_labels, all_preds) # Compute the precision
    recall = recall_score(all_labels, all_preds) # Compute the recall
    f1 = f1_score(all_labels, all_preds) # Compute the F1 score
    accuracy = accuracy_score(all_labels, all_preds) # Compute the accuracy
    specificity = selectivity  # Selectivity and Specificity are same in binary classify

    # Return a dictionary of the computed metrics
    return {
        'selectivity': selectivity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity
    }

def save_metrics_to_yaml(average_metrics: dict, directory: str) -> None:
    '''
    Function: saves calculated metrics to a yaml file
    Parameters:
        average_metrics (dict): A dictionary containing the average metrics
        directory (str): The directory to save the YAML file
    Returns: None
    '''
    formatted_metrics = {} # Initialize an empty dictionary to store the formatted metrics
    for key, value in average_metrics.items(): # Iterate over the average metrics
        mean = value['mean'] * 100 # Compute the mean and convert to percentage
        std = value['std'] * 100 # Compute the standard deviation and convert to percentage
        formatted_metrics[key] = f"{mean:.2f}% ± {std:.2f}%" # Format the metrics as a string

    yaml_path = os.path.join(directory, 'final_metrics_results.yaml') # Construct the file path for the YAML file
    os.makedirs(directory, exist_ok=True) # Create the directory if it does not exist

    with open(yaml_path, 'w') as file: # Open the file in write mode
        yaml.dump(formatted_metrics, file) # Write the formatted metrics to the file


def load_config(path: str) -> dict: 
    '''
    Function: Load the configuration from a YAML file
    Parameters:
        path (str): The path to the YAML file
    Returns:
        dict: The configuration loaded from the YAML file
    '''
    with open(path, 'r') as file: # Open the file in read mode
        return yaml.safe_load(file) # Load and return the configuration from the file


def plot_and_save_confusion_matrix(fold: int, cm, classes: list, dir: str) -> None:
    '''
    Function: Given the confusion matrix, plot and save it
    Parameters:
        fold (int): The fold number
        cm (array): The confusion matrix
        classes (list): The list of class names
        dir (str): The directory to save the confusion matrix plot
    '''
    fig, ax = plt.subplots() # Create a new figure (subplots) and axis
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues')) # Plot the confusion matrix as an image
    ax.figure.colorbar(im, ax=ax) # Add a color bar to the plot
    
    # set the axis ticks, labels, and plot title
    ax.set(xticks=np.arange(cm.shape[1]), # Set the x-ticks
           yticks=np.arange(cm.shape[0]), # Set the y-ticks
           xticklabels=classes, yticklabels=classes, # Set the x and y tick labels
           title='Confusion Matrix', # Set the title
           ylabel='True label', # Set the y-label
           xlabel='Predicted label') # Set the x-label
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor") # Rotate the x-tick labels
    
    fmt = 'd' # Set the format of the text in the plot
    thresh = cm.max() / 2. # Compute the threshold for the text color
    for i in range(cm.shape[0]): # Iterate over the rows of the confusion matrix
        for j in range(cm.shape[1]): # Iterate over the columns of the confusion matrix
            ax.text(j, i, format(cm[i, j], fmt), # Add the text to the plot
                    ha="center", va="center", # Set the horizontal and vertical alignment
                    color="white" if cm[i, j] > thresh else "black") # Set the text color based on the threshold
    fig.tight_layout() # Adjust the layout of the plot
    
    file_path = os.path.join(dir, f'confusion_matrix_fold_{fold}.png') # File path to save the plot
    plt.savefig(file_path) # Save the plot
    plt.close() # Close the plot to free up memory

def plot_metrics(metrics: dict, fold: int, save_dir: str) -> None:
    '''
    Function: Plot the training and validation metrics and save the plot
    Parameters:
        metrics (dict): A dictionary containing the training and validation metrics
        fold (int): The fold number
        save_dir (str): The directory to save
    Returns: None
    '''
    plt.figure(figsize=(10, 4)) # Create a new figure with a specific size

    plt.subplot(1, 2, 1) # Create a subplot (1 row, 2 columns, 1st plot)
    plt.plot(metrics['train_loss'], label='Train Loss') # Plot the training loss
    plt.plot(metrics['val_loss'], label='Validation Loss') # Plot the validation loss
    plt.title('Training and Validation Loss') # Set the title
    plt.xlabel('Epochs') # Set the x-label
    plt.ylabel('Loss') # Set the y-label
    plt.legend() # Add the legend

    plt.subplot(1, 2, 2) # Create a subplot (1 row, 2 columns, 2nd plot)
    plt.plot(metrics['train_accuracy'], label='Train Accuracy') # Plot the training accuracy
    plt.plot(metrics['val_accuracy'], label='Validation Accuracy') # Plot the validation accuracy
    plt.title('Training and Validation Accuracy') # Set the title
    plt.xlabel('Epochs') # Set the x-label
    plt.ylabel('Accuracy') # Set the y-label
    plt.legend() # Add the legend

    plt.tight_layout() # Adjust the layout of the plot

    # Construct the file path for the plot
    plot_filename = f"fold_{fold}.png"  # epoch+1 because epoch starts at 0
    plot_path = os.path.join(save_dir, plot_filename) # Construct the file path for the plot using os
    plt.savefig(plot_path) # Save the plot
    # print(f"Plot saved: {plot_path}")
    plt.close()  # Close the plot to free up memory

def log_confusion_matrix(labels, predictions, fold: int, cm_dir: str):
    '''
    function: save confusion matrix to csv
    Parameters:
        labels (array): The true labels
        predictions (array): The predicted labels
        fold (int): The fold number
        cm_dir (str): The directory
    Returns: None
    ''' 
    matrix_save_path = os.path.join(cm_dir, f"confusion_matrix_fold_{fold}.csv") # Construct the file path for the confusion matrix using os
    cm = confusion_matrix(labels, predictions) # Compute the confusion matrix
    df_cm = pd.DataFrame(cm, index=[i for i in range(len(set(labels)))],
                        columns=[i for i in range(len(set(predictions)))]) # create a dataframe of the confusion matrix
    df_cm.to_csv(matrix_save_path) # Save the confusion matrix to a CSV file 