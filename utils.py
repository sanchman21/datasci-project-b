import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import seaborn as sns
import pandas as pd

import numpy as np
import os



def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def plot_and_save_confusion_matrix(fold, cm, classes, dir):
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    file_path = os.path.join(dir, f'confusion_matrix_fold_{fold}.png')
    plt.savefig(file_path)
    plt.close()


def plot_metrics(metrics, fold, save_dir):
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
    plot_filename = f"fold_{fold}.png"  # epoch+1 because epoch starts at 0
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    # print(f"Plot saved: {plot_path}")
    plt.close()  # Close the plot to free up memory

def log_confusion_matrix(labels, predictions, fold, cm_dir):
    matrix_save_path = os.path.join(cm_dir, f"confusion_matrix_fold_{fold}.csv")
    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(cm, index=[i for i in range(len(set(labels)))],
                        columns=[i for i in range(len(set(predictions)))])
    df_cm.to_csv(matrix_save_path)