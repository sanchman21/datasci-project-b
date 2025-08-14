import yaml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def convert_path_to_os_specific(path: str) -> str:
    normalized_path = os.path.normpath(path)
    if os.sep == '/':
        return normalized_path.replace('\\', os.sep)
    else:
        return normalized_path.replace('/', os.sep)


def compute_metrics(cm, all_labels, all_preds) -> dict:
    TN, FP, FN, TP = cm.ravel()
    selectivity = TN / (TN + FP)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    specificity = selectivity

    return {
        'selectivity': selectivity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'specificity': specificity
    }

def save_metrics_to_yaml(average_metrics: dict, directory: str) -> None:
    formatted_metrics = {}
    for key, value in average_metrics.items():
        mean = value['mean'] * 100
        std = value['std'] * 100
        formatted_metrics[key] = f"{mean:.2f}% ± {std:.2f}%"

    yaml_path = os.path.join(directory, 'final_metrics_results.yaml')
    os.makedirs(directory, exist_ok=True)
    with open(yaml_path, 'w') as file:
        yaml.dump(formatted_metrics, file)


def load_config(path: str) -> dict: 
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def plot_and_save_confusion_matrix(fold: int, cm, classes: list, dir: str) -> None:
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

def plot_metrics(metrics: dict, fold: int, save_dir: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accuracy'], label='Train Accuracy')
    plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plot_filename = f"fold_{fold}.png"
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

def log_confusion_matrix(labels, predictions, fold: int, cm_dir: str):
    matrix_save_path = os.path.join(cm_dir, f"confusion_matrix_fold_{fold}.csv")
    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(cm, index=[i for i in range(len(set(labels)))],
                        columns=[i for i in range(len(set(predictions)))])
    df_cm.to_csv(matrix_save_path) 