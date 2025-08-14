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
import subprocess

sys.path.append('../townim') 
import utils
from utils import FixedRotation, plot_confusion_matrix, plot_save_roc_curve, save_metrics_csv
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH, MONOCYTE_NEW_NORMALS_CSV_PATH, SEGMENTED_MONOCYTE_CSV_PATH, SEGMENTED_MONOCYTE_NEW_NORMALS_CSV_PATH

cache_dir = "../cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir
mlflow.set_tracking_uri("file:./mlruns")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = "cuda"
elif torch.backends.mps.is_available():
    torch.mps.empty_cache() 
    device = "mps" 
else: 
    device = "cpu"
print(f"Using device: {device}") 

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default="monocyte_new_normals", help="image type to train the model on", 
                    choices=('neutrophil', 'monocyte', 'monocyte_new_normals', 'segmented_monocyte', 'segmented_monocyte_new_normals'))
args = parser.parse_args()
data_type = str(args.data_type)
final_epochs = 0
if data_type == 'neutrophil':
    CSV_PATH = NEUTROPHIL_CSV_PATH
elif data_type == 'monocyte':
    CSV_PATH = MONOCYTE_CSV_PATH
elif data_type == 'monocyte_new_normals':
    CSV_PATH = MONOCYTE_NEW_NORMALS_CSV_PATH
elif data_type == 'segmented_monocyte':
    CSV_PATH = SEGMENTED_MONOCYTE_CSV_PATH
elif data_type == 'segmented_monocyte_new_normals':
    CSV_PATH = SEGMENTED_MONOCYTE_NEW_NORMALS_CSV_PATH
else:
    raise ValueError("Invalid data type")

yaml_dir = f'./experiments/{data_type}/config.yaml'
with open(yaml_dir, 'r') as file:
    config = yaml.safe_load(file) 

existing_experiment = mlflow.get_experiment_by_name(data_type)
if existing_experiment is not None:
    experiment_id = existing_experiment.experiment_id 
    overwrite_exp = input(f"DO YOU WANT TO OVERWRITE EXISTING {data_type} EXPERIMENT? [Y/N]")
    if overwrite_exp.lower() == "y":
        mlflow.delete_experiment(experiment_id)
        subprocess.run(["mlflow", "gc", "--experiment-ids", experiment_id], check=True)
    else:
        print("To run the code further, you need to overwrite existing experiment. Please modify code otherwise.")
        exit()
    
mlflow.create_experiment(data_type) 
mlflow.set_experiment(experiment_name=data_type)
print(f"Created new experiment for {data_type}")

root = f"./experiments/{data_type}/train"
os.makedirs(root, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
num_epochs = config['num_epochs']
batch_size = config['batch_size']
IMAGE_SIZE = config['image_size']
use_scheduler = config['use_scheduler']
lr = config['lr']
weight_decay = config['weight_decay']
momentum = config['momentum']
start_factor = config['start_factor']
end_factor = config['end_factor']
patience = config['patience'] 

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

with mlflow.start_run(run_name="train-val") as parent_run:
    mlflow.log_artifact(yaml_dir)
    mlflow.log_params(config)

    for set_id in range(5):
        print(f"Fold: {set_id}, Data type: {data_type}")
        
        best_test_acc = 0
        best_epoch = 0
        train_losses = [] 
        val_losses = [] 
        train_accuracies = []
        val_accuracies = []
        logs = ''

        output_dir = root + f'/{data_type}_fold_{set_id}_with_TTA'
        model_dir = output_dir + "/model"
        figure_dir = output_dir + "/figures"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)
            
        shutil.copyfile('./trainer.py', os.path.join(output_dir, 'trainer.py'))
        utils.set_random_seed(420)

        test_dataset = CustomDataset('val', CSV_PATH, set_id, transform=test_transform)
        test_loaders = [
            DataLoader(CustomDataset('val', CSV_PATH, set_id, transform=transform), batch_size=8, shuffle=False, pin_memory=True, num_workers=4)
            for transform in TTAs
        ]

        train_dataset = CustomDataset('train', CSV_PATH, set_id, transform=train_transform)
        print("Training dataset stats [Normal, CMML]:", train_dataset.disease_count)
        val_dataset = CustomDataset('val', CSV_PATH, set_id, transform=test_transform)
        print("Test dataset stats [Normal, CMML]:", val_dataset.disease_count)
               
        weights = utils.make_weights_for_balanced_classes(train_dataset.labels, device)
        weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                                sampler=weighted_sampler,
                                num_workers=8, worker_init_fn=utils.worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

        num_classes = len(set(train_dataset.labels))
        model = models.resnet50(weights='IMAGENET1K_V1')
        hidden_layer_size = 512
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, num_classes)
        )

        model = model.to(device)

        count = torch.bincount(torch.tensor(train_dataset.labels)).to(device)
        class_weight = len(train_dataset.labels) / count

        print('Loss class weight:', class_weight)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=num_epochs)

        metrics_data = []
        best_val_loss = float('inf')
        best_test_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        early_stop = False

        with mlflow.start_run(run_name=f"fold_{set_id}", nested=True):
            mlflow.set_tag("train_normal_count", train_dataset.disease_count[0])
            mlflow.set_tag("train_cmml_count", train_dataset.disease_count[1])
            mlflow.set_tag("val_normal_count", val_dataset.disease_count[0])
            mlflow.set_tag("val_cmml_count", val_dataset.disease_count[1])

            for epoch in range(num_epochs): 
                if early_stop: 
                    break

                t = tqdm(enumerate(train_loader, 0), total=len(train_loader),
                        smoothing=0.9, position=0, leave=True,
                        desc="Train: Epoch: " + str(epoch + 1) + "/" + str(num_epochs))
                model.train()
                running_loss = 0.0
                all_preds, all_labels = [], []

                for i, (inputs, labels) in t:
                    inputs = inputs.to(device).float()
                    labels = labels.to(device).long()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    outputs = F.softmax(outputs, dim=-1)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

                train_loss = running_loss / len(train_loader)
                train_accuracy = accuracy_score(all_labels, all_preds)
                train_precision = precision_score(all_labels, all_preds, average='binary')
                train_recall = recall_score(all_labels, all_preds, average='binary')
                train_f1 = f1_score(all_labels, all_preds, average='binary')
                train_auroc = roc_auc_score(all_labels, all_preds)

                model.eval()
                val_loss = 0.0
                val_preds, val_labels, val_probs = [], [], []

                with torch.no_grad():
                    
                    t = tqdm(enumerate(val_loader, 0), total=len(val_loader),
                            smoothing=0.9, position=0, leave=True,
                            desc="Val: Epoch: " + str(epoch + 1) + "/" + str(num_epochs))
                    
                    for i, (inputs, labels) in t: 
                        inputs, labels = inputs.to(device).float(), labels.to(device).long()
                        outputs = model(inputs) 
                        loss = F.cross_entropy(outputs, labels)
                        val_loss += loss.item()
                        outputs = F.softmax(outputs, dim=-1)
                        preds = outputs.argmax(dim=1).cpu().numpy()
                        probs = outputs[:, 1].cpu().numpy()
                        val_preds.extend(preds)
                        val_labels.extend(labels.cpu().numpy())
                        val_probs.extend(probs)

                val_loss = val_loss / len(val_loader)
                val_accuracy = accuracy_score(val_labels, val_preds)
                val_precision = precision_score(val_labels, val_preds, average='binary')
                val_recall = recall_score(val_labels, val_preds, average='binary')
                val_f1 = f1_score(val_labels, val_preds, average='binary')
                val_auroc = roc_auc_score(val_labels, val_preds)

                if use_scheduler:
                    scheduler.step()
                    
                print(f"Epoch: {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}")
                
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

                if val_loss < best_val_loss: 
                    best_val_loss = val_loss
                    best_epoch = epoch + 1 
                    epochs_no_improve = 0 
                    torch.save(model.state_dict(), os.path.join(model_dir, f'model_fold_{set_id}.pth'))
                else:
                    epochs_no_improve += 1 
                    if epochs_no_improve == patience:
                        early_stop = True
                        print("Early Stopping....")
                        break 

            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("best_epoch", best_epoch)

            final_epochs += best_epoch

            df = pd.DataFrame(metrics_data, columns=['Epoch', 'LR', 'Train Loss', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1', 'Train AUROC', 
                                                        'Val Loss', 'Val Accuracy', 'Val Precision', 'Val Recall', 'Val F1', 'Val AUROC'])
            df.to_csv(os.path.join(output_dir, f'train_time_metrics_fold_{set_id}.csv'), index=False)

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
            
            print("Model: Resnet50")
            model = models.resnet50(weights='IMAGENET1K_V1')
            hidden_layer_size = 512
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, hidden_layer_size),
                nn.ReLU(),
                nn.Linear(hidden_layer_size, num_classes)
            )

            model = model.to(device)
            model.load_state_dict(torch.load(os.path.join(model_dir, f'model_fold_{set_id}.pth')))
            model.eval() 

            with torch.no_grad():
                correct = 0 
                preds = []
                labels = []
                logits = []
                
                for i, (inputs, targets) in enumerate(val_loaders[0]): 
                    inputs, targets = inputs.to(device).float(), targets.to(device).long() 
                    outputs = model(inputs)
                    outputs = F.softmax(outputs, dim=-1)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    preds.append(predicted.detach().cpu().numpy())
                    labels.append(targets.detach().cpu().numpy())
                    logits.append(outputs.detach().cpu().numpy().astype(np.float32))

                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0) 
                logits = np.concatenate(logits, axis=0) 

                accuracy = accuracy_score(labels, preds)
                precision = precision_score(labels, preds, average="binary")
                recall = recall_score(labels, preds, average="binary")
                f1 = f1_score(labels, preds, average="binary")
                auc = roc_auc_score(labels, logits[:, 1])

                save_metrics_csv(set_id, accuracy, precision, recall, f1, auc, os.path.join(root, "metrics_image.csv"))

                confmat_vals = confusion_matrix(labels, preds)
                plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, f"image_level_conf_mat_fold_{set_id}.png"), f"Confusion Matrix on Test [Image level] Fold: {set_id}")
                mlflow.log_artifact(os.path.join(figure_dir, f"image_level_conf_mat_fold_{set_id}.png"))

                plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, f"image_level_roc_curve_fold_{set_id}.png"))
                mlflow.log_artifact(os.path.join(figure_dir, f"image_level_roc_curve_fold_{set_id}.png"))
                print(f'[W/O TTA] Image level Accuracy (Fold {set_id}): {accuracy} AUC: {auc}')
                
                preds = []
                logits = [] 
                labels = []
                correct = 0 
                
                for i, data in enumerate(zip(*test_loaders)): 
                    inputs, targets = torch.cat([img for img,_ in data], dim=0).to(device).float(), data[0][1].to(device).long() 
                    outputs = model(inputs) 
                    outputs = F.softmax(outputs, dim=-1) 
                    outputs = outputs.reshape(len(data), int(inputs.shape[0]/len(data)), -1).mean(dim=0)

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    preds.append(predicted.detach().cpu().numpy()) 
                    labels.append(targets.detach().cpu().numpy())
                    logits.append(outputs.detach().cpu().numpy().astype(np.float32)) 

                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0) 
                logits = np.concatenate(logits, axis=0) 

                accuracy = accuracy_score(labels, preds)
                precision = precision_score(labels, preds, average="binary")
                recall = recall_score(labels, preds, average="binary")
                f1 = f1_score(labels, preds, average="binary")
                auc = roc_auc_score(labels, logits[:, 1])

                save_metrics_csv(set_id, accuracy, precision, recall, f1, auc, os.path.join(root, "metrics_image_tta.csv"))

                confmat_vals = confusion_matrix(labels, preds)
                plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, f"tta_image_level_conf_mat_fold_{set_id}.png"), f"Confusion Matrix on Test [Image level with TTA] Fold: {set_id}")
                mlflow.log_artifact(os.path.join(figure_dir, f"tta_image_level_conf_mat_fold_{set_id}.png"))

                plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(figure_dir, f"tta_image_level_roc_curve_fold_{set_id}.png"))
                mlflow.log_artifact(os.path.join(figure_dir, f"tta_image_level_roc_curve_fold_{set_id}.png"))
                print(f'[TTA] Image level Accuracy (Fold: {set_id}): {accuracy} AUC: {auc}')
                
                np.savez_compressed(os.path.join(model_dir, f'image_level_results_fold_{set_id}'),
                        labels=labels, 
                        preds=preds,
                        logits=logits,
                )

            id_patients = []
            id_patient_logits = []
            id_patient_preds = []
            id_patient_labels = []
            patient_ids = test_dataset.df['patient_id'].to_numpy()
            correct = 0 
            incorrect_ids = []

            for id in np.unique(patient_ids): 
                indices = np.where(patient_ids==id)[0] 
                id_patient_logits.append(np.mean(logits[indices,:], axis=0))
                id_patient_labels.append(np.mean(labels[indices], axis=0))
                id_patient_preds.append(id_patient_logits[-1].argmax()) 
                if id in rechecked_patient_ids: 
                    print(f"Patient Id (Rechecked): {id}, Ground Truth: {id_patient_labels[-1]}, Predicted: {id_patient_preds[-1]}")
                if id_patient_labels[-1] != id_patient_preds[-1]:
                    incorrect_ids.append((id, id_patient_labels[-1], id_patient_preds[-1]))
                correct += int(id_patient_preds[-1]==id_patient_labels[-1])
                id_patients.append(id)
                
            for id_data in incorrect_ids: 
                print(f"Patient Id: {id_data[0]}, Ground Truth: {id_data[1]}, Predicted: {id_data[2]}")

            preds = np.array(id_patient_preds)
            labels = np.array(id_patient_labels) 
            logits = np.array(id_patient_logits)

            accuracy = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, average="binary") 
            recall = recall_score(labels, preds, average="binary")
            f1 = f1_score(labels, preds, average="binary")
            auc = roc_auc_score(labels, logits[:, 1])

            save_metrics_csv(set_id, accuracy, precision, recall, f1, auc, os.path.join(root, "metrics_patient.csv"))

            confmat_vals = confusion_matrix(labels, preds)
            plot_confusion_matrix(confmat_vals, num_classes, os.path.join(figure_dir, f"patient_level_conf_mat_fold_{set_id}.png"), f"Confusion Matrix on Test [Patient level] Fold: {set_id}")
            mlflow.log_artifact(os.path.join(figure_dir, f"patient_level_conf_mat_fold_{set_id}.png"))
            
            plot_save_roc_curve(np.eye(num_classes)[np.round(labels).astype(int)], logits, os.path.join(figure_dir, f"patient_level_roc_curve_fold_{set_id}.png"))
            mlflow.log_artifact(os.path.join(figure_dir, f"patient_level_roc_curve_fold_{set_id}.png"))
            print(f'[TTA] Patient level Accuracy (Fold: {set_id}): {accuracy} AUC: {auc}')

    mlflow.log_artifact(os.path.join(root, "metrics_image.csv"))
    mlflow.log_artifact(os.path.join(root, "metrics_image_tta.csv"))
    mlflow.log_artifact(os.path.join(root, "metrics_patient.csv"))

trainval_dataset = ConcatDataset([
    CustomDataset('train', CSV_PATH, set_id, transform=train_transform),
    CustomDataset('val', CSV_PATH, set_id, transform=train_transform)
])
all_labels = train_dataset.labels + val_dataset.labels 
weights = utils.make_weights_for_balanced_classes(all_labels, device) 
weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights)) 
trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                             sampler=weighted_sampler, num_workers=8, worker_init_fn=utils.worker_init_fn) 

final_model_dir = os.path.join(root, "model")
os.makedirs(final_model_dir, exist_ok=True)
final_model = models.resnet50(weights='IMAGENET1K_V1') 
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

    torch.save(final_model.state_dict(), os.path.join(final_model_dir, 'final.pth'))
    mlflow.log_artifact(os.path.join(final_model_dir, 'final.pth'))

    test_root = f"./experiments/{data_type}/test"
    os.makedirs(test_root, exist_ok=True)
    test_figure_dir = os.path.join(test_root, "figures")
    os.makedirs(test_figure_dir, exist_ok=True)

    test_dataset = CustomDataset('test', CSV_PATH, set_id, transform=test_transform)
    test_loaders = [
        DataLoader(CustomDataset('test', CSV_PATH, set_id, transform=transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
        for transform in TTAs
    ]
    print("Test dataset stats [Normal, CMML]:", test_dataset.disease_count)

    final_model.eval()

    with torch.no_grad():
        preds, labels, logits = [], [], []
        for i, (inputs, targets) in enumerate(test_loaders[0]):
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

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")
        f1 = f1_score(labels, preds, average="binary")
        auc = roc_auc_score(labels, logits[:, 1])

        save_metrics_csv("Image W/O TTA", accuracy, precision, recall, f1, auc, os.path.join(test_root, "metrics.csv"), train=False)

        confmat_vals = confusion_matrix(labels, preds)
        plot_confusion_matrix(confmat_vals, num_classes, os.path.join(test_figure_dir, "image_level_conf_mat_test.png"), "Confusion Matrix on Test [Image level] Final Model")
        mlflow.log_artifact(os.path.join(test_figure_dir, "image_level_conf_mat_test.png"))

        plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(test_figure_dir, "image_level_roc_curve_test.png"))
        mlflow.log_artifact(os.path.join(test_figure_dir, "image_level_roc_curve_test.png"))
        print(f'[W/O TTA] Final Model Image-level Test Accuracy: {accuracy} AUC: {auc}')

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

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")
        f1 = f1_score(labels, preds, average="binary")
        auc = roc_auc_score(labels, logits[:, 1])

        save_metrics_csv("Image TTA", accuracy, precision, recall, f1, auc, os.path.join(test_root, "metrics.csv"), train=False)

        confmat_vals = confusion_matrix(labels, preds)
        plot_confusion_matrix(confmat_vals, num_classes, os.path.join(test_figure_dir, "tta_image_level_conf_mat_test.png"), "Confusion Matrix on Test [Image level with TTA] Final Model")
        mlflow.log_artifact(os.path.join(test_figure_dir, "tta_image_level_conf_mat_test.png"))

        plot_save_roc_curve(np.eye(num_classes)[labels], logits, os.path.join(test_figure_dir, "tta_image_level_roc_curve_test.png"))
        mlflow.log_artifact(os.path.join(test_figure_dir, "tta_image_level_roc_curve_test.png"))
        print(f'[TTA] Final Model Image-level Test Accuracy: {accuracy} AUC: {auc}')

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

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="binary")
    recall = recall_score(labels, preds, average="binary")
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, logits[:, 1])

    save_metrics_csv("Patient", accuracy, precision, recall, f1, auc, os.path.join(test_root, "metrics.csv"), train=False)

    confmat_vals = confusion_matrix(labels, preds)
    plot_confusion_matrix(confmat_vals, num_classes, os.path.join(test_figure_dir, "patient_level_conf_mat_test.png"), "Confusion Matrix on Test [Patient level] Final Model")
    mlflow.log_artifact(os.path.join(test_figure_dir, "patient_level_conf_mat_test.png"))

    plot_save_roc_curve(np.eye(num_classes)[np.round(labels).astype(int)], logits, os.path.join(test_figure_dir, "patient_level_roc_curve_test.png"))
    mlflow.log_artifact(os.path.join(test_figure_dir, "patient_level_roc_curve_test.png"))
    print(f'[TTA] Final Model Patient-level Test Accuracy: {accuracy} AUC: {auc}')

    mlflow.log_artifact(os.path.join(test_root, "metrics.csv"))

print("Training and testing complete!")