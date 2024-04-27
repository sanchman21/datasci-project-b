import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch import nn, optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
from MergeMasterDataset import MergeMasterDataset

def train_model(csv_file, num_epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=5e-4, momentum=0.9):
    
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    num_folds = 5
    all_metrics = []

    for fold in range(num_folds):
        print(f"Starting training for fold {fold+1}/{num_folds}")
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
        # model = model.cuda()
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

        train_dataset = MergeMasterDataset(csv_file, fold=fold, train=True, transform=transform)
        val_dataset = MergeMasterDataset(csv_file, fold=fold, train=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        for epoch in range(num_epochs):
            torch.cuda.empty_cache() #clear memory at start of every epoch

            print(f"Starting epoch {epoch+1}/{num_epochs} for fold {fold+1}")
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"Completed epoch {epoch+1}/{num_epochs} for fold {fold+1}")

        # Evaluate the model on the validation set
        print(f"Starting evaluation for fold {fold+1}")
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())

        # Calculate metrics
        precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        metrics = {
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'accuracy': accuracy,
            'auc': auc,
            'confusion_matrix': cm
        }
        all_metrics.append(metrics)
        print(f"Completed evaluation for fold {fold+1}")

    # Aggregate metrics across folds
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    std_metrics = {key: np.std([m[key] for m in all_metrics]) for key in all_metrics[0]}
    
    print("Classification Metrics:")
    print("Precision: {:.2f} ± {:.2f}".format(avg_metrics['precision'] * 100, std_metrics['precision'] * 100))
    print("Recall: {:.2f} ± {:.2f}".format(avg_metrics['recall'] * 100, std_metrics['recall'] * 100))
    print("F1 Score: {:.2f} ± {:.2f}".format(avg_metrics['fscore'] * 100, std_metrics['fscore'] * 100))
    print("Accuracy: {:.2f} ± {:.2f}".format(avg_metrics['accuracy'] * 100, std_metrics['accuracy'] * 100))
    print("AUC: {:.2f} ± {:.2f}".format(avg_metrics['auc'] * 100, std_metrics['auc'] * 100))
    print("Confusion Matrix:")
    print("TP: {:.2f} ± {:.2f}, TN: {:.2f} ± {:.2f}, FP: {:.2f} ± {:.2f}, FN: {:.2f} ± {:.2f}".format(
        np.mean([m['confusion_matrix'][1, 1] for m in all_metrics]),
        np.std([m['confusion_matrix'][1, 1] for m in all_metrics]),
        np.mean([m['confusion_matrix'][0, 0] for m in all_metrics]),
        np.std([m['confusion_matrix'][0, 0] for m in all_metrics]),
        np.mean([m['confusion_matrix'][0, 1] for m in all_metrics]),
        np.std([m['confusion_matrix'][0, 1] for m in all_metrics]),
        np.mean([m['confusion_matrix'][1, 0] for m in all_metrics]),
        np.std([m['confusion_matrix'][1, 0] for m in all_metrics])
    ))
    return all_metrics

if __name__ == '__main__':
    csv_file_path = 'merged_master.csv'
    train_model(csv_file_path)
