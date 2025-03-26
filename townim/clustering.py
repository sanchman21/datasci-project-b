import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from dataset import CustomDataset, MONOCYTE_CSV_PATH

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
test_transform = T.Compose([
    T.Resize((352, 352)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Identity layer to extract embeddings
class Identity(nn.Module):
    def forward(self, x):
        return x

# Function to extract embeddings
def extract_embeddings(model, loader, dataset, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for inputs, lbls in loader:  # Unpack only two values
            inputs = inputs.to(device).float()
            feats = model(inputs)
            embeddings.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)
    # Get patient IDs from the dataset's DataFrame
    patient_ids = dataset.df['patient_id'].to_numpy()
    return embeddings, labels, patient_ids

# Clustering for Monocyte Fold 3
data_type = 'monocyte'
set_id = 3
CSV_PATH = MONOCYTE_CSV_PATH

print(f"Clustering for {data_type} Fold {set_id}...")

# Load datasets
train_dataset = CustomDataset('train', CSV_PATH, set_id, transform=test_transform)
val_dataset = CustomDataset('val', CSV_PATH, set_id, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)

# Load fold 3 model
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features  # 2048 for ResNet-50
hidden_layer_size = 512
num_classes = 2  # Normal and CMML

# Recreate the original fc layer as it was during training
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, num_classes)
)

# Load the state dictionary
model_path = f"experiments/{data_type}/train/{data_type}_fold_{set_id}_with_TTA/model/model_fold_{set_id}.pth"
model.load_state_dict(torch.load(model_path))

# Now set fc to Identity to extract embeddings
model.fc = Identity()
model = model.to(device)

# Extract embeddings
train_embeddings, train_labels, train_patient_ids = extract_embeddings(model, train_loader, train_dataset, device)
val_embeddings, val_labels, val_patient_ids = extract_embeddings(model, val_loader, val_dataset, device)

# Cluster on train embeddings with K-means
kmeans = KMeans(n_clusters=2, random_state=123)
train_clusters = kmeans.fit_predict(train_embeddings)

# Assign val embeddings
val_clusters = kmeans.predict(val_embeddings)

# Determine majority label per cluster
majority_labels = []
for cluster in range(2):
    mask = train_clusters == cluster
    majority = np.bincount(train_labels[mask].astype(int)).argmax()
    majority_labels.append(majority)
majority_labels = np.array(majority_labels)
print(f"Cluster 0 majority label: {majority_labels[0]} (0=Normal, 1=CMML)")
print(f"Cluster 1 majority label: {majority_labels[1]} (0=Normal, 1=CMML)")

# Check misclustered val points
val_assigned_labels = majority_labels[val_clusters]
val_misclustered = val_assigned_labels != val_labels

# Visualize with t-SNE
tsne = TSNE(n_components=2, random_state=123)
embeddings_2d = tsne.fit_transform(np.vstack([train_embeddings, val_embeddings]))
train_2d = embeddings_2d[:len(train_embeddings)]
val_2d = embeddings_2d[len(train_embeddings):]

# Create DataFrame for Plotly
df = pd.DataFrame({
    'x': np.concatenate([train_2d[:, 0], val_2d[:, 0]]),
    'y': np.concatenate([train_2d[:, 1], val_2d[:, 1]]),
    'set': ['train']*len(train_2d) + ['val']*len(val_2d),
    'cluster': np.concatenate([train_clusters, val_clusters]),
    'label': np.concatenate([train_labels, val_labels]),
    'patient_id': np.concatenate([train_patient_ids, val_patient_ids]),
    'misclustered': np.concatenate([np.zeros(len(train_2d), dtype=bool), val_misclustered])
})

# Interactive plot
fig = px.scatter(
    df, x='x', y='y', color='cluster', symbol='set',
    hover_data=['patient_id', 'label', 'misclustered'],
    color_continuous_scale='Viridis',
    opacity=0.7,
    title=f'Clustering for {data_type} Fold {set_id}'
)
fig.add_scatter(
    x=df[df['misclustered']]['x'],
    y=df[df['misclustered']]['y'],
    mode='markers',
    marker=dict(color='red', size=10, symbol='x'),
    name='Misclustered'
)
fig.update_traces(marker=dict(size=8))
output_path = f'experiments/{data_type}/train/{data_type}_fold_{set_id}_with_TTA/figures/clustering_fold_{set_id}.html'
fig.write_html(output_path)

# Patient-level analysis for fold 3
incorrect_patients = []
for id in np.unique(val_patient_ids):
    indices = np.where(val_patient_ids == id)[0]
    patient_clusters = val_clusters[indices]
    patient_labels = val_labels[indices]
    patient_assigned_labels = majority_labels[patient_clusters]
    # Patient-level prediction (majority cluster assignment)
    patient_pred = np.bincount(patient_assigned_labels).argmax()
    patient_true = patient_labels[0]  # Assuming all images of a patient have the same label
    if patient_pred != patient_true:
        incorrect_patients.append((id, patient_true, patient_pred))
        # Calculate % of images misclustered
        misclustered = patient_labels != patient_assigned_labels
        percent_misclustered = 100 * np.mean(misclustered)
        print(f"Patient {id}: True={patient_true}, Pred={patient_pred}, % Images Misclustered={percent_misclustered:.2f}%")

print(f"Clustering complete! Check the plot at {output_path}")