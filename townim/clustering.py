import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
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
        for inputs, lbls in loader:
            inputs = inputs.to(device).float()
            feats = model(inputs)
            embeddings.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)
    patient_ids = dataset.df['patient_id'].to_numpy()
    return embeddings, labels, patient_ids

# Function to get logits for patient-level predictions (simplified)
def get_logits(model, loader, dataset, device):
    model.eval()
    logits = []
    labels = []
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=-1)
            logits.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())
    logits = np.concatenate(logits)
    labels = np.array(labels)
    patient_ids = dataset.df['patient_id'].to_numpy()  # Extract patient IDs in order
    return logits, labels, patient_ids

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

# Load model for embeddings
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
hidden_layer_size = 512
num_classes = 2

# First, load the model with the original fc layer to get logits
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, num_classes)
)
model_path = f"experiments/{data_type}/train/{data_type}_fold_{set_id}_with_TTA/model/model_fold_{set_id}.pth"
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# Get logits for val set
val_logits, val_labels, val_patient_ids = get_logits(model, val_loader, val_dataset, device)

# Patient-level predictions
patient_ids_unique = np.unique(val_patient_ids)
misclassified_patients = []
patient_predictions = {}  # Store predictions for each patient
for pid in patient_ids_unique:
    indices = np.where(val_patient_ids == pid)[0]
    patient_logits = np.mean(val_logits[indices], axis=0)
    patient_pred = patient_logits.argmax()
    patient_true = val_labels[indices][0].astype(int)  # Assuming all images of a patient have the same label
    patient_predictions[pid] = (patient_true, patient_pred)
    if patient_pred != patient_true:
        misclassified_patients.append(pid)

# Debug: Print the misclassified patients based on logits
print(f"Misclassified patients (based on logits): {misclassified_patients}")

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
text_output = []
for cluster in range(2):
    mask = train_clusters == cluster
    majority = np.bincount(train_labels[mask].astype(int)).argmax()
    majority_labels.append(majority)
    text = f"Cluster {cluster} majority label: {majority} (0=Normal, 1=CMML)"
    print(text)
    text_output.append(text)
majority_labels = np.array(majority_labels)

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
    'patient_id': np.concatenate([train_patient_ids, val_patient_ids])
})

# Add boolean columns for train and val
df['is_train'] = (df['set'] == 'train')
df['is_val'] = (df['set'] == 'val')

# Ensure patient_id is the same type in df and misclassified_patients
df['patient_id'] = df['patient_id'].astype(str)
misclassified_patients = [str(pid) for pid in misclassified_patients]

# Mark images as misclassified based on patient-level predictions
df['patient_misclassified'] = df['patient_id'].isin(misclassified_patients)

# Mark individual points as misclassified (image-level)
df['point_misclassified'] = False
for pid in patient_ids_unique:
    indices = np.where(val_patient_ids == pid)[0]
    patient_true, patient_pred = patient_predictions.get(pid, (None, None))
    if patient_true is not None and patient_pred is not None:
        # Find the corresponding rows in df
        df_indices = df[(df['patient_id'] == str(pid)) & (df['set'] == 'val')].index
        # Check if the point's cluster assignment matches the patient's true label
        for idx, val_idx in enumerate(indices):
            if idx < len(df_indices):
                cluster = val_clusters[val_idx]
                assigned_label = majority_labels[cluster]
                true_label = patient_true
                if assigned_label != true_label:
                    df.loc[df_indices[idx], 'point_misclassified'] = True

# Debug: Check which patients in misclassified_patients have points in val_df
val_df = df[df['set'] == 'val']
val_df = val_df[val_df['patient_id'].isin(misclassified_patients)]  # Only include misclassified patients
for pid in misclassified_patients:
    patient_points = val_df[val_df['patient_id'] == pid]
    print(f"Patient {pid}: {len(patient_points)} points in val_df, {len(patient_points[patient_points['point_misclassified']])} points misclassified")

# Create traces for the plot
traces = []

# Train points trace with fixed colors
train_df = df[df['set'] == 'train']
train_colors = np.where(train_df['cluster'] == 0, 'blue', 'green')  # Cluster 0: Blue, Cluster 1: Green
traces.append(go.Scatter(
    x=train_df['x'],
    y=train_df['y'],
    mode='markers',
    marker=dict(
        size=12,
        opacity=0.5,
        symbol='circle',
        color=train_colors
    ),
    name='Train',
    customdata=train_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val']],
    hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                  '<b>Label</b>: %{customdata[1]}<br>' +
                  '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                  '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                  '<b>Is Train</b>: %{customdata[4]}<br>' +
                  '<b>Is Val</b>: %{customdata[5]}<br>' +
                  '<b>x</b>: %{x}<br>' +
                  '<b>y</b>: %{y}<extra></extra>'
))

# Val points traces (one per misclassified patient)
val_patients = np.unique(val_df['patient_id'])
val_traces = {}

# Define a color palette excluding blue and green
patient_colors = [
    '#FF7F0E',  # Orange
    '#D62728',  # Red
    '#9467BD',  # Purple
    '#8C564B',  # Brown
    '#E377C2',  # Pink
    '#BCBD22',  # Olive
    '#17BECF',  # Cyan
    '#FFBB78',  # Light Orange
    '#98DF8A',  # Light Green (distinct from train green)
    '#C5B0D5'   # Light Purple
]
patient_color_map = {pid: patient_colors[i % len(patient_colors)] for i, pid in enumerate(val_patients)}

for pid in val_patients:
    patient_df = val_df[val_df['patient_id'] == pid]
    patient_color = patient_color_map[pid]
    val_traces[pid] = go.Scatter(
        x=patient_df['x'],
        y=patient_df['y'],
        mode='markers',
        marker=dict(
            size=14,
            opacity=1.0,
            symbol='star',
            color=patient_color
        ),
        name=f'Patient {pid}',
        customdata=patient_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val']],
        hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                      '<b>Label</b>: %{customdata[1]}<br>' +
                      '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                      '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                      '<b>Is Train</b>: %{customdata[4]}<br>' +
                      '<b>Is Val</b>: %{customdata[5]}<br>' +
                      '<b>x</b>: %{x}<br>' +
                      '<b>y</b>: %{y}<extra></extra>'
    )
    traces.append(val_traces[pid])

# Add dummy traces for cluster colors in the legend (for train clusters)
cluster_colors = ['blue', 'green']  # Cluster 0: Blue, Cluster 1: Green
for cluster in range(2):
    traces.append(go.Scatter(
        x=[None], y=[None],  # No actual points
        mode='markers',
        marker=dict(
            size=10,
            color=cluster_colors[cluster]
        ),
        name=f'Cluster {cluster}',
        visible='legendonly'  # Only show in legend
    ))

# Add a dummy trace for misclassified points (to appear in legend when a patient is selected)
traces.append(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(
        size=14,
        symbol='star',
        color='red'
    ),
    name='Misclassified',
    showlegend=False  # Hidden by default
))

# Create dropdown menu
buttons = []

# Debug: Print the patients in the dropdown
print(f"Patients in dropdown: {misclassified_patients}")

# "All" option (default view)
buttons.append(dict(
    label="All",
    method="update",
    args=[{
        "visible": [True] + [True] * len(val_patients) + [True, True] + [False],  # Show train, all val traces, clusters; hide misclassified
        "marker": [
            dict(size=12, opacity=0.5, symbol='circle', color=train_colors)  # Train
        ] + [
            dict(size=14, opacity=1.0, symbol='star', color=patient_color_map[pid])  # Val traces
            for pid in val_patients
        ] + [
            dict(size=10, color=cluster_colors[i])  # Dummy traces for legend
            for i in range(2)
        ] + [
            dict(size=14, symbol='star', color='red')  # Misclassified dummy trace
        ],
        "showlegend": [True] + [True] * len(val_patients) + [True, True] + [False]  # Show all patients in legend
    }]
))

# One option per misclassified patient
for selected_pid in misclassified_patients:
    visibility = []
    marker_styles = []
    showlegend = []
    
    # Train trace
    visibility.append(True)
    marker_styles.append(dict(size=12, opacity=0.5, symbol='circle', color=train_colors))
    showlegend.append(True)
    
    # Val traces
    for pid in val_patients:
        if pid == selected_pid:
            visibility.append(True)  # Show selected patient's points
            patient_df = val_df[val_df['patient_id'] == pid]
            patient_color = patient_color_map[pid]
            colors = np.where(patient_df['point_misclassified'], 'red', patient_color)
            marker_styles.append(dict(size=14, opacity=1.0, symbol='star', color=colors))
            showlegend.append(True)
        else:
            visibility.append(True)  # Show other val points (faded)
            marker_styles.append(dict(size=14, opacity=0.1, symbol='star', color='lightgray'))
            showlegend.append(False)  # Hide other patients in legend
    
    # Dummy traces for clusters
    visibility.extend([True, True])
    marker_styles.extend([
        dict(size=10, color=cluster_colors[0]),
        dict(size=10, color=cluster_colors[1])
    ])
    showlegend.extend([True, True])
    
    # Misclassified dummy trace
    visibility.append(True)
    marker_styles.append(dict(size=14, symbol='star', color='red'))
    showlegend.append(True)  # Show "Misclassified" in legend
    
    buttons.append(dict(
        label=f"Patient {selected_pid}",
        method="update",
        args=[{
            "visible": visibility,
            "marker": marker_styles,
            "showlegend": showlegend
        }]
    ))

# Create the figure
fig = go.Figure(data=traces)

# Add dropdown menu
fig.update_layout(
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.5,
            xanchor="center",
            y=1.15,
            yanchor="top"
        )
    ],
    title=f'Clustering for {data_type} Fold {set_id}',
    legend=dict(
        title="Legend",
        x=0.8,  # Bottom-right
        y=0.1,
        traceorder="normal"
    ),
    margin=dict(b=150),  # Add bottom margin for the text annotation
    showlegend=True
)

# Patient-level analysis for fold 3 (based on clustering, for text output)
incorrect_patients = []
for id in np.unique(val_patient_ids):
    indices = np.where(val_patient_ids == id)[0]
    patient_clusters = val_clusters[indices]
    patient_labels = val_labels[indices]
    patient_assigned_labels = majority_labels[patient_clusters]
    patient_pred = np.bincount(patient_assigned_labels).argmax()
    patient_true = patient_labels[0].astype(int)
    if patient_pred != patient_true:
        incorrect_patients.append((id, patient_true, patient_pred))
        percent_misclustered = 100 * np.mean(patient_labels.astype(int) != patient_assigned_labels)
        text = f"Patient {id}: True={patient_true}, Pred={patient_pred}, % Images Misclustered={percent_misclustered:.2f}%"
        print(text)
        text_output.append(text)

# Debug: Print the misclassified patients based on clustering
print(f"Misclassified patients (based on clustering): {[pid for pid, _, _ in incorrect_patients]}")

# Add text output as an annotation below the plot
text_annotation = "<br>".join(text_output)
fig.add_annotation(
    text=text_annotation,
    xref="paper", yref="paper",
    x=0.5, y=-0.3,
    showarrow=False,
    font=dict(size=12),
    align="left"
)

# Save the plot
output_path = f'experiments/{data_type}/train/{data_type}_fold_{set_id}_with_TTA/figures/clustering_fold_{set_id}.html'
fig.write_html(output_path)

print(f"Clustering complete! Check the plot at {output_path}")