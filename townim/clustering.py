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
    image_paths = dataset.df['image_path'].to_numpy()
    return embeddings, labels, patient_ids, image_paths

# Function to get logits for patient-level predictions
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
    patient_ids = dataset.df['patient_id'].to_numpy()
    patient_ids = patient_ids.astype(str)
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
patient_predictions = {}
for pid in patient_ids_unique:
    indices = np.where(val_patient_ids == pid)[0]
    patient_logits = np.mean(val_logits[indices], axis=0)
    patient_pred = patient_logits.argmax()
    patient_true = val_labels[indices][0].astype(int)
    patient_predictions[pid] = (patient_true, patient_pred)
    if patient_pred != patient_true:
        misclassified_patients.append(pid)

# Debug: Print the misclassified patients based on logits
print(f"Misclassified patients (based on logits): {misclassified_patients}")

# Now set fc to Identity to extract embeddings
model.fc = Identity()
model = model.to(device)

# Extract embeddings
train_embeddings, train_labels, train_patient_ids, train_image_paths = extract_embeddings(model, train_loader, train_dataset, device)
val_embeddings, val_labels, val_patient_ids, val_image_paths = extract_embeddings(model, val_loader, val_dataset, device)

# Convert patient_ids to strings
train_patient_ids = train_patient_ids.astype(str)
val_patient_ids = val_patient_ids.astype(str)

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
    'patient_id': np.concatenate([train_patient_ids, val_patient_ids]),
    'image_path': np.concatenate([train_image_paths, val_image_paths])
})

# Add boolean columns for train and val
df['is_train'] = (df['set'] == 'train')
df['is_val'] = (df['set'] == 'val')

# Ensure patient_id is the same type in df and misclassified_patients
df['patient_id'] = df['patient_id'].astype(str)
misclassified_patients = [str(pid) for pid in misclassified_patients]

# Mark images as misclassified based on patient-level predictions
df['patient_misclassified'] = df['patient_id'].isin(misclassified_patients)

# Mark individual points as misclassified (image-level, based on logits)
df['point_misclassified'] = False
for pid in patient_ids_unique:
    indices = np.where(val_patient_ids == pid)[0]
    patient_true, patient_pred = patient_predictions.get(pid, (None, None))
    if patient_true is not None and patient_pred is not None:
        df_indices = df[(df['patient_id'] == str(pid)) & (df['set'] == 'val')].index
        for idx, val_idx in enumerate(indices):
            if idx < len(df_indices):
                point_pred = val_logits[val_idx].argmax()
                true_label = patient_true
                if point_pred != true_label:
                    df.loc[df_indices[idx], 'point_misclassified'] = True

# Debug: Check which patients in misclassified_patients have points in val_df
val_df = df[df['set'] == 'val']
for pid in misclassified_patients:
    patient_points = val_df[val_df['patient_id'] == pid]
    print(f"Patient {pid}: {len(patient_points)} points in val_df, {len(patient_points[patient_points['point_misclassified']])} points misclassified")

# Create traces for the plot
traces = []

# Train points trace with fixed colors
train_df = df[df['set'] == 'train']
train_colors = np.where(train_df['cluster'] == 0, 'blue', 'green')
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
    customdata=train_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path']],
    hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                  '<b>Label</b>: %{customdata[1]}<br>' +
                  '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                  '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                  '<b>Is Train</b>: %{customdata[4]}<br>' +
                  '<b>Is Val</b>: %{customdata[5]}<br>' +
                  '<b>Image Path</b>: %{customdata[6]}<br>' +
                  '<b>x</b>: %{x}<br>' +
                  '<b>y</b>: %{y}<extra></extra>'
))

# Val points traces: Separate into correctly classified and misclassified patients
val_correct_df = val_df[~val_df['patient_misclassified']]  # Correctly classified patients
val_misclassified_df = val_df[val_df['patient_misclassified']]  # Misclassified patients

# Trace for correctly classified patients (purple stars)
if len(val_correct_df) > 0:
    traces.append(go.Scatter(
        x=val_correct_df['x'],
        y=val_correct_df['y'],
        mode='markers',
        marker=dict(
            size=14,
            opacity=1.0,
            symbol='star',
            color='#9467BD'  # Purple (not red, green, or blue)
        ),
        name='Val (Correct)',
        customdata=val_correct_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path']],
        hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                      '<b>Label</b>: %{customdata[1]}<br>' +
                      '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                      '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                      '<b>Is Train</b>: %{customdata[4]}<br>' +
                      '<b>Is Val</b>: %{customdata[5]}<br>' +
                      '<b>Image Path</b>: %{customdata[6]}<br>' +
                      '<b>x</b>: %{x}<br>' +
                      '<b>y</b>: %{y}<extra></extra>'
    ))

# Trace for misclassified patients (red stars)
if len(val_misclassified_df) > 0:
    traces.append(go.Scatter(
        x=val_misclassified_df['x'],
        y=val_misclassified_df['y'],
        mode='markers',
        marker=dict(
            size=14,
            opacity=1.0,
            symbol='star',
            color='red'
        ),
        name='Val (Misclassified)',
        customdata=val_misclassified_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path']],
        hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                      '<b>Label</b>: %{customdata[1]}<br>' +
                      '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                      '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                      '<b>Is Train</b>: %{customdata[4]}<br>' +
                      '<b>Is Val</b>: %{customdata[5]}<br>' +
                      '<b>Image Path</b>: %{customdata[6]}<br>' +
                      '<b>x</b>: %{x}<br>' +
                      '<b>y</b>: %{y}<extra></extra>'
    ))

# Add dummy traces for cluster colors in the legend
cluster_colors = ['blue', 'green']
for cluster in range(2):
    traces.append(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            size=10,
            color=cluster_colors[cluster]
        ),
        name=f'Cluster {cluster}',
        visible='legendonly'
    ))

# Create dropdown menu for all patients
val_patients = np.unique(val_df['patient_id'])  # All patients in val_df
buttons = []

# Debug: Print the patients in the dropdown
print(f"Patients in dropdown: {val_patients.tolist()}")

# "All" option (default view)
buttons.append(dict(
    label="All",
    method="update",
    args=[{
        "visible": [True, True, True, True, True],  # Train, Val (Correct), Val (Misclassified), Cluster 0, Cluster 1
        "marker": [
            dict(size=12, opacity=0.5, symbol='circle', color=train_colors),  # Train (unchanged)
            dict(size=14, opacity=1.0, symbol='star', color='#9467BD'),  # Val (Correct)
            dict(size=14, opacity=1.0, symbol='star', color='red'),  # Val (Misclassified)
            dict(size=10, color=cluster_colors[0]),  # Cluster 0
            dict(size=10, color=cluster_colors[1])   # Cluster 1
        ],
        "showlegend": [True, True, True, True, True]
    }]
))

# One option per patient (all patients)
for selected_pid in val_patients:
    visibility = []
    marker_styles = []
    showlegend = []
    
    # Train trace (always visible, never faded)
    visibility.append(True)
    marker_styles.append(dict(size=12, opacity=0.5, symbol='circle', color=train_colors))
    showlegend.append(True)
    
    # Val traces: Highlight selected patient
    patient_df = val_df[val_df['patient_id'] == selected_pid]
    colors = np.where(patient_df['point_misclassified'], 'red', '#9467BD')  # Red for misclassified points, purple for correct
    visibility.append(True)  # Val (Correct)
    visibility.append(True)  # Val (Misclassified)
    marker_styles.append(
        dict(size=14, opacity=0.1, symbol='star', color='#9467BD')
        if selected_pid not in val_correct_df['patient_id'].values
        else dict(size=14, opacity=1.0, symbol='star', color=colors)
    )
    marker_styles.append(
        dict(size=14, opacity=0.1, symbol='star', color='red')
        if selected_pid not in val_misclassified_df['patient_id'].values
        else dict(size=14, opacity=1.0, symbol='star', color=colors)
    )
    showlegend.extend([True, True])  # Keep both Val traces in legend
    
    # Dummy traces for clusters
    visibility.extend([True, True])
    marker_styles.extend([
        dict(size=10, color=cluster_colors[0]),
        dict(size=10, color=cluster_colors[1])
    ])
    showlegend.extend([True, True])
    
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
        x=0.8,
        y=0.1,
        traceorder="normal"
    ),
    margin=dict(b=150),
    showlegend=True
)

# Patient-level analysis for misclassified patients (based on logits)
text_output_misclassified = []
for pid in misclassified_patients:
    indices = np.where(val_patient_ids == pid)[0]
    if len(indices) == 0:
        print(f"Warning: Patient {pid} has no validation points for text annotation.")
        continue
    patient_labels = val_labels[indices]
    patient_logits = val_logits[indices]
    point_preds = np.argmax(patient_logits, axis=1)
    patient_true = patient_labels[0].astype(int)
    patient_pred = np.bincount(point_preds).argmax()
    percent_misclassified = 100 * np.mean(point_preds != patient_true)
    text = f"Patient {pid}: True={patient_true}, Pred={patient_pred}, % Images Misclassified={percent_misclassified:.2f}%"
    print(text)
    text_output_misclassified.append(text)

# Add text output as an annotation below the plot
text_annotation = "<br>".join(text_output + text_output_misclassified)
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