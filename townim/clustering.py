'''
This script performs clustering by extracting embeddings on different models trained on different types of data
for all folds and visualizes the embeddings using t-SNE.
'''

# import libraries
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
from utils import Identity, extract_embeddings, extract_logits

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# set constant variables (same as in trainer.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
test_transform = T.Compose([
    T.Resize((352, 352)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Clustering for Monocyte Fold 3
data_type = 'monocyte'
set_id = 3
CSV_PATH = MONOCYTE_CSV_PATH

print(f"Clustering for {data_type} Fold {set_id}...")

# load datasets and create dataloaders
train_dataset = CustomDataset('train', CSV_PATH, set_id, transform=test_transform)
val_dataset = CustomDataset('val', CSV_PATH, set_id, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)

# load model for embeddings
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
hidden_layer_size = 512
num_classes = 2

# create the original fc layer and load the model weights to avoid errors
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, num_classes)
)
model_path = f"experiments/{data_type}/train/{data_type}_fold_{set_id}_with_TTA/model/model_fold_{set_id}.pth"
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# get logits for the validation set
val_logits, val_labels, val_patient_ids = extract_logits(model, val_loader, val_dataset, device)

# compute predictions (image-level)
val_predictions = np.argmax(val_logits, axis=1)

# Patient-level predictions
patient_ids_unique = np.unique(val_patient_ids) # get unique patient ids
misclassified_patients = [] # initialise list to store misclassified patients
patient_predictions = {} # initialise dictionary to store patient predictions
for pid in patient_ids_unique: # for each patient id
    indices = np.where(val_patient_ids == pid)[0] # get indices of the patient id
    patient_logits = np.mean(val_logits[indices], axis=0) # get the mean of the logits
    patient_pred = patient_logits.argmax() # get the prediction
    patient_true = val_labels[indices][0].astype(int) # get the true label
    patient_predictions[pid] = (patient_true, patient_pred) # store the true label and prediction
    if patient_pred != patient_true: # if the prediction is incorrect
        misclassified_patients.append(pid) # add the patient id to the misclassified patients list

print(f"Misclassified patients (based on logits): {misclassified_patients}")

# set fc to Identity for extracting embeddings
model.fc = Identity()
model = model.to(device)

# extract train and val embeddings
train_embeddings, train_labels, train_patient_ids, train_image_paths = extract_embeddings(model, train_loader, train_dataset, device)
val_embeddings, val_labels, val_patient_ids, val_image_paths = extract_embeddings(model, val_loader, val_dataset, device)

# convert train and val patient ids to string
train_patient_ids = train_patient_ids.astype(str)
val_patient_ids = val_patient_ids.astype(str)

# cluster train embeddings using k-means
kmeans = KMeans(n_clusters=2, random_state=123)
train_clusters = kmeans.fit_predict(train_embeddings)

# predict val embeddings clusters
val_clusters = kmeans.predict(val_embeddings)

# visualising embeddings using t-SNE (embedding dimensionality reduction to 2D)
tsne = TSNE(n_components=2, random_state=123)
embeddings_2d = tsne.fit_transform(np.vstack([train_embeddings, val_embeddings]))
train_2d = embeddings_2d[:len(train_embeddings)] # train embeddings (2D)
val_2d = embeddings_2d[len(train_embeddings):] # val embeddings (2D)

# create a dataframe to use with plotly
df = pd.DataFrame({
    'x': np.concatenate([train_2d[:, 0], val_2d[:, 0]]),
    'y': np.concatenate([train_2d[:, 1], val_2d[:, 1]]),
    'set': ['train']*len(train_2d) + ['val']*len(val_2d),
    'cluster': np.concatenate([train_clusters, val_clusters]),
    'label': np.concatenate([train_labels, val_labels]),
    'patient_id': np.concatenate([train_patient_ids, val_patient_ids]),
    'image_path': np.concatenate([train_image_paths, val_image_paths])
})

df['prediction'] = 'N/A'  # training points have no prediction
val_indices = df[df['set'] == 'val'].index # get val indices
for i, pred in enumerate(val_predictions): # for each prediction
    if i < len(val_indices): # if the index is less than the number of val indices
        df.loc[val_indices[i], 'prediction'] = str(pred) # set the prediction

# create boolean columns for train and val (used in plotly)
df['is_train'] = (df['set'] == 'train')
df['is_val'] = (df['set'] == 'val')

df['patient_id'] = df['patient_id'].astype(str) # convert patient id to string
misclassified_patients = [str(pid) for pid in misclassified_patients] # convert misclassified patients to string

df['patient_misclassified'] = df['patient_id'].isin(misclassified_patients) # create boolean column for patient misclassified

df['point_misclassified'] = False # initialise point misclassified column to False
for pid in patient_ids_unique: # for each patient id
    indices = np.where(val_patient_ids == pid)[0] # get indices of the patient id
    patient_true, patient_pred = patient_predictions.get(pid, (None, None)) # get the true label and prediction (patient-level)
    if patient_true is not None and patient_pred is not None: # if the true label and prediction are not None
        df_indices = df[(df['patient_id'] == str(pid)) & (df['set'] == 'val')].index # get the indices of the patient id in the dataframe
        for idx, val_idx in enumerate(indices): # for each index
            if idx < len(df_indices): # if the index is less than the number of df indices
                point_pred = val_logits[val_idx].argmax() # get the point prediction (image-level)
                true_label = patient_true # get the true label
                if point_pred != true_label: # if the point prediction is incorrect
                    df.loc[df_indices[idx], 'point_misclassified'] = True # set the point misclassified column to True

# patient-level analysis (to be printed in the plot)
text_output_misclassified = [] # initialise list to store text output for misclassified patients
for pid in misclassified_patients: # for each misclassified patient
    indices = np.where(val_patient_ids == pid)[0] # get indices of the patient id
    if len(indices) == 0: # if the length of indices is 0
        print(f"Warning: Patient {pid} has no validation points for text annotation.") # print warning
        continue
    patient_labels = val_labels[indices] # get the patient labels
    patient_logits = val_logits[indices] # get the patient logits
    point_preds = np.argmax(patient_logits, axis=1) # get the point predictions
    patient_true = patient_labels[0].astype(int) # get the true label
    patient_pred = np.bincount(point_preds).argmax() # get the prediction
    percent_misclassified = 100 * np.mean(point_preds != patient_true) # get the percentage of misclassified points
    text = f"Patient {pid}: True={patient_true}, Pred={patient_pred}, % Images Misclassified={percent_misclassified:.2f}%" # create text output
    print(text) # print text output
    text_output_misclassified.append(text) # add text output to the list

# initialise traces
traces = []

# create train points traces
train_df = df[df['set'] == 'train'] # get train dataframe
train_colors = np.where(train_df['cluster'] == 0, '#00CED1', 'green')  # Cluster 0: Dark Turquoise, Cluster 1: Green
traces.append(go.Scatter( # create scatter plot of train points
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
    customdata=train_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction']],
    hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                  '<b>Ground-Truth</b>: %{customdata[1]}<br>' +
                  '<b>Prediction</b>: %{customdata[7]}<br>' +
                  '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                  '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                  '<b>Is Train</b>: %{customdata[4]}<br>' +
                  '<b>Is Val</b>: %{customdata[5]}<br>' +
                  '<b>Image Path</b>: %{customdata[6]}<br>' +
                  '<b>x</b>: %{x}<br>' +
                  '<b>y</b>: %{y}<extra></extra>'
))

val_df = df[df['set'] == 'val'] # get val dataframe
val_correct_df = val_df[~val_df['patient_misclassified']]  # correctly classified patients
val_misclassified_df = val_df[val_df['patient_misclassified']]  # misclassified patients

# trace for correctly classified patients (purple stars) used in "All" view
if len(val_correct_df) > 0: # if there are correctly classified patients
    traces.append(go.Scatter( # create scatter plot of correctly classified patients
        x=val_correct_df['x'],
        y=val_correct_df['y'],
        mode='markers',
        marker=dict(
            size=14,
            opacity=1.0,
            symbol='star',
            color='#9467BD'  # Purple
        ),
        name='Val (Correct)',
        customdata=val_correct_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction']],
        hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                      '<b>Ground-Truth</b>: %{customdata[1]}<br>' +
                      '<b>Prediction</b>: %{customdata[7]}<br>' +
                      '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                      '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                      '<b>Is Train</b>: %{customdata[4]}<br>' +
                      '<b>Is Val</b>: %{customdata[5]}<br>' +
                      '<b>Image Path</b>: %{customdata[6]}<br>' +
                      '<b>x</b>: %{x}<br>' +
                      '<b>y</b>: %{y}<extra></extra>'
    ))

# trace for misclassified patients (red stars) used in "All" view
if len(val_misclassified_df) > 0: # if there are misclassified patients
    traces.append(go.Scatter( # create scatter plot of misclassified patients
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
        customdata=val_misclassified_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction']],
        hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                      '<b>Ground-Truth</b>: %{customdata[1]}<br>' +
                      '<b>Prediction</b>: %{customdata[7]}<br>' +
                      '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                      '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                      '<b>Is Train</b>: %{customdata[4]}<br>' +
                      '<b>Is Val</b>: %{customdata[5]}<br>' +
                      '<b>Image Path</b>: %{customdata[6]}<br>' +
                      '<b>x</b>: %{x}<br>' +
                      '<b>y</b>: %{y}<extra></extra>'
    ))

# create two traces for the "patient specific" view - correct and incorrect points for each patient
val_patients = np.unique(val_df['patient_id']) # get unique patient ids
patient_correct_traces = {} # initialise dictionary to store correct traces
patient_incorrect_traces = {} # initialise dictionary to store incorrect traces
for pid in val_patients: # for each patient id
    patient_df = val_df[val_df['patient_id'] == pid] # get the patient dataframe
    correct_df = patient_df[~patient_df['point_misclassified']] # correct points
    
    if len(correct_df) > 0: # if there are correct points
        patient_correct_traces[pid] = go.Scatter( # create scatter plot of correct points
            x=correct_df['x'],
            y=correct_df['y'],
            mode='markers',
            marker=dict(
                size=14,
                opacity=1.0,
                symbol='star',
                color='#9467BD'
            ),
            name='Val (Correct)',
            customdata=correct_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction']],
            hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                          '<b>Ground-Truth</b>: %{customdata[1]}<br>' +
                          '<b>Prediction</b>: %{customdata[7]}<br>' +
                          '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                          '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                          '<b>Is Train</b>: %{customdata[4]}<br>' +
                          '<b>Is Val</b>: %{customdata[5]}<br>' +
                          '<b>Image Path</b>: %{customdata[6]}<br>' +
                          '<b>x</b>: %{x}<br>' +
                          '<b>y</b>: %{y}<extra></extra>',
            visible=False  # Hidden by default
        )
    else: # if there are no correct points
        patient_correct_traces[pid] = go.Scatter( # create a dummy trace to maintain index alignment
            x=[None], y=[None], 
            mode='markers',
            marker=dict(size=14, symbol='star', color='#9467BD'),
            name='Val (Correct)',
            visible=False
        )
    traces.append(patient_correct_traces[pid]) # append the correct traces to the traces list
    
    incorrect_df = patient_df[patient_df['point_misclassified']] # get the incorrect points
    if len(incorrect_df) > 0: # if there are incorrect points
        patient_incorrect_traces[pid] = go.Scatter( # create scatter plot of incorrect points
            x=incorrect_df['x'],
            y=incorrect_df['y'],
            mode='markers',
            marker=dict(
                size=14,
                opacity=1.0,
                symbol='star',
                color='red'
            ),
            name='Val (Incorrect)',
            customdata=incorrect_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction']],
            hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                          '<b>Ground-Truth</b>: %{customdata[1]}<br>' +
                          '<b>Prediction</b>: %{customdata[7]}<br>' +
                          '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                          '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                          '<b>Is Train</b>: %{customdata[4]}<br>' +
                          '<b>Is Val</b>: %{customdata[5]}<br>' +
                          '<b>Image Path</b>: %{customdata[6]}<br>' +
                          '<b>x</b>: %{x}<br>' +
                          '<b>y</b>: %{y}<extra></extra>',
            visible=False  # Hidden by default
        ) 
    else: # if there are no incorrect points
        patient_incorrect_traces[pid] = go.Scatter( # create a dummy trace to maintain index alignment
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=14, symbol='star', color='red'),
            name='Val (Incorrect)',
            visible=False
        )
    traces.append(patient_incorrect_traces[pid])

# create dummy traces for the legend
cluster_colors = ['#00CED1', 'green']  # Cluster 0: Dark Turquoise, Cluster 1: Green
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

# create dropdown menu for all patients
buttons = []

# append "All" option to the buttons
buttons.append(dict(
    label="All",
    method="update",
    args=[
        {
            "visible": [True, True, True] + [False] * (2 * len(val_patients)) + [True, True],  # train, val correct, val incorrect, patient traces, clusters
            "marker": [
                dict(size=12, opacity=0.5, symbol='circle', color=train_colors), # train
                dict(size=14, opacity=1.0, symbol='star', color='#9467BD'), # val correct
                dict(size=14, opacity=1.0, symbol='star', color='red'), # val incorrect
            ] + [dict()] * (2 * len(val_patients)) + [ # patient traces (placeholder)
                dict(size=10, color=cluster_colors[0]),  # cluster 0
                dict(size=10, color=cluster_colors[1])   # cluster 1
            ],
            "showlegend": [True, True, True] + [False] * (2 * len(val_patients)) + [True, True],
        },
        {
            "annotations[1].text": ""  # empty patient-specific text for "All" view
        }
    ]
))

# buttons for each patient
for idx, selected_pid in enumerate(val_patients): # for each patient id
    # initialise lists
    visibility = []
    marker_styles = []
    showlegend = []
    
    # add train traces (visible by default)
    visibility.append(True)
    marker_styles.append(dict(size=12, opacity=0.5, symbol='circle', color=train_colors))
    showlegend.append(True)
    
    # hide the val correct and val incorrect traces
    visibility.extend([False, False])
    marker_styles.extend([
        dict(),
        dict()
    ])
    showlegend.extend([False, False])
    
    # add the patient-specific traces
    for pid in val_patients: # for each patient id
        if pid == selected_pid: # if the patient id is the selected patient, add the correct and incorrect traces
            visibility.extend([True, True])
            patient_correct_df = val_df[(val_df['patient_id'] == pid) & (~val_df['point_misclassified'])]
            patient_incorrect_df = val_df[(val_df['patient_id'] == pid) & (val_df['point_misclassified'])]
            marker_styles.extend([
                dict(size=14, opacity=1.0, symbol='star', color='#9467BD') if len(patient_correct_df) > 0 else dict(),
                dict(size=14, opacity=1.0, symbol='star', color='red') if len(patient_incorrect_df) > 0 else dict()
            ])
            showlegend.extend([len(patient_correct_df) > 0, len(patient_incorrect_df) > 0])  # Show legend only if trace has points
        else: # if the patient id is not the selected patient, add dummy traces
            visibility.extend([False, False])
            marker_styles.extend([dict(), dict()])
            showlegend.extend([False, False])
    
    # add dummy traces for the clusters
    visibility.extend([True, True])
    marker_styles.extend([
        dict(size=10, color=cluster_colors[0]),
        dict(size=10, color=cluster_colors[1])
    ])
    showlegend.extend([True, True])
    
    # get the patient-specific text
    patient_true, patient_pred = patient_predictions.get(selected_pid, (None, None))
    if patient_true is not None and patient_pred is not None:
        patient_text = f"Patient {selected_pid}: True={patient_true}, Pred={patient_pred}"
    else:
        patient_text = f"Patient {selected_pid}: Data not available"
    
    buttons.append(dict( # add the button
        label=f"Patient {selected_pid}",
        method="update",
        args=[
            {
                "visible": visibility,
                "marker": marker_styles,
                "showlegend": showlegend,
            },
            {
                "annotations[1].text": patient_text
            }
        ]
    ))

# create the figure
fig = go.Figure(data=traces)

# update layout (title, legend, margin, width, height, buttons
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
        x=0.55,  # Move legend inside the plot (top-left corner)
        y=0.27,
        xanchor="left",
        yanchor="top",
        traceorder="normal"
    ),
    margin=dict(l=50, r=400, t=100, b=150),  # Increased bottom margin
    width=1500,
    height=750,
    showlegend=True,
    xaxis=dict(
        domain=[0, 0.75]  # Plot takes up 75% of the width (0 to 0.75)
    )
)

# add annotation 0 (right side of the plot)
text_annotation = "<br>".join(text_output_misclassified)
fig.add_annotation(
    text=text_annotation,
    xref="paper", yref="paper",
    x=0.8,  # Adjusted to bring text closer to the plot
    y=0.5,  # Vertically centered
    xanchor="left",  # Text starts at x and extends to the right
    yanchor="middle",  # Vertically centered
    showarrow=False,
    font=dict(size=12),
    align="left"
)

# add annotation 1 (bottom of the plot) (placeholder for default view - "All")
fig.add_annotation(
    text="",  # Start with empty text
    xref="paper", yref="paper",
    x=0.28, y=-0.1,  # Adjusted y position
    showarrow=False,
    font=dict(size=12),
    align="center"
)

# save the plot
output_path = f'experiments/{data_type}/train/{data_type}_fold_{set_id}_with_TTA/figures/clustering_fold_{set_id}.html'
fig.write_html(output_path)

print(f"Clustering complete! Check the plot at {output_path}")