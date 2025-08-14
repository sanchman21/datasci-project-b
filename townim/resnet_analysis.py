import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from dataset import CustomDataset, MONOCYTE_CSV_PATH, NEUTROPHIL_CSV_PATH, MONOCYTE_NEW_NORMALS_CSV_PATH, SEGMENTED_MONOCYTE_CSV_PATH, SEGMENTED_MONOCYTE_NEW_NORMALS_CSV_PATH
from utils import Identity, extract_embeddings, extract_logits
import json
from PIL import Image
import shutil
from tqdm import tqdm
import shap
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Perform clustering on embeddings for a specified data type and fold.')
parser.add_argument('--data_type', type=str, default='monocyte',
                    choices=['monocyte', 'neutrophil', 'monocyte_new_normals', 'segmented_monocyte_new_normals', 'segmented_monocyte'],
                    help='Data type to perform clustering on (default: monocyte)')
parser.add_argument('--fold', type=str, default='All',
                    choices=['All', '0', '1', '2', '3', '4', 'Test'],
                    help='Fold number for cross-validation (0-4), Test for final model, or All for all folds and test (default: All)')
parser.add_argument('--n_clusters', type=int, default=4,
                    help='Number of clusters for K-means (default: 4)')
args = parser.parse_args()

data_type = args.data_type
set_id = args.fold
n_clusters = args.n_clusters

CSV_PATH_DICT = {
    'monocyte': MONOCYTE_CSV_PATH,
    'neutrophil': NEUTROPHIL_CSV_PATH,
    'monocyte_new_normals': MONOCYTE_NEW_NORMALS_CSV_PATH,
    'segmented_monocyte_new_normals': SEGMENTED_MONOCYTE_NEW_NORMALS_CSV_PATH,
    'segmented_monocyte': SEGMENTED_MONOCYTE_CSV_PATH
}
CSV_PATH = CSV_PATH_DICT[data_type]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
test_transform = T.Compose([
    T.Resize((352, 352)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

folds = ['0', '1', '2', '3', '4', 'Test'] if set_id == 'All' else [set_id]

for current_fold in folds:
    fold_id = int(current_fold) if current_fold != 'Test' else None
    is_test = current_fold == 'Test'

    if not is_test:
        print(f"Clustering for {data_type} Fold {fold_id} with {n_clusters} clusters...")
        train_dataset = CustomDataset('train', CSV_PATH, fold_id, transform=test_transform)
        val_dataset = CustomDataset('val', CSV_PATH, fold_id, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
    else:
        print(f"Clustering for {data_type} Test (Final Model) with {n_clusters} clusters...")
        train_dataset = CustomDataset('train', CSV_PATH, 0, transform=test_transform)
        val_dataset_temp = CustomDataset('val', CSV_PATH, 0, transform=test_transform)
        test_dataset = CustomDataset('test', CSV_PATH, 0, transform=test_transform)
        train_val_dataset = ConcatDataset([train_dataset, val_dataset_temp])
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
        val_loader_temp = DataLoader(val_dataset_temp, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
        val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=8)
        val_dataset = test_dataset

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    hidden_layer_size = 512
    num_classes = 2

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, num_classes)
    )

    if not is_test:
        model_path = f"experiments/{data_type}/train/{data_type}_fold_{fold_id}_with_TTA/model/model_fold_{fold_id}.pth"
    else:
        model_path = f"experiments/{data_type}/train/model/final.pth"

    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        continue
    model = model.to(device)

    val_logits, val_labels, val_patient_ids = extract_logits(model, val_loader, val_dataset, device)
    val_predictions = np.argmax(val_logits, axis=1)

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

    print(f"Misclassified patients (based on logits): {misclassified_patients}")

    model.fc = Identity()
    model = model.to(device)

    if not is_test:
        train_embeddings, train_labels, train_patient_ids, train_image_paths = extract_embeddings(model, train_loader, train_dataset, device)
    else:
        train_embeddings, train_labels, train_patient_ids, train_image_paths = extract_embeddings(model, train_loader, train_dataset, device)
        val_embeddings_temp, val_labels_temp, val_patient_ids_temp, val_image_paths_temp = extract_embeddings(model, val_loader_temp, val_dataset_temp, device)
        train_embeddings = np.vstack([train_embeddings, val_embeddings_temp])
        train_labels = np.concatenate([train_labels, val_labels_temp])
        train_patient_ids = np.concatenate([train_patient_ids, val_patient_ids_temp])
        train_image_paths = np.concatenate([train_image_paths, val_image_paths_temp])

    val_embeddings, val_labels, val_patient_ids, val_image_paths = extract_embeddings(model, val_loader, val_dataset, device)

    print(f"Train set class distribution: Normal={np.sum(train_labels == 0)}, CMML={np.sum(train_labels == 1)}")
    print(f"Val set class distribution: Normal={np.sum(val_labels == 0)}, CMML={np.sum(val_labels == 1)}")

    train_patient_ids = train_patient_ids.astype(str)
    val_patient_ids = val_patient_ids.astype(str)

    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    train_clusters = kmeans.fit_predict(train_embeddings)

    val_clusters = kmeans.predict(val_embeddings)

    db_index = davies_bouldin_score(train_embeddings, train_clusters)
    
    for label in [0, 1]:
        mask = train_labels == label
        if np.sum(mask) > 0:
            class_embeddings = train_embeddings[mask]
            class_clusters = train_clusters[mask]
            class_db_index = davies_bouldin_score(class_embeddings, class_clusters) if len(np.unique(class_clusters)) > 1 else np.nan
            dbi_text = f"{class_db_index:.4f}" if not np.isnan(class_db_index) else "N/A (single cluster)"
            print(f"Davies-Bouldin Index for {'Normal' if label == 0 else 'CMML'}: {dbi_text}")
        else:
            print(f"Davies-Bouldin Index for {'Normal' if label == 0 else 'CMML'}: N/A (no samples)")

    distances_to_centroids = kmeans.transform(train_embeddings)
    assigned_distances = distances_to_centroids[np.arange(len(train_embeddings)), train_clusters]

    cluster_metadata = {str(cluster): {} for cluster in range(n_clusters)}

    text_output = []
    for cluster in range(n_clusters):
        mask = train_clusters == cluster
        if np.sum(mask) == 0:
            text = f"Cluster {cluster}: Empty"
            print(text)
            text_output.append(text)
            continue
        cluster_labels = train_labels[mask].astype(int)
        majority_label = np.bincount(cluster_labels).argmax()
        majority_count = np.sum(cluster_labels == majority_label)
        total_count = len(cluster_labels)
        majority_percentage = (majority_count / total_count) * 100
        cluster_metadata[str(cluster)]['majority_label'] = "Normal" if majority_label == 0 else "CMML"
        cluster_metadata[str(cluster)]['majority_percentage'] = float(majority_percentage)
        label_name = "Normal" if majority_label == 0 else "CMML"
        text = f"Cluster {cluster}: {label_name} ({majority_percentage:.2f}%)"
        print(text)
        text_output.append(text)

    text_output.append(f"Davies-Bouldin Index: {db_index:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_embeddings, train_labels)
    explainer = shap.LinearExplainer(clf, train_embeddings)
    shap_values = explainer.shap_values(val_embeddings)
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(feature_importance)[-10:]
    text_output.append(f"Top 10 influential embedding dimensions: {top_features.tolist()}")
    print(f"Top 10 influential embedding dimensions: {top_features.tolist()}")

    output_dir = f'experiments/{data_type}/{"train" if not is_test else "test"}/{"" if is_test else f"{data_type}_fold_{fold_id}_with_TTA/"}figures/clustering'
    os.makedirs(output_dir, exist_ok=True)
    shap_plot_path = os.path.join(output_dir, 'shap_summary.png')
    shap.summary_plot(shap_values, val_embeddings, show=False)
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    tsne = TSNE(n_components=2, random_state=123)
    embeddings_2d = tsne.fit_transform(np.vstack([train_embeddings, val_embeddings]))
    train_2d = embeddings_2d[:len(train_embeddings)]
    val_2d = embeddings_2d[len(train_embeddings):]

    df = pd.DataFrame({
        'x': np.concatenate([train_2d[:, 0], val_2d[:, 0]]),
        'y': np.concatenate([train_2d[:, 1], val_2d[:, 1]]),
        'set': ['train']*len(train_2d) + ['val']*len(val_2d),
        'cluster': np.concatenate([train_clusters, val_clusters]),
        'label': np.concatenate([train_labels, val_labels]),
        'patient_id': np.concatenate([train_patient_ids, val_patient_ids]),
        'image_path': np.concatenate([train_image_paths, val_image_paths])
    })

    df['prediction'] = 'N/A'
    val_indices = df[df['set'] == 'val'].index
    for i, pred in enumerate(val_predictions):
        if i < len(val_indices):
            df.loc[val_indices[i], 'prediction'] = str(pred)

    df['is_train'] = (df['set'] == 'train')
    df['is_val'] = (df['set'] == 'val')

    df['patient_id'] = df['patient_id'].astype(str)
    misclassified_patients = [str(pid) for pid in misclassified_patients]

    df['patient_misclassified'] = df['patient_id'].isin(misclassified_patients)

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

    output_dir = f'experiments/{data_type}/{"train" if not is_test else "test"}/{"" if is_test else f"{data_type}_fold_{fold_id}_with_TTA/"}figures/clustering'
    output_path = os.path.join(output_dir, f'clustering_fold_{fold_id}.html' if not is_test else 'clustering_final.html')
    os.makedirs(output_dir, exist_ok=True)

    K = 5

    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, num_classes)
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    def generate_gradcam(image_path, save_path, target_class=None):
        try:
            img = Image.open(os.path.join("../../data", image_path)).convert('RGB')
            img = img.resize((352, 352), Image.Resampling.LANCZOS)
            img_tensor = test_transform(img).unsqueeze(0).to(device)
            target = [ClassifierOutputTarget(target_class)] if target_class is not None else None
            grayscale_cam = cam(input_tensor=img_tensor, targets=target)[0, :]
            img_np = np.array(img) / 255.0
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=0.5)
            cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            return True
        except Exception as e:
            print(f"Error generating Grad-CAM for {image_path}: {e}")
            return False

    for cluster in range(n_clusters):
        cluster_dir = os.path.join(output_dir, f'cluster_{cluster}')
        os.makedirs(cluster_dir, exist_ok=True)

        cluster_indices = np.where(train_clusters == cluster)[0]
        if len(cluster_indices) == 0:
            print(f"Cluster {cluster} is empty, skipping image saving.")
            cluster_metadata[str(cluster)]['median_image'] = None
            cluster_metadata[str(cluster)]['median_gradcam'] = None
            cluster_metadata[str(cluster)]['top_k_images'] = []
            cluster_metadata[str(cluster)]['top_k_gradcam_paths'] = []
            continue

        cluster_distances = assigned_distances[cluster_indices]
        cluster_labels = train_labels[cluster_indices].astype(int)
        majority_label = np.bincount(cluster_labels).argmax() if len(cluster_labels) > 0 else 0

        if len(cluster_distances) > 0:
            sorted_indices = np.argsort(cluster_distances)
            median_idx = sorted_indices[len(sorted_indices) // 2]
            median_global_idx = cluster_indices[median_idx]
            median_image_path = train_image_paths[median_global_idx]
            full_median_image_path = os.path.join("../../data", median_image_path)
            median_save_path = os.path.join(cluster_dir, 'median_image.png')
            shutil.copy(full_median_image_path, median_save_path)
            cluster_metadata[str(cluster)]['median_image'] = f"experiments/{data_type}/{'test' if is_test else 'train'}/{'figures/clustering' if is_test else f'{data_type}_fold_{fold_id}_with_TTA/figures/clustering'}/cluster_{cluster}/median_image.png"

            # Generate Grad-CAM for median image with majority class as target
            median_image_rel_path = median_image_path.replace("../../data/", "")
            gradcam_save_path = os.path.join(cluster_dir, 'median_gradcam.png')
            gradcam_rel_path = f"experiments/{data_type}/{'test' if is_test else 'train'}/{'figures/clustering' if is_test else f'{data_type}_fold_{fold_id}_with_TTA/figures/clustering'}/cluster_{cluster}/median_gradcam.png"
            if generate_gradcam(median_image_rel_path, gradcam_save_path, target_class=majority_label):
                cluster_metadata[str(cluster)]['median_gradcam'] = gradcam_rel_path
            else:
                cluster_metadata[str(cluster)]['median_gradcam'] = None

            top_k_indices = np.argsort(cluster_distances)[:K]
            top_k_global_indices = cluster_indices[top_k_indices]
            top_k_paths = []
            top_k_gradcam_paths = []
            for i, global_idx in enumerate(top_k_global_indices):
                top_image_path = train_image_paths[global_idx]
                full_top_image_path = os.path.join("../../data", top_image_path)
                top_save_path = os.path.join(cluster_dir, f'{i+1}.png')
                shutil.copy(full_top_image_path, top_save_path)
                top_k_paths.append(f"experiments/{data_type}/{'test' if is_test else 'train'}/{'figures/clustering' if is_test else f'{data_type}_fold_{fold_id}_with_TTA/figures/clustering'}/cluster_{cluster}/{i+1}.png")
                # Generate Grad-CAM for top K image with majority class as target
                top_image_rel_path = top_image_path.replace("../../data/", "")
                gradcam_save_path = os.path.join(cluster_dir, f'{i+1}_gradcam.png')
                gradcam_rel_path = f"experiments/{data_type}/{'test' if is_test else 'train'}/{'figures/clustering' if is_test else f'{data_type}_fold_{fold_id}_with_TTA/figures/clustering'}/cluster_{cluster}/{i+1}_gradcam.png"
                if generate_gradcam(top_image_rel_path, gradcam_save_path, target_class=majority_label):
                    top_k_gradcam_paths.append(gradcam_rel_path)
                else:
                    top_k_gradcam_paths.append(None)
            cluster_metadata[str(cluster)]['top_k_images'] = top_k_paths
            cluster_metadata[str(cluster)]['top_k_gradcam_paths'] = top_k_gradcam_paths

    # Save misclassified images and generate Grad-CAM
    misclassified_dir = os.path.join(output_dir, 'misclassified')
    os.makedirs(misclassified_dir, exist_ok=True)
    misclassified_images = {}

    for pid in tqdm(misclassified_patients, desc="Saving misclassified images and Grad-CAM"):
        patient_dir = os.path.join(misclassified_dir, str(pid))
        os.makedirs(patient_dir, exist_ok=True)
        patient_indices = df[(df['patient_id'] == str(pid)) & (df['set'] == 'val') & (df['point_misclassified'] == True)].index
        image_list = []
        patient_true, _ = patient_predictions.get(pid, (None, None))
        for idx in patient_indices:
            img_path = df.loc[idx, 'image_path']
            full_img_path = os.path.join("../../data", img_path)
            img_name = os.path.basename(img_path)
            save_path = os.path.join(patient_dir, img_name)
            val_idx = idx - len(train_embeddings)
            image_cluster = int(val_clusters[val_idx])
            try:
                # Save original image
                shutil.copy(full_img_path, save_path)
                # Generate Grad-CAM for misclassified image
                gradcam_save_path = os.path.join(patient_dir, f"{img_name}_gradcam.png")
                gradcam_rel_path = f"experiments/{data_type}/{'test' if is_test else 'train'}/{'figures/clustering' if is_test else f'{data_type}_fold_{fold_id}_with_TTA/figures/clustering'}/misclassified/{pid}/{img_name}_gradcam.png"
                gradcam_success = generate_gradcam(img_path.replace("../../data/", ""), gradcam_save_path, target_class=patient_true)
                image_list.append({
                    'path': f"experiments/{data_type}/{'test' if is_test else 'train'}/{'figures/clustering' if is_test else f'{data_type}_fold_{fold_id}_with_TTA/figures/clustering'}/misclassified/{pid}/{img_name}",
                    'cluster': image_cluster,
                    'gradcam_path': gradcam_rel_path if gradcam_success else None
                })
            except Exception as e:
                print(f"Error processing misclassified image {full_img_path}: {e}")
                image_list.append({
                    'path': f"experiments/{data_type}/{'test' if is_test else 'train'}/{'figures/clustering' if is_test else f'{data_type}_fold_{fold_id}_with_TTA/figures/clustering'}/misclassified/{pid}/{img_name}",
                    'cluster': image_cluster,
                    'gradcam_path': None
                })
                continue
        misclassified_images[str(pid)] = {'images': image_list}

    misclassified_json_path = os.path.join(output_dir, 'misclassified_images.json')
    with open(misclassified_json_path, 'w') as f:
        json.dump(misclassified_images, f, indent=4)

    patient_to_cluster = {str(pid): int(val_clusters[i]) for i, pid in enumerate(val_patient_ids)}
    train_patient_to_cluster = {str(pid): int(train_clusters[i]) for i, pid in enumerate(train_patient_ids)}
    cluster_metadata['patient_to_cluster'] = patient_to_cluster
    cluster_metadata['train_patient_to_cluster'] = train_patient_to_cluster
    cluster_metadata['clustering_metrics'] = {'davies_bouldin_index': float(db_index)}
    cluster_metadata['feature_importance'] = {'top_features': top_features.tolist()}
    cluster_metadata['shap_plot'] = 'shap_summary.png'
    cluster_json_path = os.path.join(output_dir, 'cluster_metadata.json')
    with open(cluster_json_path, 'w') as f:
        json.dump(cluster_metadata, f, indent=4)

    traces = []
    cluster_colors = px.colors.qualitative.Plotly[:n_clusters]

    train_df = df[df['set'] == 'train']
    for cluster in range(n_clusters):
        cluster_df = train_df[train_df['cluster'] == cluster]
        if len(cluster_df) > 0:
            traces.append(go.Scatter(
                x=cluster_df['x'],
                y=cluster_df['y'],
                mode='markers',
                marker=dict(
                    size=12,
                    opacity=0.5,
                    symbol='circle',
                    color=cluster_colors[cluster]
                ),
                name=f'Cluster {cluster}',
                customdata=cluster_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction', 'cluster']],
                hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                              '<b>Ground-Truth</b>: %{customdata[1]}<br>' +
                              '<b>Prediction</b>: %{customdata[7]}<br>' +
                              '<b>Cluster</b>: %{customdata[8]}<br>' +
                              '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                              '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                              '<b>Is Train</b>: %{customdata[4]}<br>' +
                              '<b>Is Val</b>: %{customdata[5]}<br>' +
                              '<b>Image Path</b>: %{customdata[6]}<br>' +
                              '<b>x</b>: %{x}<br>' +
                              '<b>y</b>: %{y}<extra></extra>',
                visible=True
            ))
        else:
            traces.append(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=12,
                    opacity=0.5,
                    symbol='circle',
                    color=cluster_colors[cluster]
                ),
                name=f'Cluster {cluster}',
                visible=True
            ))

    val_df = df[df['set'] == 'val']
    val_correct_df = val_df[~val_df['point_misclassified']]
    val_misclassified_df = val_df[val_df['point_misclassified']]

    if len(val_correct_df) > 0:
        traces.append(go.Scatter(
            x=val_correct_df['x'],
            y=val_correct_df['y'],
            mode='markers',
            marker=dict(
                size=14,
                opacity=1.0,
                symbol='star',
                color='#9467BD'
            ),
            name='Val (Correct)' if not is_test else 'Test (Correct)',
            customdata=val_correct_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction', 'cluster']],
            hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                          '<b>Ground-Truth</b>: %{customdata[1]}<br>' +
                          '<b>Prediction</b>: %{customdata[7]}<br>' +
                          '<b>Cluster</b>: %{customdata[8]}<br>' +
                          '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                          '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                          '<b>Is Train</b>: %{customdata[4]}<br>' +
                          '<b>Is Val</b>: %{customdata[5]}<br>' +
                          '<b>Image Path</b>: %{customdata[6]}<br>' +
                          '<b>x</b>: %{x}<br>' +
                          '<b>y</b>: %{y}<extra></extra>',
            visible=True
        ))
    else:
        traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=14, symbol='star', color='#9467BD'),
            name='Val (Correct)' if not is_test else 'Test (Correct)',
            visible=True
        ))

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
            name='Val (Misclassified)' if not is_test else 'Test (Misclassified)',
            customdata=val_misclassified_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction', 'cluster']],
            hovertemplate='<b>Patient ID</b>: %{customdata[0]}<br>' +
                          '<b>Ground-Truth</b>: %{customdata[1]}<br>' +
                          '<b>Prediction</b>: %{customdata[7]}<br>' +
                          '<b>Cluster</b>: %{customdata[8]}<br>' +
                          '<b>Patient Misclassified</b>: %{customdata[2]}<br>' +
                          '<b>Point Misclassified</b>: %{customdata[3]}<br>' +
                          '<b>Is Train</b>: %{customdata[4]}<br>' +
                          '<b>Is Val</b>: %{customdata[5]}<br>' +
                          '<b>Image Path</b>: %{customdata[6]}<br>' +
                          '<b>x</b>: %{x}<br>' +
                          '<b>y</b>: %{y}<extra></extra>',
            visible=True
        ))
    else:
        traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=14, symbol='star', color='red'),
            name='Val (Misclassified)' if not is_test else 'Test (Misclassified)',
            visible=True
        ))

    val_patients = np.unique(val_df['patient_id'])
    patient_correct_traces = {}
    patient_incorrect_traces = {}
    for pid in val_patients:
        patient_df = val_df[val_df['patient_id'] == pid]
        correct_df = patient_df[~patient_df['point_misclassified']]
        if len(correct_df) > 0:
            patient_correct_traces[pid] = go.Scatter(
                x=correct_df['x'],
                y=correct_df['y'],
                mode='markers',
                marker=dict(
                    size=14,
                    opacity=1.0,
                    symbol='star',
                    color='#9467BD'
                ),
                name='Val (Correct)' if not is_test else 'Test (Correct)',
                customdata=correct_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction', 'cluster']],
                hovertemplate=(
                    '<b>Patient ID</b>: %{customdata[0]}<br>'
                    '<b>Ground-Truth</b>: %{customdata[1]}<br>'
                    '<b>Prediction</b>: %{customdata[7]}<br>'
                    '<b>Cluster</b>: %{customdata[8]}<br>'
                    '<b>Patient Misclassified</b>: %{customdata[2]}<br>'
                    '<b>Point Misclassified</b>: %{customdata[3]}<br>'
                    '<b>Is Train</b>: %{customdata[4]}<br>'
                    '<b>Is Val</b>: %{customdata[5]}<br>'
                    '<b>Image Path</b>: %{customdata[6]}<br>'
                    '<b>x</b>: %{x}<br>'
                    '<b>y</b>: %{y}<extra></extra>'
                ),
                visible=False
            )
        else:
            patient_correct_traces[pid] = go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=14, symbol='star', color='#9467BD'),
                name='Val (Correct)' if not is_test else 'Test (Correct)',
                visible=False
            )
        traces.append(patient_correct_traces[pid])
    
        incorrect_df = patient_df[patient_df['point_misclassified']]
        if len(incorrect_df) > 0:
            patient_incorrect_traces[pid] = go.Scatter(
                x=incorrect_df['x'],
                y=incorrect_df['y'],
                mode='markers',
                marker=dict(
                    size=14,
                    opacity=1.0,
                    symbol='star',
                    color='red'
                ),
                name='Val (Incorrect)' if not is_test else 'Test (Incorrect)',
                customdata=incorrect_df[['patient_id', 'label', 'patient_misclassified', 'point_misclassified', 'is_train', 'is_val', 'image_path', 'prediction', 'cluster']],
                hovertemplate=(
                    '<b>Patient ID</b>: %{customdata[0]}<br>'
                    '<b>Ground-Truth</b>: %{customdata[1]}<br>'
                    '<b>Prediction</b>: %{customdata[7]}<br>'
                    '<b>Cluster</b>: %{customdata[8]}<br>'
                    '<b>Patient Misclassified</b>: %{customdata[2]}<br>'
                    '<b>Point Misclassified</b>: %{customdata[3]}<br>'
                    '<b>Is Train</b>: %{customdata[4]}<br>'
                    '<b>Is Val</b>: %{customdata[5]}<br>'
                    '<b>Image Path</b>: %{customdata[6]}<br>'
                    '<b>x</b>: %{x}<br>'
                    '<b>y</b>: %{y}<extra></extra>'
                ),
                visible=False
            )
        else:
            patient_incorrect_traces[pid] = go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=14, symbol='star', color='red'),
                name='Val (Incorrect)' if not is_test else 'Test (Incorrect)',
                visible=False
            )
        traces.append(patient_incorrect_traces[pid])

    # Create dropdown menu
    buttons = []
    # Initialize visibility for "All" button: n_clusters traces + 2 (val correct, val misclassified) + 2*len(val_patients) (patient-specific traces)
    buttons.append(dict(
        label="All",
        method="update",
        args=[
            {
                "visible": [True] * n_clusters + [True, True] + [False] * (2 * len(val_patients)),
                "marker": [
                    dict(size=12, opacity=0.5, symbol='circle', color=cluster_colors[i]) for i in range(n_clusters)
                ] + [
                    dict(size=14, opacity=1.0, symbol='star', color='#9467BD'),
                    dict(size=14, opacity=1.0, symbol='star', color='red')
                ] + [dict()] * (2 * len(val_patients)),
                "showlegend": [True] * n_clusters + [True, True] + [False] * (2 * len(val_patients)),
            },
            {
                "annotations[1].text": ""
            }
        ]
    ))

    for idx, selected_pid in enumerate(val_patients):
        visibility = []
        marker_styles = []
        showlegend = []
        
        # Cluster traces
        for cluster in range(n_clusters):
            visibility.append(True)
            marker_styles.append(dict(size=12, opacity=0.5, symbol='circle', color=cluster_colors[cluster]))
            showlegend.append(True)
        
        # Val correct and misclassified traces
        visibility.extend([False, False])
        marker_styles.extend([dict(), dict()])
        showlegend.extend([False, False])
        
        # Patient-specific traces
        for pid in val_patients:
            if pid == selected_pid:
                visibility.extend([True, True])
                patient_correct_df = val_df[(val_df['patient_id'] == pid) & (~val_df['point_misclassified'])]
                patient_incorrect_df = val_df[(val_df['patient_id'] == pid) & (val_df['point_misclassified'])]
                marker_styles.extend([
                    dict(size=14, opacity=1.0, symbol='star', color='#9467BD') if len(patient_correct_df) > 0 else dict(),
                    dict(size=14, opacity=1.0, symbol='star', color='red') if len(patient_incorrect_df) > 0 else dict()
                ])
                showlegend.extend([len(patient_correct_df) > 0, len(patient_incorrect_df) > 0])
            else:
                visibility.extend([False, False])
                marker_styles.extend([dict(), dict()])
                showlegend.extend([False, False])
        
        patient_true, patient_pred = patient_predictions.get(selected_pid, (None, None))
        if patient_true is not None and patient_pred is not None:
            patient_text = f"Patient {selected_pid}: True={patient_true}, Pred={patient_pred}"
        else:
            patient_text = f"Patient {selected_pid}: Data not available"
        
        buttons.append(dict(
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

    # Create the figure
    fig = go.Figure(data=traces)

    # Update layout
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
        title=f'Clustering for {data_type} {"Fold " + str(fold_id) if not is_test else "Test"} with {n_clusters} Clusters',
        legend=dict(
            title="Legend",
            x=0.55,
            y=0.1,
            xanchor="left",
            yanchor="top",
            traceorder="normal"
        ),
        margin=dict(l=50, r=400, t=100, b=150),
        width=1500,
        height=750,
        showlegend=True,
        xaxis=dict(
            domain=[0, 0.75]
        )
    )

    # Add annotation for metrics (without file paths)
    text_annotation = "<br>".join(text_output + [""] + text_output_misclassified)
    fig.add_annotation(
        text=text_annotation,
        xref="paper", yref="paper",
        x=0.8,
        y=0.5,
        xanchor="left",
        yanchor="middle",
        showarrow=False,
        font=dict(size=12),
        align="left"
    )

    fig.add_annotation(
        text="",
        xref="paper", yref="paper",
        x=0.28, y=-0.1,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    # Save the plot
    fig.write_html(output_path)

    print(f"Clustering complete for {data_type} {'Fold ' + str(fold_id) if not is_test else 'Test'} with {n_clusters} clusters! Check the plot at {output_path}")

print(f"All analysis tasks complete for {data_type}!")