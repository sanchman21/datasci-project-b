import os
import json
import argparse
import pandas as pd
import numpy as np
from skimage import io, measure, feature, color
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
import mlflow
from tqdm import tqdm
import subprocess

parser = argparse.ArgumentParser(description="Morphology metrics computation")
parser.add_argument("--data_type", choices=["monocyte", "monocyte_new_normals"], required=True,
                    help="Dataset type to process")
args = parser.parse_args()

BASE_DIR = "../../"
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEGMENTATION_CSV_PATH = os.path.join("../", "datasets", "CMML Segmentation Annotation_labelbox.csv")
MODEL_PATH = os.path.join("./experiments/segmentation/models/model_full.pth")
MONOCYTE_NEW_NORMALS_CSV_PATH = os.path.join("../", "datasets", "monocyte_new_normals.csv")
MONOCYTE_CSV_PATH = os.path.join("../", "datasets", "monocyte_reassigned.csv")
CLUSTER_METADATA_PATH = './experiments/monocyte/test/figures/clustering/cluster_metadata.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

if args.data_type == "monocyte":
    CSV_PATH = MONOCYTE_CSV_PATH
else:
    CSV_PATH = MONOCYTE_NEW_NORMALS_CSV_PATH

with open(CLUSTER_METADATA_PATH, 'r') as f:
    cluster_metadata = json.load(f)

cmml_clusters = {k: v for k, v in cluster_metadata.items() if "majority_label" in v and v["majority_label"] == "CMML"}
normal_clusters = {k: v for k, v in cluster_metadata.items() if "majority_label" in v and v["majority_label"] == "Normal"}
cmml_cluster = max(cmml_clusters, key=lambda k: cmml_clusters[k]["majority_percentage"])
normal_cluster = max(normal_clusters, key=lambda k: normal_clusters[k]["majority_percentage"])

cmml_patients = [int(pid) for pid, c in cluster_metadata['train_patient_to_cluster'].items() if c == int(cmml_cluster)]
normal_patients = [int(pid) for pid, c in cluster_metadata['train_patient_to_cluster'].items() if c == int(normal_cluster)]

# rechecked patients manipulation is very important since there values were not changed in the data.
df = pd.read_csv(CSV_PATH)
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]
df['morphology'] = 1 - df['morphology']
df = df.loc[df["patient_id"] != 2209801848]
df.loc[df['patient_id'].isin(rechecked_patient_ids), 'morphology'] = 0 

cmml_df = df[df['patient_id'].isin(cmml_patients) & (df['morphology'] == 1)]
normal_df = df[df['patient_id'].isin(normal_patients) & (df['morphology'] == 0)]

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)
        self.down5 = double_conv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = double_conv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        conv1 = self.down1(x)
        x = self.pool(conv1)
        conv2 = self.down2(x)
        x = self.pool(conv2)
        conv3 = self.down3(x)
        x = self.pool(conv3)
        conv4 = self.down4(x)
        x = self.pool(conv4)
        x = self.down5(x)
        x = self.up4(x)
        x = torch.cat([F.interpolate(x, size=conv4.shape[2:]), conv4], dim=1)
        x = self.conv4(x)
        x = self.up3(x)
        x = torch.cat([F.interpolate(x, size=conv3.shape[2:]), conv3], dim=1)
        x = self.conv3(x)
        x = self.up2(x)
        x = torch.cat([F.interpolate(x, size=conv2.shape[2:]), conv2], dim=1)
        x = self.conv2(x)
        x = self.up1(x)
        x = torch.cat([F.interpolate(x, size=conv1.shape[2:]), conv1], dim=1)
        x = self.conv1(x)
        x = self.out_conv(x)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :h, :w]
        return x

model = UNet(in_channels=3, out_channels=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = Compose([Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

def compute_all_metrics(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((352, 352), Image.Resampling.LANCZOS)
        img_tensor = ToTensor()(img).unsqueeze(0)
        img_tensor = transform(img_tensor).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()[0]
        
        nucleus_mask = pred_masks[0].cpu().numpy()
        cytoplasm_mask = pred_masks[1].cpu().numpy()
        cell_mask = cytoplasm_mask  # assumption that nucleus is always inside cytoplasm
        
        nucleus_mask = (nucleus_mask > 0).astype(np.uint8)
        cytoplasm_mask = (cytoplasm_mask > 0).astype(np.uint8)
        cell_mask = (cell_mask > 0).astype(np.uint8)
        
        # True cytoplasm mask excluding nucleus
        true_cytoplasm_mask = cytoplasm_mask - nucleus_mask
        true_cytoplasm_mask = (true_cytoplasm_mask > 0).astype(np.uint8)
        
        cell_area = np.sum(cell_mask)
        props = measure.regionprops(cell_mask.astype(np.uint8))
        perimeter = props[0].perimeter if props else 0
        circularity = (4 * np.pi * cell_area) / (perimeter ** 2) if perimeter > 0 else 0
        solidity = props[0].solidity if props else 0
        
        gray_img = color.rgb2gray(np.array(img))
        nucleus_intensity = np.mean(gray_img[nucleus_mask > 0]) if np.sum(nucleus_mask) > 0 else 0
        
        glcm = feature.graycomatrix((gray_img * nucleus_mask * 255).astype(np.uint8),
                                    [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                    levels=256, symmetric=True, normed=True)
        nucleus_contrast = feature.graycoprops(glcm, 'contrast').mean() if glcm.sum() > 0 else 0
        
        glcm_cyt = feature.graycomatrix((gray_img * true_cytoplasm_mask * 255).astype(np.uint8),
                                        [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                        levels=256, symmetric=True, normed=True)
        cytoplasm_contrast = feature.graycoprops(glcm_cyt, 'contrast').mean() if glcm_cyt.sum() > 0 else 0
        
        nucleus_area = np.sum(nucleus_mask)
        cytoplasm_area = np.sum(cytoplasm_mask)
        nc_ratio = nucleus_area / cell_area
        
        return {
            'cell_area': cell_area,
            'nucleus_area': nucleus_area,
            'circularity': circularity,
            'solidity': solidity,
            'nucleus_intensity': nucleus_intensity,
            'nucleus_contrast': nucleus_contrast,
            'cytoplasm_contrast': cytoplasm_contrast,
            'nc_ratio': nc_ratio
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

cmml_metrics = []
for _, row in tqdm(cmml_df.iterrows(), total=len(cmml_df), desc="Processing CMML images"):
    image_path = os.path.join(DATA_DIR, row['image_path'])
    metrics = compute_all_metrics(image_path)
    if metrics:
        cmml_metrics.append(metrics)

normal_metrics = []
for _, row in tqdm(normal_df.iterrows(), total=len(normal_df), desc="Processing Normal images"):
    image_path = os.path.join(DATA_DIR, row['image_path'])
    metrics = compute_all_metrics(image_path)
    if metrics:
        normal_metrics.append(metrics)

def average_metrics(metrics_list):
    if not metrics_list:
        return {k: 0 for k in ['cell_area', 'nucleus_area', 'circularity', 'solidity', 'nucleus_intensity',
                               'nucleus_contrast', 'cytoplasm_contrast', 'nc_ratio']}
    return {k: np.mean([m[k] for m in metrics_list if m[k] is not None]) for k in metrics_list[0]}

avg_cmml = average_metrics(cmml_metrics)
avg_normal = average_metrics(normal_metrics)

mlflow.set_tracking_uri("file:./mlruns")
experiment_name = f"{args.data_type}-morphology-analysis"
existing_experiment = mlflow.get_experiment_by_name(experiment_name)
if existing_experiment is not None:
    experiment_id = existing_experiment.experiment_id
    overwrite_exp = input(f"DO YOU WANT TO OVERWRITE EXISTING {experiment_name} EXPERIMENT? [Y/N] ")
    if overwrite_exp.lower() == "y":
        mlflow.delete_experiment(experiment_id)
        subprocess.run(["mlflow", "gc", "--experiment-ids", experiment_id], check=True)
    else:
        print("To run the code further, you need to overwrite existing experiment. Please modify code otherwise.")
        exit()

mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name=experiment_name)
print(f"Created new experiment for {experiment_name}")

with mlflow.start_run(run_name="morphology_metrics"):
    for k, v in avg_cmml.items():
        mlflow.log_metric(f"cmml_{k}", v)
    for k, v in avg_normal.items():
        mlflow.log_metric(f"normal_{k}", v)
    mlflow.log_param("cmml_cluster", cmml_cluster)
    mlflow.log_param("normal_cluster", normal_cluster)
    mlflow.log_param("num_cmml_images", len(cmml_df))
    mlflow.log_param("num_normal_images", len(normal_df))

print(f"Average CMML Metrics ({len(cmml_df)} images):", avg_cmml)
print(f"Average Normal Metrics ({len(normal_df)} images):", avg_normal)

metrics_data = []
for metric_name in avg_cmml.keys():
    metrics_data.append({
        'metric': metric_name,
        'normal': avg_normal[metric_name],
        'cmml': avg_cmml[metric_name]
    })

metrics_df = pd.DataFrame(metrics_data)
output_csv_path = f"{args.data_type}_morphology_metrics_comparison.csv"
metrics_df.to_csv(output_csv_path, index=False)
print(f"\nMetrics comparison saved to: {output_csv_path}")
print("\nCSV Contents:")
print(metrics_df)