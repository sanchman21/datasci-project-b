import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

BASE_DIR = "../../"
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEGMENTATION_CSV_PATH = os.path.join("../", "datasets", "CMML Segmentation Annotation_labelbox.csv")
MODEL_PATH = os.path.join("./experiments/segmentation/models/model_full.pth")
MONOCYTE_CSV_PATH = os.path.join("../", "datasets", "monocyte_new_normals.csv")
RESULTS_DIR = os.path.join("./experiments/segmentation/results")
ERROR_LOG_PATH = os.path.join(RESULTS_DIR, "corrupted_images.log")
os.makedirs(RESULTS_DIR, exist_ok=True)

BACKGROUND_COLOR = np.array([253, 241, 217])

class SegmentationDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.data_dir, row['image_path'])

        try:
            with Image.open(image_path) as img:
                image = np.array(img.convert('RGB'))
        except (IOError, ValueError) as e:
            raise ValueError(f"Error: Could not load image at {image_path} due to corruption: {str(e)}")
        
        h, w = image.shape[:2]
        start_h = max(0, (h - 360) // 2)
        start_w = max(0, (w - 360) // 2)
        if h < 360 or w < 360:
            pad_h = max(0, 360 - h)
            pad_w = max(0, 360 - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        elif h > 360 or w > 360:
            image = image[start_h:start_h + 360, start_w:start_w + 360]
        
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        image = image / 255.0

        if self.transform:
            image = self.transform(image)

        return image, image_path

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

def plot_segmentation_overlay(image, pred_mask, save_path, image_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
    denorm_image = (image * std.view(3, 1, 1) + mean.view(3, 1, 1)) * 255.0
    denorm_image = denorm_image.clamp(0, 255).to(torch.uint8)
    
    cell_mask = (pred_mask[0] == 1) | (pred_mask[1] == 1)
    background_mask = ~cell_mask
    
    overlay = denorm_image.permute(1, 2, 0).cpu().numpy()
    overlay[background_mask.cpu().numpy()] = np.array(BACKGROUND_COLOR)
    
    ax.imshow(overlay.astype(np.uint8))
    ax.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_single_image(row, skip_existing=True):
    image_path = os.path.join(DATA_DIR, row['image_path'])
    
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = SegmentationDataset(pd.DataFrame([row]), DATA_DIR, transform=normalize)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    with torch.no_grad():
        for image, image_path in loader:
            image = image.to(device)
            base_name = os.path.splitext(os.path.basename(image_path[0]))[0]
            segmented_path = os.path.join(SEGMENTED_DIR, f"segmented_{base_name}.png")
            
            if skip_existing and os.path.exists(segmented_path):
                print(f"Segmented image already exists at {segmented_path}, skipping...")
                return f"segmented_{base_name}.png"
            
            try:
                outputs = model(image)
                pred_masks = (torch.sigmoid(outputs) > 0.3).float()
                
                plot_segmentation_overlay(
                    image=image[0].cpu(),
                    pred_mask=pred_masks[0].cpu(),
                    save_path=segmented_path,
                    image_path=image_path[0]
                )
                print(f"Saved segmented overlay to {segmented_path}")
                
                return segmented_path
            except ValueError as e:
                if "Could not load image" in str(e):
                    with open(ERROR_LOG_PATH, 'a') as f:
                        f.write(f"Corrupted image at {image_path[0]}: {str(e)}\n")
                    print(f"Skipping corrupted image at {image_path[0]} due to: {str(e)}")
                    return None

def process_full_dataset(csv_path, skip_existing=True):
    df = pd.read_csv(csv_path)
    
    segmented_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Segmenting images"):
        segmented_path = process_single_image(row, skip_existing)
        segmented_paths.append(segmented_path)
    
    df['image_path'] = segmented_paths
    df = df.dropna(subset=['image_path'])
    
    df.to_csv(csv_path.split(".")[0] + "_segmented.csv", index=False)
    print(f"Saved new dataset to {csv_path}")
    
    skipped_count = sum(1 for path in segmented_paths if path is None)
    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} images due to corruption or errors. Check {ERROR_LOG_PATH} for details.")

if __name__ == "__main__":
    SEGMENTED_DIR = os.path.join(BASE_DIR, "data", "segmented_images_monocyte_new_normals")
    os.makedirs(SEGMENTED_DIR, exist_ok=True)
    
    process_full_dataset(MONOCYTE_CSV_PATH, skip_existing=True)