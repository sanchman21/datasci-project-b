import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

BASE_DIR = "../../"
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEGMENTATION_CSV_PATH = os.path.join("../", "datasets", "CMML Segmentation Annotation_labelbox.csv")
MODEL_PATH = os.path.join("./experiments/segmentation/models/model_full.pth")
RESULTS_DIR = os.path.join("./experiments/segmentation/results")
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

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Could not load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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

def create_segmented_overlay(image, pred_masks, save_path):
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
    denorm_image = image * std.view(3, 1, 1) + mean.view(3, 1, 1)
    
    overlay = denorm_image.permute(1, 2, 0).clone()
    
    cell_mask = (pred_masks[0] == 1) | (pred_masks[1] == 1)
    background_mask = ~cell_mask
    
    grey_color = torch.tensor([0.8, 0.8, 0.8], device=overlay.device, dtype=overlay.dtype)
    overlay[background_mask] = grey_color
    
    overlay = overlay.clamp(0, 1).cpu().numpy()
    overlay = (overlay * 255).astype(np.uint8)
    
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

def main():
    df = pd.read_csv(SEGMENTATION_CSV_PATH)
    df = df.dropna(subset=['image_path'])
    
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    dataset = SegmentationDataset(df, DATA_DIR, transform=normalize)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    segmented_images_dir = os.path.join(RESULTS_DIR, f"segmented_images_{timestamp}")
    os.makedirs(segmented_images_dir, exist_ok=True)
    
    new_image_paths = []
    
    with torch.no_grad():
        for idx, (image, image_path) in enumerate(tqdm(dataloader, desc="Processing images")):
            image = image.to(device)
            
            outputs = model(image)
            pred_masks = (torch.sigmoid(outputs) > 0.3).float()
            
            base_name = os.path.splitext(os.path.basename(image_path[0]))[0]
            save_path = os.path.join(segmented_images_dir, f"{base_name}_segmented.png")
            
            create_segmented_overlay(
                image=image[0],
                pred_masks=pred_masks[0],
                save_path=save_path
            )
            
            relative_path = os.path.relpath(save_path, DATA_DIR)
            new_image_paths.append(relative_path)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images")
    
    df['segmented_image_path'] = new_image_paths
    output_csv_path = os.path.join("../", "datasets", "segmented_monocyte.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")
    print(f"Segmented images saved to {segmented_images_dir}")

if __name__ == "__main__":
    main()