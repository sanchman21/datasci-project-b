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
        mask_path = os.path.join(self.data_dir, row['mask_path'])

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Could not load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        data = np.load(mask_path, allow_pickle=True).item()
        mask = np.unpackbits(data['mask'])[:np.prod(data['shape'])]
        mask = mask.reshape(data['shape'])
        mask = np.transpose(mask, (2, 0, 1))
        
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        mask = torch.from_numpy(mask).float()
        
        image = image / 255.0

        if self.transform:
            image = self.transform(image)

        return image, mask, image_path

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

def plot_segmentation_results(image, true_mask, pred_mask, save_path, image_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device)
    denorm_image = image * std.view(3, 1, 1) + mean.view(3, 1, 1)
    
    axes[0, 0].imshow(denorm_image.permute(1, 2, 0).clamp(0, 1))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(true_mask[0], cmap='gray')
    axes[0, 1].set_title('True Nucleus Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(true_mask[1], cmap='gray')
    axes[0, 2].set_title('True Cytoplasm Mask')
    axes[0, 2].axis('off')
    
    axes[1, 1].imshow(pred_mask[0], cmap='gray')
    axes[1, 1].set_title('Predicted Nucleus Mask')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pred_mask[1], cmap='gray')
    axes[1, 2].set_title('Predicted Cytoplasm Mask')
    axes[1, 2].axis('off')
    
    overlay = denorm_image.permute(1, 2, 0).clone()
    
    cell_mask = (pred_mask[0] == 1) | (pred_mask[1] == 1)
    background_mask = ~cell_mask
    
    grey_color = torch.tensor([0.8, 0.8, 0.8], device=overlay.device, dtype=overlay.dtype)
    overlay[background_mask] = grey_color
    
    axes[1, 0].imshow(overlay.clamp(0, 1))
    axes[1, 0].set_title('Overlay')
    axes[1, 0].axis('off')
    
    plt.suptitle(f'Segmentation Results\n{os.path.basename(image_path)}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df = pd.read_csv(SEGMENTATION_CSV_PATH)
    df = df.dropna(subset=['mask_path', 'image_path'])
    
    _, test_df = train_test_split(df, test_size=0.15, random_state=42)
    
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    test_dataset = SegmentationDataset(test_df, DATA_DIR, transform=normalize)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with torch.no_grad():
        for idx, (image, mask, image_path) in enumerate(tqdm(test_loader, desc="Processing test set")):
            image = image.to(device)
            mask = mask.to(device)
            
            outputs = model(image)
            pred_masks = (torch.sigmoid(outputs) > 0.3).float()
            
            save_path = os.path.join(RESULTS_DIR, f"segmentation_{idx:03d}_{timestamp}.png")
            plot_segmentation_results(
                image=image[0].cpu(),
                true_mask=mask[0].cpu(),
                pred_mask=pred_masks[0].cpu(),
                save_path=save_path,
                image_path=image_path[0]
            )
            
            print(f"Saved result for image {idx+1} as {os.path.basename(save_path)}")

if __name__ == "__main__":
    main()