import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
from torchvision.transforms import Compose, Normalize
from datetime import datetime
import subprocess
from tqdm import tqdm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Warning: albumentations not installed. Falling back to basic augmentation.")
    A = None
    ToTensorV2 = None
from warnings import filterwarnings

filterwarnings("ignore")
cache_dir = "../cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir
mlflow.set_tracking_uri("file:./mlruns")

BASE_DIR = "../../"
DATA_DIR = os.path.join(BASE_DIR, 'data')
SEGMENTATION_CSV_PATH = os.path.join("../", "datasets", "CMML Segmentation Annotation_labelbox.csv")
NEW_CSV_PATH = os.path.join("../", "datasets", "CMML_Segmentation_Annotation_labelbox2.csv")
EXPERIMENT_DIR = os.path.join("./", "experiments", "segmentation")
MODEL_DIR = os.path.join(EXPERIMENT_DIR, "models")
METRICS_DIR = os.path.join(EXPERIMENT_DIR, "metrics")
CONFIG_PATH = os.path.join(EXPERIMENT_DIR, "config.yaml")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def calculate_dataset_statistics(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for images, _ in tqdm(dataloader, desc="Calculating dataset statistics"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean, std

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1 - ((2. * intersection + smooth) / (union + smooth))
    return dice.mean()

def dice_coefficient_per_class(pred, target, thresholds=[0.3, 0.3], debug=False):
    pred_prob = torch.sigmoid(pred)
    dice_scores = []
    for i in range(pred.shape[1]):
        pred_class = pred_prob[:, i] > thresholds[i]
        intersection = (pred_class * target[:, i]).sum().float()
        union = pred_class.sum().float() + target[:, i].sum().float()
        if union == 0:
            dice_scores.append(torch.tensor(1.0, device=pred.device))
        else:
            dice = (2.0 * intersection) / union
            if debug and i == 0:
                print(f"Class {i} - Dice: {dice.item():.4f}, Threshold: {thresholds[i]}")
            dice_scores.append(dice)
    return torch.stack(dice_scores)

def iou_per_class(pred, target, thresholds=[0.3, 0.3]):
    pred_prob = torch.sigmoid(pred)
    iou_scores = []
    for i in range(pred.shape[1]):
        pred_class = pred_prob[:, i] > thresholds[i]
        intersection = (pred_class * target[:, i]).sum().float()
        union = (pred_class + target[:, i]).gt(0).sum().float()
        iou_scores.append(1.0 if union == 0 else intersection / union)
    return torch.stack(iou_scores)

existing_experiment = mlflow.get_experiment_by_name("segmentation")
if existing_experiment is not None:
    overwrite_exp = input(f"DO YOU WANT TO OVERWRITE EXISTING segmentation EXPERIMENT? [Y/N]")
    if overwrite_exp.lower() == "y":
        mlflow.delete_experiment(existing_experiment.experiment_id)
        subprocess.run(["mlflow", "gc", "--experiment-ids", existing_experiment.experiment_id], check=True)
    else:
        print("Exiting due to existing experiment. Modify code or overwrite.")
        exit()
mlflow.create_experiment("segmentation")
mlflow.set_experiment("segmentation")
print("Created new segmentation experiment")

df_old = pd.read_csv(SEGMENTATION_CSV_PATH)
df_new = pd.read_csv(NEW_CSV_PATH)
df = pd.concat([df_old, df_new]).dropna(subset=['mask_path', 'image_path'])
print(f"Total samples after combining and cleaning: {len(df)}")

train_df, temp_df = train_test_split(df, train_size=0.7, random_state=42)
val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=42)

class SegmentationDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, augment=False):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        if A is not None and augment:
            self.augmentation = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_REFLECT),
                A.Affine(translate_percent=0.1, scale=(0.9, 1.1), shear=10, p=0.3, mode=cv2.BORDER_REFLECT),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.PadIfNeeded(min_height=360, min_width=360, border_mode=cv2.BORDER_REFLECT, p=1.0),
                ToTensorV2()
            ], additional_targets={'mask': 'mask'})
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
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
            
            h, w = image.shape[:2]
            start_h = max(0, (h - 360) // 2)
            start_w = max(0, (w - 360) // 2)
            if h < 360 or w < 360:
                pad_h = max(0, 360 - h)
                pad_w = max(0, 360 - w)
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
                mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            elif h > 360 or w > 360:
                image = image[start_h:start_h + 360, start_w:start_w + 360]
                mask = mask[start_h:start_h + 360, start_w:start_w + 360]
            
            image = image.astype(np.float32) / 255.0

            if self.augment and self.augmentation:
                mask = (mask > 0.5).astype(np.uint8)
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                mask = torch.where(mask > 0.5, torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32))
                mask = mask.permute(2, 0, 1) 
                image = image.to(dtype=torch.float32)
            else:
                image = torch.from_numpy(image).to(dtype=torch.float32).permute(2, 0, 1)
                mask = torch.from_numpy(mask).to(dtype=torch.float32).transpose(0, 2).transpose(1, 2)

            if image.shape[1:] != (360, 360):
                image = F.interpolate(image.unsqueeze(0), size=(360, 360), mode='bilinear', align_corners=False).squeeze(0)
            if mask.shape[1:] != (360, 360):
                mask = F.interpolate(mask.unsqueeze(0), size=(360, 360), mode='nearest').squeeze(0)

            if self.transform:
                image = self.transform(image)

            return image, mask
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {str(e)}")
            raise

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
        x = x.to(dtype=torch.float32)
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    thresholds = [0.3, 0.3]
    actual_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice_scores = torch.zeros(2, device=device)
        train_iou_scores = torch.zeros(2, device=device)
        num_train_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice_scores += dice_coefficient_per_class(outputs, masks, thresholds)
            train_iou_scores += iou_per_class(outputs, masks, thresholds)
            num_train_batches += 1
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        val_loss = 0
        val_dice_scores = torch.zeros(2, device=device)
        val_iou_scores = torch.zeros(2, device=device)
        num_val_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                best_dice = torch.zeros(2, device=device)
                for thresh in [0.2, 0.3, 0.4]:
                    dice_scores = dice_coefficient_per_class(outputs, masks, thresholds=[thresh, thresh])
                    best_dice = torch.max(best_dice, dice_scores)
                val_dice_scores += best_dice
                val_iou_scores += iou_per_class(outputs, masks, thresholds=[0.3, 0.3])
                num_val_batches += 1

        avg_train_loss = train_loss / num_train_batches
        avg_train_dice_scores = train_dice_scores / num_train_batches
        avg_train_iou_scores = train_iou_scores / num_train_batches
        avg_val_loss = val_loss / num_val_batches
        avg_val_dice_scores = val_dice_scores / num_val_batches
        avg_val_iou_scores = val_iou_scores / num_val_batches

        if scheduler:
            scheduler.step(avg_val_loss)

        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_dice_nucleus": avg_train_dice_scores[0].item(),
            "train_dice_cytoplasm": avg_train_dice_scores[1].item(),
            "train_dice_mean": avg_train_dice_scores.mean().item(),
            "val_dice_nucleus": avg_val_dice_scores[0].item(),
            "val_dice_cytoplasm": avg_val_dice_scores[1].item(),
            "val_dice_mean": avg_val_dice_scores.mean().item(),
            "train_iou_nucleus": avg_train_iou_scores[0].item(),
            "train_iou_cytoplasm": avg_train_iou_scores[1].item(),
            "train_iou_mean": avg_train_iou_scores.mean().item(),
            "val_iou_nucleus": avg_val_iou_scores[0].item(),
            "val_iou_cytoplasm": avg_val_iou_scores[1].item(),
            "val_iou_mean": avg_val_iou_scores.mean().item()
        }, step=epoch)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train Dice - Nucleus: {avg_train_dice_scores[0]:.4f}, Cytoplasm: {avg_train_dice_scores[1]:.4f}, Mean: {avg_train_dice_scores.mean():.4f}")
        print(f"Val Dice - Nucleus: {avg_val_dice_scores[0]:.4f}, Cytoplasm: {avg_val_dice_scores[1]:.4f}, Mean: {avg_val_dice_scores.mean():.4f}")
        print(f"Train IoU - Nucleus: {avg_train_iou_scores[0]:.4f}, Cytoplasm: {avg_train_iou_scores[1]:.4f}, Mean: {avg_train_iou_scores.mean():.4f}")
        print(f"Val IoU - Nucleus: {avg_val_iou_scores[0]:.4f}, Cytoplasm: {avg_val_iou_scores[1]:.4f}, Mean: {avg_val_iou_scores.mean():.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                actual_epochs = epoch + 1
                break
        actual_epochs = num_epochs if patience_counter < patience else actual_epochs

    return best_val_loss, actual_epochs

def train_full_dataset(model, full_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        num_batches = 0
        train_pbar = tqdm(full_loader, desc=f"Full Dataset Epoch {epoch+1}/{num_epochs}")
        for images, masks in train_pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_train_loss = train_loss / num_batches
        print(f"\nFull Dataset Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_full.pth"))
    print(f"Final model saved to {os.path.join(MODEL_DIR, 'model_full.pth')}")

if __name__ == "__main__":
    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_dataset = SegmentationDataset(train_df, DATA_DIR, transform=normalize, augment=True)
    val_dataset = SegmentationDataset(val_df, DATA_DIR, transform=normalize, augment=False)
    test_dataset = SegmentationDataset(test_df, DATA_DIR, transform=normalize, augment=False)
    full_dataset = SegmentationDataset(df, DATA_DIR, transform=normalize, augment=True)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    full_loader = DataLoader(full_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=2).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    config = {
        "batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "num_epochs": 100,
        "patience": 10,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "model_type": "UNet",
        "pretrained": False,
        "augmentation": {
            "horizontal_flip": 0.3,
            "vertical_flip": 0.3,
            "rotation_limit": 15,
            "affine_translate_percent": 0.1,
            "affine_scale": [0.9, 1.1],
            "affine_shear": 10,
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "gauss_noise_var_limit": [10.0, 50.0],
            "pad_if_needed": 360
        },
        "normalization": "ImageNet"
    }

    with mlflow.start_run():
        mlflow.log_params(config)
        best_val_loss, actual_epochs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
        
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_best.pth")))
        model.eval()

        test_loss = 0
        test_dice_scores = torch.zeros(2, device=device)
        test_iou_scores = torch.zeros(2, device=device)
        num_test_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Testing"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                test_loss += loss.item()
                test_dice_scores += dice_coefficient_per_class(outputs, masks, thresholds=[0.3, 0.3])
                test_iou_scores += iou_per_class(outputs, masks, thresholds=[0.3, 0.3])
                num_test_batches += 1

        avg_test_loss = test_loss / num_test_batches
        avg_test_dice_scores = test_dice_scores / num_test_batches
        avg_test_iou_scores = test_iou_scores / num_test_batches

        test_metrics = {
            "test_loss": avg_test_loss,
            "test_dice_nucleus": avg_test_dice_scores[0].item(),
            "test_dice_cytoplasm": avg_test_dice_scores[1].item(),
            "test_dice_mean": avg_test_dice_scores.mean().item(),
            "test_iou_nucleus": avg_test_iou_scores[0].item(),
            "test_iou_cytoplasm": avg_test_iou_scores[1].item(),
            "test_iou_mean": avg_test_iou_scores.mean().item()
        }
        mlflow.log_metrics(test_metrics)
        print("\nTest Metrics:")
        print(f"Loss: {avg_test_loss:.4f}")
        print(f"Dice - Nucleus: {avg_test_dice_scores[0]:.4f}, Cytoplasm: {avg_test_dice_scores[1]:.4f}, Mean: {avg_test_dice_scores.mean():.4f}")
        print(f"IoU - Nucleus: {avg_test_iou_scores[0]:.4f}, Cytoplasm: {avg_test_iou_scores[1]:.4f}, Mean: {avg_test_iou_scores.mean():.4f}")

        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_best.pth")))
        train_full_dataset(model, full_loader, criterion, optimizer, actual_epochs)

        with open(os.path.join(METRICS_DIR, "metrics.txt"), "w") as f:
            f.write("Config:\n" + "\n".join(f"{k}: {v}" for k, v in config.items() if k != "augmentation"))
            f.write("\nAugmentation:\n" + "\n".join(f"{k}: {v}" for k, v in config["augmentation"].items()))
            f.write("\n\nTest Metrics:\n" + "\n".join(f"{k}: {v:.4f}" for k, v in test_metrics.items()))
        mlflow.log_artifact(METRICS_DIR)
        mlflow.log_artifact(os.path.join(MODEL_DIR, "model_best.pth"))
        mlflow.log_artifact(os.path.join(MODEL_DIR, "model_full.pth"))
        with open(CONFIG_PATH, "w") as f:
            import yaml
            yaml.dump(config, f)
        mlflow.log_artifact(CONFIG_PATH)

    print("Training completed successfully!")