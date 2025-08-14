from torch.utils.data import Dataset
import pandas as pd, os, torch
from PIL import Image
import utils

NEUTROPHIL_CSV_PATH = '../datasets/neutrophil.csv'
MONOCYTE_CSV_PATH = '../datasets/monocyte_reassigned.csv'
MONOCYTE_NEW_NORMALS_CSV_PATH = '../datasets/monocyte_new_normals.csv'
SEGMENTED_MONOCYTE_NEW_NORMALS_CSV_PATH = '../datasets/monocyte_new_normals_segmented.csv'
SEGMENTED_MONOCYTE_CSV_PATH = '../datasets/monocyte_reassigned_segmented.csv'
CHECKED_LABELS_XLSX_PATH = '../datasets/checked_labels.xlsx'
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]

class CustomDataset(Dataset):
    def __init__(self, partition, csv_path, set_id, transform=None):
        self.DATA_DIR = "../../data"
        df = pd.read_csv(csv_path).drop_duplicates()
        df = df.loc[df[f"set{set_id}"]==partition]
        df['morphology'] = 1-df['morphology']
        df = df.loc[df["patient_id"] != 2209801848]
        self.transform = transform
        self.df = df
        self.images = []
        self.labels = []
        self.disease_count = {0: 0, 1:0}
        
        for index, row in df.iterrows():
            if csv_path == SEGMENTED_MONOCYTE_CSV_PATH:
                self.images.append(utils.convert_path_to_os_specific(str(row.image_path)))
            else:
                self.images.append(utils.convert_path_to_os_specific(str(row.image_path)))
            if row.patient_id in rechecked_patient_ids:
                self.labels.append(0)
                self.disease_count[0]+=1
            else:
                self.labels.append(int(row.morphology))
                self.disease_count[int(row.morphology)]+=1
                
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.DATA_DIR, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label, dtype=torch.float32)
        return image, label 