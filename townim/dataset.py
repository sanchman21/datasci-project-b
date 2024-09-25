from torch.utils.data import DataLoader, Dataset
import pandas as pd, os, torch
from PIL import Image
import utils

# NEUTROPHIL_CSV_PATH = '/home/tchowdhury/data/code/CMML-v2/datasets/neutrophil.csv'
# MONOCYTE_CSV_PATH = '/home/tchowdhury/data/code/CMML-v2/datasets/monocyte_reassigned.csv'
NEUTROPHIL_CSV_PATH = './datasets/neutrophil.csv'
MONOCYTE_CSV_PATH = './datasets/monocyte_reassigned.csv'
CHECKED_LABELS_XLSX_PATH = './datasets/checked_labels.xlsx'

rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]

class CustomDataset(Dataset):
    # DATA_DIR = '/home/tchowdhury/data/cmml'
    DATA_DIR = '../data'
    def __init__(self, partition, csv_path, set_id, transform=None):
        # csv_path = f'{self.DATA_DIR}/master_s0_with_feat.csv'
        df = pd.read_csv(csv_path).drop_duplicates()
        # df.dropna(axis=1, inplace= True)
        # df.dropna(axis=0, inplace= True)
        df = df.loc[df[f"set{set_id}"]==partition]
        df['morphology'] = 1-df['morphology'] # bcz normal is labelled as 1, and cmml as 0. bt it needs to be opposite
        df = df.loc[df["patient_id"] != 2209801848]
        self.transform = transform
        self.df = df
        self.images = []
        self.labels = []
        self.disease_count = {0: 0, 1:0} # {'normal', 'cmml'}
        for index, row in df.iterrows():
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

        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.float32)
            
        return image, label