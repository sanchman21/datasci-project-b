import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import utils_zhenzhuo
import os

class MergeMasterDataset(Dataset):
    DATA_DIR = "../../data"
    def __init__(self, csv_file: str, fold: int, train: bool=True, use_patient_data: bool=False, transform=None) -> None:
        self.frame = pd.read_csv(csv_file)
        self.frame = self.frame.loc[self.frame["patient_id"] != 2209801848]
        rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]
        self.frame["morphology"] = 1-self.frame["morphology"]
        self.frame.loc[self.frame["patient_id"].isin(rechecked_patient_ids), "morphology"] = 0
        self.fold = fold
        self.train = train
        self.use_patient_data = use_patient_data
        self.transform = transform
        
        fold_column = f'set{self.fold}'
        if self.train:
            self.frame = self.frame[self.frame[fold_column] == 'train']
        else:
            self.frame = self.frame[self.frame[fold_column] == 'test']

        if self.use_patient_data:
            self.columns_to_use = ['patient_id', 'image_path', 'morphology', 'Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count',
                                'Neutrophil count', 'Monocyte count', 'Platelet count', 'Blast percentage (PB)', 'LDH']
        else:
            self.columns_to_use = ['patient_id', 'image_path', 'morphology']
        
        self.frame = self.frame[self.columns_to_use]

    def __len__(self) -> int:

        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:

        img_name = os.path.join(self.DATA_DIR, self.frame["image_path"].iloc[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'morphology': self.frame.iloc[idx]['morphology']
        }

        if self.use_patient_data:
            patient_data = {
                'Age': self.frame.iloc[idx]['Age'],
                'Gender': self.frame.iloc[idx]['Gender'],
                'Haemoglobin': self.frame.iloc[idx]['Haemoglobin'],
                'MCV': self.frame.iloc[idx]['MCV'],
                'White cell count': self.frame.iloc[idx]['White cell count'],
                'Neutrophil count': self.frame.iloc[idx]['Neutrophil count'],
                'Monocyte count': self.frame.iloc[idx]['Monocyte count'],
                'Platelet count': self.frame.iloc[idx]['Platelet count'],
                'Blast percentage (PB)': self.frame.iloc[idx]['Blast percentage (PB)'],
                'LDH': self.frame.iloc[idx]['LDH']
            }
            sample.update(patient_data)
        
        return sample

