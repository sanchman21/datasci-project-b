import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ImageBasedDataset(Dataset):
    def __init__(self, csv_path1, csv_path2, transform=None):
        """
        Args:
        csv_path1 (str): master.csv
        csv_path2 (str): Accession numbers to pull neutrophils images for - DE-IDENTIFIED.xlsx
        transform (callable, optional)
        """
        self.data1 = pd.read_csv(csv_path1)
        self.data2 = pd.read_csv(csv_path2)

        self.data = pd.merge(self.data1, self.data2, left_on='patient_id', right_on='Accession number')
    
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        get the sample from the idx
        """
        item = self.data.iloc[idx]
        image = Image.open(item['image_path'])

        if self.transform:
            image = self.transform(image)

        data = {
            'image': image,
            'patient_id': item['patient_id'],
            'morphology': item['morphology'],
            'Age': item['Age'],
            'Gender': item['Gender'],
            'Haemoglobin': item['Haemoglobin'],
            'MCV': item['MCV'],
            'White cell count': item['White cell count'],
            'Neutrophil count': item['Neutrophil count'],
            'Monocyte count': item['Monocyte count'],
            'Platelet count': item['Platelet count'],
            'Blast percentage (PB)': item['Blast percentage (PB)'],
            'LDH': item['LDH']
        }
        
        return data