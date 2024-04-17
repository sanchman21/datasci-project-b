import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class PatientIdBasedDataset(Dataset):
    def __init__(self, csv_path1, csv_path2, root_dir, transform=None):
        """
        Args:
        csv_path1 (str): master.csv
        csv_path2 (str): Accession numbers to pull neutrophils images for - DE-IDENTIFIED.xlsx
        transform (callable, optional)
        root_dir (str): neutrophil_images_kevin
        """

        self.data1 = pd.read_csv(csv_path1)
        self.data2 = pd.read_csv(csv_path2)

        self.data = pd.merge(self.data1, self.data2, left_on='patient_id', right_on='Accession number')
        
        self.root_dir = root_dir
        self.transform = transform

        self.image_info = []
        for _, row in self.data.iterrows():
            patient_folder = os.path.join(self.root_dir, str(row['patient_id']))

            image_files = os.listdir(patient_folder)
            for image_file in image_files:
                self.image_info.append({
                    'patient_data': row,
                    'image_path': os.path.join(patient_folder, image_file)
                })

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_data = self.image_info[idx]
        patient_data = image_data['patient_data']
        image = Image.open(image_data['image_path'])

        if self.transform:
            image = self.transform(image)

        data = {
            'image': image,
            'patient_id': patient_data['patient_id'],
            'morphology': patient_data['morphology'],
            'Age': patient_data['Age'],
            'Gender': patient_data['Gender'],
            'Haemoglobin': patient_data['Haemoglobin'],
            'MCV': patient_data['MCV'],
            'White cell count': patient_data['White cell count'],
            'Neutrophil count': patient_data['Neutrophil count'],
            'Monocyte count': patient_data['Monocyte count'],
            'Platelet count': patient_data['Platelet count'],
            'Blast percentage (PB)': patient_data['Blast percentage (PB)'],
            'LDH': patient_data['LDH']
        }
        
        return data
