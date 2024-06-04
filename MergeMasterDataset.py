import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MergeMasterDataset(Dataset):
    def __init__(self, csv_file, fold, train=True, use_neutrophil_images=False, use_patient_data=False, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            fold (int): Index of the current fold (0 to 4 for 5 folds).
            train (bool): If True, use the training set of the fold, otherwise use the validation set.
            use_neutrophil_images (bool): Whether to include neutrophil images.
            use_patient_data (bool): Whether to include patient metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.fold = fold
        self.train = train
        self.use_neutrophil_images = use_neutrophil_images
        self.use_patient_data = use_patient_data
        self.transform = transform
        
        # Filter out neutrophil images if not used
        if not use_neutrophil_images:
            self.frame = self.frame[self.frame['dataset'] != 'neutrophil']
        
        # Select data for the specified fold
        fold_column = f'set{self.fold}'
        if self.train:
            self.frame = self.frame[self.frame[fold_column] == 'train']
        else:
            self.frame = self.frame[self.frame[fold_column] == 'test']

        # Select only relevant columns
        if self.use_patient_data:
            self.columns_to_use = ['image_path', 'morphology', 'Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count',
                                   'Neutrophil count', 'Monocyte count', 'Platelet count', 'Blast percentage (PB)', 'LDH']
        else:
            self.columns_to_use = ['image_path', 'morphology']
        
        self.frame = self.frame[self.columns_to_use]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = self.frame.iloc[idx]['image_path']
        image = Image.open(img_name).convert('RGB')

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

