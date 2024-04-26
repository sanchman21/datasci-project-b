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
        self.blood_cell_frame = pd.read_csv(csv_file)
        self.fold = fold
        self.train = train
        self.use_neutrophil_images = use_neutrophil_images
        self.use_patient_data = use_patient_data

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        # Filter out neutrophil images if not used
        if not use_neutrophil_images:
            self.blood_cell_frame = self.blood_cell_frame[
                self.blood_cell_frame[['set0', 'set1', 'set2', 'set3', 'set4']].notnull().any(axis=1)
            ]
        
        # Select data for the specified fold
        fold_column = f'set{self.fold}'
        if self.train:
            self.blood_cell_frame = self.blood_cell_frame[self.blood_cell_frame[fold_column] == 'train']
        else:
            self.blood_cell_frame = self.blood_cell_frame[self.blood_cell_frame[fold_column] == 'test']

        # Select only relevant columns
        if self.use_patient_data:
            self.columns_to_use = ['image_path', 'morphology', 'Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count',
                                   'Neutrophil count', 'Monocyte count', 'Platelet count', 'Blast percentage (PB)', 'LDH']
        else:
            self.columns_to_use = ['image_path', 'morphology']
        
        self.blood_cell_frame = self.blood_cell_frame[self.columns_to_use]

    def __len__(self):
        return len(self.blood_cell_frame)

    def __getitem__(self, idx):
        img_name = self.blood_cell_frame.iloc[idx]['image_path']
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        morphology = self.blood_cell_frame.iloc[idx]['morphology']
        sample = {'image': image, 'morphology': morphology}

        
        if self.use_patient_data:
            patient_data = {col: self.blood_cell_frame.iloc[idx][col] for col in self.columns_to_use[2:]}
            sample.update(patient_data)
        

        return image, morphology
