import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import utils

class MonocyteDataset(Dataset):
    def __init__(self, csv_file, fold, train=True, transform=None):

        self.frame = pd.read_csv(csv_file)
        self.fold = fold
        self.train = train
        self.transform = transform
        
        # Select data for the specified fold
        fold_column = f'set{self.fold}'
        if self.train:
            self.frame = self.frame[self.frame[fold_column] == 'train']
        else:
            self.frame = self.frame[self.frame[fold_column] == 'test']


        self.columns_to_use = ['image_path', 'morphology']
        self.frame = self.frame[self.columns_to_use]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        img_name = utils.convert_path_to_os_specific(self.frame.iloc[idx]['image_path'])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'morphology': self.frame.iloc[idx]['morphology']
        }

        return sample

