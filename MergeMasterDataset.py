import pandas as pd # import the pandas library
from torch.utils.data import Dataset # import the PyTorch Dataset class
from PIL import Image # import the PIL library for image manipulation
import utils_zhenzhuo # import the utility functions

class MergeMasterDataset(Dataset):
    '''
    Class: Creates a dataset using the PyTorch Dataset class for PyTorch DataLoader
    '''
    def __init__(self, csv_file: str, fold: int, train: bool=True, 
                use_neutrophil_images: bool=False, use_patient_data: bool=False, transform=None) -> None:
        """
        Parameters:
            csv_file (str): Path to the CSV file with annotations.
            fold (int): Index of the current fold (0 to 4 for 5 folds).
            train (bool): If True, use the training set of the fold, otherwise use the validation set.
            use_neutrophil_images (bool): Whether to include neutrophil images.
            use_patient_data (bool): Whether to include patient metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.frame = pd.read_csv(csv_file) # read the csv file into a pandas dataframe
        self.fold = fold # fold number to use for training or testing
        self.train = train # whether to use the dataset for training or testing
        self.use_neutrophil_images = use_neutrophil_images # whether to include neutrophil images
        self.use_patient_data = use_patient_data # whether to include patient metadata
        self.transform = transform # transformations to apply to the images if any
        
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
        
        self.frame = self.frame[self.columns_to_use] # select the columns to use for the dataset

    def __len__(self) -> int:
        '''
        function: returns the length of the dataset
        Returns: int
        '''
        return len(self.frame) # return the length of the dataset

    def __getitem__(self, idx: int) -> dict:
        '''
        function: returns the item at the specified index
        Parameters:
            idx (int): index of the item to return
        Returns: dict
        '''
        img_name = utils_zhenzhuo.convert_path_to_os_specific(self.frame.iloc[idx]['image_path']) # get the image path
        image = Image.open(img_name).convert('RGB') # open the image and convert it to RGB

        if self.transform: # apply the transformation if it exists
            image = self.transform(image) # apply the transformation to the image

        # create a dictionary with the image and the morphology
        sample = {
            'image': image,
            'morphology': self.frame.iloc[idx]['morphology']
        }

        # add patient data if it is used
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
            sample.update(patient_data) # add the patient data to the sample
        
        return sample # return the sample

