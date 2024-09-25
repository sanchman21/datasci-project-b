import pandas as pd # pandas library for data manipulation
import torch # pytorch library
from torch.utils.data import Dataset # pytorch dataset class
from PIL import Image # PIL library for image manipulation
import utils_zhenzhuo # utility functions

class MonocyteDataset(Dataset):
    '''
    Monocyte Dataset class wrapping PyTorch Dataset class
    Used to create a Dataset compatible with PyTorch DataLoader
    Attributes:
        csv_file (str): Path to the csv file with the image paths and labels
        fold (int): Fold number to use for training or testing
        train (bool): Whether to use the dataset for training or testing
        transform (torchvision.transforms): Transformations to apply to the images
    '''
    def __init__(self, csv_file: str, fold: int, train: bool=True, transform=None) -> None:
        '''
        Function: Monocyte Dataset class constructor
        Returns:
            None
        '''
        self.frame = pd.read_csv(csv_file) # read the csv file into a pandas dataframe
        self.fold = fold # fold number to use for training or testing
        self.train = train # whether to use the dataset for training or testing
        self.transform = transform # transformations to apply to the images if any
        
        # Select data for the specified fold
        fold_column = f'set{self.fold}' # fold value in the fold column based on the fold number
        if self.train: # if training, select the train data
            self.frame = self.frame[self.frame[fold_column] == 'train'] # select the rows where the fold column of that fold is 'train'
        else: # if testing, select the test data
            self.frame = self.frame[self.frame[fold_column] == 'test'] # select the rows where the fold column of that fold is 'test'


        self.columns_to_use = ['image_path', 'morphology', 'patient_id'] # columns to use for the dataset (remove unnecessary columns)
        self.frame = self.frame[self.columns_to_use] # select the columns to use for the dataset

    def __len__(self) -> int:
        '''
        Function: Get the number of rows in the dataset
        Returns:
            int: Number of rows in the dataset
        '''
        return len(self.frame) # return the number of rows in the dataset

    def __getitem__(self, idx: int) -> dict:
        '''
        Function: Get the item at the specified index
        Parameters:
            idx (int): Index of the row to get
        Returns: 
            dict: A dictionary with the image, morphology, and patient_id
        '''
        img_name = utils_zhenzhuo.convert_path_to_os_specific(self.frame.iloc[idx]['image_path']) # get the image path
        image = Image.open(img_name).convert('RGB') # open the image and convert it to RGB

        if self.transform: # if there are transformations to apply
            image = self.transform(image) # apply transformations to the image
        
        # Create a dictionary named sample with the image, morphology, and patient_id
        sample = {
            'image': image,
            'morphology': self.frame.iloc[idx]['morphology'],
            'patient_id': self.frame.iloc[idx]['patient_id']

        }

        return sample # return the sample dictionary with data for the specified index (or row)

