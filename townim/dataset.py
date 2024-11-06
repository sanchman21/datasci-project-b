'''
This script is used to create a custom dataset for the CMML modelling using only CNNs (no numerical features).
Note: Keep the data directory in the same folder as the github repository.
'''

# import libraries
from torch.utils.data import DataLoader, Dataset
import pandas as pd, os, torch
from PIL import Image
import utils

# NEUTROPHIL_CSV_PATH = '/home/tchowdhury/data/code/CMML-v2/datasets/neutrophil.csv'
# MONOCYTE_CSV_PATH = '/home/tchowdhury/data/code/CMML-v2/datasets/monocyte_reassigned.csv'
# set relative paths for useful files
NEUTROPHIL_CSV_PATH = '../datasets/neutrophil.csv'
MONOCYTE_CSV_PATH = '../datasets/monocyte_reassigned.csv'
CHECKED_LABELS_XLSX_PATH = '../datasets/checked_labels.xlsx'

# patient ids that were rechecked by the pathologist
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]

class CustomDataset(Dataset):
    '''
    This class is used to create a custom dataset for the CMML modelling using only CNNs (no numerical features).
    '''
    # DATA_DIR = '/home/tchowdhury/data/cmml'
    DATA_DIR = '../../data' # relative path to the data directory
    def __init__(self, partition, csv_path, set_id, transform=None):
        '''
        function: initializes the custom dataset
        parameters:
            partition: str, partition of the dataset (train, test)
            csv_path: str, path to the csv file containing the dataset
            set_id: int, set id of the dataset
            transform: torchvision.transforms, transformation to be applied to the images
        returns: None
        '''
        # csv_path = f'{self.DATA_DIR}/master_s0_with_feat.csv'
        df = pd.read_csv(csv_path).drop_duplicates() # read the csv file and drop duplicate rows
        df = df.loc[df[f"set{set_id}"]==partition] # filter the dataset based on the partition
        df['morphology'] = 1-df['morphology'] # since normal is labelled as 1, and cmml as 0, it needs to be reversed
        df = df.loc[df["patient_id"] != 2209801848] # remove this patient since their label is not accurate
        self.transform = transform # set the transformations
        self.df = df # set the dataframe
        self.images = [] # list to store the image paths
        self.labels = [] # list to store the labels
        self.disease_count = {0: 0, 1:0} # {'normal', 'cmml'}
        for index, row in df.iterrows(): # iterate over the dataframe rows
            self.images.append(utils.convert_path_to_os_specific(str(row.image_path))) # append the image path
            if row.patient_id in rechecked_patient_ids: # if the patient id is in the rechecked list
                self.labels.append(0) # set the label to 0
                self.disease_count[0]+=1 # increment the count of normal patients
            else:
                self.labels.append(int(row.morphology)) # append the label
                self.disease_count[int(row.morphology)]+=1 # increment the count of label in the dictionary

    def __len__(self):
        '''
        function: returns the length of the dataset
        parameters: None
        returns: int, length of the dataset
        '''
        return len(self.labels) # return the length of the labels (dataset)

    def __getitem__(self, idx):
        '''
        function: returns the image and label at the given index
        parameters: idx: int, index of the image
        returns: image, label
        '''
        img_name = os.path.join(self.DATA_DIR, self.images[idx]) # get the image path
        image = Image.open(img_name).convert("RGB") # open the image and convert to RGB
        label = self.labels[idx] # get the label
        
        # Apply transformations to the image if they exist
        if self.transform:
            image = self.transform(image)

        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.float32)
            
        return image, label 