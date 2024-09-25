import torch # import the PyTorch library
import torch.nn as nn # import the PyTorch neural network module
from torchvision import models # import the PyTorch vision models
from torchvision.models import resnet50, ResNet50_Weights # import the ResNet50 model and weights


class MultimodalClassifier(nn.Module):
    '''
    Class: Creates the Multi-modal Classifier model by wrapping nn.Module
    '''
    def __init__(self, num_patient_features: int) -> None:
        '''
        Function: Constructor for the MultimodalClassifier class
        Parameters:
            num_patient_features (int): Number of patient features to use
        Returns: Nonw
        '''
        super(MultimodalClassifier, self).__init__() # call the parent class constructor
        #loading the pre-trained resnet
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # remove the fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # add a new fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(2048 + num_patient_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )


    def forward(self, image, patient_info):
        '''
        Function: Forward pass of the model
        Parameters: 
            image (tensor): Image tensor
            patient_info (tensor): Patient information tensor
        Returns: tensor
        '''
        image_features = self.resnet(image) # get the image features (forward pass of ResNet)
        image_features = image_features.view(image_features.size(0), -1) # flatten the image features

        combined_features = torch.cat((image_features, patient_info), dim=1) # concatenate the image and patient features
        
        output = self.classifier(combined_features) # get the output from the classifier
        return output # return the output
