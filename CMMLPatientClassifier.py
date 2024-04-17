import torch
import torch.nn as nn
from torchvision import models

class PatientClassifier(nn.Module):
    def __init__(self, num_patient_features):
        super(PatientClassifier, self).__init__()
        #loading the pre-trained resnet
        self.resnet = models.resnet50(pretrained=True)
        # remove the fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Linear(2048 + num_patient_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, image, patient_info):
        image_features = self.resnet(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        combined_features = torch.cat((image_features, patient_info), dim=1)
        
        output = self.classifier(combined_features)
        return output
