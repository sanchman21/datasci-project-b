import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights

class MultimodalClassifier(nn.Module):
    
    def __init__(self, num_patient_features: int) -> None:
        super(MultimodalClassifier, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048 + num_patient_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )
        
    def forward(self, image, patient_info):
        image_features = self.resnet(image)
        image_features = image_features.view(image_features.size(0), -1)
        combined_features = torch.cat((image_features, patient_info), dim=1)
        output = self.classifier(combined_features)
        return output
