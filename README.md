|             |                |             |
|-----------------|---------------------|-------------------|
| (Selectivity)  | 72.27% ± 19.35%    | 19.35%            |
| (Precision)    | 82.44% ± 7.85%     | 7.85%             |
| (Recall)       | 90.26% ± 3.87%     | 3.87%             |
| (F1 Score)    | 86.10% ± 5.98%     | 5.98%             |
| (Accuracy)    | 83.12% ± 8.99%     | 8.99%             |
| (Specificity) | 72.27% ± 19.35%    | 19.35%            |
| (Threshold)     | 0.5                | 0.00              |


# Repository Structure
archive: archived code that is not being used anymore
configs/configs_normalizaed: config files
result_archive: archived result of training
saved_models: current result

# models overview
resnet50 means that the model used for training is the pre-trained version of resetnet50, and only images + label are used during the training process.

multimodal means that the model used for training is the modified version of the resetnet 50, this model combine the 10 features of patient metadata with 2048 features from the resnet 50 into a 2058 vector then output 2 labels

freeze_backbone means that the model will freezes the parameters of all backbone convolutional blocks (layer1 to layer4) in the ResNet-50 model in order to utilize pre-trained features in transfer learning, while keep the trainability of the initial layer and the fully connected layer to adapt to new tasks.

# 