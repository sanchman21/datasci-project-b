# readme

1. without any neutrophil images + without any patient data
2. with neutrophil images [they will be all K fold for training] + without any patient data
3. without any neutrophil image + with patient data [you will not count the patient's images if the whole row are empty]
4. with neutrophil images [they will be all K fold for training] + with patient data [you will not count the patient's images if the whole row are empty]
5. without patient data +  freeze backbone + with neutrophils + without scheduler trained 30 epochs
6. with patient data + freeze backbone + with neutrophils + without scheduler trained 30 epochs

|  | setup1 | setup2 | setup3 | setup4 | setup5 | setup6 |
| --- | --- | --- | --- | --- | --- | --- |
| Precision | 84.86% ± 8.29% | 84.72% ± 7.96% | 84.59% ± 8.20% | 84.04% ± 7.92% | 74.20% ± 7.97% | 78.85% ± 7.22% |
| Recall | 89.96% ± 2.60% | 87.39% ± 5.48% | 90.57% ± 2.82% | 89.80% ± 2.81% | 91.80% ± 2.14% | 93.86% ± 4.67% |
| F1 Score | 87.21% ± 5.39% | 90.44% ± 3.12% | 87.36% ± 5.47% | 87.37% ± 5.52% | 81.75% ± 4.18% | 85.39% ± 3.87% |
| Accuracy | 83.81% ± 7.74% | 83.99% ± 7.99% | 83.96% ± 7.77% | 84.04% ± 7.92% | 75.75% ± 5.06% | 81.08% ± 4.50% |
| Specificity / Selectivity | 73.25% ± 18.18% | 72.82% ± 18.59% | 72.87% ± 17.84% | 74.22% ± 18.09% | 52.42% ± 12.01% | 63.01% ± 9.55% |

# **Repository Structure**

archive: archived code that is not being used anymore

configs/configs_normalizaed: config files

result_archive: archived result of training

saved_models: current result

# models overview

resnet50 means that the model used for training is the pre-trained version of resetnet50, and only images + label are used during the training process.

multimodal means that the model used for training is the modified version of the resetnet 50, this model combine the 10 features of patient metadata with 2048 features from the resnet 50 into a 2058 vector then output 2 labels

freeze_backbone means that the model will freezes the parameters of all backbone convolutional blocks (layer1 to layer4) in the ResNet-50 model in order to utilize pre-trained features in transfer learning, while keep the trainability of the initial layer and the fully connected layer to adapt to new tasks.