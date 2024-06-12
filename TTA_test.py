import torch
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('neutrophil_images_kevin\\Compilation of Neutrophil Images\\2129203632\SNE_53893378.jpg').convert('RGB')

test_augmentations = transforms.Compose([
    transforms.Resize(size=(334, 334)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

test_augmented_transforms = [
    transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  # 水平翻转
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  # 垂直翻转
    transforms.Compose([transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  # 旋转90度
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  # 水平翻转 + 旋转90度
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  # 垂直翻转 + 旋转90度
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),  # 水平翻转 + 垂直翻转
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation(degrees=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # 水平翻转 + 垂直翻转 + 旋转90度
]
transformed_images = test_augmentations(img)

fig, axs = plt.subplots(1, len(transformed_images), figsize=(20, 5))
for i, img_tensor in enumerate(transformed_images):
    axs[i].imshow(transforms.ToPILImage()(img_tensor))
    axs[i].axis('off')
plt.show()
