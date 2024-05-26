from Trainer import train_model
import utils
from monocyte_resnet50 import monocyte_dataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms



if __name__ == '__main__':
    fold = 0

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = monocyte_dataset.MonocyteDataset(csv_file="monocyte_resnet50\cleaned_neutrophils.csv", fold=fold, train=True, transform=transform)

    print("Dataset size:", len(dataset))

    first_sample = dataset[0]
    print("First sample image shape:", first_sample['image'].shape)
    print("First sample morphology:", first_sample['morphology'])

    for i, sample in enumerate(dataset):
        print(f"Sample {i}: Image shape {sample['image'].shape}, Morphology {sample['morphology']}")
        if i == 3:
            break