from torchvision import transforms
from MergeMasterDataset import MergeMasterDataset

csv_file_path = 'merged_master.csv'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = MergeMasterDataset(
    csv_file=csv_file_path,
    fold=0,
    train=False,
    use_neutrophil_images=False,
    use_patient_data=False,
    transform=transform
)

print("Dataset size:", len(dataset))

for i in range(5):
    sample = dataset[i]
    print(f"Sample {i}:")
    print(f"Image path: {dataset.blood_cell_frame.iloc[i]['image_path']}")
    print(f"Morphology: {sample['morphology']}")
