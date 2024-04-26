from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length1 = len(dataset1)
        self.total_length = self.length1 + len(dataset2)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < self.length1:
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - self.length1]
