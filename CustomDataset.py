from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch


class CustomDataset(Dataset):
    def __init__(self, csv_file, fold, mode, transform=None):
        """
        Args:
            csv_file (string): CSV文件的路径。
            fold (int): 指定哪个fold作为测试集（0到4之间）。
            mode (string): 指定返回"train"还是"test"数据。
            transform (callable, optional): 可选的变换操作，用于对样本进行处理。
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.fold = fold
        self.mode = mode
        # 根据mode和fold生成一个掩码，以便于在__getitem__中使用
        self.mask = self.data_frame[f'set{fold}'] == mode

    def __len__(self):
        return len(self.data_frame[self.mask])

    def __getitem__(self, idx):
        # 应用掩码以获取正确的行
        filtered_idx = self.data_frame[self.mask].iloc[idx].name
        row = self.data_frame.iloc[filtered_idx]
        
        # 加载图像并应用转换
        img_name = row['image_path']
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        
        # 获取标签并将其转换为张量
        label = torch.tensor(row['morphology'], dtype=torch.long)

        return image, label