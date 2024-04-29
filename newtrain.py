import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
import CustomDataset
if __name__ == '__main__':
    # 数据集CSV文件路径
    csv_file_path = 'cleaned_master.csv'
    num_epochs = 100
    input_size = 352
    batch_size = 32
    num_workers = 8
    optimizer = 'sgd'
    weight_decay = 5e-4
    learning_rate = 1e-3
    learning_rate_end = 1e-5
    momentum = 0.9


    # 1. 数据预处理和数据加载器
    data_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建训练和测试数据集
    train_dataset = CustomDataset.CustomDataset(csv_file=csv_file_path, fold=0, mode='train', transform=data_transforms)
    test_dataset = CustomDataset.CustomDataset(csv_file=csv_file_path, fold=0, mode='test', transform=data_transforms)

    # 创建DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 2. 创建ResNet50模型
    model = models.resnet50(pretrained=True)

    # 修改最后的全连接层以匹配你的数据集的类别数
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Replace number_of_classes with your actual number.

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate_end)

    # 4. 训练循环
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        counter = 0
        for inputs, labels in train_loader:
            # print(counter)
            counter =  counter + 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        scheduler.step()  # Update learning rate.

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # 可以添加代码进行测试集的评估
    # ...

