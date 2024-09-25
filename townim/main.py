import numpy as np
import pandas as pd
import os, random, sys, argparse, torchvision, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
import torchvision.transforms as T
import torchmetrics
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# sys.path.append('/home/tchowdhury/data/code/CMML-v2/townim')
sys.path.append('./townim')
import utils
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH

if torch.cuda.is_available(): # if cuda is available
    torch.cuda.empty_cache() # empty the cache
    device = "cuda" # set the device to cuda
elif torch.backends.mps.is_available(): # if mps is available
    torch.mps.empty_cache() # empty the cache
    device = "mps" # set the device to mps
else: # otherwise
    torch.cpu.empty_cache() # empty the cache
    device = "cpu" # set the device to cpu
print(f"Using device: {device}") # print the device being used

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='fold_id')
args = parser.parse_args()

set_id = int(args.fold)

# Training loop
data_type = 'neutrophil' # neutrophil, monocyte
is_tta = True
num_epochs = 50 if data_type == 'neutrophil' else 100
best_test_acc = 0
best_epoch = 0
train_losses = []  # To store training losses
val_losses = []    # To store validation losses
train_accuracies = []  # To store training accuracies
val_accuracies = []    # To store validation accuracies
logs = ''
CSV_PATH = NEUTROPHIL_CSV_PATH if data_type == 'neutrophil' else MONOCYTE_CSV_PATH


# output_dir = f'./models/{data_type}_fold_{args.fold}'
output_dir = f'./models/{data_type}_fold_{args.fold}'
output_dir+='_with_TTA' if is_tta else '_without_TTA'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
# shutil.copyfile('./main.py', os.path.join(output_dir, 'main.py'))
shutil.copyfile('./townim/main.py', os.path.join(output_dir, 'main.py'))
utils.set_random_seed(123)

# Create data loaders
batch_size = 32
IMAGE_SIZE = 352
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

train_transform = T.Compose([
    # T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(90),
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

class FixedRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return T.functional.rotate(x, self.angle)


TTAs = [
    test_transform, 
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]), 
    T.Compose([T.RandomVerticalFlip(p=1.0), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]), 
    T.Compose([FixedRotation(angle=45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),  
    T.Compose([T.RandomHorizontalFlip(p=1.0), FixedRotation(angle=45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]), 
    T.Compose([T.RandomVerticalFlip(p=1.0), FixedRotation(angle=45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]), 
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]),
    T.Compose([T.RandomHorizontalFlip(p=1.0), T.RandomVerticalFlip(p=1.0), FixedRotation(angle=45), T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
]

test_dataset = CustomDataset('test', CSV_PATH, set_id, transform=test_transform)
test_loaders =[
    DataLoader(CustomDataset('test', CSV_PATH, set_id, transform=transform), batch_size=8, shuffle=False, pin_memory=True, num_workers=4)
    for transform in TTAs
]

# dataset
train_dataset = CustomDataset('train', CSV_PATH, set_id, transform=train_transform)
print("Training dataset stats [Normal, CMML]:", train_dataset.disease_count)
val_dataset = CustomDataset('test', CSV_PATH, set_id, transform=test_transform)
print("Test dataset stats [Normal, CMML]:", val_dataset.disease_count)

# For unbalanced dataset we create a weighted sampler                       
weights = utils.make_weights_for_balanced_classes(train_dataset.labels, device)
weighted_sampler = sampler.WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, 
                        sampler=weighted_sampler,
                        num_workers=4, worker_init_fn=utils.worker_init_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
print("Dataset loaded")

num_classes = len(set(train_dataset.labels))
# Define the model architecture (ResNet-50 as an example)
print("Model: Resnet50")
model = models.resnet50(weights='IMAGENET1K_V1')
hidden_layer_size = 512
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, hidden_layer_size),
    nn.ReLU(),
    # nn.BatchNorm1d(hidden_layer_size),
    # nn.Dropout(0.2),
    nn.Linear(hidden_layer_size, num_classes)
)

model = model.to(device)

# Define loss function and optimizer
count = torch.bincount(torch.tensor(train_dataset.labels)).to(device)
class_weight = len(train_dataset.labels) / count # (count*num_classes)
# class_weight = torch.tensor(
#     [1.0 / ((acc.item() + 1e-5) * cls_count.item()) for acc, cls_count in zip(val_class_accuracy, count)]
# ).to(device)

print('Loss class weight:', class_weight)
class_weight = None
# criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
params = list(model.parameters())
optimizer = optim.SGD(params, lr=1e-3, weight_decay=5e-4, momentum=0.9)
end_factor = 1e-5/1e-3
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=end_factor, total_iters=num_epochs)

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='weighted').to(device)
confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true').to(device)
class_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)

for epoch in range(num_epochs):
    t = tqdm(enumerate(train_loader, 0), total=len(train_loader), 
                smoothing=0.9, position=0, leave=True, 
                desc="Train: Epoch: "+str(epoch+1)+"/"+str(num_epochs))
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in t:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels) # criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        outputs = F.softmax(outputs, dim=-1)
        train_accuracy = accuracy(outputs, labels)
        # torchvision.utils.save_image(utils.denormalize(inputs[:30], IMAGENET_MEAN, IMAGENET_STD), 
        #                             os.path.join(output_dir, f"samples.jpg"), nrow=10, normalize=True, padding=0)
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    train_accuracy = accuracy.compute() 
    train_accuracies.append(float(train_accuracy))
    accuracy.reset()
    
    # Validation
    model.eval()
    val_correct = 0
    val_loss = 0.0
    
    with torch.no_grad():
        if not is_tta:
            t = tqdm(enumerate(val_loader, 0), total=len(val_loader), 
                    smoothing=0.9, position=0, leave=True, 
                    desc="Val: Epoch: "+str(epoch+1)+"/"+str(num_epochs))
            for i, (inputs, labels) in t:
                inputs, labels = inputs.to(device).float(), labels.to(device).long()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels) # criterion(outputs, labels)
                val_loss += loss.item()
                outputs = F.softmax(outputs, dim=-1)
                val_accuracy = accuracy(outputs, labels)
                confmat.update(outputs, labels)
                val_class_accuracy = class_accuracy(outputs, labels)
        else:
            t = tqdm(enumerate(zip(*test_loaders)), total=len(test_loaders[0]), 
                    smoothing=0.9, position=0, leave=True, 
                    desc="Val: Epoch: "+str(epoch+1)+"/"+str(num_epochs))
            for i, data in t:
                inputs, labels = torch.cat([img for img,_ in data], dim=0).to(device).float(), data[0][1].to(device).long()
                outputs = model(inputs)
                outputs = outputs.reshape(len(data), int(inputs.shape[0]/len(data)), -1).mean(dim=0)
                loss = F.cross_entropy(outputs, labels) # criterion(outputs, labels)
                val_loss += loss.item()
                outputs = F.softmax(outputs, dim=-1)
                val_accuracy = accuracy(outputs, labels)
                confmat.update(outputs, labels)
                val_class_accuracy = class_accuracy(outputs, labels)
    
    val_class_accuracy = class_accuracy.compute()   

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = accuracy.compute() 
    val_accuracies.append(float(val_accuracy))

    test_loss = val_loss # test_loss / len(test_loader)
    
    # Calculate metrics for test data
    test_accuracy = val_accuracy
    
    # scheduler
    scheduler.step()
    lr_log = f"LR: {optimizer.param_groups[0]['lr']}" # scheduler._last_lr
    print(lr_log)
    logs+=lr_log+'\n'
    
    # Print and log epoch results
    train_results = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Training Accuracy: {train_accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
    print(train_results)
    test_results = f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}"
    print(test_results)
    logs+=train_results+'\n'+test_results+'\n'
    # print(confmat.compute())
    
    # Save the model checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, f'last.pth'))
    # torchvision.utils.save_image(utils.denormalize(inputs[:30, :,:,:], IMAGENET_MEAN, IMAGENET_STD), os.path.join(output_dir, f"samples.jpg"), nrow=10, normalize=True, padding=0)
    
    if best_test_acc <= test_accuracy and epoch!=0:
        best_epoch = epoch+1
        log = f"Improve accuracy from {best_test_acc} to {test_accuracy}"
        print(log)
        logs+=log+"\n"
        best_test_acc = test_accuracy
        torch.save(model.state_dict(), os.path.join(output_dir, f'best.pth'))
        
        # fig, ax = confmat.plot()
        fig, ax = plt.subplots()
        confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)
        im = ax.imshow(confmat_vals)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('True class')

        # Loop over data dimensions and create text annotations.
        for i in range(num_classes):
            for j in range(num_classes):
                text = ax.text(j, i, confmat_vals[i, j],ha="center", va="center", color="black", fontsize=12)

        ax.set_title("Confusion Matrix on Test for best model")
        fig.savefig(os.path.join(output_dir, "conf_mat_best.png"))
        plt.close()
    
    # resetting all metrics
    accuracy.reset(); class_accuracy.reset(); confmat.reset()
    
# Save the printed outputs to a log.txt file
with open(os.path.join(output_dir, 'log.txt'), 'w') as log_file:
    log_file.write(logs)
    log_file.write(f'Best test accuracy: {best_test_acc} in epoch {best_epoch}')

# Save the loss and accuracy graphs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Validation Accuracy')

plt.savefig(os.path.join(output_dir, 'loss_accuracy_graph.png'))
plt.close()

print("CMML classifier model completed")
print("Model saved location :", output_dir)
