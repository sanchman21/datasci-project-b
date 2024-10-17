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
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb

sys.path.append('/home/tchowdhury/data/code/CMML-v2/townim')
import utils
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH


torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='fold_id')
parser.add_argument('--data_type', type=str, default='monocyte', choices=('monocyte', 'neutrophil'), help='data type')
args = parser.parse_args()

set_id = int(args.fold)

# Create data loaders
data_type = args.data_type # neutrophil, monocyte
CSV_PATH = NEUTROPHIL_CSV_PATH if data_type == 'neutrophil' else MONOCYTE_CSV_PATH
batch_size = 32
IMAGE_SIZE = 352
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)


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
    DataLoader(CustomDataset('test', CSV_PATH, set_id, transform=transform), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for transform in TTAs
]

num_classes = len(set(test_dataset.labels))
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
model_dir = f'/home/tchowdhury/data/code/CMML-v2/townim/models/{data_type}_fold_{args.fold}_without_TTA'
model.load_state_dict(torch.load(os.path.join(model_dir, f'last.pth')))
model.eval()
print(model_dir)

# collect predictions
confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='none').to(device)
auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes).to(device)
with torch.no_grad():
    preds = []
    logits = []
    labels = []
    correct = 0
    # for i, data in tqdm(enumerate(zip(*test_loaders)), total=len(test_loaders[0]), smoothing=0.9, position=0, leave=True,):
    for i, data in enumerate(zip(*test_loaders)):
        inputs, targets = torch.cat([img for img,_ in data], dim=0).to(device).float(), data[0][1].to(device).long()#torch.stack([l.squeeze(0) for _,l in data], dim=0).to(device).long()
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=-1)
        outputs = outputs.reshape(len(data), int(inputs.shape[0]/len(data)), -1).mean(dim=0)
        confmat.update(outputs, targets)
        auroc.update(outputs, targets)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        preds.append(predicted.detach().cpu().numpy())
        labels.append(targets.detach().cpu().numpy())
        logits.append(outputs.detach().cpu().numpy().astype(np.float32))
        # if i==100:break
        
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    logits = np.concatenate(logits, axis=0)
    # print(preds.shape, labels.shape, logits.shape)
    
    accuracy = 100*correct/len(test_dataset)
    auc = 100*float(auroc.compute())
    auroc.reset()
    print(f'[TTA] Image level Accuracy: {accuracy} AUC: {auc}')
    

##### patient level
# clinical_variable_df = pd.read_csv('/home/tchowdhury/data/code/CMML-v2/datasets/patients_fold.csv')
clinical_variable_df = pd.read_csv('../datasets/patients_fold.csv')
feature_columns = ['Age', 'Gender', 'Haemoglobin',
    'MCV', 'White cell count', 'Neutrophil count', 'Monocyte count',
    'Platelet count', 'Blast percentage (PB)', 'LDH'
]
target_column = 'morphology'
n=200
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'nthread': 4,
    'booster': 'gbtree',
}

set_col = f'set{set_id}'
X_train = clinical_variable_df[clinical_variable_df[set_col] == 'train'][feature_columns]#.values
y_train = clinical_variable_df[clinical_variable_df[set_col] == 'train'][target_column]#.values
X_test = clinical_variable_df[clinical_variable_df[set_col] == 'test'][feature_columns]#.values
y_test = clinical_variable_df[clinical_variable_df[set_col] == 'test'][target_column]#.values
model_xgb = xgb.XGBClassifier(**params, importance_type='gain', validate_parameters=True)
model_xgb.fit(X_train, y_train)
preds_prob_clinical_variables = model_xgb.predict_proba(X_test)
patient_ids_clinical_variables = clinical_variable_df[clinical_variable_df[set_col] == 'test']['patient_id'].values


id_patients = []
id_patient_logits = []
id_patient_preds = []
id_patient_labels = []
patient_ids = test_dataset.df['patient_id'].to_numpy()
correct = 0
# confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true').to(device)
for id in np.unique(patient_ids):
    indices = np.where(patient_ids==id)[0]
    # print(id_patient_labels[-1], id_patient_logits[-1])
    id_patients.append(id)
    # add clinical variables
    ind = np.where(patient_ids_clinical_variables==id)[0]
    # id_patient_logits.append( (np.mean(logits[indices,:], axis=0)+preds_prob_clinical_variables[ind, :][0])/2 )
    id_patient_logits.append(np.mean(np.concatenate((logits[indices,:], preds_prob_clinical_variables[ind, :]), axis=0), axis=0) )
    id_patient_labels.append(np.mean(labels[indices], axis=0))
    id_patient_preds.append(id_patient_logits[-1].argmax())
    correct += int(id_patient_preds[-1]==id_patient_labels[-1])
    
    
    
accuracy = 100*correct/len(id_patients)
auroc.update(torch.tensor(logits).to(device), torch.tensor(labels).to(device))
auc = 100*float(auroc.compute())
auroc.reset()

print(f'Patient level with clinical variable => Accuracy: {accuracy} AUC: {auc}')

preds = np.array(id_patient_preds)
labels = np.array(id_patient_labels)
logits = np.array(id_patient_logits)
# print(labels, preds.shape, labels.shape, logits.shape, len(id_patients))

np.savez_compressed(os.path.join(model_dir, 'patient_level_with_clinical_variables_results'),
    labels=labels, 
    preds=preds,
    logits=logits,
    patient_ids=np.array(id_patients)
)

# confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true').to(device)
fig, ax = plt.subplots()
confmat.update(torch.from_numpy(logits).to(device), torch.from_numpy(labels).to(device))
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

ax.set_title("Confusion Matrix on Test [Patient level]")
fig.savefig(os.path.join(model_dir, "patient_level_with_clinical_variables_conf_mat.png"))
plt.close()


