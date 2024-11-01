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

# file not used

# sys.path.append('/home/tchowdhury/data/code/CMML-v2/townim')
import utils
from dataset import CustomDataset, NEUTROPHIL_CSV_PATH, MONOCYTE_CSV_PATH

cache_dir = "../cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir # set cache directory

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=4, help='fold_id')
parser.add_argument('--data_type', type=str, default='neutrophil', choices=('monocyte', 'neutrophil'), help='data type')
args = parser.parse_args()

set_id = int(args.fold)

# Create data loaders
data_type = args.data_type # neutrophil, monocyte
CSV_PATH = NEUTROPHIL_CSV_PATH if data_type == 'neutrophil' else MONOCYTE_CSV_PATH
batch_size = 32
IMAGE_SIZE = 352
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225] # Std of ImageNet dataset (used for normalization)


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
# model_dir = f'/home/tchowdhury/data/code/CMML-v2/townim/models/{data_type}_fold_{args.fold}_without_TTA'
model_dir = f"./experiments/{data_type}/{data_type}_fold_{set_id}_with_TTA/model"
new_dir = f"./experiments/{data_type}_clinical"
new_dir_with_fold = new_dir + f"/fold_{set_id}"
os.makedirs(new_dir, exist_ok=True)
os.makedirs(new_dir_with_fold, exist_ok=True)
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
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]
clinical_variable_df = clinical_variable_df.loc[clinical_variable_df["patient_id"] != 2209801848]
clinical_variable_df.loc[clinical_variable_df["patient_id"].isin(rechecked_patient_ids), "morphology"] = 0
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

for id in np.unique(patient_ids):
    indices = np.where(patient_ids == id)[0]
    id_patients.append(id)
    
    # Average CNN logits for all images of the patient
    mean_cnn_logit = np.mean(logits[indices, :], axis=0)
    print("Mean CNN Logit for Patient:", mean_cnn_logit)
    
    ind = np.where(patient_ids_clinical_variables == id)[0]
    
    # XGB logit for the patient
    xgb_logit = preds_prob_clinical_variables[ind, :][0]
    print("XGB Logit for Patient:", xgb_logit)
    
    # Combine the averaged CNN logit with the XGB logit
    combined_logit = (mean_cnn_logit + xgb_logit) / 2
    id_patient_logits.append(combined_logit)
    
    # Average the labels for the patient to determine the ground truth label
    id_patient_labels.append(np.mean(labels[indices], axis=0))
    id_patient_preds.append(combined_logit.argmax())
    
    # Update the correct predictions count
    correct += int(id_patient_preds[-1] == id_patient_labels[-1])

# Convert lists to arrays for further processing
id_patient_logits = np.array(id_patient_logits)
id_patient_labels = np.array(id_patient_labels)
id_patient_preds = np.array(id_patient_preds)

# Calculate accuracy
accuracy = correct / len(id_patients)

# Calculate AUROC using the combined patient-level logits and labels
id_patient_labels_int = torch.tensor(id_patient_labels).round().long()
auroc.update(torch.tensor(id_patient_logits).to(device), id_patient_labels_int.to(device))
auc = float(auroc.compute())
auroc.reset()

# Save metrics to CSV
def save_metrics_csv(fold, accuracy, auc, metrics_path):
    new_metrics = pd.DataFrame([[fold, round(accuracy, 4), round(auc, 4)]], 
                                columns=["fold", "accuracy", "auroc"])
    
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        if fold in df['fold'].values:
            df.loc[df['fold'] == fold] = new_metrics
        else:
            df = pd.concat([df, new_metrics], ignore_index=True)
    else:
        df = new_metrics

    df.to_csv(metrics_path, index=False)

print(f'Patient level with clinical variable => Accuracy: {accuracy*100} AUC: {auc*100}')
save_metrics_csv(int(set_id), accuracy, auc, os.path.join(new_dir, "metrics_patient.csv"))

# Save predictions and logits for each patient
np.savez_compressed(os.path.join(new_dir_with_fold, 'patient_level_with_clinical_variables_results'),
    labels=id_patient_labels,
    preds=id_patient_preds,
    logits=id_patient_logits,
    patient_ids=np.array(id_patients)
)

# Confusion matrix calculation
confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true').to(device)
confmat.update(torch.from_numpy(id_patient_logits).to(device), torch.from_numpy(id_patient_labels).to(device))
confmat_vals = np.around(confmat.compute().cpu().detach().numpy(), 3)

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(confmat_vals)
ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xlabel('Predicted class')
ax.set_ylabel('True class')

# Annotate each cell in the confusion matrix
for i in range(num_classes):
    for j in range(num_classes):
        text = ax.text(j, i, confmat_vals[i, j], ha="center", va="center", color="black", fontsize=12)

ax.set_title("Confusion Matrix on Test [Patient level]")
fig.savefig(os.path.join(model_dir, "patient_level_with_clinical_variables_conf_mat.png"))
plt.close()


