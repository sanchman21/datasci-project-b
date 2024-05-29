import pandas as pd
import numpy as np

df = pd.read_csv('monocyte_resnet50\cleaned_neutrophils.csv')
df = df.iloc[:, 1:]

df.drop(['set0', 'set1', 'set2', 'set3', 'set4'], axis=1, inplace=True)


np.random.seed(1129)
patients = df['patient_id'].unique()
np.random.shuffle(patients)

fold_size = len(patients) // 5
folds = {f'set{i}': {} for i in range(5)}

for patient_id in patients:
    for i in range(5):
        folds[f'set{i}'][patient_id] = 'train'

for i in range(5):
    test_patients = patients[i*fold_size:(i+1)*fold_size]
    for patient_id in test_patients:
        folds[f'set{i}'][patient_id] = 'test'

for i in range(5):
    df[f'set{i}'] = df['patient_id'].map(folds[f'set{i}'])
df.to_csv('new_split.csv', index=False)
