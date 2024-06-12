import pandas as pd

df = pd.read_csv('datasets/neutrophil.csv')

folds = {'fold': [], 'type': [], 'patient_id': []}

for fold in range(5):
    fold_column = f'set{fold}'
    for partition in ['train', 'test']:
        patient_ids = df[df[fold_column] == partition]['patient_id'].unique()
        for patient_id in patient_ids:
            folds['fold'].append(fold_column)
            folds['type'].append(partition)
            folds['patient_id'].append(patient_id)

result_df = pd.DataFrame(folds)

result_df.to_csv('patient_folds.csv', index=False)

