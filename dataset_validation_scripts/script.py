import pandas as pd

patient_folds = pd.read_csv('patient_folds.csv')
target_df = pd.read_csv('datasets\monocyte.csv')

for fold in range(5):
    target_df[f'set{fold}'] = ''

for index, row in patient_folds.iterrows():
    fold = row['fold']
    fold_type = row['type']
    patient_id = row['patient_id']
    
    target_df.loc[target_df['patient_id'] == patient_id, fold] = fold_type

target_df.to_csv('datasets\monocyte_reassigned.csv', index=False)

