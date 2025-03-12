import pandas as pd 

monocyte_path = '../datasets/monocyte_reassigned.csv'
monocyte = pd.read_csv(monocyte_path)

rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125, 2209801848]
fold3_val_cmml = monocyte[(monocyte['set3'] == 'val') & (monocyte['morphology'] == 0)]['patient_id'].unique()
fold3_val_cmml = [x for x in fold3_val_cmml if x not in rechecked_patient_ids]
print(fold3_val_cmml)