import pandas as pd 

monocyte_path = '../datasets/monocyte_reassigned.csv'
monocyte = pd.read_csv(monocyte_path)

rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125, 2209801848]

fold0_train_cmml = monocyte[(monocyte['set0'] == 'train') & (monocyte['morphology'] == 0)]['patient_id'].unique().tolist()
fold1_train_cmml = monocyte[(monocyte['set1'] == 'train') & (monocyte['morphology'] == 0)]['patient_id'].unique().tolist()
fold2_train_cmml = monocyte[(monocyte['set2'] == 'train') & (monocyte['morphology'] == 0)]['patient_id'].unique().tolist()

fold3_train_cmml = monocyte[(monocyte['set3'] == 'train') & (monocyte['morphology'] == 0)]['patient_id'].unique().tolist()
fold4_train_cmml = monocyte[(monocyte['set4'] == 'train') & (monocyte['morphology'] == 0)]['patient_id'].unique().tolist()

combined_train_cmml = fold0_train_cmml + fold1_train_cmml + fold4_train_cmml + rechecked_patient_ids
unique_to_fold3 = [x for x in fold3_train_cmml if x not in combined_train_cmml]
print(unique_to_fold3)