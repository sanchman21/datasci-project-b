import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

for data_type in ("neutrophil", "monocyte", "monocyte_new_normals"):
    if data_type == "neutrophil":
        CSV_PATH = "../datasets/neutrophil.csv"
    elif data_type == "monocyte":
        CSV_PATH = "../datasets/monocyte_reassigned.csv"
    elif data_type == "monocyte_new_normals":
        CSV_PATH = "../datasets/monocyte_new_normals.csv"
    else:
        raise ValueError("Invalid data type")
    
    df = pd.read_csv(CSV_PATH)
    split_cols = ["set0", "set1", "set2", "set3", "set4"]
    df = df.drop(columns=[col for col in split_cols if col in df.columns], errors='ignore')
    patients_df = df[['patient_id', 'morphology']].drop_duplicates().reset_index(drop=True)
    train_val_patients, test_patients = train_test_split(
        patients_df,
        test_size=0.2,
        random_state=42,
        stratify=patients_df["morphology"]
    )
    test_patient_ids = set(test_patients['patient_id'])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    split_labels_list = []
    train_val_patients = train_val_patients.reset_index(drop=True)
    X = train_val_patients[['patient_id']]
    y = train_val_patients['morphology']
    
    for train_idx, val_idx in skf.split(X, y):
        labels = {}
        for idx in range(len(train_val_patients)):
            labels[train_val_patients.loc[idx, 'patient_id']] = "train"
        for idx in val_idx:
            labels[train_val_patients.loc[idx, 'patient_id']] = "val"
        split_labels_list.append(labels)
        
    for i in range(5):
        col = f"set{i}"
        df[col] = df["patient_id"].apply(lambda pid: "test" if pid in test_patient_ids 
                                        else split_labels_list[i].get(pid, "train"))
        
    df.to_csv(CSV_PATH, index=False)
    
print("Stratified 60:20:20 splits created at the patient level with a fixed test set and consistent train/val splits across sets.")