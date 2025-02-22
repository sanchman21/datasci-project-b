import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

for data_type in ("neutrophil", "monocyte", "monocyte_new_normals"): # for each data type
    if data_type == "neutrophil": # set path
        CSV_PATH = "../datasets/neutrophil.csv"
    elif data_type == "monocyte":
        CSV_PATH = "../datasets/monocyte_reassigned.csv"
    elif data_type == "monocyte_new_normals":
        CSV_PATH = "../datasets/monocyte_new_normals.csv"
    else:
        raise ValueError("Invalid data type")

    df = pd.read_csv(CSV_PATH) # read data

    split_cols = ["set0", "set1", "set2", "set3", "set4"] # split columns
    df = df.drop(columns=[col for col in split_cols if col in df.columns], errors='ignore') # remove existing split columns

    patients_df = df[['patient_id', 'morphology']].drop_duplicates().reset_index(drop=True) # patient level dataframe

    train_val_patients, test_patients = train_test_split(
        patients_df,
        test_size=0.2,
        random_state=42,
        stratify=patients_df["morphology"]
    ) # train-test split

    test_patient_ids = set(test_patients['patient_id']) # test patient ids

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # create stratified split
    split_labels_list = [] # list to store information for each split
    
    train_val_patients = train_val_patients.reset_index(drop=True) # reset index
    X = train_val_patients[['patient_id']]  
    y = train_val_patients['morphology']
    
    for train_idx, val_idx in skf.split(X, y): # for each split 
        labels = {} # store patient ids of each split
        for idx in range(len(train_val_patients)): # for each id
            labels[train_val_patients.loc[idx, 'patient_id']] = "train" # assign train
        for idx in val_idx: # for each validation id, overwrite train to val
            labels[train_val_patients.loc[idx, 'patient_id']] = "val"
        split_labels_list.append(labels)
    
    for i in range(5): # for each fold
        col = f"set{i}"
        df[col] = df["patient_id"].apply(lambda pid: "test" if pid in test_patient_ids 
                                        else split_labels_list[i].get(pid, "train")) # set split based on the informatino
    
    df.to_csv(CSV_PATH, index=False) # save updated dataset
    
print("Stratified 60:20:20 splits created at the patient level with a fixed test set and consistent train/val splits across sets.")