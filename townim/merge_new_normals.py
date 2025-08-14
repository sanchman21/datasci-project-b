import pandas as pd
import os
import random
import numpy as np

existing_dataset_path = "../datasets/monocyte_reassigned.csv"
new_images_base_path = "../../Data/Normals/"
new_cases_output_path = "../datasets/new_normals.csv"
merged_output_path = "../datasets/monocyte_new_normals.csv"
existing_df = pd.read_csv(existing_dataset_path)
new_cases = []
subfolders = ["Normals", "Test Cases (high monocyte count, but not CMML)"]

for subfolder in subfolders:
    folder_path = os.path.join(new_images_base_path, subfolder)
    
    for patient_id in os.listdir(folder_path):
        patient_path = os.path.join(folder_path, patient_id)
        
        if os.path.isdir(patient_path):
            image_counter = 0
            
            for img in os.listdir(patient_path):
                image_path = f"Normals/{subfolder}/{patient_id}/{img}"
                accession_id = patient_id
                new_cases.append([
                    image_path,
                    subfolder,
                    accession_id,
                    1,
                    "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                    "", "", "", "", "",
                    accession_id
                ])
                
columns = existing_df.columns
new_cases_df = pd.DataFrame(new_cases, columns=columns)
new_cases_df.drop_duplicates(subset=["image_path"], keep="first", inplace=True)
patients = new_cases_df["Accession number"].unique()
random.shuffle(patients)
k = 5
folds = np.array_split(patients, k)

for i in range(k):
    test_patients = set(folds[i])
    train_patients = set(patients) - test_patients
    new_cases_df[f"set{i}"] = new_cases_df["Accession number"].apply(lambda x: "test" if x in test_patients else "train")
    
merged_df = pd.concat([existing_df, new_cases_df], ignore_index=True)
merged_df.drop_duplicates(subset=["image_path"], keep="first", inplace=True)
new_cases_df.to_csv(new_cases_output_path, index=False)
merged_df.to_csv(merged_output_path, index=False)
print(f"New cases saved to {new_cases_output_path}")
print(f"Merged dataset saved to {merged_output_path}")