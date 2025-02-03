import pandas as pd
import os
import random
import numpy as np

# paths
existing_dataset_path = "../datasets/monocyte_reassigned.csv"
new_images_base_path = "../../Data/Normals/"
new_cases_output_path = "../datasets/new_normals.csv"
merged_output_path = "../datasets/monocyte_new_normals.csv"

# load existing dataset
existing_df = pd.read_csv(existing_dataset_path)

# simulate new normal cases dataset
new_cases = []
subfolders = ["Normals", "Test Cases (high monocyte count, but not CMML)"]  # subfolders containing patient data

for subfolder in subfolders:
    folder_path = os.path.join(new_images_base_path, subfolder)
    for patient_id in os.listdir(folder_path):
        patient_path = os.path.join(folder_path, patient_id)
        if os.path.isdir(patient_path):  # ensure it's a directory
            for img in os.listdir(patient_path):
                image_path = f"Normals/{subfolder}/{patient_id}/{img}"
                accession_id = patient_id  # patient id as accession number
                new_cases.append([
                    image_path,  # adjusted image path
                    subfolder,  # dataset name placeholder
                    accession_id,  # extracted patient ID
                    1,  # morphology (normal) (current dataset has 1 for normal)
                    "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",  # placeholder values
                    "", "", "", "", "",  # placeholder for split
                    accession_id  # accession number
                ])

# create DataFrame for new normal cases
columns = existing_df.columns  # ensure the same column structure
new_cases_df = pd.DataFrame(new_cases, columns=columns) # convert to dataframe

# remove duplicates based on Accession number
new_cases_df.drop_duplicates(subset=["Accession number"], keep="first", inplace=True)

# kfold Split (k=5)
patients = new_cases_df["Accession number"].unique()
random.shuffle(patients)
k = 5
folds = np.array_split(patients, k)

for i in range(k):
    test_patients = set(folds[i])
    train_patients = set(patients) - test_patients
    new_cases_df[f"set{i}"] = new_cases_df["Accession number"].apply(lambda x: "test" if x in test_patients else "train")

# merge with existing dataset
merged_df = pd.concat([existing_df, new_cases_df], ignore_index=True)
merged_df.drop_duplicates(subset=["Accession number"], keep="first", inplace=True)

# save datasets
new_cases_df.to_csv(new_cases_output_path, index=False)
merged_df.to_csv(merged_output_path, index=False)

print(f"New cases saved to {new_cases_output_path}")
print(f"Merged dataset saved to {merged_output_path}")