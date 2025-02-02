import pandas as pd
import os

# Paths
existing_dataset_path = "../datasets/monocyte_reassigned.csv"
new_images_base_path = "../../Data/Normals/"
new_cases_output_path = "../datasets/new_normals.csv"
merged_output_path = "../datasets/merged_monocyte_reassigned.csv"

# Load existing dataset
existing_df = pd.read_csv(existing_dataset_path)

# Simulate new normal cases dataset
new_cases = []
subfolders = ["Normals", "Test Cases (high monocyte count, but not CMML)"]  # Subfolders containing patient data

for subfolder in subfolders:
    folder_path = os.path.join(new_images_base_path, subfolder)
    for patient_id in os.listdir(folder_path):
        patient_path = os.path.join(folder_path, patient_id)
        if os.path.isdir(patient_path):  # Ensure it's a directory
            for img in os.listdir(patient_path):
                image_path = f"Normals/{subfolder}/{patient_id}/{img}"
                accession_id = patient_id  # Patient ID as Accession number
                new_cases.append([
                    image_path,  # Adjusted image path
                    subfolder,  # Dataset name placeholder
                    accession_id,  # Extracted patient ID
                    1,  # Morphology (normal)
                    "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",  # Placeholder values
                    "train", "train", "train", "train", "test",  # Random split (adjustable)
                    accession_id  # Accession number
                ])

# Create DataFrame for new normal cases
columns = existing_df.columns  # Ensure the same column structure
new_cases_df = pd.DataFrame(new_cases, columns=columns)

# Remove duplicates based on Accession number
new_cases_df.drop_duplicates(subset=["Accession number"], keep="first", inplace=True)

# Merge with existing dataset
merged_df = pd.concat([existing_df, new_cases_df], ignore_index=True)
merged_df.drop_duplicates(subset=["Accession number"], keep="first", inplace=True)

# Save datasets
new_cases_df.to_csv(new_cases_output_path, index=False)
merged_df.to_csv(merged_output_path, index=False)

print(f"New cases saved to {new_cases_output_path}")
print(f"Merged dataset saved to {merged_output_path}")