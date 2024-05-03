import pandas as pd
import os

df1 = pd.read_csv('cleaned_master.csv')  
df2 = pd.read_csv('cleaned_neutrophils.csv') 
columns_needed = ['image_path', 'patient_id', 'morphology', 'Age', 'Gender', 'Haemoglobin', 'MCV', 'White cell count', 'Neutrophil count', 'Monocyte count', 'Platelet count', 'Blast percentage (PB)', 'LDH']
new_entries = pd.DataFrame(columns=columns_needed)

folder_path = 'neutrophil_images_kevin\Compilation of Neutrophil Images'

for subdir in os.listdir(folder_path):
    subdir_path = os.path.join(folder_path, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(subdir_path, file)
                patient_id = int(subdir)
                if patient_id in df1['patient_id'].values:
                    df1_row = df1[df1['patient_id'] == patient_id].iloc[0]
                    if patient_id in df2['Accession number'].values:
                        df2_row = df2[df2['Accession number'] == patient_id].iloc[0]
                        new_entry = {
                            'image_path': image_path,
                            'patient_id': patient_id,
                            'morphology': df1_row['morphology'],
                            'Age': df2_row['Age'],
                            'Gender': df2_row['Gender'],
                            'Haemoglobin': df2_row['Haemoglobin'],
                            'MCV': df2_row['MCV'],
                            'White cell count': df2_row['White cell count'],
                            'Neutrophil count': df2_row['Neutrophil count'],
                            'Monocyte count': df2_row['Monocyte count'],
                            'Platelet count': df2_row['Platelet count'],
                            'Blast percentage (PB)': df2_row['Blast percentage (PB)'],
                            'LDH': df2_row['LDH']
                        }
                        new_entries = new_entries._append(new_entry, ignore_index=True)
                    else:
                        print(f"No data found in df2 for patient_id {patient_id}")
                else:
                    print(f"No data found in df1 for patient_id {patient_id}")

new_entries.to_csv('NeutrophilImages.csv', index=False)
