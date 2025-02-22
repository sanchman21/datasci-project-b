'''
This script is used to generate a summary of the dataset.
It generates three csv files: one for the training set, one for the validation set, and one for the test set, including information for each fold.
'''
import pandas as pd
import argparse

# argument parser
parser = argparse.ArgumentParser(description="Generate a dataset summary.")
parser.add_argument("--data_type", type=str, choices=["neutrophil", "monocyte", "monocyte_new_normals"], required=True, help="Specify the data type: neutrophil, monocyte, or monocyte_new_normals")
args = parser.parse_args()

data_type = args.data_type

# specify csv path
if data_type == "neutrophil":
    CSV_PATH = "../datasets/neutrophil.csv"
elif data_type == "monocyte":
    CSV_PATH = "../datasets/monocyte_reassigned.csv"
elif data_type == "monocyte_new_normals":
    CSV_PATH = "../datasets/monocyte_new_normals.csv"
else:
    raise ValueError("Invalid data type")

df = pd.read_csv(CSV_PATH)
df["morphology"] = 1 - df["morphology"]  # 1 for normal, 0 for cmml in original dataset so flip it

# the following patients had their labels changed to 0 (normal) after experts rechecked the images
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]
dont_consider = 2209801848  # this patient's label is not considered since it was not clear
df = df[df["patient_id"] != dont_consider]

# change rechecked_patient_ids morphology from 1 to 0
for patient_id in rechecked_patient_ids:
    df.loc[df["patient_id"] == patient_id, "morphology"] = 0

# ----------------- Summary for Training Set -----------------
# create a new dataframe to store the summary for train set
df2 = pd.DataFrame(columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])

# define variables to store the total values
patients_total = cmml_total = normal_total = image_total = cmml_image_total = normal_image_total = 0

for set_id in range(5):  # iterate for each fold
    num_patients = df["patient_id"].nunique()  # total number of patients
    cmml_patients = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "train")]["patient_id"].nunique()
    normal_patients = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "train")]["patient_id"].nunique()
    num_images = df[df[f"set{set_id}"] == "train"].shape[0]
    cmml_images = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "train")].shape[0]
    normal_images = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "train")].shape[0]
    
    patients_total += num_patients
    cmml_total += cmml_patients
    normal_total += normal_patients
    image_total += num_images
    cmml_image_total += cmml_images
    normal_image_total += normal_images
    
    new_df = pd.DataFrame([[num_patients, cmml_patients, normal_patients, num_images, cmml_images, normal_images]], 
                           columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
    df2 = pd.concat([df2, new_df], ignore_index=True)

new_df = pd.DataFrame([[patients_total, cmml_total, normal_total, image_total, cmml_image_total, normal_image_total]], 
                       columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
df2 = pd.concat([df2, new_df], ignore_index=True)
index = [f"Fold {i}" for i in range(0, 5)] + ["Total"]
df2.index = index
df2.rename_axis("Fold", inplace=True)
df2.to_csv(f"../datasets/summary_train_{data_type}.csv", index=True)

# ----------------- Summary for Validation Set -----------------
# create a new dataframe to store the summary for validation set
df_val = pd.DataFrame(columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])

patients_total = cmml_total = normal_total = image_total = cmml_image_total = normal_image_total = 0
for set_id in range(5):
    num_patients = df["patient_id"].nunique()
    cmml_patients = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "val")]["patient_id"].nunique()
    normal_patients = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "val")]["patient_id"].nunique()
    num_images = df[df[f"set{set_id}"] == "val"].shape[0]
    cmml_images = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "val")].shape[0]
    normal_images = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "val")].shape[0]
    
    patients_total += num_patients
    cmml_total += cmml_patients
    normal_total += normal_patients
    image_total += num_images
    cmml_image_total += cmml_images
    normal_image_total += normal_images
    
    new_df = pd.DataFrame([[num_patients, cmml_patients, normal_patients, num_images, cmml_images, normal_images]], 
                           columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
    df_val = pd.concat([df_val, new_df], ignore_index=True)

new_df = pd.DataFrame([[patients_total, cmml_total, normal_total, image_total, cmml_image_total, normal_image_total]], 
                       columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
df_val = pd.concat([df_val, new_df], ignore_index=True)
df_val.index = index
df_val.rename_axis("Fold", inplace=True)
df_val.to_csv(f"../datasets/summary_val_{data_type}.csv", index=True)

# ----------------- Summary for Test Set -----------------
# create a new dataframe to store the summary for test set, using the same column format as train and val
df_test = pd.DataFrame(columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])

patients_total = cmml_total = normal_total = image_total = cmml_image_total = normal_image_total = 0
for set_id in range(5):
    num_patients = df["patient_id"].nunique()  # total number of patients
    cmml_patients = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "test")]["patient_id"].nunique()
    normal_patients = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "test")]["patient_id"].nunique()
    num_images = df[df[f"set{set_id}"] == "test"].shape[0]
    cmml_images = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "test")].shape[0]
    normal_images = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "test")].shape[0]
    
    patients_total += num_patients
    cmml_total += cmml_patients
    normal_total += normal_patients
    image_total += num_images
    cmml_image_total += cmml_images
    normal_image_total += normal_images
    
    new_df = pd.DataFrame([[num_patients, cmml_patients, normal_patients, num_images, cmml_images, normal_images]], 
                           columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
    df_test = pd.concat([df_test, new_df], ignore_index=True)

new_df = pd.DataFrame([[patients_total, cmml_total, normal_total, image_total, cmml_image_total, normal_image_total]], 
                       columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
df_test = pd.concat([df_test, new_df], ignore_index=True)
df_test.index = index
df_test.rename_axis("Fold", inplace=True)
df_test.to_csv(f"../datasets/summary_test_{data_type}.csv", index=True)

print(f"Summaries saved for {data_type}.")