import pandas as pd
import os

data_type = "monocyte" # neutrophil, monocyte
CSV_PATH = "../datasets/neutrophil.csv" if data_type == "neutrophil" else "../datasets/monocyte_reassigned.csv"
df = pd.read_csv(CSV_PATH)
df["morphology"] = 1-df["morphology"]
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125]
dont_consider = 2209801848
df = df[df["patient_id"] != dont_consider]
# change rechecked_patient_ids morphology from 1 to 0
for patient_id in rechecked_patient_ids:
    df.loc[df["patient_id"] == patient_id, "morphology"] = 0

df2 = pd.DataFrame(columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])

patients_total = 0
cmml_total = 0
normal_total = 0
image_total = 0
cmml_image_total = 0
normal_image_total = 0

for set_id in range(5):
    num_patients = df["patient_id"].nunique()
    # set column names are f"set{set_id}"
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
    new_df = pd.DataFrame([[num_patients, cmml_patients, normal_patients, num_images, cmml_images, normal_images]], columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
    # using pd.concat
    df2 = pd.concat([df2, new_df], ignore_index=True)

new_df = pd.DataFrame([[patients_total, cmml_total, normal_total, image_total, cmml_image_total, normal_image_total]], columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
df2 = pd.concat([df2, new_df], ignore_index=True)
index = [f"Fold {i}" for i in range(0, 5)] + ["Total"]
df2.index = index
df2.rename_axis("Fold", inplace=True)
df2.to_csv(f"../datasets/summary_train_{data_type}.csv", index=True)

df3 = pd.DataFrame(columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])

patients_total = 0
cmml_total = 0
normal_total = 0
image_total = 0
cmml_image_total = 0
normal_image_total = 0
for set_id in range(5):
    num_patients = df["patient_id"].nunique()
    # set column names are f"set{set_id}"
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
    new_df = pd.DataFrame([[num_patients, cmml_patients, normal_patients, num_images, cmml_images, normal_images]], columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
    # using pd.concat
    df3 = pd.concat([df3, new_df], ignore_index=True)

new_df = pd.DataFrame([[patients_total, cmml_total, normal_total, image_total, cmml_image_total, normal_image_total]], columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
df3 = pd.concat([df3, new_df], ignore_index=True)
index = [f"Fold {i}" for i in range(0, 5)] + ["Total"]
df3.index = index
df3.rename_axis("Fold", inplace=True)
df3.to_csv(f"../datasets/summary_test_{data_type}.csv", index=True)