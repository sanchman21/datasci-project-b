'''
This script is used to generate a summary of the dataset.
It generates two csv files, one for the training set and one for the test set, including information for each fold
'''

# import libraries
import pandas as pd
import os

data_type = "monocyte" # neutrophil, monocyte
# specify CSV path (try to use relative path) 
CSV_PATH = "../datasets/neutrophil.csv" if data_type == "neutrophil" else "../datasets/monocyte_reassigned.csv"
df = pd.read_csv(CSV_PATH)
df["morphology"] = 1-df["morphology"] # 1 for normal, 0 for CMML in original dataset so flip it
# the following patients had their labels changed to 0 (normal) after experts rechecked the images
rechecked_patient_ids = [2209722160, 2209801259, 2209801421, 2209802027, 2209802125] 
dont_consider = 2209801848 # this patient's label is not considered since it was not clear
df = df[df["patient_id"] != dont_consider]
# change rechecked_patient_ids morphology from 1 to 0
for patient_id in rechecked_patient_ids:
    df.loc[df["patient_id"] == patient_id, "morphology"] = 0

# create a new dataframe to store the summary for train set
df2 = pd.DataFrame(columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])

# define variables to store the total values
patients_total = 0
cmml_total = 0
normal_total = 0
image_total = 0
cmml_image_total = 0
normal_image_total = 0

for set_id in range(5): # iterate for each fold
    num_patients = df["patient_id"].nunique() # total number of patients
    # set column names are f"set{set_id}" in the CSV file
    cmml_patients = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "train")]["patient_id"].nunique() # get the number of CMML patients in the training set
    normal_patients = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "train")]["patient_id"].nunique() # get the number of normal patients in the training set
    num_images = df[df[f"set{set_id}"] == "train"].shape[0] # get the number of images in the training set
    cmml_images = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "train")].shape[0] # get the number of CMML images in the training set
    normal_images = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "train")].shape[0] # get the number of normal images in the training set
    patients_total += num_patients # add the number of patients to the total
    cmml_total += cmml_patients # add the number of CMML patients to the total
    normal_total += normal_patients # add the number of normal patients to the total
    image_total += num_images # add the number of images to the total
    cmml_image_total += cmml_images # add the number of CMML images to the total
    normal_image_total += normal_images # add the number of normal images to the total
    # create a new dataframe to store the summary for the current fold
    new_df = pd.DataFrame([[num_patients, cmml_patients, normal_patients, num_images, cmml_images, normal_images]], columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
    df2 = pd.concat([df2, new_df], ignore_index=True) # concatenate the dataframes

# create a new dataframe of the total values
new_df = pd.DataFrame([[patients_total, cmml_total, normal_total, image_total, cmml_image_total, normal_image_total]], columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
df2 = pd.concat([df2, new_df], ignore_index=True) # concatenate the dataframe with total values
index = [f"Fold {i}" for i in range(0, 5)] + ["Total"] # create an index for the dataframe
df2.index = index # set the index
df2.rename_axis("Fold", inplace=True) # rename the axis
df2.to_csv(f"../datasets/summary_train_{data_type}.csv", index=True) # save the dataframe to a CSV file

# create a new dataframe to store the summary for test set
df3 = pd.DataFrame(columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])

# define variables to store the total values
patients_total = 0
cmml_total = 0
normal_total = 0
image_total = 0
cmml_image_total = 0
normal_image_total = 0
for set_id in range(5): # iterate for each fold
    num_patients = df["patient_id"].nunique() # total number of patients
    # set column names are f"set{set_id}" in the CSV file
    cmml_patients = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "test")]["patient_id"].nunique() # get the number of CMML patients in the test set
    normal_patients = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "test")]["patient_id"].nunique() # get the number of normal patients in the test set
    num_images = df[df[f"set{set_id}"] == "test"].shape[0] # get the number of images in the test set
    cmml_images = df[(df["morphology"] == 1) & (df[f"set{set_id}"] == "test")].shape[0] # get the number of CMML images in the test set
    normal_images = df[(df["morphology"] == 0) & (df[f"set{set_id}"] == "test")].shape[0] # get the number of normal images in the test set
    patients_total += num_patients # add the number of patients to the total
    cmml_total += cmml_patients # add the number of CMML patients to the total
    normal_total += normal_patients # add the number of normal patients to the total
    image_total += num_images # add the number of images to the total
    cmml_image_total += cmml_images # add the number of CMML images to the total
    normal_image_total += normal_images # add the number of normal images to the total
    # create a new dataframe to store the summary for the current fold
    new_df = pd.DataFrame([[num_patients, cmml_patients, normal_patients, num_images, cmml_images, normal_images]], columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
    df3 = pd.concat([df3, new_df], ignore_index=True) # concatenate the dataframes

# create a new dataframe of the total values
new_df = pd.DataFrame([[patients_total, cmml_total, normal_total, image_total, cmml_image_total, normal_image_total]], columns=["Patients", "CMML (patients)", "Normal (patients)", "Images", "CMML (images)", "Normal (images)"])
df3 = pd.concat([df3, new_df], ignore_index=True) # concatenate the dataframe with total values
index = [f"Fold {i}" for i in range(0, 5)] + ["Total"] # create an index for the dataframe
df3.index = index # set the index
df3.rename_axis("Fold", inplace=True) # rename the axis
df3.to_csv(f"../datasets/summary_test_{data_type}.csv", index=True) # save the dataframe to a CSV file