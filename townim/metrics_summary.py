'''
This script is used to print the mean and standard deviations of the metrics for each data type and level.
'''

# import libraries
import pandas as pd 
import os

# define the data types, metrics, levels, and the experiment directory
data_types = ["neutrophil", "monocyte", "neutrophil_clinical", "monocyte_clinical", "monocyte_new_normals"]
metrics = ["accuracy", "precision", "recall", "f1", "auroc"]
levels = ["image", "image_tta", "patient"]
exp_dir = "./experiments"

for data_type in data_types: # iterate over the data types
    path1 = exp_dir + f"/{data_type}" # define the path to the data type
    if os.path.exists(path1): # check if the path exists
        print(f"Data Type: {data_type}") # print the data type
        if data_type in ["neutrophil", "monocyte", "monocyte_new_normals"]:
            path1 += "/train"
        for level in levels: # iterate over the levels
            path2 = path1 + f"/metrics_{level}.csv" # define the path to the metrics file
            if os.path.exists(path2): # check if the path exists
                print(f"Level: {level}") # print the level
                df = pd.read_csv(path2) # read the metrics file
                columns = list(df.columns) # get the columns of the dataframe
                for metric in metrics: # iterate over the metrics
                    if metric in columns: # check if the metric is in the columns
                        print(f"{metric}: {round(df[metric].mean()*100, 2)} +- {round(df[metric].std()*100, 2)}") 