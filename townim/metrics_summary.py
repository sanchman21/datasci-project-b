import pandas as pd 
import os

data_types = ["neutrophil", "monocyte", "neutrophil_clinical", "monocyte_clinical"]
metrics = ["accuracy", "precision", "recall", "f1", "auroc"]
levels = ["image", "image_tta", "patient"]
exp_dir = "./experiments"

for data_type in data_types:
    path1 = exp_dir + f"/{data_type}"
    if os.path.exists(path1):
        print(f"Data Type: {data_type}")
        for level in levels:
            path2 = path1 + f"/metrics_{level}.csv"
            if os.path.exists(path2):
                print(f"Level: {level}")
                df = pd.read_csv(path2)
                columns = list(df.columns)
                for metric in metrics:
                    if metric in columns:
                        print(f"{metric}: {round(df[metric].mean()*100, 2)} +- {round(df[metric].std()*100, 2)}")