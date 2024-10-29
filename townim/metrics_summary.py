import pandas as pd 
import os

data_types = ["neutrophil", "monocyte"]
metrics = ["accuracy", "precision", "recall", "f1", "auroc"]
levels = ["image", "image_tta", "patient"]
exp_dir = "./experiments"

for data_type in data_types:
    path1 = exp_dir + f"/{data_type}"
    if os.path.exists(path1):
        print(f"Data Type: {data_type}")
        for level in levels:
            print(f"Level: {level}")
            path2 = path1 + f"/metrics_{level}.csv"
            if os.path.exists(path2):
                df = pd.read_csv(path2)
                for metric in metrics:
                    print(f"{metric}: {round(df[metric].mean()*100, 2)} +- {round(df[metric].std()*100, 2)}")