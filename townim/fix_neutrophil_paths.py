import pandas as pd 
from dataset import NEUTROPHIL_CSV_PATH

def normalize_path(path):
    if isinstance(path, str):
        components = path.split('\\')
        return '/'.join(components)
    return path

df = pd.read_csv(NEUTROPHIL_CSV_PATH)
df["image_path"] = df["image_path"].apply(normalize_path)
df.to_csv(NEUTROPHIL_CSV_PATH, index=False)
print("Paths fixed!")