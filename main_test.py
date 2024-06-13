import pandas as pd

file_path = 'datasets\monocyte_reassigned+neutrophil.csv'
df = pd.read_csv(file_path)

df_filtered = df[df['dataset'] != 'neutrophil']

filtered_file_path = 'datasets\monocyte_reassigned.csv'
df_filtered.to_csv(filtered_file_path, index=False)

print(df_filtered.head())
