import pandas as pd
import numpy as np

df = pd.read_csv('monocyte_resnet50\m_neutrophils.csv')

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

fold_size = len(df_shuffled) // 5

for i in range(5):
    df_shuffled[f'set{i}'] = None

for i in range(5):
    start_index = i * fold_size
    end_index = start_index + fold_size
    if i == 4:
        end_index = len(df_shuffled)

    df_shuffled.loc[start_index:end_index, f'set{i}'] = 'test'
    for j in range(5):
        if i != j:
            df_shuffled.loc[start_index:end_index, f'set{j}'] = 'train'

df_shuffled.to_csv('neutrophils.csv', index=False)
