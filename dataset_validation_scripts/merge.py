import pandas as pd

df1 = pd.read_csv('datasets/neutrophil.csv')
df2 = pd.read_csv('datasets\monocyte_reassigned.csv')

combined_df = pd.concat([df1, df2], ignore_index=True)

combined_df.to_csv('combined_file.csv', index=False)

