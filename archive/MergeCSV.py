import pandas as pd
import os

df1 = pd.read_csv('cleaned_master.csv')
df2 = pd.read_csv('cleaned_neutrophils.csv')
merged_df1 = pd.merge(df1, df2, left_on='patient_id', right_on='Accession number')

merged_df2 = pd.read_csv('NeutrophilImages.csv')


unique_cols = set(merged_df1.columns) - set(merged_df2.columns)
for col in unique_cols:
    merged_df2[col] = pd.NA

combined_df = pd.concat([merged_df1, merged_df2], ignore_index=True)

combined_df.to_csv('merged_master.csv', index=False)