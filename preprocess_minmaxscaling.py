import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('merged_master.csv')

features_to_scale = ['Age', 'Haemoglobin', 'MCV', 'White cell count', 'Neutrophil count', 'Monocyte count', 'Platelet count', 'Blast percentage (PB)', 'LDH']

scaler = MinMaxScaler()

df_scaled = df[features_to_scale]

df_scaled[features_to_scale] = scaler.fit_transform(df_scaled)

df[features_to_scale] = df_scaled

df.to_csv('merged_master_normalized.csv', index=False)

print(df.head())
