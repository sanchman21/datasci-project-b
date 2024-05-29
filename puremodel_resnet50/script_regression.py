import pandas as pd

df = pd.read_csv('monocyte_resnet50\merge_m.csv')


df_unique = df.drop_duplicates(subset='Accession number', keep='first')

df_unique.to_csv('no_duplication_accssion_number.csv', index=False)
