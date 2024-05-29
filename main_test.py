import pandas as pd
df = pd.read_csv('datasets\monocyte.csv')
duplicates = df.duplicated(subset='image_path', keep=False)

has_duplicates = duplicates.any()
print("是否存在重复值：", has_duplicates)
df.to_csv('datasets\patient.csv', index=False)
