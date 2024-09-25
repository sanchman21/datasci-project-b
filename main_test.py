import pandas as pd # import pandas for data manipulation

file_path = 'datasets\monocyte_reassigned+neutrophil.csv' # path to the dataset file
df = pd.read_csv(file_path) # read the dataset file into a pandas dataframe

df_filtered = df[df['dataset'] != 'neutrophil'] # filter out neutrophil images

filtered_file_path = 'datasets\monocyte_reassigned.csv' # path to save the filtered dataset
df_filtered.to_csv(filtered_file_path, index=False) # save the filtered dataset to a csv file

print(df_filtered.head()) # print the first few rows of the filtered dataset
