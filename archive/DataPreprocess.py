import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

df = pd.read_excel('neutrophil_images_kevin\Accession numbers to pull neutrophils images for - DE-IDENTIFIED.xlsx')

# 1. delete the row that only have accession number
df = df.dropna(how='all', subset=df.columns.difference(['Accession number']))

# 2. use regression to predict the missing value
imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan, max_iter=10, random_state=0)
df_numeric = df.select_dtypes(include=[np.number])
df[df_numeric.columns] = imputer.fit_transform(df_numeric)

# 3. map the gender into binary
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1})

df.to_csv('cleaned_neutrophils.csv', index=False)
df = pd.read_csv('master.csv')
df['morphology'] = df['morphology'].map({'cmml': 0, 'normal': 1})
df.to_csv('cleaned_master.csv', index=False)
