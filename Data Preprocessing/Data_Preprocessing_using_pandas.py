#1.Perform Data Preprocessing on melb_data.csv dataset using pandas with statistical perspective.

import pandas as pd
import numpy as np

df = pd.read_csv('melb_data.csv')

print("Statistical Overview:")
print(df.describe(include='all'))

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df.to_csv('melb_data_preprocessed_pandas.csv', index=False)
print("Preprocessing complete. Saved as melb_data_preprocessed_pandas.csv")
