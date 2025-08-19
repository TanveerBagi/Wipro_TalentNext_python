#1. Use-Case: House Price Prediction
    #Dataset: melb_data.csv
    #Perform the following tasks:
      #Load the data in a dataframe (Pandas)
      #Handle inappropriate data
      #Handle the missing data
      #Handle the categorical data

import pandas as pd
df = pd.read_csv("melb_data.csv")

print("Initial shape:", df.shape)
print(df.head())

if 'Address' in df.columns:
    df.drop(columns=['Address'], inplace=True)

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

df = pd.get_dummies(df, drop_first=True)

print("Final shape after preprocessing:", df.shape)
