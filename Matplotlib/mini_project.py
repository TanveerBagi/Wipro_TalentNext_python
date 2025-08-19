#1. Perform Exploratory Data Analysis for the Diabetes Dataset.
     #Dataset: Diabetes.csv
     #Perform the following tasks:
        #Load the data in the DataFrame
        #Data Pre-processing
        #Handle the Categorical Data
        #Perform Uni-variate Analysis
        #Perform Bi-variate Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Diabetes.csv')
print("First five rows of the dataset:")
print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())

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

if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("\nSummary Statistics:")
print(df.describe())

for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()

print("\nCorrelation Matrix:")
corr = df.corr()
print(corr)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

if 'Outcome' in df.columns and 'Glucose' in df.columns and 'BMI' in df.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df)
    plt.title('Glucose vs BMI colored by Outcome')
    plt.show()
