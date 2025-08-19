#1. Perform Exploratory Data Analysis for the dataset Mall_Customers.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')
print(df.head())

print("Missing values:\n", df.isnull().sum())
df = df.fillna(df.median(numeric_only=True))

cat_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("Summary Statistics:\n", df.describe())
for col in df.select_dtypes(include=['float', 'int']).columns:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram: {col}')
    plt.show()

print("Correlation Matrix:\n", df.corr())
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

if 'Age' in df.columns and 'Spending Score (1-100)' in df.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='Age', y='Spending Score (1-100)', data=df)
    plt.title('Age vs Spending Score')
    plt.show()

#2. Perform Exploratory Data Analysis for the dataset salary_data.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('salary_data.csv')
print(df.head())
print("Missing values:\n", df.isnull().sum())
df = df.fillna(df.median(numeric_only=True))

cat_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("Summary Statistics:\n", df.describe())
for col in df.select_dtypes(include=['float', 'int']).columns:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram: {col}')
    plt.show()

print("Correlation Matrix:\n", df.corr())
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

if 'YearsExperience' in df.columns and 'Salary' in df.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='YearsExperience', y='Salary', data=df)
    plt.title('YearsExperience vs Salary')
    plt.show()


#3. Perform Exploratory Data Analysis for the dataset Social Network Ads.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Social_Network_Ads.csv')
print(df.head())

print("Missing values:\n", df.isnull().sum())
df = df.fillna(df.median(numeric_only=True))

cat_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("Summary Statistics:\n", df.describe())
for col in df.select_dtypes(include=['float', 'int']).columns:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram: {col}')
    plt.show()

print("Correlation Matrix:\n", df.corr())
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

if 'Age' in df.columns and 'EstimatedSalary' in df.columns and 'Purchased' in df.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=df)
    plt.title('Age vs EstimatedSalary by Purchased Status')
    plt.show()
