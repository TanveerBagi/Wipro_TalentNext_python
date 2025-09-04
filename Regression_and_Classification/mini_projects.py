#Q1. Sales Prediction
      #Create a model which will predict the sales based on campaigning expenses.
      #Dataset : Advertising.csv  The dataset can be downloaded from https://www.kaggle.com/datasets
      #Perform the following task.
          #Load the data in the DataFrame.
          #Perform Data Preprocessing
          #Handle Categorical Data
          #Perform Exploratory Data Analysis
          #Build the model using Multiple Linear Regression
          #Use the appropriate evaluation metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Advertising.csv")

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())
# if there is categorical data in the dataset
#df = pd.get_dummies(df, drop_first=True)

sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coef_df)


