#1.Use Case: Perform the Outlier detection for the given dataset i.e. datasetExample.csv
   #Tasks:
       #a.Load the data in the DataFrame.
       #b.Detection of Outliers.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("datasetExample.csv")
print("First 5 rows:\n", data.head())

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))

print("\nOutliers Detected (True = Outlier):\n", outliers)
print("\nNumber of Outliers per column:\n", outliers.sum())

data.plot(kind='box', subplots=True, layout=(1, len(data.columns)), figsize=(12,5), sharey=False)
plt.suptitle("Outlier Detection using Boxplot")
plt.show()
