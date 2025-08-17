#1. Download the dataset and rename it to cars.csv
  #a. Import Pandas
  #b. Import the Cars Dataset and store the Pandas DataFrame in the variable cars
  #c. Inspect the first 10 rows of the DataFrame
  #d. Inspect the DataFrame by printing cars
  #e. Inspect the last 5 rows
  #f. Get some meta information on the DataFrame
import pandas as pd

cars = pd.read_csv("cars.csv")

print("First 10 Rows:\n", cars.head(10))

print("\nComplete DataFrame:\n", cars)

print("\nLast 5 Rows:\n", cars.tail())

print("\nMeta Information:\n")
print(cars.info())

#2. Download the dataset from: 50_Startups Dataset
  #a. Create DataFrame using Pandas
  #b. Read the data from 50_startups.csv file and load it into DataFrame
  #c. Check the statistical summary
  #d. Check for correlation coefficient between dependent and independent variables
import pandas as pd

startups = pd.read_csv("50_Startups.csv")

print("Dataset Preview:\n", startups.head())

print("\nStatistical Summary:\n", startups.describe())

print("\nCorrelation Matrix:\n", startups.corr())
