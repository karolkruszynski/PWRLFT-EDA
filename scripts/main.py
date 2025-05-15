import pandas as pd
import numpy as np

# Data Structure check

df = pd.read_csv("openpowerlifting.csv", low_memory=False)
print(df.head(10))

# Numbers of rows and columns
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Names of columns
print(f"Columns names: {df.columns}")
# Data types
print(f"Data types: {df.dtypes}")
# General info
print(f"General info {df.info()}")

# Basic statistics
print(f"Basic statistics: {df.describe()}")

# Missing Data
print(f"Sum of missing data {df.isnull().sum().sort_values(ascending=False)}")

# As we can see, most of the empty values are in the Squat4Kg / Bench4Kg / Deadlift4Kg columns because in these columns they refer to fourth approaches, which are granted in rare cases by the judges.
# For example, while the referees made a mistake to your disadvantage, then you are entitled to an additional approach.

