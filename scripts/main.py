import pandas as pd
import numpy as np

# STRUCTURED REVIEW OF DATA

df = pd.read_csv("openpowerlifting.csv", low_memory=False)
print(df.head(10))

# Numbers of rows and columns
print(f"Number of rows: {df.shape[0]}")
raw_data_rows = df.shape[0]
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
# % Missing Data of top 3 columns
print(f"Percentage of missing data {(df.isnull().sum() / len(df) * 100).sort_values(ascending=False)}")
# As we can see, most of the empty values are in the Squat4Kg / Bench4Kg / Deadlift4Kg columns because in these columns they refer to fourth approaches, which are granted in rare cases by the judges.
# For example, while the referees made a mistake to your disadvantage, then you are entitled to an additional approach.


# DATA CLEANING
# Crucial data: Age, Weight, Results
# If row didn't have crucial data then it will be dropped

df = df.dropna(subset=["Age", "BodyweightKg", "TotalKg"])
print(f"Number of rows after cleaning: {df.shape[0]}")
clean_data_rows = df.shape[0]
print(f"Percentage change: {round((raw_data_rows - clean_data_rows) / raw_data_rows * 100, 2)}%")

# Drop columns that will not be used in futher analysis
df = df.drop(["Best3DeadliftKg", "State", "MeetState", "MeetTown", "MeetName", "Sanctioned"], axis=1)
# Round the age 0.5 to upper
df["Age"] = np.ceil(df["Age"]).astype(int)
print(df["Age"].dtypes)

# Check unique values to get errors in data
print(f"Unique values: {df.nunique()}")
print(f"Unique countries: {df['Country'].nunique()}")
print(f"Unique sex types: {df['Sex'].unique()}")

# Fix 'Mx' Sex
df['Sex'] = df['Sex'].replace('Mx', 'M')
print(f"Unique sex types: {df['Sex'].unique()}")

# Save cleaned data
df.to_csv("cleaned_openpowerlifting.csv", index=False)