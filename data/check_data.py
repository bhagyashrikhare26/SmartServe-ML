import pandas as pd

# Load the dataset
df = pd.read_csv("data/train.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset size
print("\nDataset Shape:", df.shape)

# Display column names
print("\nColumns:", df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Basic statistics
print("\nDataset Statistics:")
print(df.describe())