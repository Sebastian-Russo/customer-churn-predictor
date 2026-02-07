"""
Phase 1.1: Explore the Telco Customer Churn dataset
"""
import pandas as pd  # Spreadsheet-like data manipulation (tables/DataFrames)
import numpy as np   # Fast math operations on arrays of numbers

# Load data
# pd.read_csv() reads CSV file and creates a DataFrame (table)
print("Loading data...")
df = pd.read_csv('../data/telco_churn.csv')

# Basic overview
# len(df) counts number of rows
# len(df.columns) counts number of columns
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"\nColumn names:\n{list(df.columns)}")  # df.columns returns column names

# First few rows
# df.head() shows first 5 rows (like peeking at top of spreadsheet)
print("\n" + "="*60)
print("SAMPLE DATA")
print("="*60)
print(df.head())

# Data types
# df.dtypes shows if each column is int, float, string, etc.
print("\n" + "="*60)
print("DATA TYPES")
print("="*60)
print(df.dtypes)

# Target variable
# df['Churn'] selects the Churn column
# .value_counts() counts how many of each unique value (Yes/No)
print("\n" + "="*60)
print("TARGET: CHURN DISTRIBUTION")
print("="*60)
print(df['Churn'].value_counts())
print("\nPercentages:")
# normalize=True converts counts to percentages (0-1), * 100 makes it 0-100
print(df['Churn'].value_counts(normalize=True) * 100)

# Missing values
# df.isnull() returns True/False for each cell (True if empty)
# .sum() counts the True values (missing cells per column)
print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
missing = df.isnull().sum()
# If any column has missing values, show them; otherwise show success message
print(missing[missing > 0] if missing.sum() > 0 else "✓ No missing values")

# Numeric features
# df.select_dtypes(include=[np.number]) filters to only numeric columns
# np.number is NumPy's way of saying "any numeric type" (int, float, etc.)
print("\n" + "="*60)
print("NUMERIC FEATURES")
print("="*60)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {numeric_cols}")
print("\nStatistics:")
# df.describe() calculates mean, min, max, std dev for numeric columns
print(df[numeric_cols].describe())

# Categorical features
# df.select_dtypes(include=['object']) filters to text/string columns
print("\n" + "="*60)
print("CATEGORICAL FEATURES")
print("="*60)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")
print(f"\nUnique values per column:")
# .nunique() counts number of unique values in each column
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n✓ Exploration complete")
