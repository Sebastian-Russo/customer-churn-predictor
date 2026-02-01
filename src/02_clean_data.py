"""
Phase 1.2: Clean the data
Fix TotalCharges, remove customerID, handle issues
"""
import pandas as pd # Spreadsheet-like data manipulation (tables/DataFrames)
import numpy as np  # Math operations on arrays

# Load data
print("Loading data...")
df = pd.read_csv('../data/telco_churn.csv')
print(f"Starting with {len(df)} rows and {len(df.columns)} columns")

# Issue 1: TotalCharges is string (should be numeric)
print("\n" + "="*60)
print("FIXING TOTALCHARGES")
print("="*60)
print(f"Current type: {df['TotalCharges'].dtype}")

# Convert to numeric (errors='coerce' turns invalid values into NaN)
# pd.to_numeric() converts strings like "123.45" to actual numbers
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"New type: {df['TotalCharges'].dtype}")

# Check if conversion created any missing values
# .isnull().sum() counts NaN (missing) values
missing_total_charges = df['TotalCharges'].isnull().sum()
print(f"Missing values after conversion: {missing_total_charges}")

if missing_total_charges > 0:
    # Show rows with missing TotalCharges
    print("\nRows with missing TotalCharges:")
    print(df[df['TotalCharges'].isnull()][['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']])

    # Strategy: If tenure is 0, TotalCharges should be MonthlyCharges
    # Otherwise, drop these rows (they're corrupted)
    print("\nFilling missing values where tenure=0...")
    # df.loc[] selects specific rows/columns to modify
    df.loc[df['TotalCharges'].isnull(), 'TotalCharges'] = df.loc[df['TotalCharges'].isnull(), 'MonthlyCharges']

    # Drop any remaining rows with missing TotalCharges
    # dropna() removes rows with NaN values
    # subset=['TotalCharges'] only checks this column
    df = df.dropna(subset=['TotalCharges'])
    print(f"Dropped {missing_total_charges} rows")

# Issue 2: Remove customerID (not useful for prediction)
print("\n" + "="*60)
print("REMOVING CUSTOMERID")
print("="*60)
print("Dropping customerID column...")
# df.drop() removes columns (axis=1) or rows (axis=0)
# inplace=True modifies df directly instead of creating a copy
df = df.drop('customerID', axis=1)
print(f"Remaining columns: {len(df.columns)}")

# Verify no missing values remain
print("\n" + "="*60)
print("FINAL DATA CHECK")
print("="*60)
missing = df.isnull().sum()
total_missing = missing.sum()
if total_missing == 0:
    print("✓ No missing values")
else:
    print(f"⚠ Still have {total_missing} missing values:")
    print(missing[missing > 0])

# Show final shape
print(f"\nFinal dataset: {len(df)} rows × {len(df.columns)} columns")

# Save cleaned data
# df.to_csv() writes DataFrame to CSV file
# index=False prevents writing row numbers as a column
print("\nSaving cleaned data...")
df.to_csv('../data/telco_churn_cleaned.csv', index=False)
print("✓ Saved to telco_churn_cleaned.csv")
