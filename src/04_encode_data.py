"""
Phase 1.4: Encode categorical variables
Convert all text to numbers so ML models can use them
"""
import pandas as pd
import numpy as np

# Load featured data
print("Loading data with engineered features...")
df = pd.read_csv('../data/telco_churn_featured.csv')
print(f"Starting with {len(df)} rows and {len(df.columns)} columns")

print("\n" + "="*60)
print("ENCODING CATEGORICAL VARIABLES")
print("="*60)

# Step 1: Identify what needs encoding
print("\n1. Current data types:")
print(df.dtypes.value_counts())

# Separate numeric and categorical
# select_dtypes() filters columns by data type
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Step 2: Encode binary Yes/No columns as 1/0
print("\n" + "="*60)
print("BINARY ENCODING (Yes/No → 1/0)")
print("="*60)

# Find columns with only Yes/No values
binary_cols = []
for col in categorical_cols:
    unique_vals = df[col].unique()
    # Check if column only has Yes/No (or No/Yes)
    if set(unique_vals) == {'Yes', 'No'}:
        binary_cols.append(col)

print(f"Binary columns: {binary_cols}")

# Encode: Yes → 1, No → 0
for col in binary_cols:
    # map() replaces values: Yes becomes 1, No becomes 0
    df[col] = df[col].map({'Yes': 1, 'No': 0})
    print(f"  {col}: Yes/No → 1/0")

# Step 2.5: Encode service columns (Yes/No/No internet service)
print("\n" + "="*60)
print("SERVICE COLUMNS ENCODING")
print("="*60)

service_encode_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']

print(f"Service columns: {service_encode_cols}")

for col in service_encode_cols:
    if col in df.columns:
        print(f"\n{col} values: {df[col].unique()}")
        # Encode: Yes → 1, No → 0, No internet service → 0
        # (No internet service = they don't have the service, same as No)
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
        print(f"  Encoded: Yes→1, No→0, No internet service→0")

# Step 3: Encode gender (Male/Female → 1/0)
print("\n" + "="*60)
print("GENDER ENCODING")
print("="*60)
if 'gender' in df.columns:
    print("Before:", df['gender'].unique())
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    print("After: Male → 1, Female → 0")

# Step 4: One-hot encode multi-category columns
# These can't be simple 0/1 because they have 3+ categories
print("\n" + "="*60)
print("ONE-HOT ENCODING (Multi-category columns)")
print("="*60)

# Columns that need one-hot encoding
onehot_cols = ['Contract', 'PaymentMethod', 'InternetService', 'MultipleLines',
               'tenure_group', 'monthly_charges_tier']

print(f"Columns to one-hot encode: {onehot_cols}")

for col in onehot_cols:
    if col in df.columns:
        print(f"\n{col}: {df[col].unique()}")

# pd.get_dummies() creates separate binary columns for each category
# Example: Contract has 3 values → creates 3 new columns (Contract_MonthToMonth, Contract_OneYear, Contract_TwoYear)
# drop_first=True removes one column to avoid redundancy (prevents multicollinearity)
df_encoded = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype=int)

print(f"\n✓ One-hot encoding complete")
print(f"  Before: {len(df.columns)} columns")
print(f"  After: {len(df_encoded.columns)} columns")

# Step 5: Verify no text columns remain
print("\n" + "="*60)
print("VERIFICATION")
print("="*60)

remaining_categorical = df_encoded.select_dtypes(include=['object']).columns.tolist()
if len(remaining_categorical) == 0:
    print("✓ All columns are numeric!")
else:
    print(f"⚠ Still have categorical columns: {remaining_categorical}")

# Show data types
print(f"\nFinal data types:")
print(df_encoded.dtypes.value_counts())

# Step 6: Separate features (X) and target (y)
print("\n" + "="*60)
print("SEPARATING FEATURES AND TARGET")
print("="*60)

# Target variable (what we're predicting)
# Note: Churn should already be encoded as 1/0 from binary encoding
target = 'Churn'
y = df_encoded[target]

# Features (everything except target)
# drop() removes the target column, leaving only features
X = df_encoded.drop(target, axis=1)

print(f"Features (X): {X.shape[0]} rows × {X.shape[1]} features")
print(f"Target (y): {y.shape[0]} values")
print(f"\nTarget distribution:")
print(y.value_counts())
print(f"  No churn (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  Churn (1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

# Show sample of encoded data
print("\n" + "="*60)
print("SAMPLE OF ENCODED DATA")
print("="*60)
print(df_encoded.head())

# Show all column names
print("\n" + "="*60)
print("ALL FEATURES")
print("="*60)
print(f"Total features: {len(X.columns)}")
print(X.columns.tolist())

# Save encoded data
print("\n" + "="*60)
print("SAVING")
print("="*60)
print("Saving encoded data...")
df_encoded.to_csv('../data/telco_churn_encoded.csv', index=False)
print("✓ Saved to telco_churn_encoded.csv")

# Also save X and y separately for easy loading in training
X.to_csv('../data/X_features.csv', index=False)
y.to_csv('../data/y_target.csv', index=False)
print("✓ Saved X_features.csv and y_target.csv")


### EncodeConverted text to numbers34 numeric features ready for ML
