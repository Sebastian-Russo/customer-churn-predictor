"""
Phase 1.3: Feature Engineering
Create new features from existing data to help the model
"""
import pandas as pd
import numpy as np

# Load cleaned data
print("Loading cleaned data...")
df = pd.read_csv('../data/telco_churn_cleaned.csv')
print(f"Starting with {len(df)} rows and {len(df.columns)} columns")

print("\n" + "="*60)
print("CREATING NEW FEATURES")
print("="*60)

# Feature 1: Customer tenure groups
# Categorize customers by how long they've been with us
# New customers (0-12 months) are most likely to churn
print("\n1. Tenure Groups:")
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 48, 72],  # Breakpoints: 0-12, 12-24, 24-48, 48-72 months
    labels=['0-1 year', '1-2 years', '2-4 years', '4-6 years']  # Labels for each group
)
# pd.cut() bins continuous numbers into categories (like making a histogram)
print(df['tenure_group'].value_counts())

# Feature 2: Is new customer (high churn risk)
# Binary flag: 1 if customer is new (< 6 months), 0 otherwise
print("\n2. New Customer Flag:")
df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
# (df['tenure'] <= 6) creates True/False, .astype(int) converts to 1/0
print(f"New customers: {df['is_new_customer'].sum()} ({df['is_new_customer'].mean()*100:.1f}%)")

# Feature 3: Average charges per month of tenure
# Shows value of customer (high spenders vs low spenders)
print("\n3. Charges per Month:")
# Avoid division by zero: if tenure=0, use 1 instead
df['charges_per_tenure'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
# .replace(0, 1) changes any 0 values to 1 to avoid dividing by zero
print(f"Average: ${df['charges_per_tenure'].mean():.2f}")
print(f"Range: ${df['charges_per_tenure'].min():.2f} - ${df['charges_per_tenure'].max():.2f}")

# Feature 4: Has multiple services
# Count how many services customer has (phone, internet, streaming, etc.)
print("\n4. Service Count:")
# Create list of service columns to count
service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']

# Count "Yes" across all service columns for each customer
# (df[service_cols] == 'Yes') creates True/False for each service
# .sum(axis=1) counts True values across columns (axis=1 means row-wise)
df['total_services'] = (df[service_cols] == 'Yes').sum(axis=1)
print(df['total_services'].value_counts().sort_index())

# Feature 5: Has multiple services flag
print("\n5. Multiple Services Flag:")
df['has_multiple_services'] = (df['total_services'] >= 2).astype(int)
print(f"Customers with 2+ services: {df['has_multiple_services'].sum()}")

# Feature 6: Monthly charges tier
# Categorize by price tier (budget, standard, premium)
print("\n6. Monthly Charges Tier:")
df['monthly_charges_tier'] = pd.cut(
    df['MonthlyCharges'],
    bins=[0, 35, 70, 120],  # Price breakpoints
    labels=['Low', 'Medium', 'High']  # Tier labels
)
print(df['monthly_charges_tier'].value_counts())

# Feature 7: Has paperless billing (often correlated with auto-pay)
print("\n7. Paperless Billing:")
df['has_paperless'] = (df['PaperlessBilling'] == 'Yes').astype(int)
print(f"Paperless customers: {df['has_paperless'].sum()}")

# Feature 8: Is senior citizen (already 0/1, but let's verify)
print("\n8. Senior Citizen:")
print(f"Senior citizens: {df['SeniorCitizen'].sum()} ({df['SeniorCitizen'].mean()*100:.1f}%)")

# Show sample of new features
print("\n" + "="*60)
print("SAMPLE WITH NEW FEATURES")
print("="*60)
# Display subset of columns to see new features
sample_cols = ['tenure', 'tenure_group', 'is_new_customer', 'MonthlyCharges',
               'monthly_charges_tier', 'total_services', 'Churn']
print(df[sample_cols].head(10))

# Final summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Original features: 20")
print(f"New features created: 8")
print(f"Total features: {len(df.columns)}")

# Save data with engineered features
print("\nSaving data with engineered features...")
df.to_csv('../data/telco_churn_featured.csv', index=False)
print("✓ Saved to telco_churn_featured.csv")
