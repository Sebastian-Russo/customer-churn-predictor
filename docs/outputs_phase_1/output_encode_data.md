$ python3 04_encode_data.py
Loading data with engineered features...
Starting with 7043 rows and 27 columns

============================================================
ENCODING CATEGORICAL VARIABLES
============================================================

1. Current data types:
str        18
int64       6
float64     3
Name: count, dtype: int64
/home/sebastian/ai-projects/customer-churn-predictor/src/04_encode_data.py:24: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
  categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

Numeric columns (9): ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'is_new_customer', 'charges_per_tenure', 'total_services', 'has_multiple_services', 'has_paperless']
Categorical columns (18): ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn', 'tenure_group', 'monthly_charges_tier']

============================================================
BINARY ENCODING (Yes/No → 1/0)
============================================================
Binary columns: ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
  Partner: Yes/No → 1/0
  Dependents: Yes/No → 1/0
  PhoneService: Yes/No → 1/0
  PaperlessBilling: Yes/No → 1/0
  Churn: Yes/No → 1/0

============================================================
GENDER ENCODING
============================================================
Before: <StringArray>
['Female', 'Male']
Length: 2, dtype: str
After: Male → 1, Female → 0

============================================================
ONE-HOT ENCODING (Multi-category columns)
============================================================
Columns to one-hot encode: ['Contract', 'PaymentMethod', 'InternetService', 'MultipleLines', 'tenure_group', 'monthly_charges_tier']

Contract: <StringArray>
['Month-to-month', 'One year', 'Two year']
Length: 3, dtype: str

PaymentMethod: <StringArray>
[         'Electronic check',              'Mailed check',
 'Bank transfer (automatic)',   'Credit card (automatic)']
Length: 4, dtype: str

InternetService: <StringArray>
['DSL', 'Fiber optic', 'No']
Length: 3, dtype: str

MultipleLines: <StringArray>
['No phone service', 'No', 'Yes']
Length: 3, dtype: str

tenure_group: <StringArray>
['0-1 year', '2-4 years', '1-2 years', '4-6 years', nan]
Length: 5, dtype: str

monthly_charges_tier: <StringArray>
['Low', 'Medium', 'High']
Length: 3, dtype: str

✓ One-hot encoding complete
  Before: 27 columns
  After: 35 columns

============================================================
VERIFICATION
============================================================
/home/sebastian/ai-projects/customer-churn-predictor/src/04_encode_data.py:89: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
  remaining_categorical = df_encoded.select_dtypes(include=['object']).columns.tolist()
⚠ Still have categorical columns: ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

Final data types:
int64      26
str         6
float64     3
Name: count, dtype: int64

============================================================
SEPARATING FEATURES AND TARGET
============================================================
Features (X): 7043 rows × 34 features
Target (y): 7043 values

Target distribution:
Churn
0    5174
1    1869
Name: count, dtype: int64
  No churn (0): 5174 (73.5%)
  Churn (1): 1869 (26.5%)

============================================================
SAMPLE OF ENCODED DATA
============================================================
   gender  SeniorCitizen  ...  monthly_charges_tier_Low  monthly_charges_tier_Medium
0       0              0  ...                         1                            0
1       1              0  ...                         0                            1
2       1              0  ...                         0                            1
3       1              0  ...                         0                            1
4       0              0  ...                         0                            0

[5 rows x 35 columns]

============================================================
ALL FEATURES
============================================================
Total features: 34
['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'is_new_customer', 'charges_per_tenure', 'total_services', 'has_multiple_services', 'has_paperless', 'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'InternetService_Fiber optic', 'InternetService_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'tenure_group_1-2 years', 'tenure_group_2-4 years', 'tenure_group_4-6 years', 'monthly_charges_tier_Low', 'monthly_charges_tier_Medium']

============================================================
SAVING
============================================================
Saving encoded data...
✓ Saved to telco_churn_encoded.csv
✓ Saved X_features.csv and y_target.csv

----------------------------------------------------------------------------------------

# Converted text to numbers34 numeric features ready for ML
