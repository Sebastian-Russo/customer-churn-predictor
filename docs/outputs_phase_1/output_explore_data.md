$ python3 01_explore_data.py
Loading data...

============================================================
DATASET OVERVIEW
============================================================
Rows: 7043
Columns: 21

Column names:
['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

============================================================
SAMPLE DATA
============================================================
   customerID  gender  SeniorCitizen  ... MonthlyCharges TotalCharges  Churn
0  7590-VHVEG  Female              0  ...          29.85        29.85     No
1  5575-GNVDE    Male              0  ...          56.95       1889.5     No
2  3668-QPYBK    Male              0  ...          53.85       108.15    Yes
3  7795-CFOCW    Male              0  ...          42.30      1840.75     No
4  9237-HQITU  Female              0  ...          70.70       151.65    Yes

[5 rows x 21 columns]

============================================================
DATA TYPES
============================================================
customerID              str
gender                  str
SeniorCitizen         int64
Partner                 str
Dependents              str
tenure                int64
PhoneService            str
MultipleLines           str
InternetService         str
OnlineSecurity          str
OnlineBackup            str
DeviceProtection        str
TechSupport             str
StreamingTV             str
StreamingMovies         str
Contract                str
PaperlessBilling        str
PaymentMethod           str
MonthlyCharges      float64
TotalCharges            str
Churn                   str
dtype: object

============================================================
TARGET: CHURN DISTRIBUTION
============================================================
Churn
No     5174
Yes    1869
Name: count, dtype: int64

Percentages:
Churn
No     73.463013
Yes    26.536987
Name: proportion, dtype: float64

============================================================
MISSING VALUES
============================================================
✓ No missing values

============================================================
NUMERIC FEATURES
============================================================
Numeric columns: ['SeniorCitizen', 'tenure', 'MonthlyCharges']

Statistics:
       SeniorCitizen       tenure  MonthlyCharges
count    7043.000000  7043.000000     7043.000000
mean        0.162147    32.371149       64.761692
std         0.368612    24.559481       30.090047
min         0.000000     0.000000       18.250000
25%         0.000000     9.000000       35.500000
50%         0.000000    29.000000       70.350000
75%         0.000000    55.000000       89.850000
max         1.000000    72.000000      118.750000

============================================================
CATEGORICAL FEATURES
============================================================
/home/sebastian/ai-projects/customer-churn-predictor/src/01_explore_data.py:59: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
  categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
Categorical columns: ['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'Churn']

Unique values per column:
customerID: 7043 unique values
gender: 2 unique values
Partner: 2 unique values
Dependents: 2 unique values
PhoneService: 2 unique values
MultipleLines: 3 unique values
InternetService: 3 unique values
OnlineSecurity: 3 unique values
OnlineBackup: 3 unique values
DeviceProtection: 3 unique values
TechSupport: 3 unique values
StreamingTV: 3 unique values
StreamingMovies: 3 unique values
Contract: 3 unique values
PaperlessBilling: 2 unique values
PaymentMethod: 4 unique values
TotalCharges: 6531 unique values
Churn: 2 unique values

✓ Exploration complete


-------------------------------------------------------------------------------------------

### Key Findings:
✅ Good News:

7,043 customers - decent dataset size
No missing values (technically)
Clear target: 73% stayed, 27% churned

⚠️ Issues Found:
1. Class Imbalance

73% No churn vs 27% Yes churn
Problem: Model might just predict "No" for everyone
Solution: We'll handle this in Phase 2

2. TotalCharges is Text (Should be Numeric!)

Shows as str type with 6531 unique values
Should be a number like MonthlyCharges
Problem: Probably has spaces or empty strings
Solution: Convert to numeric in cleaning phase

3. Too Many Categories:

customerID has 7043 unique values (useless - just an ID)
Solution: Drop it


What We Learned:
Numeric features (3):

SeniorCitizen: 0 or 1 (16% are seniors)
tenure: 0-72 months (average 32 months)
MonthlyCharges: $18-$119 (average $65)

Categorical features (17):

Binary (Yes/No): gender, Partner, Dependents, PhoneService, etc.
Multi-category: Contract (3 types), PaymentMethod (4 types)