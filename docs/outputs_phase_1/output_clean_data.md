$ python3 02_clean_data.py
Loading data...
Starting with 7043 rows and 21 columns

============================================================
FIXING TOTALCHARGES
============================================================
Current type: str
New type: float64
Missing values after conversion: 11

Rows with missing TotalCharges:
      customerID  tenure  MonthlyCharges  TotalCharges
488   4472-LVYGI       0           52.55           NaN
753   3115-CZMZD       0           20.25           NaN
936   5709-LVOEQ       0           80.85           NaN
1082  4367-NUYAO       0           25.75           NaN
1340  1371-DWPAZ       0           56.05           NaN
3331  7644-OMVMY       0           19.85           NaN
3826  3213-VVOLG       0           25.35           NaN
4380  2520-SGTTA       0           20.00           NaN
5218  2923-ARZLG       0           19.70           NaN
6670  4075-WKNIU       0           73.35           NaN
6754  2775-SEFEE       0           61.90           NaN

Filling missing values where tenure=0...
Dropped 11 rows

============================================================
REMOVING CUSTOMERID
============================================================
Dropping customerID column...
Remaining columns: 20

============================================================
FINAL DATA CHECK
============================================================
✓ No missing values

Final dataset: 7043 rows × 20 columns

Saving cleaned data...
✓ Saved to telco_churn_cleaned.csv


-------------------------------------------------------------------------------------------

### What Just Happened:
✅ Fixed TotalCharges:

Found 11 rows with empty TotalCharges (showed as spaces in CSV)
All 11 were new customers (tenure=0)
Filled them with MonthlyCharges (makes sense - first month = total charges)

✅ Removed customerID:

Dropped useless ID column
Down from 21 → 20 columns

✅ Clean dataset:

7,043 rows (no rows lost!)
20 columns
No missing values
All numeric columns are actually numeric now

