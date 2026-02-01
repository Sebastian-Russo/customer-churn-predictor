$ python3 03_feature_engineering.py
Loading cleaned data...
Starting with 7043 rows and 20 columns

============================================================
CREATING NEW FEATURES
============================================================

1. Tenure Groups:
tenure_group
4-6 years    2239
0-1 year     2175
2-4 years    1594
1-2 years    1024
Name: count, dtype: int64

2. New Customer Flag:
New customers: 1481 (21.0%)

3. Charges per Month:
Average: $64.76
Range: $13.78 - $121.40

4. Service Count:
total_services
0      80
1    2253
2     996
3    1041
4    1062
5     827
6     525
7     259
Name: count, dtype: int64

5. Multiple Services Flag:
Customers with 2+ services: 4710

6. Monthly Charges Tier:
monthly_charges_tier
High      3583
Low       1735
Medium    1725
Name: count, dtype: int64

7. Paperless Billing:
Paperless customers: 4171

8. Senior Citizen:
Senior citizens: 1142 (16.2%)

============================================================
SAMPLE WITH NEW FEATURES
============================================================
   tenure tenure_group  is_new_customer  MonthlyCharges monthly_charges_tier  total_services Churn
0       1     0-1 year                1           29.85                  Low               1    No
1      34    2-4 years                0           56.95               Medium               3    No
2       2     0-1 year                1           53.85               Medium               3   Yes
3      45    2-4 years                0           42.30               Medium               3    No
4       2     0-1 year                1           70.70                 High               1   Yes
5       8     0-1 year                0           99.65                 High               4   Yes
6      22    1-2 years                0           89.10                 High               3    No
7      10     0-1 year                0           29.75                  Low               1    No
8      28    2-4 years                0          104.80                 High               5   Yes
9      62    4-6 years                0           56.15               Medium               3    No

============================================================
SUMMARY
============================================================
Original features: 20
New features created: 8
Total features: 27

Saving data with engineered features...
✓ Saved to telco_churn_featured.csv


------------------------------------------------------------------------------

### Interesting Pattern in Sample:
Look at rows 2, 4, 5, 8 - they all churned:

Row 2: New (2 months), Medium price → Churned
Row 4: New (2 months), High price → Churned
Row 5: New (8 months), High price, 4 services → Churned
Row 8: Medium tenure (28 months), High price, 5 services → Churned

Pattern emerging: New customers + high charges = churn risk!
