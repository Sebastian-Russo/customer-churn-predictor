$ python3 07_try_xgboost.py
Loading data...
Training set: 5634 samples
Test set: 1409 samples

============================================================
BASELINE: Logistic Regression (balanced)
============================================================
Accuracy:  0.741
Precision: 0.508
Recall:    0.807
F1-Score:  0.623

============================================================
Random Forest (balanced)
============================================================
Accuracy:  0.788
Precision: 0.633
Recall:    0.476
F1-Score:  0.544

============================================================
XGBoost (default)
============================================================
Training...
Accuracy:  0.783
Precision: 0.604
Recall:    0.527
F1-Score:  0.563$ python3 07_try_xgboost.py
Loading data...
Training set: 5634 samples
Test set: 1409 samples

============================================================
BASELINE: Logistic Regression (balanced)
============================================================
Accuracy:  0.741
Precision: 0.508
Recall:    0.807
F1-Score:  0.623

============================================================
Random Forest (balanced)
============================================================
Accuracy:  0.788
Precision: 0.633
Recall:    0.476
F1-Score:  0.544

============================================================
XGBoost (default)
============================================================
Training...
Accuracy:  0.783
Precision: 0.604
Recall:    0.527
F1-Score:  0.563

============================================================
XGBoost (with class weights)
============================================================
scale_pos_weight: 2.77 (churners weighted 2.77x more)
Training...
Accuracy:  0.753
Precision: 0.523
Recall:    0.778
F1-Score:  0.626

============================================================
FINAL COMPARISON
============================================================
           Model  Accuracy  Precision   Recall  F1-Score
XGBoost Balanced  0.753016   0.523381 0.778075  0.625806
     LR Balanced  0.740951   0.507563 0.807487  0.623323
 XGBoost Default  0.782825   0.604294 0.526738  0.562857
     RF Balanced  0.787793   0.633452 0.475936  0.543511

============================================================
WINNER ANALYSIS
============================================================
Best Model: XGBoost Balanced
  F1-Score: 0.626
  Recall:   0.778 (catches 291 of 374 churners)
  Precision: 0.523

============================================================
CONFUSION MATRIX: XGBoost Balanced
============================================================

                Predicted
              Stay  Churn
Actual Stay    770    265
       Churn    83    291

True Positives (caught churners):  291 of 374 (77.8%)
False Negatives (missed churners): 83 of 374 (22.2%)
False Positives (false alarms):    265 of 1035 (25.6%)

✓ XGBoost comparison complete!

============================================================
XGBoost (with class weights)
============================================================
scale_pos_weight: 2.77 (churners weighted 2.77x more)
Training...
Accuracy:  0.753
Precision: 0.523
Recall:    0.778
F1-Score:  0.626

============================================================
FINAL COMPARISON
============================================================
           Model  Accuracy  Precision   Recall  F1-Score
XGBoost Balanced  0.753016   0.523381 0.778075  0.625806
     LR Balanced  0.740951   0.507563 0.807487  0.623323
 XGBoost Default  0.782825   0.604294 0.526738  0.562857
     RF Balanced  0.787793   0.633452 0.475936  0.543511

============================================================
WINNER ANALYSIS
============================================================
Best Model: XGBoost Balanced
  F1-Score: 0.626
  Recall:   0.778 (catches 291 of 374 churners)
  Precision: 0.523

============================================================
CONFUSION MATRIX: XGBoost Balanced
============================================================

                Predicted
              Stay  Churn
Actual Stay    770    265
       Churn    83    291

True Positives (caught churners):  291 of 374 (77.8%)
False Negatives (missed churners): 83 of 374 (22.2%)
False Positives (false alarms):    265 of 1035 (25.6%)

✓ XGBoost comparison complete!



### Final Model Comparison
ModelF1-ScoreRecallPrecisionChurners CaughtXGBoost Balanced62.6%77.8%52.3%291 / 374LR Balanced62.3%80.7% ✓50.8%302 / 374 ✓XGBoost Default56.3%52.7%60.4%197 / 374RF Balanced54.4%47.6%63.3%178 / 374

The Surprise: Logistic Regression Actually Wins!
Why LR is better despite lower F1:
Recall: 80.7% vs 77.8%

LR catches 302 churners
XGBoost catches 291 churners
LR saves 11 more customers!

Business Impact:
MetricLR BalancedXGBoost BalancedCaught churners302291Missed churners7283False alarms293265Monthly cost$7,970$8,460
LR is $490/month cheaper! ($5,880/year)

Why XGBoost Didn't Dominate

Problem is fairly linear - Simple relationships (tenure, charges → churn)
Not enough data - Only 7,043 samples (XGBoost shines with 100k+)
LR better for this balance - Better recall at acceptable precision

The lesson: More complex ≠ always better!

Final Decision: Use Logistic Regression Balanced
Reasons:

✅ Best recall (80.7% - catches most churners)
✅ Simplest model (easy to explain to business)
✅ Fastest (predictions in microseconds)
✅ Most cost-effective ($490/month cheaper)
✅ Interpretable (can show which features matter)

XGBoost advantages:

Slightly better F1 (0.3% difference - negligible)
Fewer false positives (28 fewer)

But LR's better recall wins for this business problem.

Summary: What We Learned
AttemptModelRecallResult1LR Baseline51.9%Too conservative2RF Default48.7%Even worse3LR Balanced80.7%Winner! ✓4RF Balanced47.6%Still bad5XGBoost Balanced77.8%Good, but not best
The simplest model with class weights won!