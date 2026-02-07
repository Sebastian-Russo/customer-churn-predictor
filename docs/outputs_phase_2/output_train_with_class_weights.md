$ python3 06_train_with_class_weights.py
Loading data...
Training set: 5634 samples
Test set: 1409 samples

============================================================
BASELINE: Logistic Regression (no class weights)
============================================================
Accuracy:  0.806
Precision: 0.674
Recall:    0.519
F1-Score:  0.586

============================================================
IMPROVED: Logistic Regression (balanced class weights)
============================================================
Class weights: Churners weighted ~2.7x more than non-churners
Accuracy:  0.741
Precision: 0.508
Recall:    0.807
F1-Score:  0.623

============================================================
IMPROVED: Random Forest (balanced class weights)
============================================================
Accuracy:  0.788
Precision: 0.633
Recall:    0.476
F1-Score:  0.544

============================================================
COMPARISON
============================================================
      Model  Accuracy  Precision   Recall  F1-Score
LR Baseline  0.805536   0.673611 0.518717  0.586103
LR Balanced  0.740951   0.507563 0.807487  0.623323
RF Balanced  0.787793   0.633452 0.475936  0.543511

============================================================
IMPROVEMENT ANALYSIS
============================================================
Baseline Recall:  51.9% (caught 194 of 374 churners)
Balanced Recall:  80.7% (caught 302 of 374 churners)
Improvement:      +55.7%
Additional churners saved: 108

============================================================
CONFUSION MATRIX: LR Balanced
============================================================

                Predicted
              Stay  Churn
Actual Stay    742    293
       Churn    72    302

True Positives (caught churners):  302 of 374 (80.7%)
False Negatives (missed churners): 72 of 374 (19.3%)

✓ Training with class weights complete!



### Results Analysis

MetricBaselineBalancedChangeRecall51.9%80.7%+55.7% ✓✓✓F1-Score58.6%62.3%+6.3% ✓Precision67.4%50.8%-24.6% ⚠️Accuracy80.6%74.1%-8.1% ⚠️

What Improved
Recall: 51.9% → 80.7%

Before: Caught 194 of 374 churners (missed 180)
After: Caught 302 of 374 churners (missed only 72)
Saved 108 additional customers! 🎯

F1-Score: 58.6% → 62.3%

Better overall balance despite lower precision


The Trade-off
Precision dropped: 67.4% → 50.8%

More false alarms (293 vs 94)
Predict 595 will churn, only 302 actually do

Is this acceptable?
Business calculation:
Costs:

False positives: 293 × $10/month = $2,930/month wasted
False negatives: 72 × $70/month = $5,040/month lost

Comparison to baseline:

Baseline cost: 94 × $10 + 180 × $70 = $13,540/month
Balanced cost: 293 × $10 + 72 × $70 = $7,970/month
Net savings: $5,570/month! 💰

Yes, this trade-off is worth it!

Confusion Matrix Breakdown
                Predicted
              Stay  Churn
Actual Stay    742    293  ← 293 false alarms (cost $2,930/mo)
       Churn    72    302  ← Only 72 missed! (was 180 before)
We're now catching 80.7% of churners!

Why Random Forest Didn't Improve
Random Forest with balanced weights:

Recall: only 47.6% (worse than baseline!)
Might need hyperparameter tuning

Logistic Regression wins here.

Summary
✅ Successfully improved recall from 52% → 81%
✅ Saved 108 additional customers
✅ Better F1-score (62.3%)
✅ Net cost savings: $5,570/month
The class weights worked perfectly!

