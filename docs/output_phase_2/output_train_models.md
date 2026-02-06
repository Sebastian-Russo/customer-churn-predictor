 $ python3 05_train_models.py
Loading encoded data...
Features: (7043, 34)
Target: (7043,)

Target distribution:
  Class 0: 5174 (73.5%)
  Class 1: 1869 (26.5%)

============================================================
TRAIN/TEST SPLIT
============================================================
Training set: 5634 samples
Test set: 1409 samples

Train target distribution:
  Class 0: 4139 (73.5%)
  Class 1: 1495 (26.5%)

============================================================
MODEL 1: LOGISTIC REGRESSION (Baseline)
============================================================
Training...
Accuracy:  0.806
Precision: 0.674
Recall:    0.519
F1-Score:  0.586

============================================================
MODEL 2: RANDOM FOREST
============================================================
Training...
Accuracy:  0.791
Precision: 0.639
Recall:    0.487
F1-Score:  0.552

============================================================
MODEL COMPARISON
============================================================
              Model  Accuracy  Precision   Recall  F1-Score
Logistic Regression  0.805536   0.673611 0.518717  0.586103
      Random Forest  0.790632   0.638596 0.486631  0.552352

============================================================
CONFUSION MATRIX: Logistic Regression
============================================================

                Predicted
              Stay  Churn
Actual Stay    941     94
       Churn   180    194

Explanation:
  True Negatives (correct 'stay'):   941
  False Positives (wrong 'churn'):   94
  False Negatives (missed churners): 180
  True Positives (caught churners):  194

============================================================
DETAILED REPORT: Logistic Regression
============================================================
              precision    recall  f1-score   support

        Stay       0.84      0.91      0.87      1035
       Churn       0.67      0.52      0.59       374

    accuracy                           0.81      1409
   macro avg       0.76      0.71      0.73      1409
weighted avg       0.80      0.81      0.80      1409


✓ Training complete!

Model Performance Summary
ModelAccuracyPrecisionRecallF1-ScoreLogistic Regression80.6%67.4%51.9%58.6% ✓Random Forest79.1%63.9%48.7%55.2%
Winner: Logistic Regression (better F1-score)

What These Numbers Mean
Logistic Regression Results:
Accuracy: 80.6%

Correct predictions: 1,135 out of 1,409
Better than the "dumb model" (73.5%)!

Precision: 67.4%

Predicted 288 customers would churn
194 actually did churn
94 were false alarms (67.4% = 194/288)
Business impact: Waste money on 94 customers who wouldn't leave anyway

Recall: 51.9%

374 customers actually churned
We caught 194 of them
Missed 180 (51.9% = 194/374)
Business impact: Lost 180 customers we could have saved!

F1-Score: 58.6%

Balance of precision and recall
Higher is better


Model Performance Summary
ModelAccuracyPrecisionRecallF1-ScoreLogistic Regression80.6%67.4%51.9%58.6% ✓Random Forest79.1%63.9%48.7%55.2%
Winner: Logistic Regression (better F1-score)

What These Numbers Mean
Logistic Regression Results:
Accuracy: 80.6%

Correct predictions: 1,135 out of 1,409
Better than the "dumb model" (73.5%)!

Precision: 67.4%

Predicted 288 customers would churn
194 actually did churn
94 were false alarms (67.4% = 194/288)
Business impact: Waste money on 94 customers who wouldn't leave anyway

Recall: 51.9%

374 customers actually churned
We caught 194 of them
Missed 180 (51.9% = 194/374)
Business impact: Lost 180 customers we could have saved!

F1-Score: 58.6%

Balance of precision and recall
Higher is better


Confusion Matrix Breakdown
                Predicted
              Stay  Churn
Actual Stay    941     94    ← 94 false alarms (waste $10 discount)
       Churn   180    194    ← 180 missed churners (lose $70/month each!)
The Problem: We're only catching 52% of churners!

Business Impact Analysis
Out of 374 churners in test set:

✓ Saved 194 (offered discount, they stayed)
❌ Lost 180 (didn't catch them, they left)

Costs:

False positives: 94 × $10/month = $940/month wasted
False negatives: 180 × $70/month = $12,600/month lost!

Net impact: Still losing $11,660/month on missed churners.

Why Recall Is Low (52%)
Remember the class imbalance:

73% stayed vs 27% churned
Model is biased toward predicting "stay"
It's being too conservative

What we need: Higher recall (catch more churners), even if precision drops a bit.

Next Steps to Improve
We have 3 options:
Option 1: Handle Class Imbalance

Use class weights (tell model: "churners are more important!")
Use SMOTE (create synthetic churner examples)

Option 2: Try XGBoost

Often better than Random Forest
More powerful algorithm

Option 3: Adjust Decision Threshold

Instead of predicting churn if probability > 0.5
Lower to 0.3 (catch more churners, but more false positives)

####
