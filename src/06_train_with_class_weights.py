"""
Phase 2.1: Handle class imbalance with class weights
Goal: Catch MORE churners (improve recall) by telling the model they're more important

ANALOGY:
Before: Detective treats all cases equally
After: Detective told "Missing a churner costs 7x more than a false alarm - prioritize catching them!"
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
X = pd.read_csv('../data/processed/X_features.csv')
y = pd.read_csv('../data/processed/y_target.csv').values.ravel()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples\n")

# Store results
results = []

# Baseline (no class weights) for comparison
print("="*60)
print("BASELINE: Logistic Regression (no class weights)")
print("="*60)
lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train, y_train)
y_pred_baseline = lr_baseline.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_baseline):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_baseline):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_baseline):.3f}")

results.append({
    'Model': 'LR Baseline',
    'Accuracy': accuracy_score(y_test, y_pred_baseline),
    'Precision': precision_score(y_test, y_pred_baseline),
    'Recall': recall_score(y_test, y_pred_baseline),
    'F1-Score': f1_score(y_test, y_pred_baseline)
})

# With balanced class weights
# class_weight='balanced' automatically calculates weights inversely proportional to class frequencies
# Churners (27%) get higher weight than non-churners (73%)
print("\n" + "="*60)
print("IMPROVED: Logistic Regression (balanced class weights)")
print("="*60)
print("Class weights: Churners weighted ~2.7x more than non-churners")
lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_balanced.fit(X_train, y_train)
y_pred_balanced = lr_balanced.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_balanced):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_balanced):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_balanced):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_balanced):.3f}")

results.append({
    'Model': 'LR Balanced',
    'Accuracy': accuracy_score(y_test, y_pred_balanced),
    'Precision': precision_score(y_test, y_pred_balanced),
    'Recall': recall_score(y_test, y_pred_balanced),
    'F1-Score': f1_score(y_test, y_pred_balanced)
})

# Random Forest with balanced weights
print("\n" + "="*60)
print("IMPROVED: Random Forest (balanced class weights)")
print("="*60)
rf_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_balanced.fit(X_train, y_train)
y_pred_rf_balanced = rf_balanced.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf_balanced):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_rf_balanced):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf_balanced):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_rf_balanced):.3f}")

results.append({
    'Model': 'RF Balanced',
    'Accuracy': accuracy_score(y_test, y_pred_rf_balanced),
    'Precision': precision_score(y_test, y_pred_rf_balanced),
    'Recall': recall_score(y_test, y_pred_rf_balanced),
    'F1-Score': f1_score(y_test, y_pred_rf_balanced)
})

# Comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Highlight improvements
print("\n" + "="*60)
print("IMPROVEMENT ANALYSIS")
print("="*60)
baseline_recall = results[0]['Recall']
balanced_recall = results[1]['Recall']
recall_improvement = (balanced_recall - baseline_recall) / baseline_recall * 100

print(f"Baseline Recall:  {baseline_recall:.1%} (caught {int(baseline_recall * 374)} of 374 churners)")
print(f"Balanced Recall:  {balanced_recall:.1%} (caught {int(balanced_recall * 374)} of 374 churners)")
print(f"Improvement:      {recall_improvement:+.1f}%")
print(f"Additional churners saved: {int((balanced_recall - baseline_recall) * 374)}")

# Show confusion matrix for best balanced model
best_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']

if best_model_name == 'LR Balanced':
    y_pred_best = y_pred_balanced
elif best_model_name == 'RF Balanced':
    y_pred_best = y_pred_rf_balanced
else:
    y_pred_best = y_pred_baseline

print(f"\n" + "="*60)
print(f"CONFUSION MATRIX: {best_model_name}")
print("="*60)
cm = confusion_matrix(y_test, y_pred_best)
print(f"\n                Predicted")
print(f"              Stay  Churn")
print(f"Actual Stay  {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"       Churn {cm[1,0]:5d}  {cm[1,1]:5d}")

print(f"\nTrue Positives (caught churners):  {cm[1,1]} of 374 ({cm[1,1]/374*100:.1f}%)")
print(f"False Negatives (missed churners): {cm[1,0]} of 374 ({cm[1,0]/374*100:.1f}%)")

print("\n✓ Training with class weights complete!")