"""
Phase 2.2: Try XGBoost
XGBoost = Extreme Gradient Boosting

ANALOGY:
Random Forest: 100 detectives work independently, then vote
XGBoost: 100 detectives work sequentially - each one focuses on the cases
         the previous detectives got wrong. More focused learning!
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
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

# Baseline: Logistic Regression (balanced)
print("="*60)
print("BASELINE: Logistic Regression (balanced)")
print("="*60)
lr_balanced = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_balanced.fit(X_train, y_train)
y_pred_lr = lr_balanced.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_lr):.3f}")

results.append({
    'Model': 'LR Balanced',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-Score': f1_score(y_test, y_pred_lr)
})

# Random Forest (balanced)
print("\n" + "="*60)
print("Random Forest (balanced)")
print("="*60)
rf_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_balanced.fit(X_train, y_train)
y_pred_rf = rf_balanced.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_rf):.3f}")

results.append({
    'Model': 'RF Balanced',
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1-Score': f1_score(y_test, y_pred_rf)
})

# XGBoost (default parameters)
print("\n" + "="*60)
print("XGBoost (default)")
print("="*60)
print("Training...")
xgb_default = xgb.XGBClassifier(
    n_estimators=100,      # Number of trees (like Random Forest)
    random_state=42,
    eval_metric='logloss'  # Suppress warning
)
xgb_default.fit(X_train, y_train)
y_pred_xgb_default = xgb_default.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb_default):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb_default):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_xgb_default):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_xgb_default):.3f}")

results.append({
    'Model': 'XGBoost Default',
    'Accuracy': accuracy_score(y_test, y_pred_xgb_default),
    'Precision': precision_score(y_test, y_pred_xgb_default),
    'Recall': recall_score(y_test, y_pred_xgb_default),
    'F1-Score': f1_score(y_test, y_pred_xgb_default)
})

# XGBoost with class weights
# scale_pos_weight = ratio of negative to positive samples = 5174/1869 ≈ 2.77
print("\n" + "="*60)
print("XGBoost (with class weights)")
print("="*60)
print("scale_pos_weight: 2.77 (churners weighted 2.77x more)")
print("Training...")
xgb_balanced = xgb.XGBClassifier(
    n_estimators=100,
    scale_pos_weight=2.77,  # XGBoost's version of class_weight='balanced'
    learning_rate=0.1,      # How much each tree contributes
    max_depth=5,            # How deep each tree can grow
    random_state=42,
    eval_metric='logloss'
)
xgb_balanced.fit(X_train, y_train)
y_pred_xgb_balanced = xgb_balanced.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred_xgb_balanced):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb_balanced):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_xgb_balanced):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_xgb_balanced):.3f}")

results.append({
    'Model': 'XGBoost Balanced',
    'Accuracy': accuracy_score(y_test, y_pred_xgb_balanced),
    'Precision': precision_score(y_test, y_pred_xgb_balanced),
    'Recall': recall_score(y_test, y_pred_xgb_balanced),
    'F1-Score': f1_score(y_test, y_pred_xgb_balanced)
})

# Comparison
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
results_df = pd.DataFrame(results)
# Sort by F1-Score descending
results_df = results_df.sort_values('F1-Score', ascending=False)
print(results_df.to_string(index=False))

# Highlight winner
print("\n" + "="*60)
print("WINNER ANALYSIS")
print("="*60)
winner = results_df.iloc[0]
print(f"Best Model: {winner['Model']}")
print(f"  F1-Score: {winner['F1-Score']:.3f}")
print(f"  Recall:   {winner['Recall']:.3f} (catches {int(winner['Recall']*374)} of 374 churners)")
print(f"  Precision: {winner['Precision']:.3f}")

# Confusion matrix for winner
if winner['Model'] == 'LR Balanced':
    y_pred_winner = y_pred_lr
elif winner['Model'] == 'RF Balanced':
    y_pred_winner = y_pred_rf
elif winner['Model'] == 'XGBoost Default':
    y_pred_winner = y_pred_xgb_default
else:
    y_pred_winner = y_pred_xgb_balanced

print(f"\n" + "="*60)
print(f"CONFUSION MATRIX: {winner['Model']}")
print("="*60)
cm = confusion_matrix(y_test, y_pred_winner)
print(f"\n                Predicted")
print(f"              Stay  Churn")
print(f"Actual Stay  {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"       Churn {cm[1,0]:5d}  {cm[1,1]:5d}")

print(f"\nTrue Positives (caught churners):  {cm[1,1]} of 374 ({cm[1,1]/374*100:.1f}%)")
print(f"False Negatives (missed churners): {cm[1,0]} of 374 ({cm[1,0]/374*100:.1f}%)")
print(f"False Positives (false alarms):    {cm[0,1]} of 1035 ({cm[0,1]/1035*100:.1f}%)")

print("\n✓ XGBoost comparison complete!")