"""
Phase 2: Train multiple models and compare performance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Load encoded data
print("Loading encoded data...")
# pd.read_csv() loads the CSV into a DataFrame
X = pd.read_csv('../data/X_features.csv')
y = pd.read_csv('../data/y_target.csv').values.ravel()  # .ravel() flattens to 1D array

print(f"Features: {X.shape}")  # (rows, columns)
print(f"Target: {y.shape}")    # (rows,)

# Check target distribution
print(f"\nTarget distribution:")
# np.unique() counts unique values and their frequencies
unique, counts = np.unique(y, return_counts=True)
for value, count in zip(unique, counts):
    print(f"  Class {value}: {count} ({count/len(y)*100:.1f}%)")

# Split data into training and testing sets
# train_test_split() randomly divides data into train (80%) and test (20%)
# stratify=y ensures both sets have same class distribution (73%/27%)
# random_state=42 makes the split reproducible (same split every time)
print("\n" + "="*60)
print("TRAIN/TEST SPLIT")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    stratify=y,         # Keep same class balance in both sets
    random_state=42     # Reproducible split
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Verify stratification worked
print(f"\nTrain target distribution:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
for value, count in zip(unique_train, counts_train):
    print(f"  Class {value}: {count} ({count/len(y_train)*100:.1f}%)")

# Dictionary to store all models and their results
models = {}
results = []

# Model 1: Logistic Regression (baseline)
# Simple linear model - fast but may not capture complex patterns
print("\n" + "="*60)
print("MODEL 1: LOGISTIC REGRESSION (Baseline)")
print("="*60)
print("Training...")

# Create model instance
# max_iter=1000 gives more iterations to converge
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
# .fit() learns patterns from training data
log_reg.fit(X_train, y_train)

# Make predictions on test set
# .predict() applies learned patterns to new data
y_pred_lr = log_reg.predict(X_test)

# Calculate metrics
# These measure different aspects of performance
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)  # Of predicted churners, % actually churned
recall_lr = recall_score(y_test, y_pred_lr)        # Of actual churners, % we caught
f1_lr = f1_score(y_test, y_pred_lr)                # Balance of precision and recall

print(f"Accuracy:  {accuracy_lr:.3f}")
print(f"Precision: {precision_lr:.3f}")
print(f"Recall:    {recall_lr:.3f}")
print(f"F1-Score:  {f1_lr:.3f}")

# Store results
models['Logistic Regression'] = log_reg
results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_lr,
    'Precision': precision_lr,
    'Recall': recall_lr,
    'F1-Score': f1_lr
})

# Model 2: Random Forest (ensemble method)
# Builds multiple decision trees and averages their predictions
# Usually more powerful than logistic regression
print("\n" + "="*60)
print("MODEL 2: RANDOM FOREST")
print("="*60)
print("Training...")

# n_estimators=100 means build 100 decision trees
# random_state=42 makes results reproducible
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Accuracy:  {accuracy_rf:.3f}")
print(f"Precision: {precision_rf:.3f}")
print(f"Recall:    {recall_rf:.3f}")
print(f"F1-Score:  {f1_rf:.3f}")

models['Random Forest'] = rf
results.append({
    'Model': 'Random Forest',
    'Accuracy': accuracy_rf,
    'Precision': precision_rf,
    'Recall': recall_rf,
    'F1-Score': f1_rf
})

# Comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
# pd.DataFrame() creates a table from results list
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Show confusion matrix for best model (by F1-score)
# Confusion matrix shows True Positive, False Positive, True Negative, False Negative
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

print(f"\n" + "="*60)
print(f"CONFUSION MATRIX: {best_model_name}")
print("="*60)
# confusion_matrix() creates 2x2 table of predictions vs actual
cm = confusion_matrix(y_test, y_pred_best)
print(f"\n                Predicted")
print(f"              Stay  Churn")
print(f"Actual Stay  {cm[0,0]:5d}  {cm[0,1]:5d}")
print(f"       Churn {cm[1,0]:5d}  {cm[1,1]:5d}")

print(f"\nExplanation:")
print(f"  True Negatives (correct 'stay'):   {cm[0,0]}")
print(f"  False Positives (wrong 'churn'):   {cm[0,1]}")
print(f"  False Negatives (missed churners): {cm[1,0]}")
print(f"  True Positives (caught churners):  {cm[1,1]}")

# Classification report (detailed per-class metrics)
print(f"\n" + "="*60)
print(f"DETAILED REPORT: {best_model_name}")
print("="*60)
# classification_report() shows precision/recall/f1 for each class
print(classification_report(y_test, y_pred_best, target_names=['Stay', 'Churn']))

print("\n✓ Training complete!")