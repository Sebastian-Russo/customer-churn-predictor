# Customer Churn Predictor - Action Plan

## Goal
Predict which customers will cancel their subscription based on behavior data.

---

## Phase 1: Data Exploration & Preparation

### 1.1 Explore Data
- Load CSV with pandas
- Check shape, columns, data types
- Identify target variable (Churn: Yes/No)
- Check class distribution (imbalanced?)
- Find missing values
- Understand numeric vs categorical features

### 1.2 Data Cleaning
- Handle missing values (drop or impute)
- Fix data type issues (e.g., TotalCharges might be string)
- Remove unnecessary columns (customerID)

### 1.3 Feature Engineering
- Create new features from existing ones
  - tenure_group (new/medium/long-term customer)
  - charges_per_month_of_tenure
  - has_multiple_services
- Analyze feature importance

### 1.4 Encode Categorical Variables
- Convert Yes/No → 1/0
- One-hot encode multi-category features (Contract, PaymentMethod, etc.)
- Result: All numeric data ready for ML

---

## Phase 2: Model Training

### 2.1 Train/Test Split
- Split data 80/20 (train/test)
- Stratify by Churn (maintain class balance)

### 2.2 Try Multiple Models
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Compare performance

### 2.3 Handle Class Imbalance
- Try class weights
- Try SMOTE (oversampling minority class)
- Compare results

### 2.4 Hyperparameter Tuning
- Grid search or random search
- Cross-validation
- Find best model configuration

---

## Phase 3: Model Evaluation

### 3.1 Metrics
- Accuracy (baseline)
- Precision (of predicted churners, how many actually churn?)
- Recall (of actual churners, how many did we catch?)
- F1-score (balance precision/recall)
- Confusion matrix
- ROC curve / AUC

### 3.2 Business Metrics
- Cost of false positives (intervening with non-churner)
- Cost of false negatives (missing a churner)
- ROI calculation
- Decide optimal threshold

### 3.3 Feature Importance
- Which features matter most?
- Business insights from model

---

## Phase 4: Deployment

### 4.1 Save Model
- Pickle or joblib
- Save preprocessing pipeline with model

### 4.2 Create Prediction API
**Option A: Lambda (simpler, cheaper)**
- Package model + dependencies
- Create Lambda function
- API Gateway endpoint

**Option B: SageMaker (more robust)**
- Create inference.py
- Upload model to S3
- Create endpoint

### 4.3 Build Frontend
- Reuse React template from MNIST
- Form inputs for customer features
- Display churn prediction + probability
- Show intervention recommendation

---

## Phase 5: Testing & Iteration

### 5.1 Test API
- Curl test with sample customers
- Verify predictions make sense

### 5.2 Frontend Integration
- Connect React to API
- Test end-to-end flow

### 5.3 Documentation
- README with setup instructions
- Model performance summary
- Business recommendations

---

## Key Differences from MNIST

| Aspect | MNIST | Churn |
|--------|-------|-------|
| Data type | Images | CSV/tabular |
| Preprocessing | Normalize pixels | Encode categories, handle missing data |
| Model | CNN (deep learning) | Random Forest/XGBoost (classic ML) |
| Evaluation | Accuracy | Precision/Recall/ROI |
| Features | 784 pixels | 21 mixed features |
| Explainability | Hard | Easy (feature importance) |

---

## Success Criteria

- [ ] Model achieves >80% accuracy
- [ ] Recall >70% (catch most churners)
- [ ] Precision >60% (don't waste money on false alarms)
- [ ] Positive ROI calculation
- [ ] Working API + frontend
- [ ] Can explain which features drive churn

---

## Estimated Time

- Phase 1: 1 day
- Phase 2: 1 day
- Phase 3: 0.5 day
- Phase 4: 1 day
- Phase 5: 0.5 day

**Total: 4 days**

EOF

echo "✓ Created ACTION_PLAN.md"