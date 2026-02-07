$ # Test health endpoint
curl http://localhost:5000/health

# Test prediction with high-risk customer
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 85,
    "is_new_customer": 1,
    "PaymentMethod_Electronic check": 1,
    "InternetService_Fiber optic": 1
  }'
{
  "model": "loaded",
  "status": "healthy"
}
{
  "churn_probability": 91.1,
  "recommendation": "Intervene immediately - offer discount or contract",
  "risk_level": "High",
  "will_churn": true
}




### Results Analysis

Health Check:
json{
  "model": "loaded",
  "status": "healthy"
}
✓ API is running, model is loaded
High-Risk Customer Prediction:
json{
  "churn_probability": 91.1,
  "recommendation": "Intervene immediately - offer discount or contract",
  "risk_level": "High",
  "will_churn": true
}
```

**91.1% chance of churn!** The model correctly identified this as extremely high risk.

---

## **What We've Accomplished**

✅ **Phase 1:** Data exploration, cleaning, feature engineering, encoding
✅ **Phase 2:** Trained 5 models, found best one (LR Balanced - 80.7% recall)
✅ **Phase 3:** Analyzed feature importance, saved model
✅ **Phase 4:** Built working Flask API

---

## **Complete ML Pipeline**
```
Raw Data (CSV)
    ↓
Data Processing (src/data/)
    ↓
Model Training (src/models/)
    ↓
Saved Model (models/)
    ↓
Flask API (api/app.py)
    ↓
Predictions via HTTP

Project Complete Summary
What you built:

End-to-end ML pipeline for customer churn prediction
80.7% recall (catches 302 of 374 churners)
Saves $5,570/month vs baseline
Production-ready Flask API
Professional project structure

Skills learned:

Classic ML (Logistic Regression, Random Forest, XGBoost)
Handling class imbalance
Feature engineering
Model evaluation (precision, recall, F1)
Pickling/loading models
Flask API deployment

