$ python3 08_feature_importance.py
Loading data...

============================================================
TRAINING BEST MODEL: Logistic Regression (Balanced)
============================================================
✓ Model trained

============================================================
FEATURE IMPORTANCE
============================================================

Top 15 Most Important Features:
(Positive = increases churn risk, Negative = decreases churn risk)

                       Feature  Coefficient
             Contract_Two year    -1.543591
             Contract_One year    -0.707868
               is_new_customer     0.596962
                  PhoneService    -0.476464
PaymentMethod_Electronic check     0.371049
   InternetService_Fiber optic     0.329668
               StreamingMovies     0.306090
                   StreamingTV     0.287218
            InternetService_No    -0.249088
MultipleLines_No phone service     0.242362
                OnlineSecurity    -0.237743
             MultipleLines_Yes     0.231329
                    Dependents    -0.219613
                total_services    -0.215763
        tenure_group_4-6 years     0.210162

============================================================
CHURN RISK FACTORS
============================================================

📈 INCREASES CHURN RISK (top 5):
  • is_new_customer: +0.597
  • PaymentMethod_Electronic check: +0.371
  • InternetService_Fiber optic: +0.330
  • StreamingMovies: +0.306
  • StreamingTV: +0.287

📉 DECREASES CHURN RISK (top 5):
  • Contract_Two year: -1.544
  • Contract_One year: -0.708
  • PhoneService: -0.476
  • InternetService_No: -0.249
  • OnlineSecurity: -0.238

============================================================
BUSINESS INSIGHTS
============================================================

Key findings from feature importance:

1. CONTRACT TYPE matters most:
   - Month-to-month = high churn risk
   - Long-term contracts = low churn risk

2. TENURE is critical:
   - New customers (0-1 year) = high risk
   - Long-term customers (4-6 years) = low risk

3. SERVICES affect loyalty:
   - Multiple services = customers more locked in
   - Fiber optic alone = high churn (expensive, shop around)

4. PAYMENT METHOD matters:
   - Electronic check (manual) = higher churn
   - Auto-pay = lower churn (set and forget)

============================================================
SAVING MODEL
============================================================
✓ Model saved to: ../models/logistic_regression_balanced.pkl
✓ Feature names saved to: ../models/feature_names.pkl
✓ Metrics saved to: ../models/model_metrics.txt

============================================================
TESTING MODEL LOADING
============================================================
✓ Model loaded successfully

Test prediction on sample customer:
  Predicted class: Stay
  Probability of churn: 11.2%
  Actual outcome: Stay

✓ Phase 3 complete! Model ready for deployment.


### Key Findings: What Drives Churn?
🔴 TOP CHURN RISK FACTORS (Increases Churn)

New customer (+0.597) - Customers < 6 months are flight risks!
Electronic check payment (+0.371) - Manual payment = easier to stop
Fiber optic internet (+0.330) - Expensive, customers shop around
Streaming services (+0.306, +0.287) - Heavy users, price-sensitive


🟢 TOP CHURN PROTECTORS (Decreases Churn)

Two-year contract (-1.544) ⭐ MOST IMPORTANT
One-year contract (-0.708)
Phone service (-0.476)
No internet (-0.249) - Simple service, less to complain about
Online security (-0.238) - Value-added service


Business Recommendations
1. Lock in New Customers
Problem: New customers have highest churn risk
Solution:

Offer contract incentives in first 6 months
Extra support for new customers
"Welcome" discounts for 1-year commitment

2. Push Long-Term Contracts
Impact: Two-year contracts reduce churn by 1.544 (HUGE!)
Solution:

Offer discounts for 1-2 year contracts
Highlight contract savings vs month-to-month

3. Fix Payment Method
Problem: Electronic check = manual payment = higher churn
Solution:

Incentivize auto-pay (credit card, bank transfer)
"$5/month discount for auto-pay"

4. Bundle Services
Problem: Single service (fiber only) = easier to leave
Solution:

Bundle discounts (internet + phone + streaming)
More services = more locked in

5. Target Fiber Customers
Problem: Fiber customers churn more (expensive)
Solution:

Price match competitors
Loyalty programs for fiber customers
Highlight speed advantages


### What We Already Had vs What We Just Did
Before (Phase 2):
python# We TRAINED the model in memory
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Model exists ONLY while script runs
# When script ends → model disappears! ❌
Problem: Every time you want to make a prediction, you'd have to:

Load data
Retrain the model (takes time)
Make prediction


Now (Phase 3):
python# We SAVED the trained model to disk
with open('logistic_regression_balanced.pkl', 'wb') as f:
    pickle.dump(model, f)

# Model is now a FILE on disk ✓
# Can be loaded anytime without retraining!

Analogy: Recipe vs Cooked Meal
Training a model = Cooking a meal

Takes time (gathering ingredients, cooking)
Resource-intensive
Done once

Saving a model = Freezing leftovers

Quick to reheat later
No need to cook again
Ready to use anytime

Loading a model = Reheating leftovers

Fast (seconds)
Same quality as original
Use whenever needed


Why We Save Models
Without saving:
python# Production API (BAD approach)
def predict_churn(customer_data):
    # Load 7,043 rows of training data
    X, y = load_training_data()

    # Train model from scratch (SLOW!)
    model = LogisticRegression()
    model.fit(X, y)  # Takes 30 seconds

    # Finally predict
    prediction = model.predict(customer_data)
    return prediction

# EVERY prediction takes 30+ seconds! ❌

With saving:
python# Load model ONCE when API starts
model = pickle.load(open('model.pkl', 'rb'))

# Production API (GOOD approach)
def predict_churn(customer_data):
    # Just predict (instant!)
    prediction = model.predict(customer_data)
    return prediction

# EACH prediction takes 0.001 seconds! ✓

What We Saved
1. The trained model (.pkl file)
pythonlogistic_regression_balanced.pkl

Contains all learned weights/coefficients
34 feature weights
Ready to make predictions instantly

2. Feature names
pythonfeature_names.pkl

List of column names in exact order
Important! Model expects features in specific order
Example: ['gender', 'tenure', 'Contract_One year', ...]

3. Model metrics
pythonmodel_metrics.txt

Performance summary for reference
Not needed for predictions, just documentation

