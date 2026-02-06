# What We're Building
Goal: Predict which customers will cancel their subscription (churn) based on their behavior.
Business value: Company can offer discounts/incentives to at-risk customers before they leave.
Tech stack:

Python + scikit-learn (classic ML)
pandas (data processing)
AWS Lambda or SageMaker (deployment - your choice)
React frontend (reuse your existing one!)

### Break down what we're doing and why it's different from MNIST.

### What Is Customer Churn?
Churn = When a customer cancels/leaves your service.
Example:

Netflix subscriber cancels → churn
Phone customer switches to different carrier → churn
SaaS user doesn't renew → churn

### Why companies care:

Acquiring new customers costs 5-25x more than retaining existing ones
If you can predict churn BEFORE it happens, you can intervene


### The Business Problem
Telco Company's Situation:

They have 7,043 customers
Some will cancel next month (churn)
They want to know WHO will churn so they can:

Offer discounts
Improve service
Call them to resolve issues



### Your ML model's job: Predict "Will this customer churn?" based on their data.

### What's in the Dataset?
Each row = 1 customer with features like:
FeatureExampleWhy It Matterstenure12 monthsNew customers churn moreMonthlyCharges$70.35High bills → more likely to leaveContractMonth-to-monthNo commitment = easier to leaveInternetServiceFiber opticService quality affects satisfactionTechSupportNoPoor support → frustration → churnPaymentMethodElectronic checkAuto-pay users less likely to churnChurnYes/NoThis is what we're predicting

### MNIST vs Churn: Key Differences
AspectMNIST (Image)Churn (Tabular)Data typeImages (pixels)CSV/spreadsheet rowsFeatures784 numbers (28×28 pixels)21 mixed types (numbers, categories)PreprocessingNormalize pixels, resizeHandle missing data, encode categoriesModelCNN (deep learning)Random Forest, XGBoost (classic ML)Why different models?Images need spatial patternsTabular needs feature relationshipsExplainabilityHard to explain (black box)Easy - "Tenure + high charges = churn"

### Why Classic ML (Not Deep Learning)?
For tabular data, classic ML often beats deep learning because:

Less data needed - Works with thousands of rows (vs millions for deep learning)
Faster training - Seconds vs hours
More interpretable - "Customer will churn because tenure is low AND charges are high"
Better with mixed data types - Numbers, categories, dates all together



What We'll Learn
New Concepts (Not in MNIST):

Feature engineering - Creating useful features from raw data

Example: tenure → create is_new_customer (Yes if <6 months)


Handling categorical data - Converting text to numbers

Contract = "Month-to-month" → How do we feed this to ML?


Class imbalance - Only 27% of customers churn

Can't just predict "No churn" for everyone (would be 73% accurate but useless)


Business metrics - Not just accuracy

Precision: Of customers we predicted would churn, how many actually did?
Recall: Of customers who churned, how many did we catch?
ROI: Is it worth intervening?


Tree-based models - Random Forest, XGBoost

Different from neural networks
Make decisions like: "If tenure < 6 AND charges > $70 → predict churn"




The Pipeline We'll Build
CSV Data
  ↓
Load with pandas
  ↓
Explore data (missing values, distributions)
  ↓
Feature engineering (create new useful features)
  ↓
Encode categorical variables (text → numbers)
  ↓
Train/test split
  ↓
Train Random Forest model
  ↓
Evaluate (precision, recall, ROI)
  ↓
Deploy as API (Lambda or SageMaker)
  ↓
React frontend (input customer data → predict churn)

Real-World Impact
If your model is good:

Identify 500 at-risk customers
Intervene with $10 discount/month
Save 300 of them (60% success rate)
Each customer worth $70/month

ROI Calculation:

Cost: 500 × $10 = $5,000/month
Saved revenue: 300 × $70 = $21,000/month
Net gain: $16,000/month 🎉
