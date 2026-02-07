"""
Phase 4.1: Create prediction function for deployment
This will be used in Lambda/API to make predictions on new customers

ANALOGY:
The detective has learned the patterns. Now we create a simple form:
"Tell me about a customer → I'll predict if they'll churn"
"""
import pickle
import pandas as pd
import numpy as np

def load_model():
    """Load the saved model and feature names"""
    with open('../models/logistic_regression_balanced.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('../models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return model, feature_names

def preprocess_customer_data(customer_dict, feature_names):
    """
    Convert raw customer data to model-ready format

    Input: Dictionary with customer info (like from API request)
    Output: DataFrame with all 34 features in correct order
    """
    # Start with all features as 0
    # pd.DataFrame() creates a single-row table with all features
    processed = pd.DataFrame(0, index=[0], columns=feature_names)

    # Fill in the values we have
    for key, value in customer_dict.items():
        if key in feature_names:
            processed[key] = value

    return processed

def predict_churn(customer_dict):
    """
    Main prediction function - this is what the API will call

    Args:
        customer_dict: Dictionary with customer features

    Returns:
        Dictionary with prediction and probability
    """
    # Load model
    model, feature_names = load_model()

    # Preprocess data
    X = preprocess_customer_data(customer_dict, feature_names)

    # Make prediction
    # .predict() returns 0 or 1
    prediction = model.predict(X)[0]

    # Get probability
    # .predict_proba() returns [prob_stay, prob_churn]
    probability = model.predict_proba(X)[0]
    churn_probability = probability[1]  # Probability of churning

    return {
        'will_churn': bool(prediction),
        'churn_probability': float(churn_probability),
        'risk_level': 'High' if churn_probability > 0.7 else 'Medium' if churn_probability > 0.4 else 'Low',
        'recommendation': 'Intervene immediately' if churn_probability > 0.7 else 'Monitor closely' if churn_probability > 0.4 else 'No action needed'
    }

# Test the function
if __name__ == '__main__':
    print("="*60)
    print("TESTING PREDICTION FUNCTION")
    print("="*60)

    # Test case 1: High-risk customer
    print("\n📊 Test Case 1: High-Risk Customer")
    print("-" * 60)
    high_risk_customer = {
        'gender': 0,  # Female
        'SeniorCitizen': 0,
        'Partner': 0,  # No partner
        'Dependents': 0,  # No dependents
        'tenure': 2,  # Only 2 months (new!)
        'PhoneService': 1,
        'MonthlyCharges': 85.0,  # High charges
        'TotalCharges': 170.0,
        'is_new_customer': 1,  # New customer flag
        'charges_per_tenure': 85.0,
        'total_services': 3,
        'has_multiple_services': 1,
        'has_paperless': 1,
        'PaymentMethod_Electronic check': 1,  # Manual payment
        'InternetService_Fiber optic': 1,  # Expensive service
        # Month-to-month contract (no Contract_One year or Contract_Two year)
    }

    result1 = predict_churn(high_risk_customer)
    print(f"Customer Profile: New (2 months), High charges ($85), Fiber optic, Month-to-month")
    print(f"  Prediction: {'WILL CHURN' if result1['will_churn'] else 'Will Stay'}")
    print(f"  Probability: {result1['churn_probability']*100:.1f}%")
    print(f"  Risk Level: {result1['risk_level']}")
    print(f"  Action: {result1['recommendation']}")

    # Test case 2: Low-risk customer
    print("\n📊 Test Case 2: Low-Risk Customer")
    print("-" * 60)
    low_risk_customer = {
        'gender': 1,  # Male
        'SeniorCitizen': 0,
        'Partner': 1,  # Has partner
        'Dependents': 1,  # Has dependents
        'tenure': 60,  # 5 years (loyal!)
        'PhoneService': 1,
        'MonthlyCharges': 45.0,  # Lower charges
        'TotalCharges': 2700.0,
        'is_new_customer': 0,  # Not new
        'charges_per_tenure': 45.0,
        'total_services': 2,
        'has_multiple_services': 1,
        'has_paperless': 1,
        'Contract_Two year': 1,  # Two-year contract!
        'tenure_group_4-6 years': 1,
    }

    result2 = predict_churn(low_risk_customer)
    print(f"Customer Profile: Loyal (60 months), Low charges ($45), Two-year contract")
    print(f"  Prediction: {'WILL CHURN' if result2['will_churn'] else 'Will Stay'}")
    print(f"  Probability: {result2['churn_probability']*100:.1f}%")
    print(f"  Risk Level: {result2['risk_level']}")
    print(f"  Action: {result2['recommendation']}")

    # Test case 3: Medium-risk customer
    print("\n📊 Test Case 3: Medium-Risk Customer")
    print("-" * 60)
    medium_risk_customer = {
        'gender': 0,
        'SeniorCitizen': 1,  # Senior
        'Partner': 1,
        'Dependents': 0,
        'tenure': 15,  # 15 months
        'PhoneService': 1,
        'MonthlyCharges': 70.0,
        'TotalCharges': 1050.0,
        'is_new_customer': 0,
        'charges_per_tenure': 70.0,
        'total_services': 4,
        'has_multiple_services': 1,
        'has_paperless': 1,
        'Contract_One year': 1,  # One-year contract
        'tenure_group_1-2 years': 1,
    }

    result3 = predict_churn(medium_risk_customer)
    print(f"Customer Profile: Medium tenure (15 months), Medium charges ($70), One-year contract")
    print(f"  Prediction: {'WILL CHURN' if result3['will_churn'] else 'Will Stay'}")
    print(f"  Probability: {result3['churn_probability']*100:.1f}%")
    print(f"  Risk Level: {result3['risk_level']}")
    print(f"  Action: {result3['recommendation']}")

    print("\n✓ Prediction function working! Ready for deployment.")
