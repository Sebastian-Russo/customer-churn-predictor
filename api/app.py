"""
Local Flask API for churn prediction
Run locally, accessible at http://localhost:5000
"""
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model when server starts
print("Loading model...")
with open('../models/logistic_regression_balanced.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("✓ Model loaded")

def preprocess_customer_data(customer_dict):
    """Convert raw customer data to model-ready format"""
    processed = pd.DataFrame(0, index=[0], columns=feature_names)

    for key, value in customer_dict.items():
        if key in feature_names:
            processed[key] = value

    return processed

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint

    POST /predict
    Body: {"tenure": 2, "MonthlyCharges": 85, ...}

    Returns: {"will_churn": true, "churn_probability": 85.5, ...}
    """
    try:
        # Get customer data from request
        customer_data = request.get_json()

        # Preprocess
        X = preprocess_customer_data(customer_data)

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        churn_probability = float(probability[1])

        # Determine risk level
        if churn_probability > 0.7:
            risk_level = 'High'
            recommendation = 'Intervene immediately - offer discount or contract'
        elif churn_probability > 0.4:
            risk_level = 'Medium'
            recommendation = 'Monitor closely - consider retention outreach'
        else:
            risk_level = 'Low'
            recommendation = 'No action needed'

        return jsonify({
            'will_churn': bool(prediction),
            'churn_probability': round(churn_probability * 100, 1),
            'risk_level': risk_level,
            'recommendation': recommendation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'loaded'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Churn Prediction API")
    print("="*60)
    print("Running on: http://localhost:5000")
    print("Endpoints:")
    print("  POST /predict - Make churn prediction")
    print("  GET  /health  - Check API health")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)