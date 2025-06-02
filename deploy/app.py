import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle

# Import your model classes (you'll need to include model.py in your deployment)
from model import TabTransformer, FeatureEngineering

app = Flask(__name__)

# Global variables to store loaded model and pipeline
model = None
feature_pipeline = None
device = torch.device('cpu')  # Use CPU for deployment

def load_model_and_pipeline():
    """Load the trained model and feature pipeline"""
    global model, feature_pipeline
    
    try:
        # Load feature pipeline
        feature_pipeline = FeatureEngineering.load_pipeline('configs/feature_pipeline/feature_pipeline.pkl')
        print("Feature pipeline loaded successfully")
        
        # Get model parameters from the pipeline
        cat_cardinalities = {
            feat: len(feature_pipeline.label_encoders[feat].classes_)
            for feat in feature_pipeline.cat_cols
        }
        
        # Initialize model with correct parameters
        model = TabTransformer(
            cat_cardinalities=cat_cardinalities,
            num_cont_features=len(feature_pipeline.selected_cont_cols),
            num_classes=len(feature_pipeline.target_encoder.classes_),
            dim=64,
            depth=8,
            mlp_dropout=0.3
        )
        
        # Load trained weights
        model.load_state_dict(torch.load('models/loan_model.pth', map_location=device))
        model.eval()
        print("Model loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_loan_eligibility(input_data):
    """Make prediction using the loaded model"""
    global model, feature_pipeline
    
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Transform the data using the feature pipeline
        X_cat, X_cont, _ = feature_pipeline.transform(df)
        
        # Convert to tensors
        cat_tensor = torch.tensor(X_cat, dtype=torch.long)
        cont_tensor = torch.tensor(X_cont, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(cat_tensor, cont_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Decode prediction
        prediction = feature_pipeline.target_encoder.inverse_transform([predicted_class])[0]
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'success': True
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'Current Loan Amount': float(request.form.get('current_loan_amount', 0)),
            'Term': request.form.get('term', ''),
            'Credit Score': float(request.form.get('credit_score', 0)),
            'Years in current job': request.form.get('years_current_job', ''),
            'Home Ownership': request.form.get('home_ownership', ''),
            'Annual Income': float(request.form.get('annual_income', 0)),
            'Purpose': request.form.get('purpose', ''),
            'Monthly Debt': float(request.form.get('monthly_debt', 0)),
            'Years of Credit History': float(request.form.get('years_credit_history', 0)),
            'Months since last delinquent': float(request.form.get('months_since_delinquent', 0)),
            'Number of Open Accounts': float(request.form.get('number_open_accounts', 0)),
            'Number of Credit Problems': float(request.form.get('number_credit_problems', 0)),
            'Current Credit Balance': float(request.form.get('current_credit_balance', 0)),
            'Maximum Open Credit': float(request.form.get('maximum_open_credit', 0)),
            'Bankruptcies': float(request.form.get('bankruptcies', 0)),
            'Tax Liens': float(request.form.get('tax_liens', 0))
        }
        
        # Make prediction
        result = predict_loan_eligibility(input_data)
        
        if result['success']:
            return render_template('result.html', 
                                 prediction=result['prediction'],
                                 confidence=round(result['confidence'] * 100, 2))
        else:
            return render_template('error.html', error=result['error'])
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        result = predict_loan_eligibility(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False})

if __name__ == '__main__':
    # Load model and pipeline on startup
    if load_model_and_pipeline():
        print("Starting Flask application...")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    else:
        print("Failed to load model and pipeline. Please check your files.")
