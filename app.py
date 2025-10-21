# app.py
# Description: This script creates a Flask web application to serve the
# trained diabetes prediction model. It provides an API endpoint to
# receive patient data and return a prediction with a confidence score.

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model and Scaler ---
# We use a global variable to load the model only once when the app starts.
try:
    model = joblib.load('models/diabetes_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model or scaler not found. Please run main.py to train and save them first.")
    model = None
    scaler = None

# --- Application Routes ---

@app.route('/')
def home():
    """Render the home page with the prediction form."""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring."""
    model_status = 'loaded' if model is not None and scaler is not None else 'not_loaded'
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'environment': os.environ.get('FLASK_ENV', 'development')
    })

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    """
    Handle prediction requests from the web form.
    Validates input data, scales it, and returns the model's prediction.
    """
    if not model or not scaler:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Please run "python main.py" to train and save the model first.',
            'code': 'MODEL_NOT_LOADED'
        }), 503

    try:
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must contain JSON data',
                'code': 'INVALID_REQUEST_FORMAT'
            }), 400

        # Validate that all required fields are present
        required_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(feature in patient_data for feature in required_features):
            missing_fields = [f for f in required_features if f not in patient_data]
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'code': 'MISSING_FIELDS'
            }), 400

        # The order of features must match the training data
        try:
            feature_values = [
                float(patient_data['Pregnancies']),
                float(patient_data['Glucose']),
                float(patient_data['BloodPressure']),
                float(patient_data['SkinThickness']),
                float(patient_data['Insulin']),
                float(patient_data['BMI']),
                float(patient_data['DiabetesPedigreeFunction']),
                float(patient_data['Age'])
            ]
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'Invalid data types',
                'message': 'All values must be valid numbers',
                'code': 'INVALID_DATA_TYPE'
            }), 422

        # Convert to a numpy array for scaling
        final_features = np.array(feature_values).reshape(1, -1)

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_probability = model.predict_proba(scaled_features)

        # Get confidence score with proper class mapping
        classes = model.classes_
        pred_class = int(prediction[0])
        class_idx = np.where(classes == pred_class)[0][0]
        confidence_score = float(prediction_probability[0][class_idx])

        # Return the result as JSON
        return jsonify({
            'prediction': pred_class,
            'confidence': confidence_score,
            'risk_level': 'High Risk' if pred_class == 1 else 'Low Risk'
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred during prediction',
            'code': 'INTERNAL_ERROR'
        }), 500

if __name__ == '__main__':
    # Production-ready configuration
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )
