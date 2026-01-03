# app.py
# Description: Flask web application serving the diabetes prediction model.
# Provides REST API endpoints for predictions with comprehensive error handling,
# logging, CORS support, and input validation for production deployment.

import os
import math
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
application = Flask(__name__)
app = application

# Enable CORS for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Application configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size
app.config['JSON_SORT_KEYS'] = False

# Global variables for model and scaler
prediction_model = None
feature_scaler = None

# Model paths
MODEL_DIRECTORY = 'models'
MODEL_FILE_PATH = os.path.join(MODEL_DIRECTORY, 'diabetes_model.pkl')
SCALER_FILE_PATH = os.path.join(MODEL_DIRECTORY, 'scaler.pkl')

# Required feature names for validation
REQUIRED_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Validation rules with clinical ranges
VALIDATION_RULES = {
    'Pregnancies': {'min': 0, 'max': 20, 'type': int, 'unit': ''},
    'Glucose': {'min': 50, 'max': 600, 'type': float, 'unit': 'mg/dL'},
    'BloodPressure': {'min': 40, 'max': 250, 'type': float, 'unit': 'mmHg'},
    'SkinThickness': {'min': 1, 'max': 100, 'type': float, 'unit': 'mm'},
    'Insulin': {'min': 0, 'max': 1000, 'type': float, 'unit': 'mu U/ml'},
    'BMI': {'min': 10, 'max': 80, 'type': float, 'unit': 'kg/m²'},
    'DiabetesPedigreeFunction': {'min': 0.0, 'max': 3.0, 'type': float, 'unit': ''},
    'Age': {'min': 1, 'max': 120, 'type': int, 'unit': 'years'}
}

def validate_patient_data(data):
    """
    Validate patient data for clinical plausibility.
    
    Args:
        data: Dictionary containing patient medical data
        
    Returns:
        tuple: (is_valid, error_messages, cleaned_data)
    """
    errors = []
    cleaned_data = {}
    
    for field, rules in VALIDATION_RULES.items():
        # Check if field is present
        if field not in data:
            errors.append(f"Missing required field: {field}")
            continue
        
        # Try to convert to appropriate type
        try:
            value = rules['type'](data[field])
        except (ValueError, TypeError):
            errors.append(f"{field} must be a valid number")
            continue
        
        # Check for NaN/Infinity
        if not isinstance(value, int):
            if math.isnan(value) or math.isinf(value):
                errors.append(f"{field} cannot be NaN or infinity")
                continue
        
        # Check value ranges
        if value < rules['min'] or value > rules['max']:
            unit = rules.get('unit', '')
            unit_str = f" {unit}" if unit else ""
            errors.append(
                f"{field} must be between {rules['min']} and {rules['max']}{unit_str}"
            )
            continue
        
        cleaned_data[field] = value
    
    is_valid = len(errors) == 0
    return is_valid, errors, cleaned_data

def load_model_and_scaler():
    """
    Load the trained machine learning model and feature scaler.
    
    Returns:
        tuple: (model, scaler) if successful, (None, None) otherwise
    """
    global prediction_model, feature_scaler
    
    try:
        if os.path.exists(MODEL_FILE_PATH) and os.path.exists(SCALER_FILE_PATH):
            prediction_model = joblib.load(MODEL_FILE_PATH)
            feature_scaler = joblib.load(SCALER_FILE_PATH)
            logger.info("Model and scaler loaded successfully")
            return prediction_model, feature_scaler
        else:
            logger.error(f"Model files not found at {MODEL_DIRECTORY}")
            logger.error("Please run main.py to train and save the model first")
            return None, None
    except Exception as error:
        logger.error(f"Error loading model: {str(error)}")
        return None, None

# Load model on startup
load_model_and_scaler()

@app.route('/')
def home():
    """
    Render the home page with the prediction interface.
    
    Returns:
        HTML: Rendered index.html template
    """
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring application status.
    
    Returns:
        JSON: Application health status
    """
    model_status = "loaded" if prediction_model is not None else "not_loaded"
    scaler_status = "loaded" if feature_scaler is not None else "not_loaded"
    
    return jsonify({
        'status': 'healthy' if prediction_model and feature_scaler else 'degraded',
        'model': model_status,
        'scaler': scaler_status
    }), 200 if prediction_model and feature_scaler else 503

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    """
    Handle diabetes prediction requests with comprehensive validation.
    
    Expects JSON payload with patient medical data.
    Returns prediction result with confidence score or detailed validation errors.
    
    Returns:
        JSON: Prediction result with confidence score or error message
    """
    # Check if model is loaded
    if prediction_model is None or feature_scaler is None:
        logger.error("Prediction attempted but model not loaded")
        return jsonify({
            'error': 'Model not available',
            'message': 'Please run \"python main.py\" to train and save the model first.',
            'code': 'MODEL_NOT_LOADED'
        }), 503
    
    try:
        # Parse request data
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must contain JSON data',
                'code': 'INVALID_REQUEST_FORMAT'
            }), 400
        
        # Comprehensive validation
        is_valid, validation_errors, cleaned_data = validate_patient_data(patient_data)
        
        if not is_valid:
            return jsonify({
                'error': 'Invalid input data',
                'message': 'Please correct the following issues:',
                'validation_errors': validation_errors,
                'code': 'VALIDATION_FAILED'
            }), 422
        
        # Convert to feature array in correct order
        feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        feature_values = [cleaned_data[field] for field in feature_order]
        input_features = np.array(feature_values).reshape(1, -1)
        
        # Scale features
        scaled_features = feature_scaler.transform(input_features)
        
        # Make prediction
        prediction = prediction_model.predict(scaled_features)
        prediction_probabilities = prediction_model.predict_proba(scaled_features)
        
        # Extract confidence score for predicted class
        predicted_class = int(prediction[0])
        
        # Get confidence score with proper class mapping
        classes = prediction_model.classes_
        class_idx = np.where(classes == predicted_class)[0][0]
        confidence_score = float(prediction_probabilities[0][class_idx])
        
        # Log prediction
        logger.info(f"Prediction made: class={predicted_class}, confidence={confidence_score:.2f}")
        
        # Return prediction result
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence_score,
            'risk_level': 'High Risk' if predicted_class == 1 else 'Low Risk',
            'model_info': {
                'classes': classes.tolist(),
                'feature_order': feature_order
            }
        }), 200
        
    except Exception as unexpected_error:
        logger.error(f"Unexpected error during prediction: {str(unexpected_error)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred during prediction',
            'code': 'INTERNAL_ERROR'
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """
    Handle 404 errors.
    
    Args:
        error: The error object
        
    Returns:
        JSON: Error message with 404 status
    """
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_server_error(error):
    """
    Handle 500 errors.
    
    Args:
        error: The error object
        
    Returns:
        JSON: Error message with 500 status
    """
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Determine environment
    environment = os.getenv('ENVIRONMENT', 'development')
    debug_mode = environment == 'development'
    
    # Get port from environment or use default
    port = int(os.getenv('PORT', 5000))
    
    logger.info(f"Starting application in {environment} mode on port {port}")
    
    # Run application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )
