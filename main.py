# main.py
# Description: Complete machine learning pipeline for diabetes prediction.
# Handles data loading, preprocessing, model training, evaluation, and persistence.
# Generates performance metrics and SHAP feature importance visualizations.

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from preprocessing import clean_and_preprocess_data
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
DATASET_FILE = 'diabetes.csv'
MODEL_DIRECTORY = 'models'
STATIC_DIRECTORY = 'static'
MODEL_FILE_PATH = os.path.join(MODEL_DIRECTORY, 'diabetes_model.pkl')
SCALER_FILE_PATH = os.path.join(MODEL_DIRECTORY, 'scaler.pkl')
CONFUSION_MATRIX_PATH = os.path.join(STATIC_DIRECTORY, 'confusion_matrix.png')
SHAP_PLOT_PATH = os.path.join(STATIC_DIRECTORY, 'shap_feature_importance.png')

# Model hyperparameters
MODEL_CONFIG = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Training configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = 'Outcome'

def ensure_directories_exist():
    """
    Create necessary directories if they don't exist.
    """
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    os.makedirs(STATIC_DIRECTORY, exist_ok=True)
    logger.info(f"Ensured directories exist: {MODEL_DIRECTORY}, {STATIC_DIRECTORY}")

def load_dataset(file_path):
    """
    Load the diabetes dataset from CSV file.
    
    Args:
        file_path: Path to the CSV dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    try:
        dataset = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully from {file_path}")
        logger.info(f"Dataset shape: {dataset.shape}")
        return dataset
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
        logger.error("Please place the diabetes.csv file in the root directory")
        raise

def split_dataset(processed_data):
    """
    Split the dataset into training and testing sets.
    
    Args:
        processed_data: Preprocessed dataframe with features and target
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("Splitting dataset into training and testing sets")
    
    feature_columns = processed_data.drop(TARGET_COLUMN, axis=1)
    target_column = processed_data[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        feature_columns, 
        target_column, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=target_column
    )
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Testing set size: {X_test.shape}")
    logger.info(f"Training set class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Testing set class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train the XGBoost classifier model.
    
    Args:
        X_train: Training feature data
        y_train: Training target data
        
    Returns:
        XGBClassifier: Trained model
    """
    logger.info("Initializing and training XGBoost model")
    
    classifier = XGBClassifier(**MODEL_CONFIG)
    classifier.fit(X_train, y_train)
    
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    logger.info(f"Training accuracy: {train_accuracy * 100:.2f}%")
    logger.info("Model training completed successfully")
    
    return classifier

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained classifier
        X_test: Testing feature data
        y_test: Testing target data
        
    Returns:
        tuple: (predictions, test_accuracy)
    """
    logger.info("Evaluating model performance on test data")
    
    predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    
    logger.info(f"Testing accuracy: {test_accuracy * 100:.2f}%")
    logger.info("\nDetailed Classification Report:")
    
    classification_rep = classification_report(y_test, predictions, target_names=['No Diabetes', 'Diabetes'])
    print(classification_rep)
    
    return predictions, test_accuracy

def generate_confusion_matrix(y_test, predictions, save_path):
    """
    Generate and save confusion matrix visualization.
    
    Args:
        y_test: True labels
        predictions: Predicted labels
        save_path: Path to save the confusion matrix plot
    """
    logger.info("Generating confusion matrix visualization")
    
    conf_matrix = confusion_matrix(y_test, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Diabetes', 'Diabetes'],
        yticklabels=['No Diabetes', 'Diabetes'],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Diabetes Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")

def generate_shap_feature_importance(model, X_test, save_path):
    """
    Generate SHAP feature importance plot.
    
    Args:
        model: Trained model
        X_test: Test feature data
        save_path: Path to save the SHAP plot
    """
    logger.info("Generating SHAP feature importance visualization")
    
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title("Feature Importance (SHAP Analysis)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP feature importance plot saved to {save_path}")
    except Exception as error:
        logger.error(f"Error generating SHAP plot: {str(error)}")
        logger.warning("Continuing without SHAP visualization")

def save_model_and_scaler(model, scaler):
    """
    Save the trained model and scaler to disk.
    
    Args:
        model: Trained classifier
        scaler: Fitted feature scaler
    """
    logger.info("Saving trained model and scaler")
    
    joblib.dump(model, MODEL_FILE_PATH)
    joblib.dump(scaler, SCALER_FILE_PATH)
    
    model_size = os.path.getsize(MODEL_FILE_PATH) / 1024
    scaler_size = os.path.getsize(SCALER_FILE_PATH) / 1024
    
    logger.info(f"Model saved to {MODEL_FILE_PATH} ({model_size:.2f} KB)")
    logger.info(f"Scaler saved to {SCALER_FILE_PATH} ({scaler_size:.2f} KB)")

def train_evaluate_and_save_model():
    """
    Execute the complete machine learning pipeline:
    - Load and preprocess data
    - Split into train/test sets
    - Train XGBoost model
    - Evaluate performance
    - Generate visualizations
    - Save model artifacts
    """
    logger.info("=" * 60)
    logger.info("Starting Diabetes Prediction Model Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Ensure directories exist
        ensure_directories_exist()
        
        # Step 2: Load dataset
        diabetes_dataset = load_dataset(DATASET_FILE)
        
        # Step 3: Preprocess data
        logger.info("Preprocessing dataset")
        processed_data, feature_scaler = clean_and_preprocess_data(diabetes_dataset)
        logger.info("Data preprocessing completed successfully")
        
        # Step 4: Split dataset
        X_train, X_test, y_train, y_test = split_dataset(processed_data)
        
        # Step 5: Train model
        trained_model = train_model(X_train, y_train)
        
        # Step 6: Evaluate model
        predictions, test_accuracy = evaluate_model(trained_model, X_test, y_test)
        
        # Step 7: Generate confusion matrix
        generate_confusion_matrix(y_test, predictions, CONFUSION_MATRIX_PATH)
        
        # Step 8: Generate SHAP feature importance
        generate_shap_feature_importance(trained_model, X_test, SHAP_PLOT_PATH)
        
        # Step 9: Save model and scaler
        save_model_and_scaler(trained_model, feature_scaler)
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
        logger.info("=" * 60)
        
    except Exception as error:
        logger.error(f"Pipeline failed with error: {str(error)}")
        raise

if __name__ == '__main__':
    train_evaluate_and_save_model()
