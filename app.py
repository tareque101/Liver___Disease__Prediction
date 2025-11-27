from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import pymysql
import os
import warnings
import traceback
from datetime import datetime

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ✅ SIMPLE MYSQL SETUP - REMOVED FLASK-MYSQLDB
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'liver_disease_db'
}

# ✅ SIMPLE MYSQL CONNECTION FUNCTION
def get_mysql_connection():
    try:
        connection = pymysql.connect(
            host=mysql_config['host'],
            user=mysql_config['user'],
            password=mysql_config['password'],
            database=mysql_config['database'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"❌ MySQL connection failed: {e}")
        return None

# ✅ TEST MYSQL CONNECTION
def test_mysql_connection():
    try:
        connection = get_mysql_connection()
        if connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            connection.close()
            print("✅ MySQL connection successful!")
            return True
        return False
    except Exception as e:
        print(f"❌ MySQL connection failed: {e}")
        return False

# Global variables
model = None
feature_names = None
model_info = {
    "model_type": "extreme Gradient Boosting",
    "accuracy": 0.8018433179723502,
    "auc_score": 0.8793747876316684,
    "dataset_size": 722,
    "training_date": "2025-11-22 12:48:23"
}

# Expected features for liver disease prediction (will be loaded from file)
expected_features = []


def load_model_and_features():
    """Load the liver disease prediction model and feature names"""
    global model, feature_names, expected_features

    try:
        # Load feature names first
        print("Loading feature_names (2).pkl...")
        feature_names = joblib.load('feature_names (2).pkl')
        print(f"✅ Feature names loaded: {feature_names}")

        # Set expected features
        expected_features = feature_names
        model_info["features"] = expected_features

        # Load the main model
        print("Loading Liver_disease_model.pkl...")
        model = joblib.load('Liver_disease_model.pkl')
        print(f"✅ Model loaded: {type(model)}")

        # Verify the model expects correct number of features
        if hasattr(model, 'n_features_in_'):
            print(f"📊 Model expects {model.n_features_in_} features")
            expected_count = len(expected_features)
            if model.n_features_in_ != expected_count:
                print(f"⚠️ Warning: Model expects {model.n_features_in_} features, but we have {expected_count}")

        return True

    except Exception as e:
        print(f"❌ Failed to load model or features: {e}")
        print("💡 Creating demo model for testing...")

        # Use default features if feature file not found
        if not expected_features:
            expected_features = [
                "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
                "Alkaline_Phosphotase", "Alamine_Aminotransferase",
                "Aspartate_Aminotransferase", "Total_Protiens",
                "Albumin", "Albumin_and_Globulin_Ratio"
            ]
            model_info["features"] = expected_features

        # Create a simple demo model for testing
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        # Create a dummy model for demo purposes
        X, y = make_classification(n_samples=100, n_features=len(expected_features), random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        print("✅ Demo model created for testing")
        return False


# Load model and features
if load_model_and_features():
    print("🚀 Liver Disease Model loaded successfully!")
    print(f"📋 Expected features: {expected_features}")
    print(f"🔢 Number of features: {len(expected_features)}")
else:
    print("⚠️ Using demo model - replace with actual trained model")

# ✅ TEST MYSQL CONNECTION WHEN APP STARTS
print("🔌 Testing MySQL connection...")
mysql_working = test_mysql_connection()

@app.route('/')
def index():
    return render_template('index.html',
                           model_loaded=model is not None,
                           feature_count=len(expected_features),
                           model_info=model_info)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'})

    try:
        print("📥 Received prediction request...")

        # Get form data
        form_data = request.form
        print(f"📋 Form fields received: {list(form_data.keys())}")

        # Build input data with all expected features
        input_data = {}

        for feature in expected_features:
            value = form_data.get(feature, '').strip()
            if value == '':
                # Set default values based on medical norms
                default_values = {
                    'Age': 45.0,
                    'Gender': 1.0,  # Male
                    'Total_Bilirubin': 0.8,
                    'Direct_Bilirubin': 0.2,
                    'Alkaline_Phosphotase': 200.0,
                    'Alamine_Aminotransferase': 25.0,
                    'Aspartate_Aminotransferase': 30.0,
                    'Total_Protiens': 6.5,
                    'Albumin': 3.5,
                    'Albumin_and_Globulin_Ratio': 1.0
                }
                # Use feature name as key (handle different naming conventions)
                feature_key = feature.replace(' ', '_')  # Handle spaces in feature names
                input_data[feature] = default_values.get(feature, default_values.get(feature_key, 0.0))
            else:
                try:
                    # Handle gender conversion
                    if feature.lower() == 'gender':
                        if value.lower() in ['male', 'm', '1']:
                            input_data[feature] = 1.0
                        elif value.lower() in ['female', 'f', '0']:
                            input_data[feature] = 0.0
                        else:
                            input_data[feature] = float(value)
                    else:
                        input_data[feature] = float(value)
                except ValueError:
                    input_data[feature] = 0.0

        print(f"🔢 Input data prepared with {len(input_data)} features")
        print(f"📊 Sample values: { {k: v for k, v in list(input_data.items())[:3]} }")

        # Create DataFrame with exact expected features order
        input_df = pd.DataFrame([input_data])[expected_features]

        print(f"📊 Input DataFrame shape: {input_df.shape}")
        print(f"✅ Expected shape: (1, {len(expected_features)})")

        # Make prediction
        prediction = model.predict(input_df)[0]
        print(f"📈 Raw prediction: {prediction}")

        # Get probabilities
        try:
            probability = model.predict_proba(input_df)[0]
            confidence = max(probability)
            print(f"📊 Probabilities: {probability}")

            # Determine which probability corresponds to which class
            if hasattr(model, 'classes_'):
                classes = model.classes_
                print(f"🎯 Model classes: {classes}")

                if len(classes) == 2:
                    if classes[1] == 1 or 'disease' in str(classes[1]).lower():
                        disease_prob = probability[1]
                        healthy_prob = probability[0]
                    else:
                        disease_prob = probability[0]
                        healthy_prob = probability[1]
                else:
                    disease_prob = probability[1] if len(probability) > 1 else probability[0]
                    healthy_prob = probability[0]
            else:
                disease_prob = probability[1] if len(probability) > 1 else probability[0]
                healthy_prob = probability[0]

        except Exception as e:
            print(f"⚠️ Probability error: {e}")
            confidence = 0.5
            disease_prob = 0.5 if prediction == 1 else 0.5
            healthy_prob = 0.5 if prediction == 0 else 0.5

        # Determine result
        prediction_label = 'Liver Disease' if prediction == 1 else 'Healthy'
        risk_level = 'high' if disease_prob > 0.7 else 'medium' if disease_prob > 0.3 else 'low'

        print(f"🎯 Final prediction: {prediction_label}")
        print(f"📊 Confidence: {confidence:.2f}")
        print(f"🩺 Risk level: {risk_level}")

        # ✅ UPDATED MYSQL DATABASE SAVING - SIMPLE VERSION
        try:
            connection = get_mysql_connection()
            if connection:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO patient_records 
                        (age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase, 
                         alamine_aminotransferase, aspartate_aminotransferase, total_protiens, 
                         albumin, albumin_globulin_ratio, prediction_result, confidence, risk_level)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        input_data.get('Age'),
                        input_data.get('Gender'),
                        input_data.get('Total_Bilirubin'),
                        input_data.get('Direct_Bilirubin'),
                        input_data.get('Alkaline_Phosphotase'),
                        input_data.get('Alamine_Aminotransferase'),
                        input_data.get('Aspartate_Aminotransferase'),
                        input_data.get('Total_Protiens'),
                        input_data.get('Albumin'),
                        input_data.get('Albumin_and_Globulin_Ratio'),
                        prediction_label,
                        f"{confidence * 100:.1f}%",
                        risk_level
                    ))
                connection.commit()
                connection.close()
                print("✅ Data saved to MySQL database!")
            else:
                print("❌ No MySQL connection - skipping database save")

        except Exception as db_error:
            print(f"❌ Failed to save to database: {db_error}")

        result = {
            'prediction': prediction_label,
            'risk_level': risk_level,
            'confidence': f"{confidence * 100:.1f}%",
            'class_probabilities': {
                'disease': float(disease_prob),
                'healthy': float(healthy_prob)
            },
            'input_features': input_data
        }

        return jsonify(result)

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if model else 'no_model',
        'model_loaded': model is not None,
        'expected_features': len(expected_features),
        'model_type': model_info['model_type'],
        'model_accuracy': model_info['accuracy'],
        'features_loaded_from_file': feature_names is not None,
        'mysql_working': mysql_working
    })


@app.route('/model-info')
def model_info_route():
    info = {
        'model_type': model_info['model_type'],
        'accuracy': model_info['accuracy'],
        'auc_score': model_info['auc_score'],
        'expected_features': expected_features,
        'expected_feature_count': len(expected_features),
        'dataset_size': model_info['dataset_size'],
        'training_date': model_info['training_date'],
        'features_loaded_from_file': feature_names is not None,
        'mysql_working': mysql_working
    }

    if hasattr(model, 'n_features_in_'):
        info['n_features_in'] = model.n_features_in_
    if hasattr(model, 'classes_'):
        info['classes'] = model.classes_.tolist()

    return jsonify(info)


@app.route('/feature-ranges')
def feature_ranges():
    """Return medical reference ranges for features"""
    # Common medical reference ranges for liver function tests
    ranges = {
        'Age': {'min': 1, 'max': 100, 'normal': 'N/A', 'unit': 'years'},
        'Gender': {'options': ['Male (1)', 'Female (0)'], 'unit': 'category'},
        'Total_Bilirubin': {'min': 0.1, 'max': 30.0, 'normal': '0.1-1.2', 'unit': 'mg/dL'},
        'Direct_Bilirubin': {'min': 0.1, 'max': 15.0, 'normal': '0.1-0.3', 'unit': 'mg/dL'},
        'Alkaline_Phosphotase': {'min': 50, 'max': 1500, 'normal': '44-147', 'unit': 'U/L'},
        'Alamine_Aminotransferase': {'min': 5, 'max': 500, 'normal': '7-56', 'unit': 'U/L'},
        'Aspartate_Aminotransferase': {'min': 5, 'max': 500, 'normal': '5-40', 'unit': 'U/L'},
        'Total_Protiens': {'min': 3.0, 'max': 9.0, 'normal': '6.0-8.3', 'unit': 'g/dL'},
        'Albumin': {'min': 1.0, 'max': 5.0, 'normal': '3.4-5.4', 'unit': 'g/dL'},
        'Albumin_and_Globulin_Ratio': {'min': 0.1, 'max': 3.0, 'normal': '1.0-2.0', 'unit': 'ratio'}
    }

    # Create a dynamic ranges dictionary based on actual features
    dynamic_ranges = {}
    for feature in expected_features:
        # Try to find matching range, use generic if not found
        feature_key = feature.replace(' ', '_')  # Handle spaces
        if feature_key in ranges:
            dynamic_ranges[feature] = ranges[feature_key]
        else:
            # Create generic range for unknown features
            dynamic_ranges[feature] = {'min': 0, 'max': 100, 'normal': 'N/A', 'unit': 'units'}

    return jsonify(dynamic_ranges)


if __name__ == '__main__':
    # Try different ports - 5000 is blocked by Windows system process
    available_ports = [5001, 5002, 5003, 8080, 8000, 3000, 5050, 5055]

    for port in available_ports:
        try:
            print(f"🚀 Attempting to start Liver Disease Prediction Server on http://localhost:{port}")
            print(f"📊 Model Type: {model_info['model_type']}")
            print(f"🎯 Accuracy: {model_info['accuracy']:.1%}")
            print(f"📈 AUC Score: {model_info['auc_score']:.1%}")
            print(f"🔢 Expected features: {expected_features}")
            print(f"📋 Number of features: {len(expected_features)}")
            print(f"📁 Features loaded from file: {feature_names is not None}")
            print(f"🔌 MySQL Working: {mysql_working}")
            print("-" * 60)

            app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
            break

        except OSError as e:
            if "attempt was made to access a socket in a way forbidden" in str(e) or "Address already in use" in str(e):
                print(f"❌ Port {port} is not available, trying next port...")
                continue
            else:
                print(f"❌ Unexpected error on port {port}: {e}")
                raise e
    else:
        print("❌ All common ports are occupied. Please free up a port and try again.")
        print("💡 You can also try running on a specific port like 5050:")
        print("   Change the code to: app.run(port=5050)")