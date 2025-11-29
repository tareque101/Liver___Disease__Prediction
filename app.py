from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
from datetime import datetime
from sqlalchemy import text

app = Flask(__name__)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/liver_disease_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class PatientRecord(db.Model):
    __tablename__ = 'patient_records'
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    total_bilirubin = db.Column(db.Float, nullable=False)
    direct_bilirubin = db.Column(db.Float, nullable=False)
    alkaline_phosphotase = db.Column(db.Float, nullable=False)
    alamine_aminotransferase = db.Column(db.Float, nullable=False)
    aspartate_aminotransferase = db.Column(db.Float, nullable=False)
    total_protiens = db.Column(db.Float, nullable=False)
    albumin = db.Column(db.Float, nullable=False)
    albumin_globulin_ratio = db.Column(db.Float, nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

# Global variables
model = None
expected_features = []
model_info = {
    "model_type": "Extreme Gradient Boosting",
    "accuracy": 0.801,
    "auc_score": 0.879,
    "dataset_size": 722
}

def init_database():
    try:
        with app.app_context():
            db.session.execute(text('SELECT 1'))
            print("✅ Database connected")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

def save_prediction(patient_data, prediction_result, confidence, risk_level):
    try:
        with app.app_context():
            record = PatientRecord(
                age=int(patient_data['Age']),
                gender='Male' if patient_data['Gender'] == 1 else 'Female',
                total_bilirubin=patient_data['Total_Bilirubin'],
                direct_bilirubin=patient_data['Direct_Bilirubin'],
                alkaline_phosphotase=patient_data['Alkaline_Phosphotase'],
                alamine_aminotransferase=patient_data['Alamine_Aminotransferase'],
                aspartate_aminotransferase=patient_data['Aspartate_Aminotransferase'],
                total_protiens=patient_data['Total_Protiens'],
                albumin=patient_data['Albumin'],
                albumin_globulin_ratio=patient_data['Albumin_and_Globulin_Ratio'],
                prediction_result=prediction_result,
                confidence=float(confidence),
                risk_level=risk_level
            )
            db.session.add(record)
            db.session.commit()
            return record.id
    except Exception as e:
        print(f"❌ Database save failed: {e}")
        return None

def load_model():
    global model, expected_features
    try:
        feature_names = joblib.load('feature_names (2).pkl')
        expected_features = feature_names
        model_info["features"] = expected_features
        model = joblib.load('Liver_disease_model.pkl')
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        expected_features = [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Protiens",
            "Albumin", "Albumin_and_Globulin_Ratio"
        ]
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=len(expected_features), random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        print("✅ Demo model created")
        return False

# Initialize
print("🚀 Starting Liver Disease Prediction System...")
init_database()
load_model()

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model is not None, model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'})

    try:
        form_data = request.form
        input_data = {}
        
        for feature in expected_features:
            value = form_data.get(feature, '').strip()
            if not value:
                return jsonify({'error': f'Please fill in {feature}'})
            
            if feature == 'Gender':
                if value.lower() in ['male', 'm', '1']:
                    input_data[feature] = 1
                elif value.lower() in ['female', 'f', '0']:
                    input_data[feature] = 0
                else:
                    return jsonify({'error': 'Please select Male or Female'})
            else:
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    return jsonify({'error': f'Invalid number for {feature}'})

        input_df = pd.DataFrame([input_data])[expected_features]
        prediction = model.predict(input_df)[0]
        
        try:
            probabilities = model.predict_proba(input_df)[0]
            confidence = max(probabilities)
            disease_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        except:
            confidence = 0.8
            disease_prob = 0.8 if prediction == 1 else 0.2

        prediction_label = 'Liver Disease' if prediction == 1 else 'Healthy'
        risk_level = 'high' if disease_prob > 0.7 else 'medium' if disease_prob > 0.3 else 'low'

        record_id = save_prediction(input_data, prediction_label, confidence, risk_level)

        result = {
            'prediction': prediction_label,
            'risk_level': risk_level,
            'confidence': f"{confidence * 100:.1f}%",
            'class_probabilities': {
                'disease': float(disease_prob),
                'healthy': float(1 - disease_prob)
            },
            'input_features': input_data,
            'record_id': record_id
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/history')
def history():
    try:
        with app.app_context():
            records = PatientRecord.query.order_by(PatientRecord.created_at.desc()).limit(10).all()
            records_list = [{
                'id': r.id,
                'age': r.age,
                'gender': r.gender,
                'prediction_result': r.prediction_result,
                'confidence': r.confidence,
                'risk_level': r.risk_level,
                'created_at': r.created_at.isoformat() if r.created_at else None
            } for r in records]
            return jsonify({'success': True, 'records': records_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/feature-ranges')
def feature_ranges():
    ranges = {
        'Age': {'normal': 'N/A', 'unit': 'years'},
        'Total_Bilirubin': {'normal': '0.1-1.2', 'unit': 'mg/dL'},
        'Direct_Bilirubin': {'normal': '0.1-0.3', 'unit': 'mg/dL'},
        'Alkaline_Phosphotase': {'normal': '44-147', 'unit': 'U/L'},
        'Alamine_Aminotransferase': {'normal': '7-56', 'unit': 'U/L'},
        'Aspartate_Aminotransferase': {'normal': '5-40', 'unit': 'U/L'},
        'Total_Protiens': {'normal': '6.0-8.3', 'unit': 'g/dL'},
        'Albumin': {'normal': '3.4-5.4', 'unit': 'g/dL'},
        'Albumin_and_Globulin_Ratio': {'normal': '1.0-2.0', 'unit': 'ratio'}
    }
    return jsonify(ranges)

if __name__ == '__main__':
    for port in [5001, 5002, 5003, 8080]:
        try:
            print(f"🚀 Server starting on http://localhost:{port}")
            app.run(debug=True, host='127.0.0.1', port=port, use_reloader=False)
            break
        except OSError:
            print(f"❌ Port {port} not available")