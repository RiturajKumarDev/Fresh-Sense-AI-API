from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
import json

app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)

# Global variables
clf = None
reg = None
vegetable_encoder = None
models_loaded = False

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_models():
    global clf, reg, vegetable_encoder, models_loaded
    try:
        # Look for models in different possible locations
        model_paths = [
            os.path.join(PROJECT_ROOT, 'vegetable_classifier.pkl'),
            os.path.join(PROJECT_ROOT, 'api', 'vegetable_classifier.pkl'),
            'vegetable_classifier.pkl',
            '/tmp/vegetable_classifier.pkl'
        ]
        
        reg_paths = [
            os.path.join(PROJECT_ROOT, 'vegetable_regressor.pkl'),
            os.path.join(PROJECT_ROOT, 'api', 'vegetable_regressor.pkl'),
            'vegetable_regressor.pkl',
            '/tmp/vegetable_regressor.pkl'
        ]
        
        encoder_paths = [
            os.path.join(PROJECT_ROOT, 'vegetable_encoder.pkl'),
            os.path.join(PROJECT_ROOT, 'api', 'vegetable_encoder.pkl'),
            'vegetable_encoder.pkl',
            '/tmp/vegetable_encoder.pkl'
        ]
        
        # Find existing files
        clf_path = next((p for p in model_paths if os.path.exists(p)), None)
        reg_path = next((p for p in reg_paths if os.path.exists(p)), None)
        encoder_path = next((p for p in encoder_paths if os.path.exists(p)), None)
        
        if clf_path and reg_path and encoder_path:
            clf = joblib.load(clf_path)
            reg = joblib.load(reg_path)
            vegetable_encoder = joblib.load(encoder_path)
            models_loaded = True
            print(f"✅ Models loaded successfully")
            print(f"Classifier: {clf_path}")
            print(f"Regressor: {reg_path}")
            print(f"Encoder: {encoder_path}")
        else:
            print("❌ Could not find model files")
            print(f"Checked paths - Classifier: {model_paths}")
            print(f"Regressor: {reg_paths}")
            print(f"Encoder: {encoder_paths}")
            
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        models_loaded = False

# Load models when the module loads
load_models()

@app.route('/')
def home():
    """Serve the HTML page"""
    try:
        return send_from_directory('../static', 'index.html')
    except:
        return jsonify({
            "message": "Vegetable Freshness API is running",
            "status": "ok" if models_loaded else "error",
            "models_loaded": models_loaded,
            "endpoints": {
                "/": "This info page",
                "/health": "Health check",
                "/predict": "POST endpoint for predictions",
                "/dashboard": "Web dashboard"
            }
        })

@app.route('/dashboard')
def dashboard():
    """Alternative route for the dashboard"""
    try:
        return send_from_directory('../static', 'index.html')
    except:
        return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded. Please check server logs."}), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = ['vegetable_type', 'ethanol', 'co2', 'temp', 'humidity', 
                          'days_passed', 'weight', 'color_score']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400
        
        # Process vegetable type
        vegetable_type = str(data['vegetable_type']).strip().lower()
        try:
            vegetable_encoded = vegetable_encoder.transform([vegetable_type])[0]
        except Exception as e:
            return jsonify({"error": f"Unknown vegetable type: {vegetable_type}"}), 400
        
        # Prepare features
        try:
            features = [[
                float(data['ethanol']),
                float(data['co2']),
                float(data['temp']),
                float(data['humidity']),
                float(data['days_passed']),
                vegetable_encoded,
                float(data['weight']),
                float(data['color_score'])
            ]]
        except ValueError as e:
            return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400
        
        # Make predictions
        status = clf.predict(features)[0]
        remaining_days = max(0, float(reg.predict(features)[0]))
        
        # Calculate freshness percentage (assuming 6 days is baseline for 100%)
        freshness = max(0, min(100, round((remaining_days / 6) * 100)))
        
        return jsonify({
            "status": str(status),
            "remaining_days": round(remaining_days, 2),
            "freshness": freshness,
            "vegetable_type": vegetable_type
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# This is required for Vercel
app = app