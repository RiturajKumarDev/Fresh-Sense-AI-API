from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import json

app = Flask(__name__)
CORS(app)

# Global variables
clf = None
reg = None
vegetable_encoder = None
models_loaded = False

def load_models():
    global clf, reg, vegetable_encoder, models_loaded
    try:
        # Try multiple paths for model files
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        model_files = {
            'classifier': 'vegetable_classifier.pkl',
            'regressor': 'vegetable_regressor.pkl',
            'encoder': 'vegetable_encoder.pkl'
        }
        
        for name, filename in model_files.items():
            # Check in current directory and parent directory
            for path in ['.', '..', base_path, '/tmp']:
                full_path = os.path.join(path, filename)
                if os.path.exists(full_path):
                    print(f"Found {name} at: {full_path}")
                    if name == 'classifier':
                        clf = joblib.load(full_path)
                    elif name == 'regressor':
                        reg = joblib.load(full_path)
                    elif name == 'encoder':
                        vegetable_encoder = joblib.load(full_path)
                    break
        
        if clf is not None and reg is not None and vegetable_encoder is not None:
            models_loaded = True
            print("✅ All models loaded successfully")
        else:
            print("❌ Could not load all models")
            
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        models_loaded = False

# Load models
load_models()

@app.route('/')
def home():
    return jsonify({
        "name": "FreshSense AI API",
        "version": "1.0.0",
        "status": "active",
        "models_loaded": models_loaded,
        "endpoints": {
            "GET /": "API Information",
            "GET /health": "Health Check",
            "POST /predict": "Make Predictions"
        },
        "example_post_request": {
            "vegetable_type": "tomato",
            "ethanol": 0.5,
            "co2": 400,
            "temp": 22,
            "humidity": 65,
            "days_passed": 2,
            "weight": 150,
            "color_score": 8
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "environment": "vercel"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({
            "error": "Models not loaded. Please ensure model files are present."
        }), 503
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract and validate required fields
        required = ['vegetable_type', 'ethanol', 'co2', 'temp', 'humidity', 
                   'days_passed', 'weight', 'color_score']
        
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        
        # Process vegetable type
        veg_type = str(data['vegetable_type']).lower().strip()
        try:
            veg_encoded = vegetable_encoder.transform([veg_type])[0]
        except:
            return jsonify({"error": f"Invalid vegetable type: {veg_type}"}), 400
        
        # Prepare features
        features = [[
            float(data['ethanol']),
            float(data['co2']),
            float(data['temp']),
            float(data['humidity']),
            float(data['days_passed']),
            veg_encoded,
            float(data['weight']),
            float(data['color_score'])
        ]]
        
        # Make predictions
        status = clf.predict(features)[0]
        remaining_days = float(reg.predict(features)[0])
        
        # Calculate freshness (assuming 6 days = 100% fresh)
        freshness = max(0, min(100, round((remaining_days / 6) * 100)))
        
        return jsonify({
            "success": True,
            "vegetable_type": veg_type,
            "status": str(status),
            "remaining_days": round(remaining_days, 2),
            "freshness_percentage": freshness
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel requires this
app = app