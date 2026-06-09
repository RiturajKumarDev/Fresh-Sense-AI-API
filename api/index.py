from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
import json

app = Flask(__name__, static_folder="../static", static_url_path="/static")
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
            "classifier": "vegetable_classifier.pkl",
            "regressor": "vegetable_regressor.pkl",
            "encoder": "vegetable_encoder.pkl",
        }

        for name, filename in model_files.items():
            # Check in current directory and parent directory
            for path in [".", "..", base_path, "/tmp"]:
                full_path = os.path.join(path, filename)
                if os.path.exists(full_path):
                    print(f"Found {name} at: {full_path}")
                    if name == "classifier":
                        clf = joblib.load(full_path)
                    elif name == "regressor":
                        reg = joblib.load(full_path)
                    elif name == "encoder":
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


@app.route("/")
def serve_index():
    """Serve the main HTML page"""
    try:
        return send_from_directory("../static", "index.html")
    except:
        return jsonify(
            {
                "name": "FreshSense AI API",
                "version": "1.0.0",
                "status": "active",
                "models_loaded": models_loaded,
                "message": "API is running. Use /dashboard to access web interface",
            }
        )


@app.route("/dashboard")
def dashboard():
    """Alternative route for dashboard"""
    try:
        return send_from_directory("../static", "index.html")
    except:
        return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy" if models_loaded else "degraded",
            "models_loaded": models_loaded,
            "message": "API is operational" if models_loaded else "Models not loaded",
        }
    )


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if not models_loaded:
        return (
            jsonify({"error": "Models are loading. Please try again in a moment."}),
            503,
        )

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract and validate required fields
        required = [
            "vegetable_type",
            "ethanol",
            "co2",
            "temp",
            "humidity",
            "days_passed",
            "weight",
            "color_score",
        ]

        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        # Process vegetable type
        veg_type = str(data["vegetable_type"]).lower().strip()
        try:
            veg_encoded = vegetable_encoder.transform([veg_type])[0]
        except:
            return (
                jsonify(
                    {
                        "error": f"Invalid vegetable type: '{veg_type}'. Please check available types."
                    }
                ),
                400,
            )

        # Prepare features
        try:
            features = [
                [
                    float(data["ethanol"]),
                    float(data["co2"]),
                    float(data["temp"]),
                    float(data["humidity"]),
                    float(data["days_passed"]),
                    veg_encoded,
                    float(data["weight"]),
                    float(data["color_score"]),
                ]
            ]
        except ValueError as e:
            return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400

        # Make predictions
        status = clf.predict(features)[0]
        remaining_days = float(reg.predict(features)[0])

        # Calculate freshness (assuming 6 days = 100% fresh)
        freshness = max(0, min(100, round((remaining_days / 6) * 100)))

        # Determine quality level
        if freshness >= 70:
            quality = "Excellent"
            recommendation = "Consume soon for best taste"
        elif freshness >= 40:
            quality = "Good"
            recommendation = "Use within 2-3 days"
        elif freshness >= 20:
            quality = "Fair"
            recommendation = "Consume immediately"
        else:
            quality = "Poor"
            recommendation = "Not recommended for consumption"

        return jsonify(
            {
                "success": True,
                "vegetable_type": veg_type,
                "status": str(status),
                "remaining_days": round(remaining_days, 2),
                "freshness_percentage": freshness,
                "quality": quality,
                "recommendation": recommendation,
                "confidence": "High",
            }
        )

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error. Please try again."}), 500


# For local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

# Vercel requires this
app = app
