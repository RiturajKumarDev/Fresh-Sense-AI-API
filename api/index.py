from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
import sys

app = Flask(__name__, static_folder="../static", static_url_path="")
CORS(app)

# Global variables
clf = None
reg = None
vegetable_encoder = None
models_loaded = False

# Get the absolute path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_models():
    global clf, reg, vegetable_encoder, models_loaded
    try:
        # Try different locations for model files
        possible_locations = [
            PROJECT_ROOT,
            os.path.join(PROJECT_ROOT, "api"),
            os.path.join(PROJECT_ROOT, "models"),
            "/tmp",
            ".",
        ]

        for location in possible_locations:
            clf_path = os.path.join(location, "vegetable_classifier.pkl")
            reg_path = os.path.join(location, "vegetable_regressor.pkl")
            encoder_path = os.path.join(location, "vegetable_encoder.pkl")

            if all(os.path.exists(p) for p in [clf_path, reg_path, encoder_path]):
                clf = joblib.load(clf_path)
                reg = joblib.load(reg_path)
                vegetable_encoder = joblib.load(encoder_path)
                models_loaded = True
                print(f"✅ Models loaded from {location}")
                return

        print("❌ Model files not found in any location")
        print(f"Checked locations: {possible_locations}")

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        models_loaded = False


# Load models
load_models()


@app.route("/")
def home():
    """Serve the HTML page or API info"""
    try:
        return send_from_directory("../static", "index.html")
    except:
        return jsonify(
            {
                "message": "Vegetable Freshness API",
                "status": "operational" if models_loaded else "degraded",
                "models_loaded": models_loaded,
                "endpoints": {
                    "/health": "Health check endpoint",
                    "/predict": "POST endpoint for predictions",
                    "/dashboard": "Web dashboard",
                },
            }
        )


@app.route("/dashboard")
def dashboard():
    """Dashboard route"""
    try:
        return send_from_directory("../static", "index.html")
    except:
        return jsonify({"error": "Dashboard not found"}), 404


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy" if models_loaded else "unhealthy",
            "models_loaded": models_loaded,
            "python_version": sys.version,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 503

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract and validate data
        vegetable_type = str(data.get("vegetable_type", "")).strip().lower()
        if not vegetable_type:
            return jsonify({"error": "vegetable_type required"}), 400

        try:
            vegetable_encoded = vegetable_encoder.transform([vegetable_type])[0]
        except:
            return jsonify({"error": f"Unknown vegetable: {vegetable_type}"}), 400

        # Prepare features
        try:
            features = [
                [
                    float(data["ethanol"]),
                    float(data["co2"]),
                    float(data["temp"]),
                    float(data["humidity"]),
                    float(data["days_passed"]),
                    vegetable_encoded,
                    float(data["weight"]),
                    float(data["color_score"]),
                ]
            ]
        except KeyError as e:
            return jsonify({"error": f"Missing field: {str(e)}"}), 400
        except ValueError as e:
            return jsonify({"error": f"Invalid number: {str(e)}"}), 400

        # Predict
        status = str(clf.predict(features)[0])
        remaining_days = float(reg.predict(features)[0])
        remaining_days = max(0, remaining_days)

        # Calculate freshness (assuming 6 days = 100%)
        freshness = max(0, min(100, int((remaining_days / 6) * 100)))

        return jsonify(
            {
                "status": status,
                "remaining_days": round(remaining_days, 2),
                "freshness": freshness,
                "vegetable_type": vegetable_type,
            }
        )

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


# Required for Vercel
app = app
