from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import sys

# Add the current directory to path for Vercel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Global variables to store models
clf = None
reg = None
vegetable_encoder = None
models_loaded = False


# Load models when the module is imported
def load_models():
    global clf, reg, vegetable_encoder, models_loaded
    try:
        # Try different paths for Vercel
        model_paths = [
            "vegetable_classifier.pkl",
            "/tmp/vegetable_classifier.pkl",
            "../vegetable_classifier.pkl",
            os.path.join(os.path.dirname(__file__), "vegetable_classifier.pkl"),
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "vegetable_classifier.pkl"
            ),
        ]

        reg_paths = [
            "vegetable_regressor.pkl",
            "/tmp/vegetable_regressor.pkl",
            "../vegetable_regressor.pkl",
            os.path.join(os.path.dirname(__file__), "vegetable_regressor.pkl"),
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "vegetable_regressor.pkl"
            ),
        ]

        encoder_paths = [
            "vegetable_encoder.pkl",
            "/tmp/vegetable_encoder.pkl",
            "../vegetable_encoder.pkl",
            os.path.join(os.path.dirname(__file__), "vegetable_encoder.pkl"),
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "vegetable_encoder.pkl"
            ),
        ]

        # Try to find the models
        clf_path = None
        for path in model_paths:
            if os.path.exists(path):
                clf_path = path
                break

        reg_path = None
        for path in reg_paths:
            if os.path.exists(path):
                reg_path = path
                break

        encoder_path = None
        for path in encoder_paths:
            if os.path.exists(path):
                encoder_path = path
                break

        if clf_path and reg_path and encoder_path:
            clf = joblib.load(clf_path)
            reg = joblib.load(reg_path)
            vegetable_encoder = joblib.load(encoder_path)
            models_loaded = True
            print(
                f"Models loaded successfully from {clf_path}, {reg_path}, {encoder_path}"
            )
        else:
            print(f"Could not find model files. Checked paths:")
            print(f"Classifier paths: {model_paths}")
            print(f"Regressor paths: {reg_paths}")
            print(f"Encoder paths: {encoder_paths}")

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        models_loaded = False


# Load models at startup
load_models()


@app.route("/dashboard")
def dashboard():
    return send_from_directory(".", "index.html")


# Or make it the default route
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/")
def home():
    if not models_loaded:
        return (
            jsonify(
                {
                    "message": "Vegetable Freshness API - Models not loaded",
                    "status": "error",
                    "error": "Models failed to load. Please check server logs.",
                }
            ),
            500,
        )
    return jsonify({"message": "Vegetable Freshness API is running", "status": "ok"})


@app.route("/health")
def health():
    if not models_loaded:
        return jsonify({"status": "error", "message": "Models not loaded"}), 500
    return jsonify({"status": "ok", "models_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded. Please check server logs."}), 500

    try:
        d = request.get_json()

        if not d:
            return jsonify({"error": "No JSON data provided"}), 400

        vegetable_type_raw = str(d.get("vegetable_type", "")).strip().lower()

        if not vegetable_type_raw:
            return jsonify({"error": "vegetable_type is required"}), 400

        # Check if vegetable type exists in encoder
        try:
            vegetable_type = vegetable_encoder.transform([vegetable_type_raw])[0]
        except Exception as e:
            return (
                jsonify(
                    {
                        "error": f"Unknown vegetable type: {vegetable_type_raw}. Please use a valid vegetable type."
                    }
                ),
                400,
            )

        # Extract and validate all required fields
        required_fields = [
            "ethanol",
            "co2",
            "temp",
            "humidity",
            "days_passed",
            "weight",
            "color_score",
        ]
        features_dict = {}

        for field in required_fields:
            if field not in d:
                return jsonify({"error": f"Missing field: {field}"}), 400
            try:
                features_dict[field] = float(d[field])
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid numeric value for {field}"}), 400

        features = [
            [
                features_dict["ethanol"],
                features_dict["co2"],
                features_dict["temp"],
                features_dict["humidity"],
                features_dict["days_passed"],
                vegetable_type,
                features_dict["weight"],
                features_dict["color_score"],
            ]
        ]

        status = str(clf.predict(features)[0])
        days = float(reg.predict(features)[0])

        # Ensure days is non-negative
        days = max(0, days)

        # 6 days = 100% freshness
        freshness = max(0, min(100, round((days / 6) * 100)))

        return jsonify(
            {"status": status, "remaining_days": round(days, 2), "freshness": freshness}
        )

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error during prediction"}), 500


# Vercel requires this variable
app = app
