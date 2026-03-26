from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Global variables for models
clf = None
reg = None
fruit_encoder = None
storage_encoder = None


def load_models():
    """Load models with error handling"""
    global clf, reg, fruit_encoder, storage_encoder

    try:
        # Check if model files exist
        required_files = [
            "classifier.pkl",
            "regressor.pkl",
            "fruit_encoder.pkl",
            "storage_encoder.pkl",
        ]

        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(
                    f"Model file {file} not found. Make sure it's in the correct directory."
                )

        clf = joblib.load("classifier.pkl")
        reg = joblib.load("regressor.pkl")
        fruit_encoder = joblib.load("fruit_encoder.pkl")
        storage_encoder = joblib.load("storage_encoder.pkl")

        print("✅ Models loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e


# Load models when the app starts
load_models()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render"""
    return jsonify(
        {
            "status": "healthy",
            "models_loaded": clf is not None and reg is not None,
            "service": "Fresh Sense AI",
        }
    )


@app.route("/", methods=["GET"])
def home():
    """Home endpoint"""
    return jsonify(
        {
            "message": "Hello! Welcome to Fresh Sense AI",
            "endpoints": {
                "/health": "GET - Check service health",
                "/predict": "POST - Make freshness predictions",
            },
            "status": "active",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    try:
        # Validate models are loaded
        if clf is None or reg is None:
            return jsonify({"error": "Models not loaded properly"}), 503

        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate required fields
        required_fields = [
            "fruit_type",
            "storage_type",
            "ethanol",
            "co2",
            "voc",
            "temp",
            "humidity",
            "days_passed",
            "weight",
            "color_score",
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return (
                jsonify(
                    {"error": f"Missing required fields: {', '.join(missing_fields)}"}
                ),
                400,
            )

        # Convert and validate fruit type
        fruit_type_raw = str(data.get("fruit_type", "")).strip().lower()
        if not fruit_type_raw:
            return jsonify({"error": "fruit_type cannot be empty"}), 400

        # Convert and validate storage type
        storage_type_raw = str(data.get("storage_type", "")).strip().lower()
        if not storage_type_raw:
            return jsonify({"error": "storage_type cannot be empty"}), 400

        # Transform categorical variables
        try:
            fruit_type = fruit_encoder.transform([fruit_type_raw])[0]
            storage_type = storage_encoder.transform([storage_type_raw])[0]
        except Exception as e:
            return (
                jsonify(
                    {
                        "error": f"Invalid fruit_type or storage_type: {str(e)}",
                        "valid_fruit_types": (
                            list(fruit_encoder.classes_) if fruit_encoder else None
                        ),
                        "valid_storage_types": (
                            list(storage_encoder.classes_) if storage_encoder else None
                        ),
                    }
                ),
                400,
            )

        # Convert numeric values
        try:
            features = [
                [
                    float(data["ethanol"]),
                    float(data["co2"]),
                    float(data["voc"]),
                    float(data["temp"]),
                    float(data["humidity"]),
                    float(data["days_passed"]),
                    fruit_type,
                    storage_type,
                    float(data["weight"]),
                    float(data["color_score"]),
                ]
            ]
        except ValueError as e:
            return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400

        # Make predictions
        try:
            status = str(clf.predict(features)[0])
            days = float(reg.predict(features)[0])
        except Exception as e:
            return jsonify({"error": f"Prediction error: {str(e)}"}), 500

        # Calculate freshness (6 days = 100% freshness)
        freshness = max(0, min(100, round((days / 6) * 100, 2)))

        # Return prediction
        return jsonify(
            {
                "status": status,
                "remaining_days": round(days, 2),
                "freshness": freshness,
                "prediction_details": {
                    "fruit_type": fruit_type_raw,
                    "storage_type": storage_type_raw,
                    "days_passed": float(data["days_passed"]),
                },
            }
        )

    except Exception as e:
        # Log error for debugging
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    app.run(host="0.0.0.0", port=port, debug=debug_mode)
