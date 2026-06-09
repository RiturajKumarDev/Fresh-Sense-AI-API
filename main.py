from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Models load at startup with error handling
try:
    clf = joblib.load("vegetable_classifier.pkl")
    reg = joblib.load("vegetable_regressor.pkl")
    vegetable_encoder = joblib.load("vegetable_encoder.pkl")
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    models_loaded = False


@app.route("/")
def home():
    if not models_loaded:
        return (
            jsonify(
                {
                    "message": "Vegetable Freshness API - Models not loaded",
                    "status": "error",
                }
            ),
            500,
        )
    return jsonify({"message": "Vegetable Freshness API is running", "status": "ok"})


@app.route("/health")
def health():
    if not models_loaded:
        return jsonify({"status": "error", "message": "Models not loaded"}), 500
    return jsonify({"status": "ok"})


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
        print(f"Prediction error: {str(e)}")  # Log error for debugging
        return jsonify({"error": "Internal server error during prediction"}), 500


# For local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# For Vercel serverless deployment
app.debug = False
