from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
import sys
import traceback

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
        # Find model files in multiple locations
        search_paths = [
            PROJECT_ROOT,
            os.path.join(PROJECT_ROOT, "api"),
            os.path.join(PROJECT_ROOT, "models"),
            "/tmp",
            ".",
            os.getcwd(),
        ]

        for location in search_paths:
            clf_path = os.path.join(location, "vegetable_classifier.pkl")
            reg_path = os.path.join(location, "vegetable_regressor.pkl")
            encoder_path = os.path.join(location, "vegetable_encoder.pkl")

            if (
                os.path.exists(clf_path)
                and os.path.exists(reg_path)
                and os.path.exists(encoder_path)
            ):
                print(f"Found models in: {location}")
                clf = joblib.load(clf_path)
                reg = joblib.load(reg_path)
                vegetable_encoder = joblib.load(encoder_path)
                models_loaded = True
                print("✅ Models loaded successfully")
                return

        print("❌ Model files not found. Checked paths:")
        for loc in search_paths:
            print(f"  - {loc}")

        # List files in root directory for debugging
        print("\nFiles in root directory:")
        for file in os.listdir(PROJECT_ROOT):
            if file.endswith(".pkl"):
                print(f"  - {file}")

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        traceback.print_exc()
        models_loaded = False


# Load models on startup
load_models()


@app.route("/")
@app.route("/dashboard")
def home():
    """Serve the HTML dashboard"""
    try:
        # Try to serve from static folder
        return send_from_directory("../static", "index.html")
    except Exception as e:
        # If static file not found, return API info
        return jsonify(
            {
                "message": "Vegetable Freshness Prediction API",
                "version": "1.0.0",
                "status": "operational" if models_loaded else "degraded",
                "models_loaded": models_loaded,
                "endpoints": {
                    "GET /": "API information",
                    "GET /health": "Health check",
                    "POST /predict": "Make predictions",
                    "GET /dashboard": "Web interface",
                },
                "example_request": {
                    "vegetable_type": "tomato",
                    "ethanol": 0.5,
                    "co2": 400,
                    "temp": 22,
                    "humidity": 65,
                    "days_passed": 2,
                    "weight": 150,
                    "color_score": 8,
                },
            }
        )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy" if models_loaded else "unhealthy",
            "models_loaded": models_loaded,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
    )


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded. Please check logs."}), 503

    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate required fields
        required_fields = [
            "vegetable_type",
            "ethanol",
            "co2",
            "temp",
            "humidity",
            "days_passed",
            "weight",
            "color_score",
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        # Process vegetable type
        veg_type = str(data["vegetable_type"]).strip().lower()
        if not veg_type:
            return jsonify({"error": "vegetable_type cannot be empty"}), 400

        try:
            veg_encoded = vegetable_encoder.transform([veg_type])[0]
        except Exception as e:
            return (
                jsonify(
                    {
                        "error": f"Unknown vegetable type: '{veg_type}'. Please check available types."
                    }
                ),
                400,
            )

        # Build features array
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
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid numeric value: {str(e)}"}), 400

        # Make predictions
        status = clf.predict(features)[0]
        remaining_days = reg.predict(features)[0]
        remaining_days = max(0, float(remaining_days))

        # Calculate freshness percentage (6 days baseline)
        freshness = max(0, min(100, round((remaining_days / 6) * 100)))

        return jsonify(
            {
                "status": str(status),
                "remaining_days": round(remaining_days, 2),
                "freshness": freshness,
                "vegetable_type": veg_type,
                "confidence": "high",  # You can add actual confidence scores if available
            }
        )

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# For local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# This is required for Vercel
app = app
