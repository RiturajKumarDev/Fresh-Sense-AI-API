from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Models load at startup
clf = joblib.load("classifier.pkl")
reg = joblib.load("regressor.pkl")
fruit_encoder = joblib.load("fruit_encoder.pkl")
storage_encoder = joblib.load("storage_encoder.pkl")


@app.route("/")
def home():
    return jsonify({"message": "Fruit Freshness API is running"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = request.get_json()

        if not d:
            return jsonify({"error": "No JSON data provided"}), 400

        fruit_type_raw = str(d.get("fruit_type", "")).strip().lower()
        storage_type_raw = str(d.get("storage_type", "")).strip().lower()

        if not fruit_type_raw or not storage_type_raw:
            return jsonify({"error": "fruit_type and storage_type are required"}), 400

        fruit_type = fruit_encoder.transform([fruit_type_raw])[0]
        storage_type = storage_encoder.transform([storage_type_raw])[0]

        features = [
            [
                float(d["ethanol"]),
                float(d["co2"]),
                float(d["voc"]),
                float(d["temp"]),
                float(d["humidity"]),
                float(d["days_passed"]),
                fruit_type,
                storage_type,
                float(d["weight"]),
                float(d["color_score"]),
            ]
        ]

        status = str(clf.predict(features)[0])
        days = float(reg.predict(features)[0])

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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
