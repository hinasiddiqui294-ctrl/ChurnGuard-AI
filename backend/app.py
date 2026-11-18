
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Find project folders
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "backend" / "churn_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

# Get the exact input columns used during training
FEATURES = list(
    model.named_steps["preprocessor"].feature_names_in_
)

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to ChurnGuard AI API",
        "group": "ChurnX AI",
        "status": "Backend running successfully"
    })

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "Random Forest",
        "features": len(FEATURES)
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Please send customer data"}), 400

        # Check required customer fields
        missing = [col for col in FEATURES if col not in data]

        if missing:
            return jsonify({
                "error": "Missing customer fields",
                "missing_fields": missing
            }), 400

        # Arrange input in training column order
        customer = pd.DataFrame(
            [[data[col] for col in FEATURES]],
            columns=FEATURES
        )

        # Make prediction
        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])

        result = "Churn" if prediction == 1 else "No Churn"

        return jsonify({
            "prediction": result,
            "churn_probability": round(probability * 100, 2),
            "message": (
                "Customer is at risk of leaving"
                if prediction == 1
                else "Customer is likely to stay"
            )
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
