from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Define app FIRST
app = Flask(__name__)

# Load model
model = joblib.load("churn_model.pkl")

# Home route
@app.route('/')
def home():
    return "🎉 Customer Churn Prediction API is running!"

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
