from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)  # ❗ This must come BEFORE route definitions

# Load the trained model
model = joblib.load("churn_model.pkl")

@app.route('/')
def home():
    return "🎉 Customer Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Drop irrelevant columns like customerID if they exist
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
