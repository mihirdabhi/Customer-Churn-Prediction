from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)  # 🛠 Move this line to the top before @app.route

@app.route('/')
def home():
    return "🎉 Customer Churn Prediction API is running!"

# Load the trained model
model = joblib.load("churn_model.pkl")

# Define prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Predict churn
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
