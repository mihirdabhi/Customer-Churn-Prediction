from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load model and get required feature names
model = joblib.load("churn_model.pkl")
required_features = list(model.feature_names_in_)  # Ensures input feature alignment

@app.route('/')
def home():
    return "🎉 Customer Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Drop irrelevant columns
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        # Reorder columns to match training and fill any missing with 0
        df = df.reindex(columns=required_features, fill_value=0)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 3)
        })

    except Exception as e:
        error_msg = traceback.format_exc()
        print(" ERROR during prediction:\n", error_msg)
        return jsonify({"error": error_msg})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
