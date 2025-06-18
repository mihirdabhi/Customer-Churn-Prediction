from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load model and required features
model = joblib.load("churn_model.pkl")
required_features = list(model.feature_names_in_)

@app.route('/')
def home():
    return "🎉 Customer Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Drop irrelevant column
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        # Define encoding mappings for categorical features
        mappings = {
            'gender': {'Female': 0, 'Male': 1},
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'PhoneService': {'Yes': 1, 'No': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0},
            'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
            'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
            'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'PaymentMethod': {
                'Electronic check': 0,
                'Mailed check': 1,
                'Bank transfer (automatic)': 2,
                'Credit card (automatic)': 3
            }
        }

        # Apply the encodings
        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Align features and fill missing with 0
        df = df.reindex(columns=required_features, fill_value=0)

        # Predict
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 3)
        })

    except Exception as e:
        error_msg = traceback.format_exc()
        print("ERROR during prediction:\n", error_msg)
        return jsonify({"error": error_msg})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
