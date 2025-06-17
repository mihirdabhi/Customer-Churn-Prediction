# 📊 Customer Churn Prediction – End-to-End Machine Learning Project

This project builds a complete machine learning pipeline to analyze and predict customer churn for a telecom company. It includes data exploration, feature engineering, model training, evaluation, and deployment preparation using Python and industry-standard tools.

---

## ✅ Key Objectives

- Understand and visualize churn trends in customer data
- Clean and preprocess raw data into model-ready format
- Train classification models (Random Forest, XGBoost)
- Evaluate performance using classification metrics
- Prepare for model deployment (API, Docker, CI/CD)
- Create insightful visualizations for business teams

---

## 🛠 Tech Stack

| Category        | Tools Used                              |
|----------------|------------------------------------------|
| Programming     | Python, Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning| scikit-learn, XGBoost                   |
| Deployment Prep | Docker, joblib                          |
| Automation      | GitHub Actions                          |
| Visualization   | Tableau (or Matplotlib/Seaborn)         |
| Storage         | CSV (simulate S3), PostgreSQL (optional) |

---

## 📁 Project Structure

```
customer-churn-prediction/
├── EDA.ipynb                   # Data cleaning + visual insights
├── 02_model_training.ipynb     # Training + evaluation notebook
├── cleaned_churn_data.csv      # Cleaned dataset ready for modeling
├── churn_model.pkl             # Trained ML model (saved)
├── requirements.txt            # Dependencies
├── Dockerfile                  # Containerization setup
├── schema.sql                  # SQL table structure
└── .github/workflows/main.yml  # CI/CD workflow
```

---

## 📊 Sample Visualizations

| Churn by Contract | Correlation Heatmap |
|-------------------|---------------------|
| ![contract](images/churn_by_contract.png) | ![corr](images/correlation_heatmap.png) |

---

## 📈 Model Metrics (Random Forest)

- **Accuracy**: 0.80
- **Precision**: 0.78
- **Recall**: 0.72
- **ROC AUC**: 0.85

---

## 💡 Business Insights

- Month-to-month contracts show higher churn compared to longer-term plans.
- Customers using fiber optics tend to churn more than DSL or no service.
- Tenure and total charges are strong indicators of churn probability.

---

## 🚀 Next Improvements

- Build a REST API to serve predictions
- Add retraining automation with Airflow or Lambda
- Deploy a Streamlit dashboard or Tableau dashboard for stakeholders

---

## 👤 Author

**Mihir Dabhi**  
📫 Email: mihirdabhi143@gmail.com  
🐦 Twitter: [@iammihirdabhi](https://twitter.com/iammihirdabhi)

---

> *Feel free to fork this project or use it as a starting point for your own machine learning pipelines.*
