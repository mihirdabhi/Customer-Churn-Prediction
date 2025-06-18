# 📊 Customer Churn Prediction & Business Insights Platform – Full-Stack ML Project

This project builds a complete **end-to-end machine learning pipeline** to analyze and predict customer churn for a telecom company. It includes data collection, ETL workflows using AWS, model development, cloud-based deployment, and interactive dashboards for insights.

---

## ✅ Key Objectives

- Automate churn data ingestion and transformation (AWS Lambda + RDS)
- Train and evaluate ML models (Random Forest, XGBoost)
- Serve real-time predictions via Flask API (hosted on Render)
- Containerize app with Docker
- Automate CI/CD with GitHub Actions
- Visualize insights with Tableau (or AWS QuickSight)

---

## 🛠 Tech Stack

| Category        | Tools Used                              |
|----------------|------------------------------------------|
| Programming     | Python, Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning| scikit-learn, XGBoost                   |
| Deployment Prep | Flask, Docker, Render, joblib           |
| Automation      | GitHub Actions, AWS Lambda              |
| Visualization   | Tableau, SQL (RDS)                      |
| Storage         | AWS S3, PostgreSQL (RDS), CSV           |

---

## 📁 Project Structure

```
customer-churn-prediction/
├── etl_lambda/
│   ├── etl_lambda_s3_to_s3.py
│   ├── etl_lambda_s3_to_rds.py
│   └── requirements.txt
├── model/
│   ├── train_model.ipynb
│   └── churn_model.pkl
├── app/
│   └── app.py
├── dashboard/
│   └── tableau_screenshots/
├── Dockerfile
├── .github/workflows/
│   └── ci.yml
├── schema.sql
├── requirements.txt
└── README.md
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

- ✅ Flask API already deployed with Render
- 🔄 Scheduled retraining via Lambda or Airflow (in progress)
- 📊 Tableau dashboard + SQL-based KPIs from RDS
- 📦 Add MLflow tracking for experiments (optional)
- 🐳 Push Docker image to AWS ECR

---

## 👤 Author

**Mihir Dabhi**  
📫 Email: mihirdabhi143@gmail.com  
🐦 Twitter: [@iammihirdabhi](https://twitter.com/iammihirdabhi)

---

> *This full-stack ML project is designed to simulate real-world churn management and help recruiters see your data + engineering + product mindset. Fork it or extend it!*
