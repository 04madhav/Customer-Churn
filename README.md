# 📊 Customer Churn Prediction

A machine learning model to predict customer churn using customer data, achieving ~80% accuracy. Developed a Streamlit app for real-time churn probability predictions to support data-driven retention strategies.

## 🚀 Features
- Preprocessing pipeline (numeric + categorical)
- Logistic Regression (baseline) – easy to swap with RandomForest/XGBoost
- Streamlit app for CSV upload and predictions
- Batch + single JSON prediction support

## 📂 Project Structure
customer-churn/
│── data/ # dataset (local, not on GitHub)
│ └── template_for_predictions.csv
│── models/ # trained models, confusion matrix, metadata
│── train.py # train the churn model
│── predict.py # batch/JSON predictions
│── app.py # Streamlit app
│── requirements.txt
│── README.md


## ⚡ How to Run
```bash
# install dependencies
pip install -r requirements.txt

# train model
python train.py --data data/telco.csv

# run streamlit app
streamlit run app.py
