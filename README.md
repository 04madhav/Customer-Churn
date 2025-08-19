# 📊 Customer Churn Prediction

A machine learning project to predict whether a customer is likely to churn.

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
