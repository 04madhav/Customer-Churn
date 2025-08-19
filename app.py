# app.py
import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.joblib")

st.title("📊 Customer Churn Predictor")
st.write("Upload a CSV with the same columns as the training features to get churn predictions.")

# Download template
tpl_path = Path("data/template_for_predictions.csv")
if tpl_path.exists():
    st.download_button("Download CSV Template", tpl_path.read_bytes(), file_name="template_for_predictions.csv")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
        
    st.write("Preview:", df.head())

    try:
        model = load_model()
    except Exception as e:
        st.error("Model not found. Train first: `python train.py`")
        st.stop()

    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred

    st.subheader("Results")
    st.write(out.head())

    # Basic summary
    churn_rate = pred.mean()
    st.metric("Predicted Churn Rate", f"{churn_rate:.1%}")

    # Download results
    buf = BytesIO()
    out.to_csv(buf, index=False)
    st.download_button("Download Predictions", buf.getvalue(), file_name="predictions.csv", mime="text/csv")
