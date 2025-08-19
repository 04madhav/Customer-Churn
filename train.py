# train.py
import argparse, json, os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt


RANDOM_STATE = 42


def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)


def load_or_generate_dataset(data_path: str) -> pd.DataFrame:
    """
    Load Telco dataset if available; otherwise generate a synthetic telco-like dataset.
    """
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"[INFO] Loaded dataset from {data_path} with shape {df.shape}")
        return df

    # Synthetic fallback (minimal but realistic)
    n = 6000
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "customerID": [f"C{100000+i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.85, 0.15]),
        "Partner": rng.choice(["Yes", "No"], n, p=[0.45, 0.55]),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.35, 0.65]),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n, p=[0.45, 0.45, 0.10]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.3, 0.6, 0.1]),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n, p=[0.35, 0.55, 0.10]),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n, p=[0.4, 0.5, 0.10]),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n, p=[0.4, 0.5, 0.10]),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n, p=[0.35, 0.55, 0.10]),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n, p=[0.45, 0.45, 0.10]),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n, p=[0.45, 0.45, 0.10]),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.25, 0.15]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.65, 0.35]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.4, 0.25, 0.2, 0.15]
        ),
    })
    # Monetary signals
    base = rng.normal(75, 30, n).clip(18, 150)
    df["MonthlyCharges"] = base.round(2)
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"] * rng.uniform(0.9, 1.1, n)).round(2)

    # Generate churn with logic: shorter tenure, month-to-month, higher monthly charges => more churn
    logits = (
        -2.0
        + 0.02 * (df["MonthlyCharges"] - 70)
        - 0.03 * df["tenure"]
        + df["Contract"].map({"Month-to-month": 1.2, "One year": -0.7, "Two year": -1.0})
    )
    probs = 1 / (1 + np.exp(-logits))
    df["Churn"] = np.where(rng.random(n) < probs, "Yes", "No")

    synth_path = "data/synthetic_telco.csv"
    df.to_csv(synth_path, index=False)
    print(f"[WARN] {data_path} not found. Generated synthetic dataset at {synth_path} with shape {df.shape}")
    return df


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix TotalCharges type (convert blanks to NaN, then numeric)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fill NaN with 0 (or mean, but 0 makes sense for tenure=0 customers)
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop obvious non-features
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df



def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found.")
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


def build_pipeline(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model = Pipeline(steps=[("preprocess", pre), ("clf", clf)])
    return model


def plot_and_save_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # write numbers
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(args):
    ensure_dirs()
    df = load_or_generate_dataset(args.data)
    df = clean_telco(df)
    X, y = split_xy(df)

    cat_cols, num_cols = get_column_types(X)
    print(f"[INFO] Categorical: {len(cat_cols)} | Numerical: {len(num_cols)}")
    print(f"[INFO] Sample categorical columns: {cat_cols[:5]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = build_pipeline(cat_cols, num_cols)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, digits=4)

    print("\n=== Classification Report ===")
    print(report)
    print("ROC-AUC:", round(auc, 4))
    print("Confusion Matrix:\n", cm)

    # Save artifacts
    joblib.dump(model, args.model)
    print(f"[OK] Model saved to {args.model}")

    meta = {
        "features": X.columns.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "target": "Churn",
        "auc": float(auc),
    }
    with open("models/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[OK] Meta saved to models/meta.json")

    # Save template CSV for inference
    template = pd.DataFrame(columns=X.columns.tolist())
    template.to_csv("data/template_for_predictions.csv", index=False)
    print("[OK] Template saved to data/template_for_predictions.csv")

    # Save confusion matrix plot
    plot_and_save_confusion_matrix(cm, ["No", "Yes"], "models/confusion_matrix.png")
    print("[OK] Confusion matrix plot saved to models/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--data", type=str, default="data/telco.csv", help="Path to CSV dataset")
    parser.add_argument("--model", type=str, default="models/churn_model.joblib", help="Output model path")
    main(parser.parse_args())
