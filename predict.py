# predict.py
import argparse, json
import pandas as pd
import joblib
from pathlib import Path


def load_model(path: str):
    return joblib.load(path)


def predict_from_csv(model_path: str, input_csv: str, out_csv: str):
    model = load_model(model_path)
    df = pd.read_csv(input_csv)
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = pred
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] Predictions saved to {out_csv}")


def predict_from_json(model_path: str, json_path_or_str: str):
    model = load_model(model_path)
    # support both a path to JSON or a JSON string
    try:
        if Path(json_path_or_str).exists():
            payload = json.loads(Path(json_path_or_str).read_text())
        else:
            payload = json.loads(json_path_or_str)
    except Exception as e:
        raise ValueError("Provide a valid JSON path or JSON string.") from e

    df = pd.DataFrame([payload])
    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)
    print(json.dumps({"churn_proba": float(proba), "churn_pred": pred}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict churn from CSV or JSON")
    ap.add_argument("--model", default="models/churn_model.joblib")
    ap.add_argument("--csv", help="Path to CSV for batch prediction")
    ap.add_argument("--out", default="predictions.csv", help="Output CSV for batch prediction")
    ap.add_argument("--json", help="Path to JSON file or raw JSON string for single prediction")
    args = ap.parse_args()

    if args.csv:
        predict_from_csv(args.model, args.csv, args.out)
    elif args.json:
        predict_from_json(args.model, args.json)
    else:
        print("Provide --csv <file> or --json <file_or_string>.")
