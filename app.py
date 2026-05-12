import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "fraud_xgb.joblib"


@st.cache_resource
def load_model():
    path = Path(os.environ.get("FRAUD_MODEL_PATH", DEFAULT_MODEL))
    return joblib.load(path)


def main() -> None:
    st.set_page_config(page_title="Fraud scoring", layout="wide")
    st.title("Credit card fraud scoring")

    model_path = Path(os.environ.get("FRAUD_MODEL_PATH", DEFAULT_MODEL))
    if not model_path.is_file():
        st.error(
            f"Model not found at `{model_path}`. Train first: `python train.py`, "
            "or set `FRAUD_MODEL_PATH` to your joblib file."
        )
        st.stop()

    model = load_model()
    required_cols = list(model.feature_names_in_)

    uploaded = st.file_uploader("Upload CSV (same features as training)", type=["csv"])
    threshold = st.slider("Fraud probability threshold", 0.0, 1.0, 0.5, 0.01)

    if uploaded is None:
        st.info("Upload a CSV to score transactions.")
        return

    df = pd.read_csv(uploaded, encoding="latin-1")
    has_class = "Class" in df.columns

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing[:10]}{'…' if len(missing) > 10 else ''}")
        st.stop()

    X = df[required_cols]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["fraud_probability"] = proba
    out["predicted_fraud"] = pred

    st.subheader("Results")
    st.dataframe(out, use_container_width=True)

    fraud_count = int(pred.sum())
    st.metric("Flagged as fraud (at current threshold)", fraud_count)

    if has_class:
        y_true = df["Class"].astype(int).values
        st.subheader("Evaluation (Class column present)")
        st.metric("ROC-AUC", f"{roc_auc_score(y_true, proba):.4f}")
        st.metric("PR-AUC", f"{average_precision_score(y_true, proba):.4f}")
        st.text(classification_report(y_true, pred, digits=4))
        cm = confusion_matrix(y_true, pred)
        st.write("Confusion matrix (rows=true, cols=pred):")
        st.dataframe(cm, use_container_width=False)


if __name__ == "__main__":
    main()
