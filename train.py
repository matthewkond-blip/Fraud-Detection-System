import os
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "fraud_xgb.joblib"


def main() -> None:
    data_path = Path(os.environ.get("FRAUD_DATA_CSV", DEFAULT_DATA))
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Place creditcard.csv in data/ or set FRAUD_DATA_CSV."
        )

    df = pd.read_csv(data_path, encoding="latin-1")
    if "Class" not in df.columns:
        raise ValueError("CSV must include a 'Class' column for training.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC (average precision):", average_precision_score(y_test, y_proba))
    print()
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
