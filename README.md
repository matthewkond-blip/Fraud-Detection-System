# Credit card fraud detection (XGBoost + Streamlit)

## What this project is

This is a small end-to-end **machine learning** workflow for **binary fraud classification**: you train an **XGBoost** model on historical transactions labeled as legitimate or fraud, save the trained model to disk, then run a **Streamlit** web app that scores new transaction rows from an uploaded CSV and lets you tune a **probability threshold** for flagging fraud.

The training code is aimed at the classic **Kaggle-style credit card dataset** (anonymized PCA features `V1`–`V28`, plus `Time`, `Amount`, and a `Class` label: 0 = normal, 1 = fraud). The same code can train on any CSV with the same column layout and a `Class` target column.

## How it works

1. **Training ([`train.py`](train.py))**  
   - Loads a CSV (default: [`data/creditcard.csv`](data/creditcard.csv)).  
   - Splits rows into train and test with **stratified sampling** so both sets keep a similar fraud rate.  
   - Applies **SMOTE** only on the **training** split to oversample the minority (fraud) class; the test set stays untouched so metrics reflect real imbalance.  
   - Fits a scikit-learn **Pipeline**: **StandardScaler** (zero mean, unit variance per feature) then **XGBClassifier** (gradient boosted trees).  
   - Prints **ROC-AUC**, **PR-AUC** (average precision), a **classification report**, and a **confusion matrix** on the held-out test set.  
   - Saves the **fitted pipeline** (scaler + model) to [`models/fraud_xgb.joblib`](models/fraud_xgb.joblib). **SMOTE is not saved**; at scoring time you pass real rows only, as in production.

2. **App ([`app.py`](app.py))**  
   - Loads the saved joblib model once (cached).  
   - You upload a CSV. The app requires the **same feature columns** the model was trained on (everything except `Class`; see below).  
   - It outputs each row’s **fraud probability** and a **binary prediction** based on your chosen threshold.  
   - If the upload includes a **`Class`** column, the app also shows evaluation metrics (ROC-AUC, PR-AUC, report, confusion matrix) for that file against the threshold.

3. **Schema contract**  
   The model only understands the **exact feature names and order** it learned (`feature_names_in_` on the pipeline). New data must match that schema (same columns as training, aside from optional `Class`). Different column names or a different feature set need either a **column mapping** / **retrain** on the new data—not arbitrary “similar” files.

**Deprecated entrypoint:** [`main.py`](main.py) only prints a pointer to `train.py` and exits; use `train.py` for training.

---

## Setup

1. Create a virtual environment (recommended), then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add the dataset: copy `creditcard.csv` into [`data/creditcard.csv`](data/creditcard.csv), or set `FRAUD_DATA_CSV` to your file path (see [`data/README.md`](data/README.md)).

## Train

Writes [`models/fraud_xgb.joblib`](models/fraud_xgb.joblib).

```bash
python train.py
```

## App

```bash
streamlit run app.py
```

Upload a CSV with the same feature columns as training (all columns except `Class`). If `Class` is present, the app shows evaluation metrics for that upload.

---

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `FRAUD_DATA_CSV` | `train.py` | Full path to the training CSV (overrides default `data/creditcard.csv`). |
| `FRAUD_MODEL_PATH` | `app.py` | Full path to the saved joblib model (overrides default `models/fraud_xgb.joblib`). |

On Windows PowerShell, for a single session:

```powershell
$env:FRAUD_DATA_CSV = "C:\path\to\creditcard.csv"
python train.py
```

```powershell
$env:FRAUD_MODEL_PATH = "C:\path\to\my_model.joblib"
streamlit run app.py
```
