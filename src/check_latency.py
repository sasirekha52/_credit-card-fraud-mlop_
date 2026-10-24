import time

import joblib
import pandas as pd

from preprocess import FEATURE_COLUMNS

val = pd.read_csv("data/processed/val.csv")
row = val[FEATURE_COLUMNS].iloc[[0]]  # one transaction, like one API request


def measure(pipe, label):
    pipe.predict_proba(row)  # warm-up call
    start = time.perf_counter()
    for _ in range(100):
        pipe.predict_proba(row)
    ms = (time.perf_counter() - start) / 100 * 1000
    print(f"{label}: {ms:.1f} ms per single prediction")


rf = joblib.load("models/random_forest_smote.joblib")
measure(rf, "random_forest_smote (n_jobs=-1, as saved)")
rf.named_steps["model"].set_params(n_jobs=1)
measure(rf, "random_forest_smote (n_jobs=1)")

xgb = joblib.load("models/xgboost_none.joblib")
measure(xgb, "xgboost_none (as saved)")
xgb.named_steps["model"].set_params(n_jobs=1)
measure(xgb, "xgboost_none (n_jobs=1)")