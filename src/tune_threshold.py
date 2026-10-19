from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluate import evaluate
from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]


def candidates() -> dict:
    """Best imbalance strategy per model family, chosen from validation PR-AUC."""
    return {
        "logreg_class_weight": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest_smote": Pipeline([
            ("smote", SMOTE(sampling_strategy=0.1, random_state=42)),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=5,
                n_jobs=-1, random_state=42)),
        ]),
        "xgboost_none": Pipeline([
            ("model", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                eval_metric="aucpr", n_jobs=-1, random_state=42)),
        ]),
    }


def best_threshold_f2(y_true, scores) -> float:
    """Threshold that maximises F2 (recall weighted twice as much as precision)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    denom = 4 * precision + recall
    f2 = np.divide(5 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(f2)])


rows = []
for name, pipe in candidates().items():
    print(f"Training {name} ...")
    pipe.fit(X_train, y_train)
    scores = pipe.predict_proba(X_val)[:, 1]
    threshold = best_threshold_f2(y_val, scores)
    metrics = evaluate(y_val, scores, threshold=threshold)
    rows.append({"model": name, "threshold": threshold, **metrics})
    joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")

results = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 220)
print("\nValidation metrics at the F2-optimal threshold (optimistic: tuned on this same data)")
print(results.round(4))
