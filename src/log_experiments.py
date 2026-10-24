import hashlib
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from evaluate import best_threshold_f2, evaluate
from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-card-fraud")

MODELS = {
    "logreg_class_weight": ("LogisticRegression", "class_weight"),
    "random_forest_smote": ("RandomForest", "smote"),
    "xgboost_none": ("XGBoost", "none"),
}

val = pd.read_csv("data/processed/val.csv")
X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_hash(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


for name, (family, strategy) in MODELS.items():
    pipe = joblib.load(f"models/{name}.joblib")
    scores = pipe.predict_proba(X_val)[:, 1]
    threshold = best_threshold_f2(y_val, scores)
    metrics = evaluate(y_val, scores, threshold=threshold)

    with mlflow.start_run(run_name=name):
        mlflow.set_tags({
            "model_family": family,
            "imbalance_strategy": strategy,
            "git_commit": git_commit(),
            "train_data_hash": file_hash("data/processed/train.csv"),
            "evaluated_on": "validation",
        })
        mlflow.log_params(pipe.named_steps["model"].get_params())
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        print(f"Logged {name}: PR-AUC={metrics['pr_auc']:.4f}, threshold={threshold:.4f}")