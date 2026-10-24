import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from evaluate import best_threshold_f2, evaluate
from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_NAME = "fraud-detector"
SOURCE_MODEL = Path("models/random_forest_smote.joblib")
TEST_RESULT_FILE = Path("reports/test_evaluation.json")

# Guard: the test set is for ONE evaluation of the chosen model.
if TEST_RESULT_FILE.exists():
    raise SystemExit(
        f"{TEST_RESULT_FILE} already exists. The test set has already been used. "
        "Delete that file only if you are fixing a bug, and note it in the README."
    )

val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")
X_val, y_val = val[FEATURE_COLUMNS], val[TARGET_COLUMN]
X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

pipe = joblib.load(SOURCE_MODEL)
pipe.named_steps["model"].set_params(n_jobs=1)  # fast single-row serving

# Threshold comes from VALIDATION data only
threshold = best_threshold_f2(y_val, pipe.predict_proba(X_val)[:, 1])
val_metrics = evaluate(y_val, pipe.predict_proba(X_val)[:, 1], threshold)

# The single test-set evaluation
test_metrics = evaluate(y_test, pipe.predict_proba(X_test)[:, 1], threshold)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-card-fraud")

with mlflow.start_run(run_name="final_random_forest_smote"):
    mlflow.set_tags({"stage": "final", "evaluated_on": "test (once)"})
    mlflow.log_param("threshold", threshold)
    mlflow.log_metrics({f"val_{k}": float(v) for k, v in val_metrics.items()})
    mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})
    info = mlflow.sklearn.log_model(
        pipe,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        input_example=X_val.head(5),
    )
    version = str(info.registered_model_version)

# Mark this version as the one the API should serve
MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)

# Save what the API will need
joblib.dump(pipe, "models/final_model.joblib")
Path("reports").mkdir(exist_ok=True)
Path("models/model_info.json").write_text(json.dumps({
    "model_name": MODEL_NAME,
    "model_version": version,
    "threshold": threshold,
    "features": FEATURE_COLUMNS,
}, indent=2))
TEST_RESULT_FILE.write_text(json.dumps({
    "model_version": version, "threshold": threshold,
    "validation": val_metrics, "test": test_metrics,
}, indent=2))

print(f"Registered {MODEL_NAME} version {version} with alias 'champion'")
print(f"Threshold (chosen on validation): {threshold:.4f}")
print("\nValidation:", {k: round(v, 4) for k, v in val_metrics.items()})
print("TEST      :", {k: round(v, 4) for k, v in test_metrics.items()})