import mlflow
import mlflow.sklearn
import pandas as pd

from preprocess import FEATURE_COLUMNS

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/fraud-detector@champion")

row = pd.read_csv("data/processed/test.csv")[FEATURE_COLUMNS].iloc[[0]]
print("Fraud probability:", round(float(model.predict_proba(row)[0, 1]), 6))