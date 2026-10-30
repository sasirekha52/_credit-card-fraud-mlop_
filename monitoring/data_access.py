import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import pandas as pd
import joblib
from sqlalchemy import text

from api.db import engine
from preprocess import FEATURE_COLUMNS


def load_reference(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """A fixed sample of the TRAINING features: what the model learned from."""
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    return train[FEATURE_COLUMNS].sample(n=n, random_state=seed)


def load_current() -> pd.DataFrame:
    """Logged feature values from PostgreSQL, one row per transaction_id."""
    query = text(
        """
        SELECT p.transaction_id, f.features
        FROM predictions p JOIN prediction_features f ON f.request_id = p.request_id
        ORDER BY p.created_at
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    ids = pd.Series([r["transaction_id"] for r in rows])
    frame = pd.json_normalize([r["features"] for r in rows])
    repeated = (ids.duplicated(keep="last") & ids.notna()).values  # keep latest copy only
    return frame.loc[~repeated, FEATURE_COLUMNS].reset_index(drop=True)

def load_validation_scores() -> pd.DataFrame:
    """Model scores on the validation set: unseen data, so a fair reference for live scores."""
    model = joblib.load(ROOT / "models" / "final_model.joblib")
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    return pd.DataFrame({"fraud_probability": model.predict_proba(val[FEATURE_COLUMNS])[:, 1]})


def load_current_scores() -> pd.DataFrame:
    """Logged fraud probabilities, one row per transaction_id (latest copy)."""
    query = text("SELECT transaction_id, fraud_probability FROM predictions ORDER BY created_at")
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    if not rows:
        return pd.DataFrame(columns=["fraud_probability"])
    ids = pd.Series([r["transaction_id"] for r in rows])
    scores = pd.Series([r["fraud_probability"] for r in rows], dtype=float)
    repeated = (ids.duplicated(keep="last") & ids.notna()).values
    return pd.DataFrame({"fraud_probability": scores[~repeated].reset_index(drop=True)})