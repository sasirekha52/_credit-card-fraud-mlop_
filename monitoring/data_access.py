import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

import pandas as pd
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