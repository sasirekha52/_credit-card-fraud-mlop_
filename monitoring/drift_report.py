import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))  # so "preprocess" can be imported

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sqlalchemy import text

from api.db import engine
from preprocess import FEATURE_COLUMNS

REPORT_DIR = ROOT / "reports" / "monitoring"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MIN_CURRENT_ROWS = 30

# Reference: a fixed sample of the TRAINING data (what the model learned from)
reference = pd.read_csv(ROOT / "data" / "processed" / "train.csv")[FEATURE_COLUMNS]
reference = reference.sample(n=20_000, random_state=42)

# Current: feature values the API actually received, one row per transaction
query = text(
    """
    SELECT p.transaction_id, p.created_at, f.features
    FROM predictions p JOIN prediction_features f ON f.request_id = p.request_id
    ORDER BY p.created_at
    """
)
with engine.connect() as conn:
    rows = conn.execute(query).mappings().all()

meta = pd.DataFrame({"transaction_id": [r["transaction_id"] for r in rows]})
current = pd.json_normalize([r["features"] for r in rows])
current = current.loc[~meta["transaction_id"].duplicated(keep="last").values, FEATURE_COLUMNS]
print(f"Reference rows: {len(reference)}   Current rows (unique transactions): {len(current)}")

if len(current) < MIN_CURRENT_ROWS:
    raise SystemExit(f"Need at least {MIN_CURRENT_ROWS} rows for a meaningful drift check.")

definition = DataDefinition(numerical_columns=FEATURE_COLUMNS)
reference_ds = Dataset.from_pandas(reference, data_definition=definition)
current_ds = Dataset.from_pandas(current, data_definition=definition)

# Evidently 0.7: report.run(current, reference)
snapshot = Report([DataDriftPreset()]).run(current_ds, reference_ds)

snapshot.save_html(str(REPORT_DIR / "drift_report.html"))
(REPORT_DIR / "drift_report.json").write_text(snapshot.json())
print(f"Saved reports to {REPORT_DIR}")

# Show the real structure so the alert logic can be written against it
result = snapshot.dict()
print("\nTop-level keys:", list(result.keys()))
print(json.dumps(result, indent=2, default=str)[:3500])