import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # so "api" can be imported

import pandas as pd
from sqlalchemy import select

from api.db import SessionLocal
from api.db_models import Prediction

test = pd.read_csv("data/processed/test.csv")

with SessionLocal() as session:
    # 1. Fill in labels for simulated predictions that don't have one yet.
    pending = session.scalars(
        select(Prediction).where(
            Prediction.source == "simulated", Prediction.actual_label.is_(None)
        )
    ).all()
    now = datetime.now(timezone.utc)
    attached = 0
    for p in pending:
        try:
            row_id = int(p.transaction_id.split("-")[1])
            p.actual_label = int(test.loc[row_id, "Class"])
            p.label_updated_at = now
            attached += 1
        except (AttributeError, IndexError, ValueError, KeyError):
            continue  # not a simulator ID, leave it unlabeled
    session.commit()
    print(f"Labels attached this run: {attached}")

    # 2. Measure performance on labeled simulated predictions.
    rows = session.execute(
        select(
            Prediction.transaction_id,
            Prediction.created_at,
            Prediction.decision,
            Prediction.fraud_probability,
            Prediction.actual_label,
        ).where(Prediction.source == "simulated", Prediction.actual_label.is_not(None))
    ).all()

df = pd.DataFrame(rows, columns=["transaction_id", "created_at", "decision", "probability", "actual"])
if df.empty:
    raise SystemExit("No labeled simulated predictions yet. Run src\\simulate.py first.")

# The same test row may have been sent more than once; keep the latest prediction per transaction.
df = df.sort_values("created_at").drop_duplicates("transaction_id", keep="last")
flagged = df["decision"] == "flag_for_review"
fraud = df["actual"] == 1

tp = int((flagged & fraud).sum())
fp = int((flagged & ~fraud).sum())
fn = int((~flagged & fraud).sum())
tn = int((~flagged & ~fraud).sum())

print(f"\nUnique simulated transactions with labels: {len(df)}")
print(f"Confirmed fraud among them: {int(fraud.sum())}")
print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
if tp + fn:
    print(f"Recall: {tp}/{tp + fn} = {tp / (tp + fn):.2f}")
if tp + fp:
    print(f"Precision: {tp}/{tp + fp} = {tp / (tp + fp):.2f}")
print(
    "\nCaution: the simulator over-samples fraud, and these samples are small. "
    "Do not quote these numbers as the model's real-world performance."
)