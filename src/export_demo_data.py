"""Export a small sample of dashboard data for the Streamlit Cloud demo."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
from sqlalchemy import text

from api.db import engine

OUT = ROOT / "dashboard" / "demo_data"
OUT.mkdir(parents=True, exist_ok=True)

query = text("""
    SELECT request_id, transaction_id, created_at, fraud_probability, threshold,
           decision, risk_level, model_version, latency_ms, amount, source, actual_label
    FROM predictions ORDER BY created_at DESC LIMIT 5000
""")
with engine.connect() as conn:
    df = pd.DataFrame(conn.execute(query).mappings().all())

labeled = df[df["actual_label"].notna()]
recent = df.drop(labeled.index).head(max(0, 2000 - len(labeled)))
sample = pd.concat([labeled, recent]).sort_values("created_at", ascending=False)
sample.to_csv(OUT / "predictions.csv", index=False)

files = [
    (ROOT / "models" / "model_info.json", "model_info.json"),
    (ROOT / "reports" / "test_evaluation.json", "test_evaluation.json"),
    (ROOT / "reports" / "monitoring" / "drift_status.json", "drift_status.json"),
]
for src, name in files:
    if src.exists():
        shutil.copy(src, OUT / name)
    else:
        print("MISSING:", src)

if not (OUT / "model_info.json").exists():
    sys.exit("model_info.json is required. Fix the MISSING file above and rerun.")
info = json.loads((OUT / "model_info.json").read_text())
(OUT / "health.json").write_text(json.dumps({
    "status": "snapshot", "model_name": info["model_name"],
    "model_version": info["model_version"], "model_loaded": True, "database": "snapshot"}, indent=2))
(OUT / "metrics.json").write_text(json.dumps(
    {"p95_latency_ms": round(float(sample["latency_ms"].quantile(0.95)), 1)}, indent=2))
print(len(sample), "rows exported to", OUT)