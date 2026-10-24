import json
from pathlib import Path

import pandas as pd

from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

test = pd.read_csv("data/processed/test.csv")
out = Path("api/sample_requests")
out.mkdir(parents=True, exist_ok=True)

for name, label in [("legit", 0), ("fraud", 1)]:
    row = test[test[TARGET_COLUMN] == label].iloc[0]
    payload = {
        "transaction_id": f"sample-{name}",
        "features": {c: float(row[c]) for c in FEATURE_COLUMNS},
    }
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print("wrote", out / f"{name}.json")