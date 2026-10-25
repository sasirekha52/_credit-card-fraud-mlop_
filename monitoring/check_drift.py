import argparse
import json
from datetime import datetime, timezone

from data_access import ROOT, FEATURE_COLUMNS, load_current, load_reference
from drift_alerts import DriftConfig, evaluate_drift

parser = argparse.ArgumentParser(description="Check logged inputs for data drift.")
parser.add_argument("--min-rows", type=int, default=200)
parser.add_argument("--share-alert", type=float, default=0.30)
parser.add_argument("--quantile", type=float, default=0.95)
args = parser.parse_args()

config = DriftConfig(min_rows=args.min_rows, share_alert=args.share_alert, null_quantile=args.quantile)
reference, current = load_reference(), load_current()
result = evaluate_drift(reference, current, FEATURE_COLUMNS, config)
result["checked_at"] = datetime.now(timezone.utc).isoformat()

out_dir = ROOT / "reports" / "monitoring"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "drift_status.json").write_text(json.dumps(result, indent=2))

print(f"Status: {result['status'].upper()}   rows checked: {result['rows']} (minimum {config.min_rows})")
if result["status"] == "insufficient_data":
    print("Not enough logged transactions for a trustworthy drift verdict. Run the simulator for more.")
else:
    print(f"Share of columns above their noise threshold: {result['drift_share']} (alert at {config.share_alert})")
    print("Most shifted columns (score):", result["top_columns"])
    print("This compares logged inputs with the TRAINING data. For simulated data, a gap largely reflects the time shift between the training and test periods.")