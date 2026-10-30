import argparse
import json
from datetime import datetime, timezone

from data_access import (
    FEATURE_COLUMNS, ROOT, load_current, load_current_scores,
    load_reference, load_validation_scores,
)
from drift_alerts import DriftConfig, evaluate_drift

parser = argparse.ArgumentParser(description="Check logged inputs and scores for drift.")
parser.add_argument("--min-rows", type=int, default=200)
parser.add_argument("--share-alert", type=float, default=0.30)
parser.add_argument("--quantile", type=float, default=0.95)
args = parser.parse_args()

config = DriftConfig(min_rows=args.min_rows, share_alert=args.share_alert, null_quantile=args.quantile)

features = evaluate_drift(load_reference(), load_current(), FEATURE_COLUMNS, config)

ref_scores, cur_scores = load_validation_scores(), load_current_scores()
scores = evaluate_drift(ref_scores, cur_scores, ["fraud_probability"], config)

model_info = json.loads((ROOT / "models" / "model_info.json").read_text())
threshold = model_info["threshold"]
scores["flag_rate_reference"] = round(float((ref_scores["fraud_probability"] >= threshold).mean()), 4)
scores["flag_rate_current"] = (
    round(float((cur_scores["fraud_probability"] >= threshold).mean()), 4) if len(cur_scores) else None
)

result = {"checked_at": datetime.now(timezone.utc).isoformat(), "features": features, "scores": scores}
out_dir = ROOT / "reports" / "monitoring"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "drift_status.json").write_text(json.dumps(result, indent=2))

print(f"INPUT FEATURES  status: {features['status'].upper()}   rows: {features['rows']} (minimum {config.min_rows})")
if features["status"] != "insufficient_data":
    print(f"  share of columns above noise threshold: {features['drift_share']} (alert at {config.share_alert})")
    print(f"  most shifted columns: {features['top_columns']}")
print(f"PREDICTION SCORES  status: {scores['status'].upper()}   rows: {scores['rows']}")
print(f"  flag rate: reference (validation) {scores['flag_rate_reference']}  vs  current {scores['flag_rate_current']}")
print("Reference for features = training data; for scores = validation data. Simulated data over-samples fraud, so a higher flag rate is expected.")