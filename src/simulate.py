import argparse
import random
import statistics
import time

import pandas as pd
import requests

from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

API_URL = "http://127.0.0.1:8000"

parser = argparse.ArgumentParser(description="Replay SIMULATED transactions to the API.")
parser.add_argument("--n", type=int, default=100, help="number of transactions to send")
parser.add_argument("--delay", type=float, default=0.5, help="seconds between transactions")
parser.add_argument("--fraud-share", type=float, default=0.05, help="share of fraud rows (demo only)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

test = pd.read_csv("data/processed/test.csv")
fraud_rows = test.index[test[TARGET_COLUMN] == 1].tolist()
legit_rows = test.index[test[TARGET_COLUMN] == 0].tolist()
rng = random.Random(args.seed)

# Stop early with a clear message if the API is not reachable.
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
except requests.RequestException as exc:
    raise SystemExit(f"Cannot reach the API at {API_URL} ({type(exc).__name__}). Is uvicorn running?")
print(f"API health: {health['status']} (database: {health['database']})")
print(f"Sending {args.n} SIMULATED transactions (not a live bank feed)\n")

latencies, flagged, errors = [], 0, 0
for i in range(args.n):
    pool = fraud_rows if rng.random() < args.fraud_share else legit_rows
    row_id = rng.choice(pool)
    row = test.loc[row_id]
    payload = {
        "transaction_id": f"sim-{row_id}",  # row number in test.csv, used later to attach labels
        "source": "simulated",
        "features": {c: float(row[c]) for c in FEATURE_COLUMNS},
    }
    start = time.perf_counter()
    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        errors += 1
        print(f"[{i + 1}/{args.n}] request failed ({type(exc).__name__})")
        time.sleep(args.delay)
        continue
    round_trip_ms = (time.perf_counter() - start) * 1000
    result = resp.json()
    latencies.append(round_trip_ms)
    flagged += result["decision"] == "flag_for_review"
    print(
        f"[{i + 1}/{args.n}] {payload['transaction_id']:<12} "
        f"p={result['fraud_probability']:.4f}  {result['decision']:<16} "
        f"round trip {round_trip_ms:6.1f} ms"
    )
    time.sleep(args.delay)

print("\n--- Summary (simulated data) ---")
print(f"Sent OK: {len(latencies)}  Failed: {errors}  Flagged for review: {flagged}")
if latencies:
    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    print(f"Round trip ms: mean {statistics.mean(latencies):.1f}, median {statistics.median(latencies):.1f}, p95 {p95:.1f}")