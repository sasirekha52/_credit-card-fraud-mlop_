import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

THRESHOLD = 0.1
N = 92
REPEATS = 30
rng = np.random.default_rng(0)

train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
reference = train.sample(n=20_000, random_state=42)[FEATURE_COLUMNS]
ref_std = reference.std().replace(0, 1)


def distances(sample: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {c: wasserstein_distance(reference[c], sample[c]) / ref_std[c] for c in FEATURE_COLUMNS}
    )


def run(name: str, make_sample) -> None:
    shares, v3 = [], []
    for _ in range(REPEATS):
        d = distances(make_sample())
        shares.append((d >= THRESHOLD).mean())
        v3.append(d["V3"])
    print(f"{name:<44} median share of columns flagged: {np.median(shares):.2f}   median V3 score: {np.median(v3):.2f}")


train_other = train.drop(reference.index, errors="ignore")
test_legit = test[test[TARGET_COLUMN] == 0]
test_fraud = test[test[TARGET_COLUMN] == 1]

print(f"{N} rows per sample, {REPEATS} repeats, threshold {THRESHOLD}\n")
run("A: more TRAINING rows (no real drift)", lambda: train_other.sample(n=N, random_state=int(rng.integers(1e9)))[FEATURE_COLUMNS])
run("B: TEST-period legit rows only", lambda: test_legit.sample(n=N, random_state=int(rng.integers(1e9)))[FEATURE_COLUMNS])
run(
    "C: TEST-period rows, ~5% fraud (like simulator)",
    lambda: pd.concat([test_legit.sample(n=N - 5, random_state=int(rng.integers(1e9))),
                       test_fraud.sample(n=5, random_state=int(rng.integers(1e9)))])[FEATURE_COLUMNS],
)