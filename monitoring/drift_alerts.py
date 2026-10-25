from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


@dataclass(frozen=True)
class DriftConfig:
    min_rows: int = 200          # below this, no verdict is given
    null_quantile: float = 0.95  # noise level used as each column's threshold
    null_draws: int = 100        # how many no-drift samples are drawn to measure noise
    share_alert: float = 0.30    # alert when this share of columns drifts
    seed: int = 0


def drift_scores(reference: pd.DataFrame, current: pd.DataFrame, columns) -> pd.Series:
    """Wasserstein distance between distributions, divided by the reference std dev."""
    scores = {}
    for col in columns:
        std = reference[col].std()
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        scores[col] = wasserstein_distance(reference[col], current[col]) / std
    return pd.Series(scores)


def null_thresholds(reference: pd.DataFrame, columns, n: int, config: DriftConfig) -> pd.Series:
    """Per-column score that pure sampling noise exceeds only (1 - quantile) of the time."""
    if n >= len(reference) // 2:
        raise ValueError("Current sample is too large for this reference to calibrate noise.")
    rng = np.random.default_rng(config.seed)
    draws = []
    for _ in range(config.null_draws):
        mask = np.zeros(len(reference), dtype=bool)
        mask[rng.choice(len(reference), size=n, replace=False)] = True
        draws.append(drift_scores(reference[~mask], reference[mask], columns))
    return pd.DataFrame(draws).quantile(config.null_quantile)


def evaluate_drift(reference: pd.DataFrame, current: pd.DataFrame, columns, config: DriftConfig) -> dict:
    n = len(current)
    base = {"rows": n, "min_rows": config.min_rows, "share_alert": config.share_alert}
    if n < config.min_rows:
        return {**base, "status": "insufficient_data", "drift_share": None, "drifted_columns": []}

    scores = drift_scores(reference, current, columns)
    thresholds = null_thresholds(reference, columns, n, config)
    drifted = scores > thresholds
    share = float(drifted.mean())
    ranked = (scores / thresholds).sort_values(ascending=False)
    return {
        **base,
        "status": "alert" if share >= config.share_alert else "ok",
        "drift_share": round(share, 3),
        "drifted_columns": [c for c in ranked.index if drifted[c]],
        "top_columns": {c: round(float(scores[c]), 3) for c in ranked.index[:5]},
    }