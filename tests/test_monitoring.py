import numpy as np
import pandas as pd
import pytest

from monitoring.drift_alerts import DriftConfig, drift_scores, evaluate_drift

COLUMNS = [f"x{i}" for i in range(10)]
CONFIG = DriftConfig(min_rows=100, null_draws=50, share_alert=0.5)


def make(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(loc=shift, scale=1.0, size=(n, len(COLUMNS))), columns=COLUMNS)


def test_identical_distributions_score_low():
    scores = drift_scores(make(3000, seed=1), make(3000, seed=2), COLUMNS)
    assert (scores < 0.1).all()


def test_shifted_distribution_scores_high():
    scores = drift_scores(make(3000, seed=1), make(500, shift=1.0, seed=3), COLUMNS)
    assert (scores > 0.8).all()


def test_too_few_rows_gives_no_verdict():
    result = evaluate_drift(make(3000, seed=1), make(50, seed=4), COLUMNS, CONFIG)
    assert result["status"] == "insufficient_data"
    assert result["drift_share"] is None


def test_no_drift_is_ok():
    result = evaluate_drift(make(3000, seed=1), make(300, seed=5), COLUMNS, CONFIG)
    assert result["status"] == "ok"


def test_real_drift_raises_alert():
    result = evaluate_drift(make(3000, seed=1), make(300, shift=1.0, seed=6), COLUMNS, CONFIG)
    assert result["status"] == "alert"
    assert len(result["drifted_columns"]) == len(COLUMNS)


def test_alert_threshold_is_configurable():
    lenient = DriftConfig(min_rows=100, null_draws=50, share_alert=1.1)  # share can never reach 1.1
    result = evaluate_drift(make(3000, seed=1), make(300, shift=1.0, seed=6), COLUMNS, lenient)
    assert result["status"] == "ok"


def test_reference_too_small_to_calibrate_raises():
    with pytest.raises(ValueError):
        evaluate_drift(make(100, seed=1), make(60, seed=2), COLUMNS, DriftConfig(min_rows=10))