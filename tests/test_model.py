import pandas as pd

from api.model_loader import load_model


def test_model_loads_with_metadata():
    pipeline, info = load_model()
    assert hasattr(pipeline, "predict_proba")
    assert {"model_name", "model_version", "threshold", "features"} <= set(info)
    assert len(info["features"]) == 29


def test_probability_is_between_zero_and_one():
    pipeline, info = load_model()
    row = pd.DataFrame([{c: 0.0 for c in info["features"]}])
    probability = pipeline.predict_proba(row)[0, 1]
    assert 0.0 <= probability <= 1.0