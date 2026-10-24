import json
from pathlib import Path

import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model():
    """Load the final pipeline and its metadata (version, threshold, feature list)."""
    info = json.loads((MODEL_DIR / "model_info.json").read_text())
    pipeline = joblib.load(MODEL_DIR / "final_model.joblib")
    return pipeline, info