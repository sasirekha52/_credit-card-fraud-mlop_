import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # so "api" can be imported

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from api.db import engine

API_URL = "http://127.0.0.1:8000"
THRESHOLD_FILE = ROOT / "models" / "model_info.json"

st.set_page_config(page_title="Fraud Risk Monitor", layout="wide")


# ---------- data loading ----------
@st.cache_data(ttl=5)
def load_predictions() -> pd.DataFrame | None:
    """Latest 5000 predictions from PostgreSQL, or None if the database is unreachable."""
    query = text(
        """
        SELECT request_id, transaction_id, created_at, fraud_probability, threshold,
               decision, risk_level, model_version, latency_ms, amount, source, actual_label
        FROM predictions ORDER BY created_at DESC LIMIT 5000
        """
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
    except SQLAlchemyError:
        return None
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


@st.cache_data(ttl=5)
def api_get(path: str) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


# ---------- header ----------
st.title("Credit Card Fraud Risk Monitor")
st.caption(
    "Predictions are risk signals from a statistical model, not proof of fraud. "
    "Transactions marked 'simulated' are replayed historical data, not a live bank feed."
)
if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

df = load_predictions()
health = api_get("/health")
metrics = api_get("/metrics")

# ---------- system status ----------
st.subheader("System status")
c1, c2, c3, c4 = st.columns(4)
if health is None:
    c1.metric("API", "unreachable")
else:
    c1.metric("API", health["status"])
    c2.metric("Model", f"{health['model_name']} v{health['model_version']}" if health["model_loaded"] else "not loaded")
    c3.metric("Database", health["database"])
if metrics and metrics.get("p95_latency_ms") is not None:
    c4.metric("Model latency p95 (ms)", metrics["p95_latency_ms"])
if health is None:
    st.warning("The API is not reachable. History below comes straight from the database.")

if df is None:
    st.error("The database is unavailable, so prediction history cannot be shown.")
    st.stop()
if df.empty:
    st.info("No predictions logged yet. Run `python src\\simulate.py` to generate some.")
    st.stop()

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
sources = st.sidebar.multiselect("Source", sorted(df["source"].unique()), default=sorted(df["source"].unique()))
decisions = st.sidebar.multiselect("Decision", sorted(df["decision"].unique()), default=sorted(df["decision"].unique()))
risks = st.sidebar.multiselect("Risk level", ["low", "medium", "high"], default=["low", "medium", "high"])
min_d, max_d = df["created_at"].min().date(), df["created_at"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start, end = (date_range if len(date_range) == 2 else (date_range[0], date_range[0]))

view = df[
    df["source"].isin(sources)
    & df["decision"].isin(decisions)
    & df["risk_level"].isin(risks)
    & (df["created_at"].dt.date >= start)
    & (df["created_at"].dt.date <= end)
]
st.sidebar.caption(f"{len(view)} of {len(df)} logged predictions shown")
if view.empty:
    st.info("No predictions match the current filters.")
    st.stop()

# ---------- activity ----------
st.subheader("Prediction activity")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions processed", len(view))
k2.metric("Flagged for review (predicted)", int((view["decision"] == "flag_for_review").sum()))
k3.metric("Median model latency (ms)", round(float(view["latency_ms"].median()), 1))
k4.metric("Confirmed fraud (unique transactions)",
          int(view[view["actual_label"] == 1]["transaction_id"].nunique()))
st.caption(
    "'Flagged' is what the model predicted. 'Confirmed fraud' counts only rows that "
    "have a ground-truth label, and most rows have none yet."
)

left, right = st.columns(2)
with left:
    fig = px.histogram(view, x="fraud_probability", nbins=40, log_y=True,
                       title="Fraud probability distribution (log count)")
    fig.add_vline(x=float(view["threshold"].iloc[0]), line_dash="dash", annotation_text="threshold")
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.scatter(view, x="amount", y="fraud_probability", color="decision",
                     hover_data=["transaction_id", "risk_level"],
                     title="Transaction amount vs fraud probability")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    counts = view["risk_level"].value_counts().reindex(["low", "medium", "high"], fill_value=0).reset_index()
    counts.columns = ["risk_level", "count"]
    st.plotly_chart(px.bar(counts, x="risk_level", y="count", title="Predictions by risk level"),
                    use_container_width=True)
with right:
    lat = view.sort_values("created_at")
    st.plotly_chart(px.line(lat, x="created_at", y="latency_ms", title="Model latency over time (ms)"),
                    use_container_width=True)

# ---------- ground truth ----------
st.subheader("Ground-truth evaluation")
labeled = df[df["actual_label"].notna() & df["source"].isin(sources)].copy()
labeled = labeled.sort_values("created_at").drop_duplicates("transaction_id", keep="last")
if labeled.empty:
    st.info(
        "No confirmed labels yet, so model performance is **unknown**. "
        "For simulated data, run `python src\\attach_labels.py`."
    )
else:
    flagged = labeled["decision"] == "flag_for_review"
    fraud = labeled["actual_label"] == 1
    tp, fp = int((flagged & fraud).sum()), int((flagged & ~fraud).sum())
    fn, tn = int((~flagged & fraud).sum()), int((~flagged & ~fraud).sum())
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Labeled transactions", len(labeled))
    g2.metric("Confirmed fraud caught", f"{tp} of {tp + fn}")
    g3.metric("Recall", f"{tp / (tp + fn):.2f}" if tp + fn else "n/a")
    g4.metric("Precision", f"{tp / (tp + fp):.2f}" if tp + fp else "n/a")
    st.caption(
        f"TP={tp}  FP={fp}  FN={fn}  TN={tn}. Computed only on labeled rows. Simulated data "
        "over-samples fraud and the sample is small, so do not read these as real-world performance."
    )

# ---------- model info ----------
st.subheader("Model information")
try:
    info = json.loads(THRESHOLD_FILE.read_text())
    evaluation = json.loads((ROOT / "reports" / "test_evaluation.json").read_text())
    t = evaluation["test"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registered model", f"{info['model_name']} v{info['model_version']}")
    m2.metric("Decision threshold", round(info["threshold"], 4))
    m3.metric("Held-out test recall", f"{t['recall']:.2f} ({t['tp']} of {t['tp'] + t['fn']})")
    m4.metric("Held-out test precision", f"{t['precision']:.2f}")
    st.caption(
        "From the one-time evaluation on the chronological test set (52 fraud cases). "
        "Compare experiments with: mlflow ui --backend-store-uri sqlite:///mlflow.db"
    )
except (OSError, KeyError, json.JSONDecodeError):
    st.info("Model information files were not found.")
    
# ---------- drift ----------
st.subheader("Data drift and monitoring")
try:
    drift = json.loads((ROOT / "reports" / "monitoring" / "drift_status.json").read_text())
    for label, part in [("Input features vs training data", drift["features"]),
                        ("Prediction scores vs validation data", drift["scores"])]:
        text_line = f"**{label}:** {part['status'].replace('_', ' ')} ({part['rows']} rows checked)"
        if part["status"] == "alert":
            st.warning(text_line + f", {part['drift_share']:.0%} of columns above their noise threshold")
        elif part["status"] == "ok":
            st.success(text_line)
        else:
            st.info(text_line + f", needs at least {part['min_rows']}")
    st.caption(
        f"Last check: {drift['checked_at']}. Update with `python monitoring\\check_drift.py`. "
        "Simulated data replays test-period transactions, so an alert mainly reflects the time "
        "shift from the training period, not a fault. Drift is not the same as lower accuracy: "
        "true performance is only known for rows with labels."
    )
except (OSError, KeyError, json.JSONDecodeError):
    st.info("No drift check has been run yet. Run `python monitoring\\check_drift.py`.")

# ---------- history ----------
st.subheader("Prediction history")
history = view.sort_values("created_at", ascending=False).head(200).copy()
history["actual_label"] = history["actual_label"].map({0: "legitimate", 1: "fraud"}).fillna("unknown")
st.dataframe(
    history[["created_at", "transaction_id", "source", "fraud_probability", "decision",
             "risk_level", "amount", "latency_ms", "actual_label"]],
    use_container_width=True, hide_index=True,
)