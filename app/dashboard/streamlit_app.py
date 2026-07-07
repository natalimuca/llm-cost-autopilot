"""Cost + quality dashboard. The headline number is savings_pct.

Run: streamlit run app/dashboard/streamlit_app.py
"""
import sqlite3
import sys
from pathlib import Path

# `streamlit run` executes this file directly, which only puts this file's
# own directory (app/dashboard) on sys.path -- not the project root. Add the
# root explicitly so `from app...` imports resolve the same as they do under
# uvicorn (which imports app.main as a module from the project root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.db.database import DB_PATH, get_stats, init_db

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")

init_db()
stats = get_stats()

st.title("LLM Cost Autopilot")

if stats["total_requests"] == 0:
    st.info("No requests logged yet. Send some traffic through POST /v1/completions first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Requests routed", f"{stats['total_requests']:,}")
col2.metric("Actual cost", f"${stats['total_cost_usd']:.4f}")
col3.metric(
    "Cost if everything went to the top-tier model",
    f"${stats['reference_cost_usd']:.4f}",
)
col4.metric(
    "You saved",
    f"${stats['savings_usd']:.4f}",
    delta=f"{stats['savings_pct']:.1f}% cheaper",
)

st.markdown(f"## {stats['savings_pct']:.1f}% cost reduction vs. sending everything to the top-tier model")

left, right = st.columns(2)

with left:
    st.subheader("Routing distribution")
    dist_df = pd.DataFrame(
        {"model": list(stats["routing_distribution"].keys()), "count": list(stats["routing_distribution"].values())}
    )
    st.plotly_chart(px.pie(dist_df, names="model", values="count"), use_container_width=True)

with right:
    st.subheader("Escalation rate")
    st.metric("Escalated requests", f"{stats['escalation_count']} ({stats['escalation_rate_pct']:.1f}%)")
    if stats["avg_quality_score"] is not None:
        st.metric("Avg verifier quality score", f"{stats['avg_quality_score']:.2f} / 5")
    else:
        st.info("No verifier scores yet.")

st.subheader("Requests over time")
with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql(
        "SELECT timestamp, routed_model, cost_usd, quality_score, escalated FROM requests ORDER BY timestamp",
        conn,
    )
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

daily = df.groupby("date").agg(requests=("cost_usd", "count"), cost=("cost_usd", "sum")).reset_index()
st.plotly_chart(px.bar(daily, x="date", y="cost", hover_data=["requests"], title="Daily cost (USD)"), use_container_width=True)

if df["quality_score"].notna().any():
    st.subheader("Quality score distribution")
    st.plotly_chart(px.histogram(df.dropna(subset=["quality_score"]), x="quality_score", nbins=5), use_container_width=True)

st.subheader("Recent requests")
st.dataframe(df.tail(50).sort_values("timestamp", ascending=False), use_container_width=True)
