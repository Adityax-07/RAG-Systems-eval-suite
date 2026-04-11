"""Streamlit dashboard for the LLM Evaluation Framework.

Run with:
    streamlit run dashboard/app.py

What this shows:
    1. System overview — all systems you've evaluated and their latest scores
    2. Score trends — how a system's metrics change over time (catch regressions!)
    3. Run details — drill into any single run to see per-example scores
    4. Run comparison — side-by-side diff of two runs

Teaching note: Streamlit turns Python functions into web UIs. The key insight
is that Streamlit reruns your entire script on every user interaction. This
means you write UI code exactly like normal code — no callbacks, no state
management boilerplate. Use st.cache_data to avoid re-querying the DB.
"""

import sys
from pathlib import Path
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_framework.storage.database import EvalDatabase
from eval_framework.config import get_settings

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Eval Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Database connection ──────────────────────────────────────────────────────
@st.cache_resource
def get_db() -> EvalDatabase:
    """Cache the DB connection so it's not recreated on every rerun."""
    # Use absolute path so this works regardless of which directory
    # streamlit is launched from.
    db_path = Path(__file__).parent.parent / "data" / "results.db"
    return EvalDatabase(str(db_path))


# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🔬 LLM Eval Framework")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["System Overview", "Score Trends", "Run Details", "Compare Runs"],
    index=0,
)

db = get_db()
all_systems = db.list_systems()

if not all_systems:
    st.info(
        "**No evaluation runs found yet.**\n\n"
        "Run your first evaluation from the project root:\n\n"
        "```bash\n"
        "python examples/rag_eval.py\n"
        "```\n\n"
        "Or using the CLI:\n\n"
        "```bash\n"
        "python cli/main.py rag-eval data/knowledge_base.txt\n"
        "```\n\n"
        "Results will appear here automatically once a run completes."
    )
    st.stop()

# ─── Page: System Overview ────────────────────────────────────────────────────
if page == "System Overview":
    st.title("System Overview")
    st.markdown("Latest evaluation scores across all systems.")

    all_runs = db.list_all_runs(limit=200)

    # Build a summary DataFrame: one row per system (most recent run)
    systems_latest = {}
    for run in all_runs:
        name = run["system_name"]
        if name not in systems_latest:
            systems_latest[name] = run

    rows = []
    for system_name, run in systems_latest.items():
        scores = run["summary_scores"]
        row = {
            "System": system_name,
            "Last Run": run["timestamp"][:19],
            "Examples": run["total_examples"],
        }
        for metric in ["faithfulness", "relevance", "completeness", "hallucination_rate"]:
            row[metric.replace("_", " ").title()] = (
                f"{scores[metric]:.2f}" if metric in scores else "—"
            )
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Radar chart — compare all systems on a single chart
    st.markdown("### Metric Comparison (Radar Chart)")
    metrics_to_plot = ["faithfulness", "relevance", "completeness"]

    fig = go.Figure()
    for system_name, run in systems_latest.items():
        scores = run["summary_scores"]
        values = [scores.get(m, 0) for m in metrics_to_plot]
        values.append(values[0])  # Close the radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_to_plot + [metrics_to_plot[0]],
            fill="toself",
            name=system_name,
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Page: Score Trends ───────────────────────────────────────────────────────
elif page == "Score Trends":
    st.title("Score Trends Over Time")
    st.markdown("Track how your system's quality changes across runs.")

    selected_system = st.selectbox("Select system", all_systems)
    runs = db.get_runs_for_system(selected_system)

    if len(runs) < 2:
        st.info(
            f"Only {len(runs)} run(s) for '{selected_system}'. "
            "Run more evaluations to see trends."
        )

    if runs:
        # Build a DataFrame with one row per run, columns = metrics
        df_rows = []
        for run in reversed(runs):  # Oldest first for charts
            scores = run["summary_scores"]
            row = {"timestamp": run["timestamp"][:19], "run_id": run["id"][:8]}
            row.update(scores)
            df_rows.append(row)
        df = pd.DataFrame(df_rows)

        # Line chart for each metric
        metric_cols = [c for c in df.columns if c not in ("timestamp", "run_id")]
        if metric_cols:
            fig = px.line(
                df,
                x="timestamp",
                y=metric_cols,
                title=f"Metric Trends: {selected_system}",
                markers=True,
                range_y=[0, 1],
            )
            fig.update_layout(
                yaxis_title="Score (0-1)",
                xaxis_title="Run Date",
                legend_title="Metric",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Raw data table
        with st.expander("Raw data"):
            st.dataframe(df, use_container_width=True)


# ─── Page: Run Details ────────────────────────────────────────────────────────
elif page == "Run Details":
    st.title("Run Details")
    st.markdown("Inspect individual evaluation results.")

    selected_system = st.selectbox("Select system", all_systems)
    runs = db.get_runs_for_system(selected_system)

    if not runs:
        st.warning("No runs for this system.")
        st.stop()

    run_options = {
        f"{r['timestamp'][:19]} — {r['id'][:8]}": r["id"] for r in runs
    }
    selected_run_label = st.selectbox("Select run", list(run_options.keys()))
    run_id = run_options[selected_run_label]

    run = db.get_run(run_id)
    if not run:
        st.error("Run not found.")
        st.stop()

    # Summary metrics
    st.markdown("### Summary Scores")
    scores = run["summary_scores"]
    cols = st.columns(len(scores))
    for col, (metric, score) in zip(cols, scores.items()):
        delta_color = "normal"
        col.metric(
            label=metric.replace("_", " ").title(),
            value=f"{score:.2f}",
        )

    st.markdown(f"**Run ID:** `{run_id}` | **Examples:** {run['total_examples']} | **Cost:** ${run['total_cost']:.4f}")

    # Per-metric result distribution
    st.markdown("### Score Distributions")
    results = db.get_results_for_run(run_id)
    if results:
        df = pd.DataFrame(results)
        metrics_in_run = df["metric"].unique()

        for metric in metrics_in_run:
            metric_df = df[df["metric"] == metric]
            fig = px.histogram(
                metric_df,
                x="score",
                nbins=20,
                title=f"{metric} — score distribution ({len(metric_df)} examples)",
                range_x=[0, 1],
                color_discrete_sequence=["#636EFA"],
            )
            fig.update_layout(height=250, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Lowest-scoring examples (useful for debugging)
        st.markdown("### Lowest-Scoring Examples")
        st.markdown("These are the examples where the system performed worst — useful for debugging.")
        low_scores = df.nsmallest(10, "score")[["metric", "score", "reasoning"]]
        st.dataframe(low_scores, use_container_width=True, hide_index=True)


# ─── Page: Compare Runs ───────────────────────────────────────────────────────
elif page == "Compare Runs":
    st.title("Compare Runs")
    st.markdown("Side-by-side comparison of two evaluation runs.")

    all_runs = db.list_all_runs(limit=50)
    run_labels = {
        f"{r['system_name']} — {r['timestamp'][:19]} ({r['id'][:8]})": r["id"]
        for r in all_runs
    }
    labels = list(run_labels.keys())

    if len(labels) < 2:
        st.warning("Need at least 2 evaluation runs to compare.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        label1 = st.selectbox("Baseline run", labels, index=0)
    with col2:
        label2 = st.selectbox("Comparison run", labels, index=min(1, len(labels) - 1))

    if label1 == label2:
        st.warning("Select two different runs to compare.")
        st.stop()

    run_id_1 = run_labels[label1]
    run_id_2 = run_labels[label2]

    comparison = db.compare_runs(run_id_1, run_id_2)

    # Delta bar chart
    metrics = list(comparison["metrics"].keys())
    deltas = [comparison["metrics"][m]["delta"] for m in metrics]
    colors = ["#00CC96" if d >= 0 else "#EF553B" for d in deltas]

    fig = go.Figure(go.Bar(
        x=metrics,
        y=deltas,
        marker_color=colors,
        text=[f"{d:+.3f}" for d in deltas],
        textposition="auto",
    ))
    fig.update_layout(
        title="Score Delta (Comparison − Baseline)",
        yaxis_title="Delta",
        height=350,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="white"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    df_rows = []
    for metric, data in comparison["metrics"].items():
        df_rows.append({
            "Metric": metric,
            "Baseline": f"{data['run1_score']:.3f}",
            "Comparison": f"{data['run2_score']:.3f}",
            "Delta": f"{data['delta']:+.3f}",
            "Improved": "✅" if data["improved"] else "❌",
        })
    st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)
