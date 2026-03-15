import json
from pathlib import Path
from typing import Optional, List

import pandas as pd
import streamlit as st


def find_metrics_files(project_root: Path) -> List[Path]:
    """Return all metrics.json files under artifacts/*/model_trainer/metrics.json."""
    artifacts_dir = project_root / "artifacts"
    if not artifacts_dir.exists():
        return []
    files = list(artifacts_dir.glob("*/model_trainer/metrics.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def get_latest_metrics_path(project_root: Path) -> Optional[Path]:
    files = find_metrics_files(project_root)
    return files[0] if files else None


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_metrics_table(data: dict) -> pd.DataFrame:
    all_metrics = data.get("all_model_metrics", {})
    if not all_metrics:
        return pd.DataFrame()

    df = pd.DataFrame(all_metrics).T.reset_index().rename(columns={"index": "model"})
    preferred_cols = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "pr_auc",
        "roc_auc",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    return df[cols]


def main() -> None:
    st.set_page_config(page_title="Accuracy Paradox Dashboard", layout="wide")
    st.title("Accuracy Paradox Dashboard")

    # Project root (one level above streamlit_app)
    project_root = Path(__file__).resolve().parents[1]

    st.caption(f"Project root: `{project_root}`")

    # Auto-discover metrics files
    metrics_files = find_metrics_files(project_root)
    latest_path = get_latest_metrics_path(project_root)

    with st.sidebar:
        st.header("Data Source")
        if metrics_files:
            options = [str(p) for p in metrics_files]
            default_index = 0
            selected = st.selectbox("Select metrics.json from artifacts", options, index=default_index)
            metrics_path_str = st.text_input("Or paste custom metrics.json path", value=selected)
        else:
            metrics_path_str = st.text_input("Paste metrics.json path", value="")
            st.info("No artifacts metrics found yet. Run `python main.py` first.")

    if not metrics_path_str:
        st.warning("Provide a metrics.json path.")
        st.stop()

    metrics_path = Path(metrics_path_str)
    if not metrics_path.exists():
        st.error(f"File not found: {metrics_path}")
        if latest_path:
            st.info(f"Latest detected metrics: {latest_path}")
        st.stop()

    try:
        data = load_json(metrics_path)
    except Exception as e:
        st.error(f"Failed to read JSON: {e}")
        st.stop()

    # Top summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", data.get("best_model_name", "N/A"))
    best_f1 = data.get("best_model_f1", None)
    col2.metric("Best F1", f"{best_f1:.4f}" if isinstance(best_f1, (int, float)) else "N/A")
    col3.metric("Bento Tag", data.get("bento_model_tag", "N/A"))

    # Accuracy paradox message
    st.subheader("Accuracy Paradox Insight")
    paradox = data.get("accuracy_paradox_demo", {})
    if paradox:
        baseline_acc = paradox.get("baseline_accuracy", None)
        baseline_recall = paradox.get("baseline_recall", None)
        baseline_f1 = paradox.get("baseline_f1", None)
        message = paradox.get("message", "")
        st.error(
            f"Dummy baseline accuracy: `{baseline_acc}` | recall: `{baseline_recall}` | f1: `{baseline_f1}`"
        )
        if message:
            st.write(message)
    else:
        st.info("No `accuracy_paradox_demo` section found in metrics.json.")

    # Metrics table
    st.subheader("Model Comparison")
    df = build_metrics_table(data)
    if df.empty:
        st.warning("No `all_model_metrics` found in metrics.json.")
    else:
        st.dataframe(df, use_container_width=True)

        # Simple charts
        numeric_cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "pr_auc", "roc_auc"] if c in df.columns]
        if numeric_cols:
            st.subheader("Metric Charts")
            metric_to_plot = st.selectbox("Select metric", numeric_cols, index=numeric_cols.index("f1_score") if "f1_score" in numeric_cols else 0)
            chart_df = df[["model", metric_to_plot]].set_index("model")
            st.bar_chart(chart_df)

    with st.expander("Raw JSON"):
        st.json(data)


if __name__ == "__main__":
    main()