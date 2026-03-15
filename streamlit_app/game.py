import sys
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# set page config FIRST
st.set_page_config(page_title="Accuracy Paradox Game", layout="wide")
st.title("🎮 Accuracy Paradox Game")

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    XGB_IMPORT_ERROR = ""
except Exception as e:
    XGB_AVAILABLE = False
    XGB_IMPORT_ERROR = str(e)

if not XGB_AVAILABLE:
    st.sidebar.warning("XGBoost unavailable in current environment.")
    st.sidebar.code(f"Python: {sys.executable}")
    st.sidebar.caption(f"Import error: {XGB_IMPORT_ERROR}")


def get_model(name: str, use_balancing: bool, y_train=None):
    if name == "Dummy (most_frequent)":
        return DummyClassifier(strategy="most_frequent")

    if name == "LogisticRegression":
        cw = "balanced" if use_balancing else None
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)

    if name == "RandomForest":
        cw = "balanced_subsample" if use_balancing else None
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=2,
            class_weight=cw,
            random_state=42,
            n_jobs=-1,
        )

    if name == "XGBoost":
        if not XGB_AVAILABLE:
            raise ValueError("XGBoost is not installed")
        scale_pos = 1.0
        if use_balancing and y_train is not None:
            scale_pos = (np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
        return XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=float(scale_pos),
        )

    raise ValueError(f"Unknown model: {name}")


def evaluate_model(model, X_train, y_train, X_test, y_test, threshold=0.5):
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_prob = None
        y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        "pr_auc": average_precision_score(y_test, y_prob) if y_prob is not None else np.nan,
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

    return metrics, cm


# ---------- Sidebar ----------
st.sidebar.header("Controls")
n_samples = st.sidebar.slider("Samples", 1000, 20000, 6000, 1000)
minority_pct = st.sidebar.slider("Minority Class %", 0.2, 20.0, 1.0, 0.1)
class_sep = st.sidebar.slider("Class Separation", 0.2, 3.0, 1.0, 0.1)
flip_y = st.sidebar.slider("Label Noise", 0.0, 0.2, 0.01, 0.01)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.5, 0.05)
use_balancing = st.sidebar.checkbox("Use class balancing", value=True)

model_options = ["Dummy (most_frequent)", "LogisticRegression", "RandomForest"]
if XGB_AVAILABLE:
    model_options.append("XGBoost")
model_name = st.sidebar.selectbox("Model", model_options)

show_full_compare = st.sidebar.checkbox("Show full model comparison", value=True)

# ---------- Data ----------
weights = [1 - minority_pct / 100.0, minority_pct / 100.0]
X, y = make_classification(
    n_samples=n_samples,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_clusters_per_class=2,
    weights=weights,
    class_sep=class_sep,
    flip_y=flip_y,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=42
)

# ---------- Selected model ----------
try:
    model = get_model(model_name, use_balancing, y_train=y_train)
    metrics, cm = evaluate_model(model, X_train, y_train, X_test, y_test, threshold=threshold)
except Exception as e:
    st.error(f"Training failed for {model_name}: {e}")
    st.stop()

# ---------- UI: KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
c2.metric("Recall", f"{metrics['recall']:.4f}")
c3.metric("F1", f"{metrics['f1_score']:.4f}")
c4.metric("PR-AUC", "N/A" if np.isnan(metrics["pr_auc"]) else f"{metrics['pr_auc']:.4f}")

# ---------- Confusion Matrix ----------
st.subheader("Confusion Matrix")
st.dataframe(
    pd.DataFrame(
        [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    ),
    use_container_width=True
)

st.subheader("Confusion Components Graph")
st.bar_chart(pd.DataFrame([cm], index=[model_name]))

# ---------- Comparison: Selected vs Dummy ----------
st.subheader("Comparison Graph (Selected vs Dummy Baseline)")
try:
    dummy_model = get_model("Dummy (most_frequent)", use_balancing=False)
    dummy_metrics, _ = evaluate_model(dummy_model, X_train, y_train, X_test, y_test, threshold=0.5)

    compare_df = pd.DataFrame([
        {"model": "Dummy (most_frequent)", **dummy_metrics},
        {"model": model_name, **metrics},
    ])
    compare_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "pr_auc", "roc_auc"]
    st.dataframe(compare_df[["model"] + compare_cols], use_container_width=True)
    st.bar_chart(compare_df.set_index("model")[["accuracy", "precision", "recall", "f1_score", "pr_auc"]].fillna(0))
except Exception as e:
    st.warning(f"Could not build selected-vs-dummy comparison: {e}")

# ---------- Full comparison ----------
if show_full_compare:
    st.subheader("Full Model Comparison Graph")
    full_models = ["Dummy (most_frequent)", "LogisticRegression", "RandomForest"]
    if XGB_AVAILABLE:
        full_models.append("XGBoost")

    rows = []
    for mname in full_models:
        try:
            m = get_model(mname, use_balancing=use_balancing, y_train=y_train)
            m_metrics, _ = evaluate_model(m, X_train, y_train, X_test, y_test, threshold=threshold)
            rows.append({"model": mname, **m_metrics})
        except Exception as e:
            rows.append({"model": mname, "error": str(e)})

    full_df = pd.DataFrame(rows)

    if "error" in full_df.columns:
        err_df = full_df[full_df["error"].notna()]
        if not err_df.empty:
            st.warning("Some models failed:")
            st.dataframe(err_df[["model", "error"]], use_container_width=True)

    metric_cols = [c for c in ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "pr_auc", "roc_auc"] if c in full_df.columns]
    good_df = full_df.dropna(subset=["model"]).copy()

    st.dataframe(good_df[["model"] + metric_cols], use_container_width=True)

    metric_pick = st.selectbox("Pick metric for model comparison graph", metric_cols, index=metric_cols.index("f1_score"))
    st.bar_chart(good_df.set_index("model")[[metric_pick]].fillna(0))

st.subheader("All Metrics (Selected Model)")
st.dataframe(pd.DataFrame([metrics]), use_container_width=True)