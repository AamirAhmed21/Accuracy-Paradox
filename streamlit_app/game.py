import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import requests
import time
import matplotlib.pyplot as plt
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
    roc_curve,
    auc,
)

# set page config FIRST
st.set_page_config(page_title="Accuracy Paradox Game", layout="wide")
st.title("🎮 Accuracy Paradox Game")

# Option Smote
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception as e:
    SMOTE_AVAILABLE = False
    st.sidebar.warning("SMOTE unavailable in current environment.")
    st.sidebar.code(f"Python: {sys.executable}")
    st.sidebar.caption(f"Import error: {str(e)}")

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

    return metrics, cm, y_prob


@st.cache_data(show_spinner=False)
def get_train_test_data(n_samples, minority_pct, class_sep, flip_y, test_size):
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
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


@st.cache_data(show_spinner=False)
def build_scenario_database(class_sep):
    scenario_ratios = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    scenario_rows = []

    for ratio in scenario_ratios:
        w_s = [1 - ratio / 100, ratio / 100]
        Xs, ys = make_classification(
            n_samples=5000,
            n_features=20,
            n_informative=8,
            n_redundant=4,
            n_clusters_per_class=2,
            weights=w_s,
            class_sep=class_sep,
            flip_y=0.01,
            random_state=42,
        )
        Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
            Xs, ys, test_size=0.2, stratify=ys, random_state=42
        )

        for do_bal_s in [False, True]:
            try:
                m_s = get_model("LogisticRegression", do_bal_s, y_train=ytr_s)
                m_met_s, _, _ = evaluate_model(m_s, Xtr_s, ytr_s, Xte_s, yte_s, threshold=0.5)
                scenario_rows.append({
                    "minority_%": ratio,
                    "balancing": "Class Weight" if do_bal_s else "No Balancing",
                    "accuracy": round(m_met_s["accuracy"], 4),
                    "recall": round(m_met_s["recall"], 4),
                    "f1_score": round(m_met_s["f1_score"], 4),
                    "pr_auc": round(m_met_s["pr_auc"], 4),
                })
            except Exception:
                pass

    return pd.DataFrame(scenario_rows)


@st.cache_data(show_spinner=False)
def build_resolution_experiment(
    selected_model_name,
    n_samples,
    class_sep,
    flip_y,
    test_size,
    threshold,
    smote_available,
):
    experiment_rows = []

    exp_model_name = selected_model_name
    if exp_model_name == "Dummy (most_frequent)":
        exp_model_name = "XGBoost" if XGB_AVAILABLE else "LogisticRegression"

    experiment_cases = [
        {"case": "A) 1% minority + No balancing", "minority_pct": 1.0, "use_balancing": False, "use_smote": False},
        {"case": "B) 1% minority + Class weight", "minority_pct": 1.0, "use_balancing": True, "use_smote": False},
        {"case": "C) 10% minority + No balancing", "minority_pct": 10.0, "use_balancing": False, "use_smote": False},
    ]

    if smote_available:
        experiment_cases.append(
            {"case": "D) 1% minority + SMOTE", "minority_pct": 1.0, "use_balancing": False, "use_smote": True}
        )

    for exp_case in experiment_cases:
        Xtr_e, Xte_e, ytr_e, yte_e = get_train_test_data(
            n_samples=n_samples,
            minority_pct=exp_case["minority_pct"],
            class_sep=class_sep,
            flip_y=flip_y,
            test_size=test_size,
        )

        Xtr_model_e, ytr_model_e = Xtr_e, ytr_e
        if exp_case["use_smote"] and smote_available:
            sm_e = SMOTE(random_state=42)
            Xtr_model_e, ytr_model_e = sm_e.fit_resample(Xtr_e, ytr_e)

        model_e = get_model(
            exp_model_name,
            exp_case["use_balancing"],
            y_train=ytr_model_e,
        )
        met_e, cm_e, _ = evaluate_model(
            model_e,
            Xtr_model_e,
            ytr_model_e,
            Xte_e,
            yte_e,
            threshold=threshold,
        )
        experiment_rows.append(
            {
                "experiment": exp_case["case"],
                "model_used": exp_model_name,
                "minority_%": exp_case["minority_pct"],
                "accuracy": round(met_e["accuracy"], 4),
                "recall": round(met_e["recall"], 4),
                "f1_score": round(met_e["f1_score"], 4),
                "pr_auc": round(met_e["pr_auc"], 4),
                "tp": int(cm_e["TP"]),
                "fn": int(cm_e["FN"]),
            }
        )

    return pd.DataFrame(experiment_rows)


# ---------- Sidebar ----------
st.sidebar.header("Controls")
n_samples = st.sidebar.slider("Samples", 1000, 20000, 6000, 1000)
minority_pct = st.sidebar.slider("Minority Class %", 0.2, 20.0, 1.0, 0.1)
class_sep = st.sidebar.slider("Class Separation", 0.2, 3.0, 1.0, 0.1)
flip_y = st.sidebar.slider("Label Noise", 0.0, 0.2, 0.01, 0.01)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
threshold = st.sidebar.slider("Decision Threshold", 0.05, 0.95, 0.5, 0.05)
balancing_options = ["No balancing", "Class weight (balancing)"]
if SMOTE_AVAILABLE:
    balancing_options.append("SMOTE (oversampling Minority)")
balancing_method = st.sidebar.selectbox("Balancing Method", balancing_options, index=0)
use_balancing = balancing_method == "Class weight (balancing)"
use_smote = SMOTE_AVAILABLE and balancing_method == "SMOTE (oversampling Minority)"
if not SMOTE_AVAILABLE:
    st.sidebar.caption("💡 pip install imbalanced-learn to enable SMOTE")

model_options = ["Dummy (most_frequent)", "LogisticRegression", "RandomForest"]
if XGB_AVAILABLE:
    model_options.append("XGBoost")
model_name = st.sidebar.selectbox("Model", model_options)

# ---------- BentoML API Sidebar ----------
st.sidebar.subheader("BentoML API")
use_bento_api = st.sidebar.checkbox("Use BentoML API", value=False)
bento_api_url = st.sidebar.text_input("Predict URL", "http://127.0.0.1:3000/predict")

# ---------- Data ----------
X_train, X_test, y_train, y_test = get_train_test_data(
    n_samples=n_samples,
    minority_pct=minority_pct,
    class_sep=class_sep,
    flip_y=flip_y,
    test_size=test_size,
)
show_full_compare = st.sidebar.checkbox("Show full model comparison", value=False)
# ---------- Apply resampling ----------
X_train_model, y_train_model = X_train, y_train
if use_smote:
    try:
        smote = SMOTE(random_state=42)
        X_train_model, y_train_model = smote.fit_resample(X_train, y_train)
        minority_before = int(np.sum(y_train == 1))
        minority_after = int(np.sum(y_train_model == 1))
        st.info(f"✅ SMOTE applied: minority class grew from {minority_before} → {minority_after} samples")
    except Exception as e:
        st.warning(f"SMOTE failed: {e}. Using original data.")
        X_train_model, y_train_model = X_train, y_train
        
# ---------- Selected model ----------
try:
    model = get_model(model_name, use_balancing, y_train=y_train_model)
    metrics, cm, y_prob = evaluate_model(model, X_train_model, y_train_model, X_test, y_test, threshold=threshold)
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
dummy_prob = None
try:
    dummy_model = get_model("Dummy (most_frequent)", use_balancing=False)
    dummy_metrics, _, dummy_prob = evaluate_model(dummy_model, X_train_model, y_train_model, X_test, y_test, threshold=0.5)
    compare_df = pd.DataFrame([
        {"model": "Dummy (most_frequent)", **dummy_metrics},
        {"model": model_name, **metrics},
    ])
    compare_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "pr_auc", "roc_auc"]
    st.dataframe(compare_df[["model"] + compare_cols], use_container_width=True)
    st.bar_chart(compare_df.set_index("model")[["accuracy", "precision", "recall", "f1_score", "pr_auc"]].fillna(0))

    # ROC Curve comparison for clear visual difference
    if (dummy_prob is not None) and (y_prob is not None):
        st.subheader("ROC Curve (Selected vs Dummy)")
        fpr_d, tpr_d, _ = roc_curve(y_test, dummy_prob)
        fpr_s, tpr_s, _ = roc_curve(y_test, y_prob)

        auc_d = auc(fpr_d, tpr_d)
        auc_s = auc(fpr_s, tpr_s)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr_d, tpr_d, linestyle="--", label=f"Dummy (AUC={auc_d:.3f})")
        ax.plot(fpr_s, tpr_s, label=f"{model_name} (AUC={auc_s:.3f})")
        ax.plot([0, 1], [0, 1], "k:", label="Random baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-AUC Comparison")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
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
            m = get_model(mname, use_balancing=use_balancing, y_train=y_train_model)
            m_metrics, _, _ = evaluate_model(m, X_train_model, y_train_model, X_test, y_test, threshold=threshold)
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

# ---------- Experimental Proof (before vs after fix) ----------
st.divider()
st.subheader("🧪 Experimental Proof: Why Lower Accuracy Can Be Better")
st.caption(
    "Runs controlled experiments to show: high accuracy can fail minority detection, and balancing/minority increase improves useful prediction."
)

experiment_threshold = st.slider(
    "Experiment Threshold (independent from main threshold)",
    min_value=0.05,
    max_value=0.95,
    value=0.30,
    step=0.05,
)
st.caption("Tip: use 0.20–0.35 to make minority detection differences easier to see.")

if st.button("Run resolution experiment"):
    with st.spinner("Running controlled before-vs-after experiments..."):
        exp_df = build_resolution_experiment(
            selected_model_name=model_name,
            n_samples=n_samples,
            class_sep=class_sep,
            flip_y=flip_y,
            test_size=test_size,
            threshold=experiment_threshold,
            smote_available=SMOTE_AVAILABLE,
        )

    display_cols = [
        "experiment",
        "model_used",
        "minority_%",
        "accuracy",
        "recall",
        "f1_score",
        "pr_auc",
        "tp",
        "fn",
    ]
    st.dataframe(exp_df[display_cols], use_container_width=True)

    st.subheader("Minority Detection Metrics (what matters)")
    st.bar_chart(exp_df.set_index("experiment")[["recall", "f1_score", "pr_auc"]])

    st.subheader("Accuracy (can be misleading)")
    st.bar_chart(exp_df.set_index("experiment")[["accuracy"]])

    best_recall_row = exp_df.loc[exp_df["recall"].idxmax()]
    worst_recall_row = exp_df.loc[exp_df["recall"].idxmin()]
    st.success(
        f"Best minority detection: {best_recall_row['experiment']} (Recall={best_recall_row['recall']:.4f}, F1={best_recall_row['f1_score']:.4f}, TP={int(best_recall_row['tp'])}, FN={int(best_recall_row['fn'])})"
    )
    st.warning(
        f"Misleading high-accuracy case: {worst_recall_row['experiment']} (Recall={worst_recall_row['recall']:.4f}, TP={int(worst_recall_row['tp'])}, FN={int(worst_recall_row['fn'])}, Accuracy={worst_recall_row['accuracy']:.4f})"
    )
    st.info(
        "Conclusion: a model with slightly lower accuracy but much higher Recall/F1 is better for imbalanced data."
    )

# ---------- Resolution Methods Comparison (Requirement 1) ----------
st.divider()
st.subheader("🔬 Resolution Methods Comparison")
st.caption("Compares the same model trained three ways: No Balancing vs Class Weight vs SMOTE.")

resolution_rows = []
methods_to_run = [("No Balancing", False, False), ("Class Weight", True, False)]
if SMOTE_AVAILABLE:
    methods_to_run.append(("SMOTE", False, True))

for method_name, do_cw, do_smote in methods_to_run:
    try:
        Xtr_r, ytr_r = X_train, y_train
        if do_smote:
            smote_r = SMOTE(random_state=42)
            Xtr_r, ytr_r = smote_r.fit_resample(X_train, y_train)
        m_r = get_model(model_name, do_cw, y_train=ytr_r)
        m_met_r, _, _ = evaluate_model(m_r, Xtr_r, ytr_r, X_test, y_test, threshold=threshold)
        resolution_rows.append({"Method": method_name, **m_met_r})
    except Exception as e_r:
        resolution_rows.append({"Method": method_name, "error": str(e_r)})

res_df = pd.DataFrame(resolution_rows)
show_res_cols = [c for c in ["accuracy", "recall", "f1_score", "pr_auc"] if c in res_df.columns]
if show_res_cols:
    st.dataframe(res_df[["Method"] + show_res_cols].fillna(0), use_container_width=True)
    st.bar_chart(res_df.set_index("Method")[show_res_cols].fillna(0))
    st.caption("⚠️ Key finding: Accuracy drops with balancing, but Recall and F1 rise — proving accuracy was masking failure.")

# ---------- Imbalance Scenario Database (Requirement 3) ----------
st.divider()
st.subheader("📊 Imbalance Scenario Database")
st.caption("Synthetic dataset tool demonstrating the paradox across 6 imbalance scenarios (10% → 0.2% minority).")

with st.expander("▶ Run Scenario Database Analysis", expanded=False):
    st.caption("This analysis is compute-heavy. Run it only when you need it.")
    if "scenario_df" not in st.session_state:
        st.session_state.scenario_df = None

    if st.button("Run scenario analysis"):
        with st.spinner("Generating scenarios and training models..."):
            st.session_state.scenario_df = build_scenario_database(class_sep)

    scen_df = st.session_state.scenario_df
    if scen_df is not None and not scen_df.empty:
        st.dataframe(scen_df, use_container_width=True)

        no_bal_df = scen_df[scen_df["balancing"] == "No Balancing"].set_index("minority_%").sort_index(ascending=False)
        bal_df_s = scen_df[scen_df["balancing"] == "Class Weight"].set_index("minority_%").sort_index(ascending=False)

        st.subheader("Accuracy vs Recall — No Balancing (Paradox Worsening)")
        st.line_chart(no_bal_df[["accuracy", "recall"]])
        st.caption("As minority % shrinks, accuracy stays near 100% but recall collapses to 0.")

        both_f1 = pd.DataFrame({
            "No Balancing (F1)": no_bal_df["f1_score"],
            "Class Weight (F1)": bal_df_s["f1_score"],
        })
        st.subheader("F1-Score: No Balancing vs Class Weight Across All Scenarios")
        st.line_chart(both_f1)
        st.caption("Class Weight consistently improves F1 across all imbalance levels.")
    else:
        st.info("Click 'Run scenario analysis' to generate scenario results.")

    export_dir = os.path.join("artifacts", "scenario_database")
    if st.button("Export scenario database to CSV files"):
        os.makedirs(export_dir, exist_ok=True)

        if scen_df is not None and not scen_df.empty:
            scen_df.to_csv(os.path.join(export_dir, "scenario_metrics_summary.csv"), index=False)

        scenario_ratios = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
        meta_rows = []
        for ratio in scenario_ratios:
            w = [1 - ratio / 100, ratio / 100]
            Xs, ys = make_classification(
                n_samples=5000,
                n_features=20,
                n_informative=8,
                n_redundant=4,
                n_clusters_per_class=2,
                weights=w,
                class_sep=class_sep,
                flip_y=0.01,
                random_state=42,
            )
            df_s = pd.DataFrame(Xs, columns=[f"feature_{i}" for i in range(Xs.shape[1])])
            df_s["target"] = ys.astype(int)

            file_name = f"scenario_{str(ratio).replace('.', 'p')}_minority.csv"
            file_path = os.path.join(export_dir, file_name)
            df_s.to_csv(file_path, index=False)

            meta_rows.append(
                {
                    "minority_pct": ratio,
                    "file_name": file_name,
                    "rows": len(df_s),
                    "cols": df_s.shape[1],
                }
            )

        pd.DataFrame(meta_rows).to_csv(
            os.path.join(export_dir, "scenario_dataset_index.csv"), index=False
        )
        st.success(f"Scenario database exported to: {export_dir}")

st.divider()
st.subheader("🧪 Missing Data Simulation")
st.caption("Tests how missing values affect model quality and how imputation helps.")

missing_pct = st.selectbox("Missingness level", [0, 5, 10, 20], index=1)

if st.button("Run missing-data experiment"):
    rng = np.random.default_rng(42)
    X_train_miss = X_train.copy()
    X_test_miss = X_test.copy()

    if missing_pct > 0:
        tr_mask = rng.random(X_train_miss.shape) < (missing_pct / 100.0)
        te_mask = rng.random(X_test_miss.shape) < (missing_pct / 100.0)
        X_train_miss[tr_mask] = np.nan
        X_test_miss[te_mask] = np.nan

    col_means = np.nanmean(X_train_miss, axis=0)
    inds_tr = np.where(np.isnan(X_train_miss))
    inds_te = np.where(np.isnan(X_test_miss))
    X_train_miss[inds_tr] = np.take(col_means, inds_tr[1])
    X_test_miss[inds_te] = np.take(col_means, inds_te[1])

    m_base = get_model("LogisticRegression", False, y_train=y_train)
    met_base, _, _ = evaluate_model(m_base, X_train, y_train, X_test, y_test, threshold=0.5)

    m_miss = get_model("LogisticRegression", False, y_train=y_train)
    met_miss, _, _ = evaluate_model(
        m_miss, X_train_miss, y_train, X_test_miss, y_test, threshold=0.5
    )

    out = pd.DataFrame(
        [
            {
                "setting": "No missing data",
                "accuracy": met_base["accuracy"],
                "recall": met_base["recall"],
                "f1": met_base["f1_score"],
                "pr_auc": met_base["pr_auc"],
            },
            {
                "setting": f"{missing_pct}% missing + mean imputation",
                "accuracy": met_miss["accuracy"],
                "recall": met_miss["recall"],
                "f1": met_miss["f1_score"],
                "pr_auc": met_miss["pr_auc"],
            },
        ]
    )
    st.dataframe(out, use_container_width=True)
    st.bar_chart(out.set_index("setting")[["recall", "f1", "pr_auc"]])

# ---------- BentoML API Section ----------
if use_bento_api:
    st.subheader("🔌 Live API Inference (BentoML)")

    # API status check
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check API Status"):
            try:
                r = requests.get(bento_api_url.replace("/predict", "/healthz"), timeout=5)
                if r.ok:
                    st.success("API is running ✅")
                else:
                    st.warning(f"API responded with status: {r.status_code}")
            except Exception:
                st.error("API is not reachable. Start it with: bentoml serve inference_service:AccuracyParadoxService")
    with col2:
        sample_idx = st.number_input(
            "Sample index from test set",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1,
        )

    # Show sample features
    with st.expander("Sample features (X_test row)"):
        st.write(X_test[int(sample_idx)].tolist())
        st.write(f"Actual label: **{int(y_test[int(sample_idx)])}**")

    # API call + history
    if "api_history" not in st.session_state:
        st.session_state.api_history = []
    if st.button("Predict via API"):
        payload = {"features": X_test[int(sample_idx)].tolist()}
        try:
            t0 = time.time()
            resp = requests.post(bento_api_url, json=payload, timeout=30)
            latency_ms = (time.time() - t0) * 1000

            if resp.ok:
                data = resp.json()
                pred = data.get("prediction")
                prob = data.get("probability", 0.0)
                prob = float(prob)
                prob_pct = prob * 100.0
                actual = int(y_test[int(sample_idx)])
                correct = pred == actual

                st.success(
                    f"Prediction: **{pred}** | Probability (raw): **{prob:.10f}** | Probability (%): **{prob_pct:.6f}%**"
                )
                if prob == 0.0:
                    st.warning("API returned exactly 0.0 probability for class 1. This means the deployed model is highly confident in class 0 for this sample.")
                elif prob_pct < 0.01:
                    st.info("Probability is very small (less than 0.01%), so it may look like 0 when rounded.")
                st.caption(f"Latency: {latency_ms:.2f} ms")

                if y_prob is not None and int(sample_idx) < len(y_prob):
                    local_prob = float(y_prob[int(sample_idx)])
                    st.caption(
                        f"Local app model probability for same sample: {local_prob:.10f} ({local_prob * 100.0:.6f}%)"
                    )
                if correct:
                    st.info("✅ Correct prediction")
                else:
                    st.warning(f"❌ Wrong prediction (actual = {actual})")

                # Save to history
                st.session_state.api_history.append({
                    "sample_idx": int(sample_idx),
                    "prediction": pred,
                    "probability": round(prob, 4),
                    "actual": actual,
                    "correct": correct,
                    "latency_ms": round(latency_ms, 2),
                })
            else:
                st.error(f"API Error: {resp.text}")
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.info("Make sure BentoML service is running: bentoml serve inference_service:AccuracyParadoxService --port 3000")

    # API call history table
    if st.session_state.api_history:
        st.subheader("API Call History")
        history_df = pd.DataFrame(st.session_state.api_history)
        st.dataframe(history_df, use_container_width=True)

        if st.button("Clear History"):
            st.session_state.api_history = []
            st.rerun()
