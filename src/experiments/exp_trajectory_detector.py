"""
Trajectory-Based Hallucination Detection
==========================================
Uses multi-scale confidence trajectories (7 Pythia sizes) to detect
high-confidence false claims that single-model confidence misses.

Core hypothesis: True claims show stable/increasing confidence with scale.
False-but-transmissible claims show a different trajectory shape (early
plateau, non-monotonicity, divergence).

Experiments:
  T1: Trajectory vs single-point classification (LOO AUC comparison)
  T2: Hard case analysis (high-confidence false claims)
  T3: Feature importance (which trajectory features matter?)
  T4: Minimal trajectory (practical 2-model detector)
  T5: Category-specific analysis (general, medical, mandela, settled)
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import load_records
from src.scaling import MODEL_REGISTRY, SCALING_MODELS_ALL, PARAM_COUNTS
from src.scaling_viz import MODEL_COLORS, model_display_name

SCALES = SCALING_MODELS_ALL  # ["160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]

# Output dirs
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "trajectory_detector"
FIGURES_DIR = PROJECT_ROOT / "figures" / "trajectory_detector"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# Step 1: Pool all confidence data across experiments and scales
# ===================================================================

def load_all_items() -> pd.DataFrame:
    """
    Load confidence values for all labeled items across all 7 scales.

    Returns DataFrame with columns:
        item_id, text, is_true, category, source,
        conf_160m, conf_410m, ..., conf_12b
    """
    rows = []

    # ----- A1 Truth pairs (Exp 2) -----
    # Load each scale's results, index by label
    truth_by_scale = {}
    for size in SCALES:
        path = PROJECT_ROOT / "data" / "results" / "scaling" / f"a1_truth_{size}.jsonl"
        if path.exists():
            records = load_records(path)
            truth_by_scale[size] = {r.label: r for r in records}

    # Collect unique labels
    if truth_by_scale:
        all_labels = set()
        for recs in truth_by_scale.values():
            all_labels.update(recs.keys())

        for label in sorted(all_labels):
            # Check this label exists at all scales
            confs = {}
            text = None
            category = None
            for size in SCALES:
                if size in truth_by_scale and label in truth_by_scale[size]:
                    rec = truth_by_scale[size][label]
                    confs[f"conf_{size}"] = rec.mean_top1_prob
                    if text is None:
                        text = rec.text
                        category = rec.category

            if len(confs) == len(SCALES) and category in ("true", "false"):
                base_id = label.rsplit("_", 1)[0]
                rows.append({
                    "item_id": f"truth_{label}",
                    "text": text,
                    "is_true": 1 if category == "true" else 0,
                    "category": "general_fact",
                    "source": "exp2_truth",
                    "pair_id": f"truth_{base_id}",
                    **confs,
                })

    # ----- Medical pairs (Exp 9) -----
    med_by_scale = {}
    for size in SCALES:
        path = PROJECT_ROOT / "data" / "results" / "exp9" / f"medical_pairs_{size}.jsonl"
        if path.exists():
            records = load_records(path)
            med_by_scale[size] = {r.label: r for r in records}

    if med_by_scale:
        all_labels = set()
        for recs in med_by_scale.values():
            all_labels.update(recs.keys())

        for label in sorted(all_labels):
            confs = {}
            text = None
            category = None
            for size in SCALES:
                if size in med_by_scale and label in med_by_scale[size]:
                    rec = med_by_scale[size][label]
                    confs[f"conf_{size}"] = rec.mean_top1_prob
                    if text is None:
                        text = rec.text
                        category = rec.category

            if len(confs) == len(SCALES):
                is_true = 1 if "true" in category else 0
                base_id = label.rsplit("_", 1)[0]
                rows.append({
                    "item_id": f"med_{label}",
                    "text": text,
                    "is_true": is_true,
                    "category": "medical",
                    "source": "exp9_medical",
                    "pair_id": f"med_{base_id}",
                    **confs,
                })

    # ----- Mandela expanded (raw framing only) -----
    mandela_by_scale = {}
    for size in SCALES:
        path = PROJECT_ROOT / "data" / "results" / "mandela" / "expanded" / f"expanded_{size}.jsonl"
        if path.exists():
            records = load_records(path)
            mandela_by_scale[size] = {r.label: r for r in records}

    if mandela_by_scale:
        all_labels = set()
        for recs in mandela_by_scale.values():
            all_labels.update(recs.keys())

        # Only use "raw" framing to avoid double-counting
        for label in sorted(all_labels):
            if "_raw_" not in label:
                continue

            confs = {}
            text = None
            category = None
            for size in SCALES:
                if size in mandela_by_scale and label in mandela_by_scale[size]:
                    rec = mandela_by_scale[size][label]
                    confs[f"conf_{size}"] = rec.mean_top1_prob
                    if text is None:
                        text = rec.text
                        category = rec.category

            if len(confs) == len(SCALES):
                is_true = 1 if category == "mandela_correct" else 0
                # Extract item_id from label like "star_wars_raw_wrong"
                parts = label.rsplit("_raw_", 1)
                base_id = parts[0]
                version = parts[1] if len(parts) > 1 else "unknown"
                rows.append({
                    "item_id": f"mandela_{label}",
                    "text": text,
                    "is_true": is_true,
                    "category": "mandela",
                    "source": "mandela_expanded",
                    "pair_id": f"mandela_{base_id}",
                    **confs,
                })

    # ----- Settled science (from A2 contested, settled category only) -----
    contested_by_scale = {}
    for size in SCALES:
        path = PROJECT_ROOT / "data" / "results" / "scaling" / f"a2_contested_{size}.jsonl"
        if path.exists():
            records = load_records(path)
            contested_by_scale[size] = {r.label: r for r in records}

    if contested_by_scale:
        all_labels = set()
        for recs in contested_by_scale.values():
            all_labels.update(recs.keys())

        for label in sorted(all_labels):
            confs = {}
            text = None
            category = None
            for size in SCALES:
                if size in contested_by_scale and label in contested_by_scale[size]:
                    rec = contested_by_scale[size][label]
                    confs[f"conf_{size}"] = rec.mean_top1_prob
                    if text is None:
                        text = rec.text
                        category = rec.category

            if len(confs) == len(SCALES) and category == "settled":
                rows.append({
                    "item_id": f"settled_{label}",
                    "text": text,
                    "is_true": 1,  # settled science = true
                    "category": "settled_science",
                    "source": "exp3_settled",
                    "pair_id": f"settled_{label}",  # no pair, stand-alone
                    **confs,
                })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} items across {len(df['source'].unique())} sources")
    print(f"  True: {(df['is_true'] == 1).sum()}, False: {(df['is_true'] == 0).sum()}")
    print(f"  Categories: {dict(df['category'].value_counts())}")
    return df


# ===================================================================
# Step 2: Trajectory feature extraction
# ===================================================================

def count_reversals(arr):
    """Count direction changes in a sequence."""
    diffs = np.diff(arr)
    signs = np.sign(diffs)
    # Remove zeros (no change)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int(np.sum(np.diff(signs) != 0))


def extract_trajectory_features(row, scales=SCALES) -> dict:
    """Extract trajectory features from a single item's scaling data."""
    raw = np.array([row[f"conf_{s}"] for s in scales])
    n = len(raw)
    x = np.arange(n)

    features = {}

    # Basic
    features["mean_conf"] = np.mean(raw)
    features["max_conf"] = np.max(raw)
    features["min_conf"] = np.min(raw)
    features["conf_range"] = np.max(raw) - np.min(raw)
    features["conf_6.9b"] = row["conf_6.9b"]
    features["conf_12b"] = row["conf_12b"]

    # Trajectory shape
    slope, intercept = np.polyfit(x, raw, 1)
    features["slope"] = slope

    poly2 = np.polyfit(x, raw, 2)
    features["curvature"] = poly2[0]  # quadratic coefficient

    mono_r, _ = spearmanr(x, raw)
    features["monotonicity"] = mono_r if not np.isnan(mono_r) else 0.0

    # Stability
    features["variance"] = np.var(raw)
    features["std"] = np.std(raw)
    diffs = np.diff(raw)
    features["max_jump"] = np.max(np.abs(diffs))
    features["mean_abs_jump"] = np.mean(np.abs(diffs))

    # Late-stage behavior (2.8B, 6.9B, 12B)
    late_raw = raw[4:]  # last 3 points
    if len(late_raw) >= 2:
        late_slope, _ = np.polyfit(np.arange(len(late_raw)), late_raw, 1)
        features["late_slope"] = late_slope
    else:
        features["late_slope"] = 0.0

    # Early-stage behavior (160M, 410M, 1B)
    early_raw = raw[:3]
    if len(early_raw) >= 2:
        early_slope, _ = np.polyfit(np.arange(len(early_raw)), early_raw, 1)
        features["early_slope"] = early_slope
    else:
        features["early_slope"] = 0.0

    # Non-monotonicity signals
    features["n_reversals"] = count_reversals(raw)
    early_mean = np.mean(raw[:3])
    late_mean = np.mean(raw[4:])
    features["early_late_ratio"] = early_mean / (late_mean + 1e-10)
    features["early_late_diff"] = early_mean - late_mean
    features["peak_position"] = int(np.argmax(raw))
    features["trough_position"] = int(np.argmin(raw))

    # Residual variance (deviation from linear fit)
    linear_fit = slope * x + intercept
    features["residual_variance"] = np.var(raw - linear_fit)

    # Ratio-based features (practical detector)
    features["ratio_160m_6.9b"] = row["conf_160m"] / (row["conf_6.9b"] + 1e-10)
    features["ratio_160m_12b"] = row["conf_160m"] / (row["conf_12b"] + 1e-10)
    features["diff_160m_6.9b"] = row["conf_160m"] - row["conf_6.9b"]
    features["diff_160m_12b"] = row["conf_160m"] - row["conf_12b"]

    return features


# ===================================================================
# Step 3: Classification helpers
# ===================================================================

def run_loo_classification(X, y, name="model"):
    """Leave-one-out cross-validated classification."""
    loo = LeaveOneOut()
    y_pred_proba = np.zeros(len(y))

    scaler = StandardScaler()

    for train_idx, test_idx in loo.split(X):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
        clf.fit(X_train, y[train_idx])
        y_pred_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y, y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)
    acc = accuracy_score(y, y_pred)

    return {"auc": auc, "accuracy": acc, "y_pred_proba": y_pred_proba, "name": name}


def run_stratified_cv(X, y, name="model", n_splits=5):
    """Stratified k-fold for more stable estimate (backup for LOO)."""
    n_splits = min(n_splits, min(np.bincount(y)))
    if n_splits < 2:
        return run_loo_classification(X, y, name)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_proba = np.zeros(len(y))
    scaler = StandardScaler()

    for train_idx, test_idx in skf.split(X, y):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
        clf.fit(X_train, y[train_idx])
        y_pred_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y, y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)
    acc = accuracy_score(y, y_pred)

    return {"auc": auc, "accuracy": acc, "y_pred_proba": y_pred_proba, "name": name}


# ===================================================================
# Experiment T1: Trajectory vs Single-Point
# ===================================================================

def run_t1(df, feat_df):
    """Compare trajectory features vs single-point confidence for classification."""
    print("\n" + "=" * 70)
    print("T1: TRAJECTORY vs SINGLE-POINT CLASSIFICATION")
    print("=" * 70)

    y = df["is_true"].values

    # Baseline: single-model confidence at each scale
    baselines = {}
    for size in SCALES:
        col = f"conf_{size}"
        X_single = df[[col]].values
        result = run_loo_classification(X_single, y, name=f"single_{size}")
        baselines[size] = result
        print(f"  Single {size:>5s}: AUC={result['auc']:.3f}, Acc={result['accuracy']:.1%}")

    # Trajectory: all engineered features
    trajectory_cols = [c for c in feat_df.columns if c not in
                       ["item_id", "text", "is_true", "category", "source", "pair_id"]
                       and not c.startswith("conf_")]
    X_trajectory = feat_df[trajectory_cols].values
    traj_result = run_loo_classification(X_trajectory, y, name="trajectory_all")
    print(f"\n  Trajectory (all features): AUC={traj_result['auc']:.3f}, "
          f"Acc={traj_result['accuracy']:.1%}")

    # Trajectory subset: shape-only features (no absolute confidence)
    shape_cols = ["slope", "curvature", "monotonicity", "variance", "max_jump",
                  "late_slope", "early_slope", "n_reversals", "early_late_ratio",
                  "early_late_diff", "residual_variance"]
    shape_cols = [c for c in shape_cols if c in feat_df.columns]
    X_shape = feat_df[shape_cols].values
    shape_result = run_loo_classification(X_shape, y, name="trajectory_shape")
    print(f"  Trajectory (shape only): AUC={shape_result['auc']:.3f}, "
          f"Acc={shape_result['accuracy']:.1%}")

    # Combined: best single-point + trajectory shape
    combined_cols = ["conf_6.9b"] + shape_cols
    combined_cols = [c for c in combined_cols if c in feat_df.columns]
    X_combined = feat_df[combined_cols].values
    combined_result = run_loo_classification(X_combined, y, name="combined")
    print(f"  Combined (6.9B + shape): AUC={combined_result['auc']:.3f}, "
          f"Acc={combined_result['accuracy']:.1%}")

    # Improvement
    best_single_auc = max(b["auc"] for b in baselines.values())
    best_single_size = max(baselines, key=lambda s: baselines[s]["auc"])
    delta = traj_result["auc"] - best_single_auc
    print(f"\n  Best single-point: {best_single_size} (AUC={best_single_auc:.3f})")
    print(f"  Trajectory improvement: {delta:+.3f} AUC")

    results = {
        "baselines": {s: {"auc": r["auc"], "accuracy": r["accuracy"]}
                      for s, r in baselines.items()},
        "trajectory_all": {"auc": traj_result["auc"], "accuracy": traj_result["accuracy"]},
        "trajectory_shape": {"auc": shape_result["auc"], "accuracy": shape_result["accuracy"]},
        "combined": {"auc": combined_result["auc"], "accuracy": combined_result["accuracy"]},
        "improvement": delta,
        "best_single": best_single_size,
    }

    # --- Plot: AUC comparison ---
    _plot_t1_auc(baselines, traj_result, shape_result, combined_result)
    _plot_t1_roc(baselines, traj_result, shape_result, combined_result, y)

    return results


def _plot_t1_auc(baselines, traj_result, shape_result, combined_result):
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(12, 6))

    names = []
    aucs = []
    colors = []

    for size in SCALES:
        names.append(f"Single\n{model_display_name(size)}")
        aucs.append(baselines[size]["auc"])
        colors.append(MODEL_COLORS.get(size, "#999"))

    names.append("Shape\nOnly")
    aucs.append(shape_result["auc"])
    colors.append("#2196F3")

    names.append("All\nTrajectory")
    aucs.append(traj_result["auc"])
    colors.append("#4CAF50")

    names.append("Combined\n(6.9B+shape)")
    aucs.append(combined_result["auc"])
    colors.append("#FF9800")

    x = np.arange(len(names))
    bars = ax.bar(x, aucs, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("LOO AUC", fontsize=12)
    ax.set_title("T1: Trajectory vs Single-Point Truth Detection", fontsize=14)
    ax.set_ylim(0.4, max(aucs) + 0.05)

    for i, v in enumerate(aucs):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "t1_auc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_t1_roc(baselines, traj_result, shape_result, combined_result, y):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Best single-point
    best_size = max(baselines, key=lambda s: baselines[s]["auc"])
    fpr, tpr, _ = roc_curve(y, baselines[best_size]["y_pred_proba"])
    ax.plot(fpr, tpr, color=MODEL_COLORS.get(best_size, "#999"), linewidth=2,
            label=f"Best single ({model_display_name(best_size)}, "
                  f"AUC={baselines[best_size]['auc']:.3f})")

    # Trajectory
    fpr, tpr, _ = roc_curve(y, traj_result["y_pred_proba"])
    ax.plot(fpr, tpr, color="#4CAF50", linewidth=2.5,
            label=f"Trajectory (AUC={traj_result['auc']:.3f})")

    # Shape only
    fpr, tpr, _ = roc_curve(y, shape_result["y_pred_proba"])
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2, linestyle="--",
            label=f"Shape only (AUC={shape_result['auc']:.3f})")

    # Combined
    fpr, tpr, _ = roc_curve(y, combined_result["y_pred_proba"])
    ax.plot(fpr, tpr, color="#FF9800", linewidth=2.5,
            label=f"Combined (AUC={combined_result['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("T1: ROC Curves — Truth Detection", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "t1_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Experiment T2: Hard Case Analysis
# ===================================================================

def run_t2(df, feat_df):
    """Analyze cases where 6.9B confidence is misleading."""
    print("\n" + "=" * 70)
    print("T2: HARD CASE ANALYSIS")
    print("=" * 70)

    y = df["is_true"].values

    # Median confidence at 6.9B
    median_conf = df["conf_6.9b"].median()
    print(f"  Median 6.9B confidence: {median_conf:.4f}")

    # Hard cases: false claims with ABOVE-median confidence at 6.9B
    hard_mask = (df["is_true"] == 0) & (df["conf_6.9b"] > median_conf)
    hard_cases = df[hard_mask].copy()
    n_hard = len(hard_cases)
    print(f"  Hard cases (false + high confidence): {n_hard}")

    if n_hard == 0:
        print("  No hard cases found!")
        return {"n_hard": 0}

    # Single-point would classify these as TRUE (above median)
    # How many does trajectory correctly classify as FALSE?

    # Run full trajectory classifier
    trajectory_cols = [c for c in feat_df.columns if c not in
                       ["item_id", "text", "is_true", "category", "source", "pair_id"]
                       and not c.startswith("conf_")]
    X_all = feat_df[trajectory_cols].values
    scaler = StandardScaler()

    # LOO: get predictions for each item
    loo = LeaveOneOut()
    y_pred_proba = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X_all):
        X_train = scaler.fit_transform(X_all[train_idx])
        X_test = scaler.transform(X_all[test_idx])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
        clf.fit(X_train, y[train_idx])
        y_pred_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    # How many hard cases caught by trajectory?
    hard_indices = df.index[hard_mask].tolist()
    hard_caught = sum(1 for i in hard_indices if y_pred_proba[i] < 0.5)
    catch_rate = hard_caught / n_hard if n_hard > 0 else 0

    print(f"  Trajectory correctly reclassified: {hard_caught}/{n_hard} ({catch_rate:.1%})")

    # Details for each hard case
    hard_details = []
    for idx in hard_indices:
        row = df.iloc[idx]
        detail = {
            "item_id": row["item_id"],
            "text": row["text"][:80],
            "category": row["category"],
            "conf_6.9b": float(row["conf_6.9b"]),
            "trajectory_pred": float(y_pred_proba[idx]),
            "caught": y_pred_proba[idx] < 0.5,
        }
        hard_details.append(detail)

    print(f"\n  Hard case details:")
    for d in sorted(hard_details, key=lambda x: x["trajectory_pred"]):
        caught_str = "CAUGHT" if d["caught"] else "MISSED"
        print(f"    [{caught_str}] {d['item_id']:<40s} "
              f"6.9B={d['conf_6.9b']:.4f} traj_p={d['trajectory_pred']:.3f}")

    results = {
        "n_hard": n_hard,
        "n_caught": hard_caught,
        "catch_rate": catch_rate,
        "median_conf_6.9b": float(median_conf),
        "details": hard_details,
    }

    # --- Plot: Hard case trajectories ---
    _plot_t2_trajectories(df, hard_indices, y_pred_proba)

    return results


def _plot_t2_trajectories(df, hard_indices, y_pred_proba):
    """Plot scaling trajectories for hard cases."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    params = [PARAM_COUNTS[s] for s in SCALES]

    # Left: caught hard cases (trajectory correctly identifies as false)
    ax = axes[0]
    caught = [i for i in hard_indices if y_pred_proba[i] < 0.5]
    for idx in caught:
        row = df.iloc[idx]
        confs = [row[f"conf_{s}"] for s in SCALES]
        label = row["item_id"].replace("truth_", "").replace("med_", "").replace("mandela_", "")
        ax.plot(params, confs, "o-", linewidth=1.5, markersize=5, alpha=0.7,
                label=label[:25])
    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=11)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title(f"Hard Cases: CAUGHT by trajectory ({len(caught)})", fontsize=12,
                 color="#4CAF50")
    if caught:
        ax.legend(fontsize=6, loc="best", ncol=2)

    # Right: missed hard cases
    ax = axes[1]
    missed = [i for i in hard_indices if y_pred_proba[i] >= 0.5]
    for idx in missed:
        row = df.iloc[idx]
        confs = [row[f"conf_{s}"] for s in SCALES]
        label = row["item_id"].replace("truth_", "").replace("med_", "").replace("mandela_", "")
        ax.plot(params, confs, "o-", linewidth=1.5, markersize=5, alpha=0.7,
                label=label[:25])
    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=11)
    ax.set_ylabel("Confidence", fontsize=11)
    ax.set_title(f"Hard Cases: MISSED ({len(missed)})", fontsize=12,
                 color="#F44336")
    if missed:
        ax.legend(fontsize=6, loc="best", ncol=2)

    fig.suptitle("T2: High-Confidence False Claims — Trajectory Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "t2_hard_case_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Experiment T3: Feature Importance
# ===================================================================

def run_t3(df, feat_df):
    """Determine which trajectory features matter most."""
    print("\n" + "=" * 70)
    print("T3: FEATURE IMPORTANCE")
    print("=" * 70)

    y = df["is_true"].values
    trajectory_cols = [c for c in feat_df.columns if c not in
                       ["item_id", "text", "is_true", "category", "source", "pair_id"]
                       and not c.startswith("conf_")]

    X = feat_df[trajectory_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit on full data for coefficient inspection
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
    clf.fit(X_scaled, y)

    coefs = pd.DataFrame({
        "feature": trajectory_cols,
        "coefficient": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort_values("abs_coef", ascending=False)

    print(f"\n  Feature coefficients (positive = predicts TRUE):")
    for _, row in coefs.iterrows():
        sign = "+" if row["coefficient"] > 0 else "-"
        print(f"    {sign} {row['feature']:<25s} {row['coefficient']:+8.4f}")

    # Permutation importance
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(clf, X_scaled, y, n_repeats=30, random_state=42)
    perm_df = pd.DataFrame({
        "feature": trajectory_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print(f"\n  Permutation importance:")
    for _, row in perm_df.head(10).iterrows():
        print(f"    {row['feature']:<25s} {row['importance_mean']:+.4f} "
              f"(+/- {row['importance_std']:.4f})")

    # Single-feature AUCs
    print(f"\n  Single-feature LOO AUCs:")
    single_aucs = {}
    for col in trajectory_cols:
        X_single = feat_df[[col]].values
        result = run_loo_classification(X_single, y, name=col)
        single_aucs[col] = result["auc"]

    sorted_aucs = sorted(single_aucs.items(), key=lambda x: x[1], reverse=True)
    for name, auc in sorted_aucs[:10]:
        print(f"    {name:<25s} AUC={auc:.3f}")

    results = {
        "coefficients": coefs.to_dict("records"),
        "permutation_importance": perm_df.to_dict("records"),
        "single_feature_aucs": single_aucs,
    }

    # --- Plot ---
    _plot_t3_importance(coefs, perm_df, single_aucs)

    return results


def _plot_t3_importance(coefs, perm_df, single_aucs):
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Left: Logistic regression coefficients
    ax = axes[0]
    top = coefs.head(12)
    colors = ["#4CAF50" if c > 0 else "#F44336" for c in top["coefficient"]]
    ax.barh(range(len(top)), top["coefficient"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=9)
    ax.set_xlabel("Coefficient")
    ax.set_title("Logistic Regression Coefficients\n(+ = predicts TRUE)")
    ax.invert_yaxis()

    # Middle: Permutation importance
    ax = axes[1]
    top_perm = perm_df.head(12)
    ax.barh(range(len(top_perm)), top_perm["importance_mean"],
            xerr=top_perm["importance_std"], color="#2196F3", alpha=0.8)
    ax.set_yticks(range(len(top_perm)))
    ax.set_yticklabels(top_perm["feature"], fontsize=9)
    ax.set_xlabel("Permutation Importance")
    ax.set_title("Permutation Importance")
    ax.invert_yaxis()

    # Right: Single-feature AUCs
    ax = axes[2]
    sorted_items = sorted(single_aucs.items(), key=lambda x: x[1], reverse=True)[:12]
    names = [x[0] for x in sorted_items]
    vals = [x[1] for x in sorted_items]
    ax.barh(range(len(names)), vals, color="#FF9800", alpha=0.8)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LOO AUC")
    ax.set_title("Single-Feature AUCs")
    ax.invert_yaxis()

    fig.suptitle("T3: Feature Importance Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "t3_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Experiment T4: Minimal Trajectory (2-model detector)
# ===================================================================

def run_t4(df, feat_df):
    """Test all 2-model pairs to find the practical minimal detector."""
    print("\n" + "=" * 70)
    print("T4: MINIMAL TRAJECTORY — 2-MODEL DETECTOR")
    print("=" * 70)

    y = df["is_true"].values

    # Baseline: best single model
    best_single_auc = 0
    best_single_size = ""
    for size in SCALES:
        X_single = df[[f"conf_{size}"]].values
        r = run_loo_classification(X_single, y, name=size)
        if r["auc"] > best_single_auc:
            best_single_auc = r["auc"]
            best_single_size = size

    print(f"  Best single model: {best_single_size} (AUC={best_single_auc:.3f})")

    # Test all pairs
    pair_results = {}
    print(f"\n  2-Model pair AUCs:")
    print(f"  {'Small':<8s} {'Large':<8s} {'AUC':<8s} {'vs best single':>15s}")
    print(f"  {'-'*42}")

    for i, small in enumerate(SCALES):
        for large in SCALES[i+1:]:
            # Features: ratio, difference, both confidences
            X_pair = np.column_stack([
                df[f"conf_{small}"].values,
                df[f"conf_{large}"].values,
                df[f"conf_{small}"].values / (df[f"conf_{large}"].values + 1e-10),
                df[f"conf_{small}"].values - df[f"conf_{large}"].values,
            ])

            r = run_loo_classification(X_pair, y, name=f"{small}+{large}")
            pair_results[(small, large)] = r["auc"]
            delta = r["auc"] - best_single_auc
            marker = "**" if delta > 0.02 else "*" if delta > 0 else ""
            print(f"  {small:<8s} {large:<8s} {r['auc']:<8.3f} {delta:+.3f} {marker}")

    # Best pair
    best_pair = max(pair_results, key=pair_results.get)
    best_pair_auc = pair_results[best_pair]
    print(f"\n  Best pair: {best_pair[0]} + {best_pair[1]} "
          f"(AUC={best_pair_auc:.3f}, delta={best_pair_auc - best_single_auc:+.3f})")

    # Simple ratio detector: just small/large ratio
    print(f"\n  Simple ratio detectors (1 feature):")
    ratio_results = {}
    for small in SCALES[:3]:  # 160m, 410m, 1b
        for large in SCALES[4:]:  # 2.8b, 6.9b, 12b
            ratio = df[f"conf_{small}"].values / (df[f"conf_{large}"].values + 1e-10)
            X_ratio = ratio.reshape(-1, 1)
            r = run_loo_classification(X_ratio, y, name=f"ratio_{small}/{large}")
            ratio_results[(small, large)] = r["auc"]
            delta = r["auc"] - best_single_auc
            print(f"    {small}/{large}: AUC={r['auc']:.3f} ({delta:+.3f})")

    results = {
        "best_single": {"size": best_single_size, "auc": best_single_auc},
        "pair_results": {f"{k[0]}+{k[1]}": v for k, v in pair_results.items()},
        "best_pair": {"small": best_pair[0], "large": best_pair[1], "auc": best_pair_auc},
        "ratio_results": {f"{k[0]}/{k[1]}": v for k, v in ratio_results.items()},
    }

    # --- Plot: Heatmap of pair AUCs ---
    _plot_t4_heatmap(pair_results, best_single_auc)

    return results


def _plot_t4_heatmap(pair_results, best_single_auc):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 8))

    # Build matrix
    n = len(SCALES)
    matrix = np.full((n, n), np.nan)
    for (s1, s2), auc in pair_results.items():
        i = SCALES.index(s1)
        j = SCALES.index(s2)
        matrix[i, j] = auc

    labels = [model_display_name(s) for s in SCALES]
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=labels, yticklabels=labels,
                ax=ax, vmin=0.45, vmax=max(pair_results.values()) + 0.02,
                linewidths=0.5, linecolor="white",
                mask=np.isnan(matrix))

    ax.set_xlabel("Larger Model", fontsize=11)
    ax.set_ylabel("Smaller Model", fontsize=11)
    ax.set_title(f"T4: 2-Model Pair AUCs (best single={best_single_auc:.3f})", fontsize=13)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "t4_minimal_pair_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Experiment T5: Category-Specific Analysis
# ===================================================================

def run_t5(df, feat_df):
    """Run trajectory analysis within each claim category."""
    print("\n" + "=" * 70)
    print("T5: CATEGORY-SPECIFIC ANALYSIS")
    print("=" * 70)

    y_all = df["is_true"].values
    trajectory_cols = [c for c in feat_df.columns if c not in
                       ["item_id", "text", "is_true", "category", "source", "pair_id"]
                       and not c.startswith("conf_")]

    categories = df["category"].unique()
    cat_results = {}

    for cat in sorted(categories):
        mask = df["category"] == cat
        df_cat = df[mask]
        feat_cat = feat_df[mask]
        y_cat = df_cat["is_true"].values

        n_true = (y_cat == 1).sum()
        n_false = (y_cat == 0).sum()
        print(f"\n  {cat}: {len(df_cat)} items (true={n_true}, false={n_false})")

        if n_true < 2 or n_false < 2:
            print(f"    SKIPPED (insufficient class balance)")
            cat_results[cat] = {"n": len(df_cat), "skipped": True}
            continue

        # Single-point
        X_single = df_cat[["conf_6.9b"]].values
        single_r = run_loo_classification(X_single, y_cat, name=f"{cat}_single")
        print(f"    Single 6.9B: AUC={single_r['auc']:.3f}")

        # Trajectory
        X_traj = feat_cat[trajectory_cols].values
        traj_r = run_loo_classification(X_traj, y_cat, name=f"{cat}_traj")
        print(f"    Trajectory:  AUC={traj_r['auc']:.3f}")

        delta = traj_r["auc"] - single_r["auc"]
        print(f"    Improvement: {delta:+.3f}")

        cat_results[cat] = {
            "n": len(df_cat),
            "n_true": int(n_true),
            "n_false": int(n_false),
            "single_auc": single_r["auc"],
            "trajectory_auc": traj_r["auc"],
            "improvement": delta,
        }

    # --- Plot ---
    _plot_t5_categories(cat_results)

    return cat_results


def _plot_t5_categories(cat_results):
    sns.set_theme(style="whitegrid", palette="muted")

    cats = [c for c, r in cat_results.items() if not r.get("skipped")]
    if not cats:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(cats))
    width = 0.35

    single_aucs = [cat_results[c]["single_auc"] for c in cats]
    traj_aucs = [cat_results[c]["trajectory_auc"] for c in cats]

    ax.bar(x - width/2, single_aucs, width, label="Single (6.9B)",
           color="#F44336", alpha=0.8)
    ax.bar(x + width/2, traj_aucs, width, label="Trajectory",
           color="#4CAF50", alpha=0.8)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={cat_results[c]['n']})" for c in cats], fontsize=9)
    ax.set_ylabel("LOO AUC", fontsize=12)
    ax.set_title("T5: Category-Specific Performance", fontsize=14)
    ax.legend(fontsize=10)

    for i, (s, t) in enumerate(zip(single_aucs, traj_aucs)):
        delta = t - s
        ax.text(i, max(s, t) + 0.01, f"{delta:+.3f}", ha="center", fontsize=8,
                color="#4CAF50" if delta > 0 else "#F44336")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "t5_category_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Example trajectory visualization
# ===================================================================

def plot_example_trajectories(df):
    """Plot example scaling trajectories: true vs false."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    params = [PARAM_COUNTS[s] for s in SCALES]

    # Sample up to 15 true and 15 false items
    true_items = df[df["is_true"] == 1].sample(n=min(15, len(df[df["is_true"] == 1])),
                                                random_state=42)
    false_items = df[df["is_true"] == 0].sample(n=min(15, len(df[df["is_true"] == 0])),
                                                  random_state=42)

    # Left: True claims
    ax = axes[0]
    for _, row in true_items.iterrows():
        confs = [row[f"conf_{s}"] for s in SCALES]
        ax.plot(params, confs, "o-", linewidth=1, markersize=3, alpha=0.5, color="#4CAF50")
    # Mean trajectory
    all_true_confs = np.array([[row[f"conf_{s}"] for s in SCALES]
                               for _, row in df[df["is_true"] == 1].iterrows()])
    mean_true = np.mean(all_true_confs, axis=0)
    ax.plot(params, mean_true, "o-", linewidth=3, markersize=8, color="#1B5E20",
            label=f"Mean (n={len(all_true_confs)})")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Confidence")
    ax.set_title("True Claims", fontsize=13, color="#4CAF50")
    ax.legend()

    # Right: False claims
    ax = axes[1]
    for _, row in false_items.iterrows():
        confs = [row[f"conf_{s}"] for s in SCALES]
        ax.plot(params, confs, "o-", linewidth=1, markersize=3, alpha=0.5, color="#F44336")
    all_false_confs = np.array([[row[f"conf_{s}"] for s in SCALES]
                                for _, row in df[df["is_true"] == 0].iterrows()])
    mean_false = np.mean(all_false_confs, axis=0)
    ax.plot(params, mean_false, "o-", linewidth=3, markersize=8, color="#B71C1C",
            label=f"Mean (n={len(all_false_confs)})")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Confidence")
    ax.set_title("False Claims", fontsize=13, color="#F44336")
    ax.legend()

    fig.suptitle("Scaling Trajectories: True vs False Claims", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "example_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def run_experiment():
    total_start = time.time()

    print("=" * 70)
    print("TRAJECTORY-BASED HALLUCINATION DETECTION")
    print("=" * 70)

    # Step 1: Load data
    print("\n[Step 1] Loading all items...")
    df = load_all_items()

    # Save master dataset
    df.to_csv(RESULTS_DIR / "all_items_with_trajectories.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'all_items_with_trajectories.csv'}")

    # Step 2: Extract features
    print("\n[Step 2] Extracting trajectory features...")
    feat_rows = []
    for idx, row in df.iterrows():
        feats = extract_trajectory_features(row)
        feats.update(row.to_dict())
        feat_rows.append(feats)

    feat_df = pd.DataFrame(feat_rows)
    print(f"  Features extracted: {len([c for c in feat_df.columns if c not in df.columns])} "
          f"trajectory features per item")

    # Example trajectories
    print("\n[Viz] Plotting example trajectories...")
    plot_example_trajectories(df)

    # Step 3: Run experiments
    t1_results = run_t1(df, feat_df)
    t2_results = run_t2(df, feat_df)
    t3_results = run_t3(df, feat_df)
    t4_results = run_t4(df, feat_df)
    t5_results = run_t5(df, feat_df)

    # ===================================================================
    # Grand Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("TRAJECTORY DETECTOR — GRAND SUMMARY")
    print("=" * 70)

    best_single = t1_results["baselines"][t1_results["best_single"]]["auc"]
    traj_auc = t1_results["trajectory_all"]["auc"]
    shape_auc = t1_results["trajectory_shape"]["auc"]
    combined_auc = t1_results["combined"]["auc"]

    print(f"\n  T1: Classification")
    print(f"      Best single-point:  AUC = {best_single:.3f}")
    print(f"      Shape features:     AUC = {shape_auc:.3f} ({shape_auc - best_single:+.3f})")
    print(f"      All trajectory:     AUC = {traj_auc:.3f} ({traj_auc - best_single:+.3f})")
    print(f"      Combined:           AUC = {combined_auc:.3f} ({combined_auc - best_single:+.3f})")

    print(f"\n  T2: Hard Cases")
    print(f"      {t2_results.get('n_caught', 0)}/{t2_results.get('n_hard', 0)} "
          f"high-confidence false claims caught ({t2_results.get('catch_rate', 0):.1%})")

    if not t4_results.get("best_pair", {}).get("auc"):
        print(f"\n  T4: Best 2-model pair: N/A")
    else:
        bp = t4_results["best_pair"]
        print(f"\n  T4: Best 2-model pair: {bp['small']} + {bp['large']} "
              f"(AUC={bp['auc']:.3f})")

    # Verdicts
    improvement = traj_auc - best_single
    print(f"\n  {'='*60}")
    if improvement > 0.05:
        print(f"  STRONG: Trajectory beats single-point by {improvement:+.3f} AUC")
    elif improvement > 0:
        print(f"  MINIMUM VIABLE: Trajectory improves by {improvement:+.3f} AUC")
    else:
        print(f"  NULL: Trajectory does not beat single-point ({improvement:+.3f})")

    catch_rate = t2_results.get("catch_rate", 0)
    if catch_rate > 0.3:
        print(f"  STRONG: {catch_rate:.0%} of hard cases caught")
    elif catch_rate > 0:
        print(f"  WEAK: Only {catch_rate:.0%} of hard cases caught")
    else:
        print(f"  NULL: No hard cases caught")

    # Save all results
    all_results = {
        "t1_classification": {k: v for k, v in t1_results.items()
                              if k != "baselines" or True},
        "t2_hard_cases": {k: v for k, v in t2_results.items() if k != "details"},
        "t3_feature_importance": {
            "top_features": t3_results.get("coefficients", [])[:5],
            "top_single_aucs": dict(sorted(t3_results.get("single_feature_aucs", {}).items(),
                                           key=lambda x: x[1], reverse=True)[:5]),
        },
        "t4_minimal_pairs": t4_results,
        "t5_categories": t5_results,
    }

    # Clean for JSON serialization
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_clean(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = RESULTS_DIR / "trajectory_detector_results.json"
    with open(results_path, "w") as f:
        json.dump(_clean(all_results), f, indent=2)
    print(f"\n  Results saved: {results_path}")

    total_time = time.time() - total_start
    fig_count = len(list(FIGURES_DIR.glob("*.png")))
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    run_experiment()
