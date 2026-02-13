"""
Experiment B: Shape Analysis
=============================
Go beyond mean confidence — extract shape features from confidence curves
and test whether richer features improve classification.

Uses existing Phase 1 data (no model inference needed).

B1: Feature extraction from confidence traces
B2: Shape-based classification (compare to mean-only)
B3: Unsupervised clustering
"""

import sys
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import load_records, ConfidenceRecord
from src.utils import RESULTS_DIR, SHAPES_RESULTS_DIR, SHAPES_FIGURES_DIR


# ---------------------------------------------------------------------------
# B1: Shape Feature Extraction
# ---------------------------------------------------------------------------

def extract_shape_features(record: ConfidenceRecord) -> dict:
    """Extract shape features from a single confidence trace."""
    probs = np.array([t.top1_prob for t in record.tokens])
    ents = np.array([t.entropy for t in record.tokens])

    if len(probs) < 3:
        # Too short for meaningful shape analysis
        return None

    # Consecutive differences
    diffs = np.diff(probs)
    ent_diffs = np.diff(ents)

    # --- Volatility features ---
    features = {
        "confidence_variance": float(np.var(probs)),
        "confidence_std": float(np.std(probs)),
        "entropy_variance": float(np.var(ents)),
        "max_confidence_drop": float(np.min(diffs)) if len(diffs) > 0 else 0.0,
        "max_confidence_spike": float(np.max(diffs)) if len(diffs) > 0 else 0.0,
        "n_transitions": int(np.sum(np.abs(diffs) > 0.15)),
        "mean_abs_delta": float(np.mean(np.abs(diffs))) if len(diffs) > 0 else 0.0,
    }

    # --- Shape features ---
    positions = np.arange(len(probs))
    if len(probs) >= 2:
        slope, intercept, r_val, p_val, std_err = stats.linregress(positions, probs)
        features["confidence_slope"] = float(slope)
        features["slope_r_squared"] = float(r_val ** 2)
    else:
        features["confidence_slope"] = 0.0
        features["slope_r_squared"] = 0.0

    # Head vs tail
    n = len(probs)
    k = max(1, n // 3)
    features["head_confidence"] = float(np.mean(probs[:k]))
    features["tail_confidence"] = float(np.mean(probs[-k:]))
    features["head_tail_ratio"] = (features["tail_confidence"] /
                                    (features["head_confidence"] + 1e-10))

    # --- Recovery features ---
    min_pos = int(np.argmin(probs))
    features["min_pos_relative"] = float(min_pos / max(1, n - 1))
    if min_pos < n - 1:
        recovery = float(np.mean(probs[min_pos + 1:]) - probs[min_pos])
        features["recovery_after_min"] = recovery
    else:
        features["recovery_after_min"] = 0.0

    # Recovery after drops > 1 std
    threshold = np.std(probs)
    drop_positions = np.where(diffs < -threshold)[0]
    recoveries = []
    for dp in drop_positions:
        if dp + 1 < len(probs):
            rec = probs[dp + 1] - probs[dp]  # how much it bounces back
            recoveries.append(rec)
    features["mean_drop_recovery"] = float(np.mean(recoveries)) if recoveries else 0.0

    # --- Distribution features ---
    features["entropy_confidence_corr"] = float(
        np.corrcoef(probs, ents)[0, 1]) if len(probs) > 2 else 0.0

    # Top-5 spread: how concentrated is the model's prediction?
    top5_spreads = []
    for t in record.tokens:
        if len(t.top5_probs) >= 2:
            top5_spreads.append(t.top5_probs[0] - t.top5_probs[-1])
    features["mean_top5_spread"] = float(np.mean(top5_spreads)) if top5_spreads else 0.0

    # Skew and kurtosis
    if len(probs) >= 4:
        features["confidence_skew"] = float(stats.skew(probs))
        features["confidence_kurtosis"] = float(stats.kurtosis(probs))
    else:
        features["confidence_skew"] = 0.0
        features["confidence_kurtosis"] = 0.0

    # --- Existing summary features (for comparison) ---
    features["mean_confidence"] = float(record.mean_top1_prob)
    features["mean_entropy"] = float(record.mean_entropy)

    return features


# ---------------------------------------------------------------------------
# B2: Classification with shape features
# ---------------------------------------------------------------------------

def run_classification_comparison(df: pd.DataFrame, y: np.ndarray,
                                  task_name: str):
    """Compare mean-only vs shape-only vs combined classifiers."""
    results = {}
    scaler = StandardScaler()

    # Define feature sets
    mean_features = ["mean_confidence", "mean_entropy"]
    shape_features = [c for c in df.columns
                      if c not in mean_features + ["label", "category"]]
    all_features = mean_features + shape_features

    clf = LogisticRegression(random_state=42, max_iter=2000)

    for name, cols in [("Mean-only", mean_features),
                       ("Shape-only", shape_features),
                       ("Combined", all_features)]:
        X = scaler.fit_transform(df[cols].values)
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        results[name] = {
            "accuracy": scores.mean(),
            "std": scores.std(),
            "features": cols,
        }
        print(f"  {name:>12s}: {scores.mean():.1%} +/- {scores.std():.1%} "
              f"({len(cols)} features)")

    # Feature importance from combined model
    X_all = scaler.fit_transform(df[all_features].values)
    clf.fit(X_all, y)
    importances = pd.Series(np.abs(clf.coef_[0]) if clf.coef_.ndim > 1
                            else np.abs(clf.coef_),
                            index=all_features)
    importances = importances.sort_values(ascending=False)

    return results, importances


# ---------------------------------------------------------------------------
# B3: Unsupervised clustering
# ---------------------------------------------------------------------------

def run_clustering(df: pd.DataFrame, categories: list[str]):
    """K-means clustering on shape features, compare to true categories."""
    shape_cols = [c for c in df.columns
                  if c not in ["label", "category", "mean_confidence", "mean_entropy"]]
    X = StandardScaler().fit_transform(df[shape_cols].values)

    # Try different k values
    silhouettes = {}
    for k in range(2, min(6, len(X))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        silhouettes[k] = sil
        print(f"  k={k}: silhouette={sil:.3f}")

    # Best k
    best_k = max(silhouettes, key=silhouettes.get)
    print(f"  Best k={best_k} (silhouette={silhouettes[best_k]:.3f})")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X)

    # PCA projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    return X_pca, cluster_labels, silhouettes, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_feature_distributions(df: pd.DataFrame, categories: list[str],
                               save_path: Path):
    """Violin plots of top shape features by category."""
    # Select most interesting features
    feat_cols = ["confidence_variance", "confidence_slope", "recovery_after_min",
                 "max_confidence_drop", "n_transitions", "entropy_variance",
                 "head_tail_ratio", "confidence_kurtosis"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    n = len(feat_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(feat_cols):
        ax = axes[i]
        for cat in categories:
            vals = df[df["category"] == cat][col].values
            parts = ax.violinplot([vals], positions=[categories.index(cat)],
                                   showmeans=True)
            for pc in parts["bodies"]:
                pc.set_alpha(0.6)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([c[:10] for c in categories], fontsize=8, rotation=30)
        ax.set_title(col, fontsize=10)

    for i in range(len(feat_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Shape Feature Distributions by Category", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_classifier_comparison(results: dict, task_name: str, save_path: Path):
    """Bar chart comparing classifier accuracies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    stds = [results[n]["std"] for n in names]

    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(names, accs, yerr=stds, color=colors[:len(names)],
                  edgecolor="white", capsize=5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.1%}", ha="center", fontsize=11)

    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title(f"Classifier Comparison: {task_name}")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Chance (50%)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, task_name: str,
                            save_path: Path, top_n: int = 15):
    """Horizontal bar chart of feature importance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    top = importances.head(top_n)
    colors = ["#F44336" if "mean" in f else "#2196F3" for f in top.index]

    ax.barh(range(len(top)), top.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("|Coefficient|")
    ax.set_title(f"Feature Importance: {task_name}")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#F44336", label="Mean features"),
                        Patch(color="#2196F3", label="Shape features")],
              fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clustering(X_pca, cluster_labels, true_categories, category_names,
                    var_explained, save_path: Path):
    """PCA projection colored by cluster and by true category (side by side)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: colored by cluster
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
                           cmap="Set1", s=60, alpha=0.7, edgecolors="white")
    ax1.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
    ax1.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")
    ax1.set_title("K-Means Clusters")
    ax1.legend(*scatter1.legend_elements(), title="Cluster", fontsize=8)

    # Right: colored by true category
    cat_to_num = {c: i for i, c in enumerate(category_names)}
    cat_nums = [cat_to_num[c] for c in true_categories]
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cat_nums,
                           cmap="Set2", s=60, alpha=0.7, edgecolors="white")
    ax2.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)")
    ax2.set_title("True Categories")
    handles = [plt.scatter([], [], c=plt.cm.Set2(cat_to_num[c] / len(category_names)),
               s=60, label=c) for c in category_names]
    ax2.legend(handles=handles, fontsize=8)

    fig.suptitle("Shape Feature Clustering vs True Categories", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette(silhouettes: dict, save_path: Path):
    """Silhouette score vs k."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(silhouettes.keys())
    scores = [silhouettes[k] for k in ks]
    ax.plot(ks, scores, "o-", linewidth=2, markersize=8, color="#2196F3")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Optimal Number of Clusters")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 65)
    print("EXPERIMENT B: Shape Analysis")
    print("=" * 65)
    start_time = time.time()

    # ===================================================================
    # B1: Feature extraction from Phase 1 data
    # ===================================================================
    print("\n--- B1: Shape Feature Extraction ---")

    # Load Phase 1 results
    exp2_records = load_records(RESULTS_DIR / "exp2_truth.jsonl")
    exp3_records = load_records(RESULTS_DIR / "exp3_contested.jsonl")

    print(f"  Loaded {len(exp2_records)} records from exp2 (truth)")
    print(f"  Loaded {len(exp3_records)} records from exp3 (contested)")

    # Extract features for all records
    all_features = []
    all_labels = []
    all_categories = []

    for rec in exp2_records:
        feats = extract_shape_features(rec)
        if feats is not None:
            feats["label"] = rec.label
            feats["category"] = rec.category
            all_features.append(feats)

    for rec in exp3_records:
        feats = extract_shape_features(rec)
        if feats is not None:
            feats["label"] = rec.label
            feats["category"] = rec.category
            all_features.append(feats)

    df = pd.DataFrame(all_features)
    df.to_csv(SHAPES_RESULTS_DIR / "shape_features.csv", index=False)
    print(f"  Extracted {len(df.columns) - 2} features for {len(df)} records")
    print(f"  Saved to {SHAPES_RESULTS_DIR / 'shape_features.csv'}")

    # Print feature summary
    feat_cols = [c for c in df.columns if c not in ["label", "category"]]
    print(f"\n  Shape features ({len(feat_cols)} total):")
    for col in feat_cols:
        print(f"    {col:<28s} mean={df[col].mean():.4f}  std={df[col].std():.4f}")

    # ===================================================================
    # B2: Classification comparison
    # ===================================================================
    print("\n--- B2: Shape-Based Classification ---")

    # Exp 2: True vs False classification
    print("\n  [Exp 2] True vs False:")
    df_exp2 = df[df["category"].isin(["true", "false"])].copy()
    if len(df_exp2) > 10:
        y_exp2 = (df_exp2["category"] == "true").astype(int).values
        feat_df2 = df_exp2.drop(columns=["label", "category"])
        results_exp2, imp_exp2 = run_classification_comparison(
            feat_df2.assign(label=df_exp2["label"].values,
                            category=df_exp2["category"].values),
            y_exp2, "Truth Detection")
        # Fix: need to pass just numeric columns
        results_exp2, imp_exp2 = run_classification_comparison(feat_df2, y_exp2,
                                                               "Truth Detection")

        plot_classifier_comparison(
            results_exp2, "Truth vs False",
            SHAPES_FIGURES_DIR / "b2_truth_classifier_comparison.png")
        plot_feature_importance(
            imp_exp2, "Truth Detection",
            SHAPES_FIGURES_DIR / "b2_truth_feature_importance.png")
    else:
        print(f"    Skipped: only {len(df_exp2)} records")

    # Exp 3: Settled vs Non-settled (binary)
    print("\n  [Exp 3] Settled vs Non-settled:")
    df_exp3 = df[df["category"].isin(["settled", "mostly_settled",
                                       "contested", "unknown"])].copy()
    if len(df_exp3) > 10:
        y_exp3 = (df_exp3["category"].isin(["settled", "mostly_settled"])).astype(int).values
        feat_df3 = df_exp3.drop(columns=["label", "category"])
        results_exp3, imp_exp3 = run_classification_comparison(
            feat_df3, y_exp3, "Knowledge Level")

        plot_classifier_comparison(
            results_exp3, "Settled vs Contested",
            SHAPES_FIGURES_DIR / "b2_contested_classifier_comparison.png")
        plot_feature_importance(
            imp_exp3, "Knowledge Level",
            SHAPES_FIGURES_DIR / "b2_contested_feature_importance.png")

    # ===================================================================
    # B3: Unsupervised clustering
    # ===================================================================
    print("\n--- B3: Unsupervised Clustering ---")

    # Use all records for clustering
    print("\n  Clustering all records by shape features:")
    feat_only = df.drop(columns=["label", "category"])
    X_pca, cluster_labels, silhouettes, var_explained = run_clustering(
        feat_only, df["category"].tolist())

    plot_silhouette(silhouettes,
                    SHAPES_FIGURES_DIR / "b3_silhouette.png")

    unique_cats = sorted(df["category"].unique())
    plot_clustering(X_pca, cluster_labels, df["category"].tolist(),
                    unique_cats, var_explained,
                    SHAPES_FIGURES_DIR / "b3_clustering_pca.png")

    # Feature distributions by category (exp2 true/false)
    if len(df_exp2) > 5:
        plot_feature_distributions(
            df_exp2.assign(category=df[df["category"].isin(["true", "false"])]["category"].values),
            ["true", "false"],
            SHAPES_FIGURES_DIR / "b1_truth_feature_distributions.png")

    # Feature distributions by category (exp3 knowledge levels)
    if len(df_exp3) > 5:
        plot_feature_distributions(
            df_exp3.assign(category=df[df["category"].isin(
                ["settled", "mostly_settled", "contested", "unknown"])]["category"].values),
            ["settled", "mostly_settled", "contested", "unknown"],
            SHAPES_FIGURES_DIR / "b1_contested_feature_distributions.png")

    # ===================================================================
    # Summary
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(SHAPES_FIGURES_DIR.glob("*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT B COMPLETE")
    print("=" * 65)
    print(f"  Features extracted: {len(feat_cols)}")
    print(f"  Records analyzed: {len(df)}")
    if len(df_exp2) > 10:
        print(f"  Truth classifier (combined): {results_exp2.get('Combined', {}).get('accuracy', 0):.1%}")
    if len(df_exp3) > 10:
        print(f"  Knowledge classifier (combined): {results_exp3.get('Combined', {}).get('accuracy', 0):.1%}")
    print(f"  Figures: {fig_count} plots saved")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    run_experiment()
