"""
Qwen2.5-32B Cross-Architecture Validation
============================================
Run the same truth, medical, and Mandela prompts through Qwen2.5-32B
(base model, not instruct) to test whether confidence patterns are
Pythia-specific or generalize across architectures.

Key questions:
  1. Does Qwen 32B show the same true > false confidence gap?
  2. Does the medical domain signal transfer?
  3. Does the Mandela calibration (model confidence ~ human prevalence) hold?

Model: Qwen/Qwen2.5-32B (base, 32.76B params, float16 on MPS)
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import torch

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text, unload_model
from src.schema import save_records, load_records, ConfidenceRecord

# Import prompts
from src.experiments.exp2_truth import PAIRS as TRUTH_PAIRS
from src.experiments.exp9_medical_validation import MEDICAL_PAIRS
from src.experiments.exp_mandela_expanded import (
    LINGUISTIC_ITEMS, _make_texts, filter_raw, _records_to_pairs,
)

# Model config
MODEL_NAME = "Qwen/Qwen2.5-32B"
DTYPE = torch.float16  # ~65GB on MPS; fits in 96GB M3 Ultra

# Output dirs
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "qwen32b"
FIGURES_DIR = PROJECT_ROOT / "figures" / "qwen32b"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# ROC helper (same as Pythia experiments)
# ===================================================================

def compute_roc(scores, labels):
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]
    tp, fp = 0, 0
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    tpr_list, fpr_list = [0.0], [0.0]
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    auc = np.trapezoid(tpr_list, fpr_list)
    return np.array(fpr_list), np.array(tpr_list), auc


# ===================================================================
# Experiment 1: Truth vs Falsehood
# ===================================================================

def run_truth(force: bool = False) -> dict:
    output_path = RESULTS_DIR / "truth_qwen32b.jsonl"

    if output_path.exists() and not force:
        print(f"  [truth] Loading from cache...")
        records = load_records(output_path)
    else:
        records = []
        print(f"\n  [truth] Analyzing {len(TRUTH_PAIRS)} pairs ({len(TRUTH_PAIRS)*2} texts)...")
        start = time.time()

        for pair in tqdm(TRUTH_PAIRS, desc="  truth", leave=False):
            true_rec = analyze_fixed_text(
                pair["true"], category="true", label=f"{pair['id']}_true",
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            false_rec = analyze_fixed_text(
                pair["false"], category="false", label=f"{pair['id']}_false",
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            records.extend([true_rec, false_rec])

        elapsed = time.time() - start
        print(f"  [truth] Done in {elapsed:.1f}s ({elapsed/len(TRUTH_PAIRS):.1f}s/pair)")

        if output_path.exists():
            output_path.unlink()
        save_records(records, output_path)

    # Compute metrics
    by_id = defaultdict(dict)
    for r in records:
        base_id = r.label.rsplit("_", 1)[0]
        by_id[base_id][r.category] = r

    deltas = []
    for pid, pair in by_id.items():
        if "true" in pair and "false" in pair:
            deltas.append(pair["true"].mean_top1_prob - pair["false"].mean_top1_prob)

    deltas = np.array(deltas)
    wins = int(np.sum(deltas > 0))
    win_rate = wins / len(deltas)
    stat, p_val = stats.wilcoxon(deltas, alternative="greater") if len(deltas) > 5 else (0, 1)
    cohens_d = np.mean(deltas) / (np.std(deltas) + 1e-10)

    # AUC
    features = np.array([r.mean_top1_prob for r in records])
    labels = np.array([1 if r.category == "true" else 0 for r in records])
    fpr, tpr, auc = compute_roc(features, labels)

    metrics = {
        "n_pairs": len(deltas),
        "win_rate": win_rate,
        "wins": wins,
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "auc": float(auc),
        "mean_delta": float(np.mean(deltas)),
        "roc": (fpr, tpr, auc),
    }

    print(f"  [truth] Win rate: {wins}/{len(deltas)} ({win_rate:.1%}), "
          f"AUC: {auc:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")
    return metrics


# ===================================================================
# Experiment 2: Medical Pairs
# ===================================================================

def run_medical(force: bool = False) -> dict:
    output_path = RESULTS_DIR / "medical_qwen32b.jsonl"

    if output_path.exists() and not force:
        print(f"  [medical] Loading from cache...")
        records = load_records(output_path)
    else:
        records = []
        print(f"\n  [medical] Analyzing {len(MEDICAL_PAIRS)} pairs ({len(MEDICAL_PAIRS)*2} texts)...")
        start = time.time()

        for pair in tqdm(MEDICAL_PAIRS, desc="  medical", leave=False):
            true_rec = analyze_fixed_text(
                pair["true"], category="medical_true",
                label=f"{pair['id']}_true",
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            true_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                                 "version": "true"}

            false_rec = analyze_fixed_text(
                pair["false"], category="medical_false",
                label=f"{pair['id']}_false",
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            false_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                                  "version": "false"}

            records.extend([true_rec, false_rec])

        elapsed = time.time() - start
        print(f"  [medical] Done in {elapsed:.1f}s")

        if output_path.exists():
            output_path.unlink()
        save_records(records, output_path)

    # Compute metrics
    by_id = defaultdict(dict)
    for r in records:
        pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
        version = r.metadata.get("version", "true" if "true" in r.label else "false")
        by_id[pid][version] = r

    deltas = []
    for pid, versions in by_id.items():
        if "true" in versions and "false" in versions:
            deltas.append(versions["true"].mean_top1_prob - versions["false"].mean_top1_prob)

    deltas = np.array(deltas)
    wins = int(np.sum(deltas > 0))
    win_rate = wins / len(deltas)
    t_stat, p_val = stats.ttest_1samp(deltas, 0) if len(deltas) > 1 else (0, 1)
    cohens_d = np.mean(deltas) / (np.std(deltas) + 1e-10) if np.std(deltas) > 0 else 0

    metrics = {
        "n_pairs": len(deltas),
        "win_rate": win_rate,
        "wins": wins,
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "mean_delta": float(np.mean(deltas)),
    }

    print(f"  [medical] Win rate: {wins}/{len(deltas)} ({win_rate:.1%}), "
          f"p={p_val:.4f}, d={cohens_d:.3f}")
    return metrics


# ===================================================================
# Experiment 3: Mandela Expanded
# ===================================================================

def run_mandela(force: bool = False) -> dict:
    output_path = RESULTS_DIR / "mandela_qwen32b.jsonl"

    if output_path.exists() and not force:
        print(f"  [mandela] Loading from cache...")
        records = load_records(output_path)
    else:
        records = []
        n_texts = sum(len(_make_texts(item)) * 2 for item in LINGUISTIC_ITEMS)
        print(f"\n  [mandela] Analyzing {len(LINGUISTIC_ITEMS)} items ({n_texts} texts)...")
        start = time.time()

        for item in tqdm(LINGUISTIC_ITEMS, desc="  mandela", leave=False):
            for framing_name, wrong_text, correct_text in _make_texts(item):
                w_rec = analyze_fixed_text(
                    wrong_text,
                    category="mandela_wrong",
                    label=f"{item['id']}_{framing_name}_wrong",
                    model_name=MODEL_NAME, revision="main", dtype=DTYPE,
                )
                w_rec.metadata = {
                    "item_id": item["id"],
                    "framing": framing_name,
                    "version": "wrong",
                    "human_ratio": item["human_ratio"],
                    "human_wrong_pct": item["human_wrong_pct"],
                    "human_correct_pct": item["human_correct_pct"],
                    "source": item["source"],
                }

                c_rec = analyze_fixed_text(
                    correct_text,
                    category="mandela_correct",
                    label=f"{item['id']}_{framing_name}_correct",
                    model_name=MODEL_NAME, revision="main", dtype=DTYPE,
                )
                c_rec.metadata = {
                    "item_id": item["id"],
                    "framing": framing_name,
                    "version": "correct",
                    "human_ratio": item["human_ratio"],
                    "human_wrong_pct": item["human_wrong_pct"],
                    "human_correct_pct": item["human_correct_pct"],
                    "source": item["source"],
                }

                records.extend([w_rec, c_rec])

        elapsed = time.time() - start
        print(f"  [mandela] Done in {elapsed:.1f}s ({len(records)} records)")

        if output_path.exists():
            output_path.unlink()
        save_records(records, output_path)

    # Compute correlation metrics
    pairs = _records_to_pairs(records)
    raw = filter_raw(pairs)

    h = [v["human_ratio"] for v in raw]
    m = [v["confidence_ratio"] for v in raw]
    r, p = stats.pearsonr(h, m)
    rho, rho_p = stats.spearmanr(h, m)

    metrics = {
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "n_items": len(raw),
    }

    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  [mandela] r={r:.3f} (p={p:.4f}){sig}  rho={rho:.3f} (p={rho_p:.4f})")
    return metrics


# ===================================================================
# Visualization
# ===================================================================

def plot_comparison(truth_m, medical_m, mandela_m):
    """Side-by-side Pythia 6.9B vs Qwen 32B comparison."""
    sns.set_theme(style="whitegrid", palette="muted")

    # --- Bar chart: win rates ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Load Pythia 6.9B metrics for comparison
    pythia_truth = {"win_rate": 0.90, "auc": 0.678}   # from Phase 2 results
    pythia_medical = {"win_rate": 0.88}
    pythia_mandela = {"spearman_rho": 0.652}

    # Panel 1: Truth win rates
    ax = axes[0]
    names = ["Pythia 6.9B", "Qwen 32B"]
    vals = [pythia_truth["win_rate"], truth_m["win_rate"]]
    colors = ["#d73027", "#7a0177"]
    ax.bar(names, vals, color=colors, edgecolor="white")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Win Rate")
    ax.set_title("Truth Detection")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")

    # Panel 2: Medical win rates
    ax = axes[1]
    vals = [pythia_medical["win_rate"], medical_m["win_rate"]]
    ax.bar(names, vals, color=colors, edgecolor="white")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Win Rate")
    ax.set_title("Medical Domain")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")

    # Panel 3: Mandela Spearman rho
    ax = axes[2]
    vals = [pythia_mandela["spearman_rho"], mandela_m["spearman_rho"]]
    ax.bar(names, vals, color=colors, edgecolor="white")
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Mandela Calibration")
    ax.set_ylim(-0.2, 1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("Cross-Architecture Comparison: Pythia 6.9B vs Qwen2.5-32B",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "qwen_vs_pythia_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mandela_scatter(mandela_m):
    """Mandela scatter: human prevalence vs Qwen confidence ratio."""
    # Load the records to get per-item data
    records = load_records(RESULTS_DIR / "mandela_qwen32b.jsonl")
    pairs = _records_to_pairs(records)
    raw = filter_raw(pairs)

    if not raw:
        return

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(9, 9))

    h = [v["human_ratio"] for v in raw]
    m = [v["confidence_ratio"] for v in raw]

    ax.scatter(h, m, s=100, color="#7a0177", zorder=5, alpha=0.8)
    for x, y, v in zip(h, m, raw):
        ax.annotate(v["item_id"], (x, y), fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")

    # Best-fit line
    slope, intercept, _, _, _ = stats.linregress(h, m)
    fit_x = np.linspace(min(h) - 0.02, max(h) + 0.02, 100)
    fit_y = slope * fit_x + intercept
    ax.plot(fit_x, fit_y, "-", color="#F44336", alpha=0.5, linewidth=1.5)

    r, p = stats.pearsonr(h, m)
    rho, rho_p = stats.spearmanr(h, m)
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""

    ax.text(0.05, 0.95,
            f"n = {len(raw)} linguistic items\n"
            f"Pearson r = {r:.3f} (p = {p:.4f}){sig}\n"
            f"Spearman rho = {rho:.3f} (p = {rho_p:.4f})",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Human Prevalence Ratio", fontsize=12)
    ax.set_ylabel("Model Confidence Ratio", fontsize=12)
    ax.set_title("Mandela Calibration — Qwen2.5-32B", fontsize=14)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.1, 1.0)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "qwen_mandela_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def run_experiment(force: bool = False):
    total_start = time.time()

    print("=" * 70)
    print("QWEN2.5-32B CROSS-ARCHITECTURE VALIDATION")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Dtype: {DTYPE}")
    print(f"Params: 32.76B")
    print()
    print("Experiments:")
    n_mandela = sum(len(_make_texts(item)) * 2 for item in LINGUISTIC_ITEMS)
    print(f"  1. Truth/false pairs: {len(TRUTH_PAIRS)} pairs = {len(TRUTH_PAIRS)*2} texts")
    print(f"  2. Medical pairs: {len(MEDICAL_PAIRS)} pairs = {len(MEDICAL_PAIRS)*2} texts")
    print(f"  3. Mandela expanded: {len(LINGUISTIC_ITEMS)} items = {n_mandela} texts")
    total = len(TRUTH_PAIRS)*2 + len(MEDICAL_PAIRS)*2 + n_mandela
    print(f"  Total: {total} forward passes")

    # Run all three
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENTS")
    print("=" * 70)

    truth_m = run_truth(force=force)
    medical_m = run_medical(force=force)
    mandela_m = run_mandela(force=force)

    unload_model()

    # ===================================================================
    # Comparison Table
    # ===================================================================
    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE COMPARISON")
    print("=" * 70)

    print(f"\n  {'Experiment':<25s} {'Metric':<20s} {'Pythia 6.9B':<15s} "
          f"{'Qwen 32B':<15s} {'Delta':<10s}")
    print("  " + "-" * 85)
    print(f"  {'Truth detection':<25s} {'Win rate':<20s} {'90.0%':<15s} "
          f"{truth_m['win_rate']:.1%}{'':<10s} {truth_m['win_rate'] - 0.90:+.1%}")
    print(f"  {'Truth detection':<25s} {'AUC':<20s} {'0.678':<15s} "
          f"{truth_m['auc']:.3f}{'':<10s} {truth_m['auc'] - 0.678:+.3f}")
    print(f"  {'Truth detection':<25s} {'p-value':<20s} {'<0.001':<15s} "
          f"{truth_m['p_value']:.4f}{'':<10s}")
    print(f"  {'Truth detection':<25s} {'Cohen d':<20s} {'1.01':<15s} "
          f"{truth_m['cohens_d']:.3f}{'':<10s}")
    print(f"  {'Medical domain':<25s} {'Win rate':<20s} {'88.0%':<15s} "
          f"{medical_m['win_rate']:.1%}{'':<10s} {medical_m['win_rate'] - 0.88:+.1%}")
    print(f"  {'Medical domain':<25s} {'p-value':<20s} {'0.010':<15s} "
          f"{medical_m['p_value']:.4f}{'':<10s}")
    print(f"  {'Mandela calibration':<25s} {'Spearman rho':<20s} {'0.652':<15s} "
          f"{mandela_m['spearman_rho']:.3f}{'':<10s} "
          f"{mandela_m['spearman_rho'] - 0.652:+.3f}")
    print(f"  {'Mandela calibration':<25s} {'Pearson r':<20s} {'0.572':<15s} "
          f"{mandela_m['pearson_r']:.3f}{'':<10s}")

    # Verdicts
    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)

    if truth_m["win_rate"] > 0.70:
        print(f"  TRUTH: Signal TRANSFERS to Qwen ({truth_m['win_rate']:.1%})")
    elif truth_m["win_rate"] > 0.55:
        print(f"  TRUTH: Weak transfer ({truth_m['win_rate']:.1%})")
    else:
        print(f"  TRUTH: Does NOT transfer ({truth_m['win_rate']:.1%})")

    if medical_m["win_rate"] > 0.70:
        print(f"  MEDICAL: Signal TRANSFERS ({medical_m['win_rate']:.1%})")
    elif medical_m["win_rate"] > 0.55:
        print(f"  MEDICAL: Weak transfer ({medical_m['win_rate']:.1%})")
    else:
        print(f"  MEDICAL: Does NOT transfer ({medical_m['win_rate']:.1%})")

    if mandela_m["spearman_p"] < 0.05:
        print(f"  MANDELA: Calibration TRANSFERS "
              f"(rho={mandela_m['spearman_rho']:.3f}, p={mandela_m['spearman_p']:.4f})")
    elif mandela_m["spearman_rho"] > 0.3:
        print(f"  MANDELA: Trending but not significant "
              f"(rho={mandela_m['spearman_rho']:.3f}, p={mandela_m['spearman_p']:.4f})")
    else:
        print(f"  MANDELA: Does NOT transfer "
              f"(rho={mandela_m['spearman_rho']:.3f}, p={mandela_m['spearman_p']:.4f})")

    # Plots
    print("\n  Generating plots...")
    plot_comparison(truth_m, medical_m, mandela_m)
    plot_mandela_scatter(mandela_m)

    # Save results
    results = {
        "model": MODEL_NAME,
        "dtype": str(DTYPE),
        "truth": {k: v for k, v in truth_m.items() if k != "roc"},
        "medical": medical_m,
        "mandela": mandela_m,
    }
    results_path = RESULTS_DIR / "qwen32b_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - total_start
    fig_count = len(list(FIGURES_DIR.glob("*.png")))
    print(f"\n  Results: {results_path}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_experiment(force=args.force)
