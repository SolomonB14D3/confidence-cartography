"""
Experiment A1: Truth vs Falsehood Scaling
==========================================
Run the same 40 true/false pairs from Exp 2 across all Pythia model sizes.

Key question: Does truth detection improve with scale?
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text, unload_model
from src.schema import save_records, load_records, ConfidenceRecord
from src.scaling import (
    MODEL_REGISTRY, SCALING_MODELS, PARAM_COUNTS,
    get_scaling_output_path, load_scaling_results, print_runtime_estimates,
)
from src.scaling_viz import (
    plot_scaling_law, plot_scaling_heatmap, plot_roc_overlay,
    plot_pvalue_cascade, MODEL_COLORS, model_display_name,
)
from src.utils import SCALING_FIGURES_DIR

# Import the exact same pairs from Phase 1
from src.experiments.exp2_truth import PAIRS


# ---------------------------------------------------------------------------
# ROC/AUC computation (adapted from exp2)
# ---------------------------------------------------------------------------

def compute_roc(scores, labels):
    """Compute ROC curve and AUC for a single score dimension."""
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]
    tp, fp = 0, 0
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
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


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_single_model(size: str, force: bool = False) -> list[ConfidenceRecord]:
    """Run truth/falsehood analysis for a single model size."""
    output_path = get_scaling_output_path("a1_truth", size)

    if output_path.exists() and not force:
        print(f"  [{size}] Results exist, loading from cache...")
        return load_records(output_path)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []

    print(f"\n  [{size}] Analyzing {len(PAIRS)} pairs with {model_name} (dtype={dtype})...")
    start = time.time()

    for pair in tqdm(PAIRS, desc=f"  {size}", leave=False):
        true_rec = analyze_fixed_text(
            pair["true"], category="true", label=f"{pair['id']}_true",
            model_name=model_name, revision="main", dtype=dtype,
        )
        false_rec = analyze_fixed_text(
            pair["false"], category="false", label=f"{pair['id']}_false",
            model_name=model_name, revision="main", dtype=dtype,
        )
        records.extend([true_rec, false_rec])

    elapsed = time.time() - start
    print(f"  [{size}] Done in {elapsed:.1f}s ({len(records)} records)")

    # Save
    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)

    return records


def compute_metrics(records: list[ConfidenceRecord]) -> dict:
    """Compute truth detection metrics from a set of records."""
    # Split into true and false
    by_id = defaultdict(dict)
    for r in records:
        base_id = r.label.rsplit("_", 1)[0]
        if r.category == "true":
            by_id[base_id]["true"] = r
        else:
            by_id[base_id]["false"] = r

    # Compute pair-level metrics
    delta_probs = []
    true_probs_list = []
    false_probs_list = []

    for pid, pair in by_id.items():
        if "true" in pair and "false" in pair:
            delta = pair["true"].mean_top1_prob - pair["false"].mean_top1_prob
            delta_probs.append(delta)
            true_probs_list.append(pair["true"].mean_top1_prob)
            false_probs_list.append(pair["false"].mean_top1_prob)

    delta_probs = np.array(delta_probs)
    n_pairs = len(delta_probs)

    # Win rate
    wins = int(np.sum(delta_probs > 0))
    win_rate = wins / n_pairs if n_pairs > 0 else 0

    # Wilcoxon signed-rank test
    if n_pairs > 5:
        stat, p_val = stats.wilcoxon(delta_probs, alternative="greater")
    else:
        p_val = 1.0

    # Cohen's d
    cohens_d = np.mean(delta_probs) / (np.std(delta_probs) + 1e-10)

    # AUC
    features = []
    labels = []
    for r in records:
        features.append(r.mean_top1_prob)
        labels.append(1 if r.category == "true" else 0)
    features = np.array(features)
    labels = np.array(labels)
    fpr, tpr, auc = compute_roc(features, labels)

    return {
        "n_pairs": n_pairs,
        "win_rate": win_rate,
        "wins": wins,
        "p_value": p_val,
        "cohens_d": cohens_d,
        "auc": auc,
        "mean_delta": float(np.mean(delta_probs)),
        "median_delta": float(np.median(delta_probs)),
        "mean_true_prob": float(np.mean(true_probs_list)),
        "mean_false_prob": float(np.mean(false_probs_list)),
        "roc": (fpr, tpr, auc),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(models: list[str] = None, force: bool = False):
    models = models or SCALING_MODELS

    print("=" * 65)
    print("EXPERIMENT A1: Truth vs Falsehood Scaling")
    print("=" * 65)
    print(f"Models: {', '.join(models)}")
    print(f"Pairs per model: {len(PAIRS)}")
    print_runtime_estimates(len(PAIRS) * 2)

    start_time = time.time()
    all_results = {}
    all_metrics = {}

    for size in models:
        records = run_single_model(size, force=force)
        all_results[size] = records
        metrics = compute_metrics(records)
        all_metrics[size] = metrics
        unload_model()

        # Print summary for this size
        print(f"  [{size}] Win rate: {metrics['wins']}/{metrics['n_pairs']} "
              f"({metrics['win_rate']:.1%}), "
              f"AUC: {metrics['auc']:.3f}, "
              f"p={metrics['p_value']:.4f}, "
              f"d={metrics['cohens_d']:.3f}")

    # ===================================================================
    # Summary Table
    # ===================================================================
    print("\n" + "=" * 65)
    print("SCALING SUMMARY")
    print("=" * 65)

    sizes_done = [s for s in models if s in all_metrics]
    print(f"\n{'Size':<8} {'Params':<12} {'Win%':<8} {'AUC':<8} "
          f"{'p-value':<10} {'Cohen d':<9} {'Mean dP':<9}")
    print("-" * 64)
    for size in sizes_done:
        m = all_metrics[size]
        params = PARAM_COUNTS[size]
        print(f"{size:<8} {params/1e6:>8.0f}M  {m['win_rate']:<8.1%} "
              f"{m['auc']:<8.3f} {m['p_value']:<10.4f} "
              f"{m['cohens_d']:<9.3f} {m['mean_delta']:<+9.4f}")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING SCALING PLOTS")
    print("=" * 65)

    fig_dir = SCALING_FIGURES_DIR

    # 1. AUC scaling
    print("\n[1/5] AUC scaling law...")
    plot_scaling_law(
        sizes_done,
        {"AUC": [all_metrics[s]["auc"] for s in sizes_done]},
        ylabel="AUC",
        title="Truth Detection AUC vs Model Size",
        save_path=fig_dir / "a1_auc_scaling.png",
        hline=0.5, hline_label="Random",
    )

    # 2. Delta P scaling
    print("[2/5] Delta P scaling...")
    plot_scaling_law(
        sizes_done,
        {"Mean delta P": [all_metrics[s]["mean_delta"] for s in sizes_done]},
        ylabel="Mean P(true) - P(false)",
        title="Truth-Falsehood Confidence Gap vs Model Size",
        save_path=fig_dir / "a1_delta_scaling.png",
        hline=0.0, hline_label="No gap",
    )

    # 3. Win rate + Cohen's d
    print("[3/5] Multi-metric scaling...")
    plot_scaling_law(
        sizes_done,
        {
            "Win Rate": [all_metrics[s]["win_rate"] for s in sizes_done],
            "Cohen's d": [all_metrics[s]["cohens_d"] for s in sizes_done],
        },
        ylabel="Value",
        title="Truth Detection Metrics vs Model Size",
        save_path=fig_dir / "a1_metrics_scaling.png",
        hline=0.5, hline_label="Chance",
    )

    # 4. ROC overlay
    print("[4/5] ROC overlay...")
    roc_data = {s: all_metrics[s]["roc"] for s in sizes_done}
    plot_roc_overlay(
        roc_data,
        title="ROC Curves: Truth Detection Across Model Sizes",
        save_path=fig_dir / "a1_roc_overlay.png",
    )

    # 5. P-value cascade
    print("[5/5] P-value cascade...")
    plot_pvalue_cascade(
        sizes_done,
        {"Wilcoxon (true > false)": [all_metrics[s]["p_value"] for s in sizes_done]},
        title="Statistical Significance vs Model Size",
        save_path=fig_dir / "a1_pvalue_cascade.png",
    )

    # ===================================================================
    # Final Summary
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(fig_dir.glob("a1_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT A1 COMPLETE")
    print("=" * 65)
    print(f"  Models analyzed: {len(sizes_done)}")
    print(f"  Total records: {sum(len(all_results[s]) for s in sizes_done)}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s")

    # Key finding
    if len(sizes_done) >= 2:
        first_auc = all_metrics[sizes_done[0]]["auc"]
        last_auc = all_metrics[sizes_done[-1]]["auc"]
        print(f"\n  Scaling verdict: AUC {first_auc:.3f} ({sizes_done[0]}) → "
              f"{last_auc:.3f} ({sizes_done[-1]})")
        if last_auc > first_auc + 0.05:
            print("  → Truth detection IMPROVES with scale")
        elif last_auc < first_auc - 0.05:
            print("  → Truth detection DEGRADES with scale")
        else:
            print("  → Truth detection is SCALE-INVARIANT")


if __name__ == "__main__":
    run_experiment()
