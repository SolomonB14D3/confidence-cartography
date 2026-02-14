"""
B14: Regime Stability Across Training Checkpoints
===================================================
Do regime 2 items show more confidence oscillation during training
than regime 1 items?

Regime 1: Truth/false probes where model gets it right (true > false)
  → Uses existing A3 checkpoint data (3 true_fact + 3 false_statement probes)
Regime 2: Mandela linguistic items where model gets it wrong (popular > correct)
  → NEW: runs 13 mandela items × 14 checkpoints × 6 model sizes

Stability metrics per item across checkpoints:
  - Variance, reversals, mean absolute delta, max jump
  - Monotonicity, late-training variance, convergence ratio
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

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text, unload_model
from src.schema import save_records, load_records, ConfidenceRecord
from src.scaling import MODEL_REGISTRY
from src.experiments.exp4_training import CHECKPOINTS
from src.experiments.exp_mandela_expanded import LINGUISTIC_ITEMS

# Output
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "checkpoint_stability"
FIGURES_DIR = PROJECT_ROOT / "figures" / "checkpoint_stability"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SCALING_DIR = PROJECT_ROOT / "data" / "results" / "scaling"

MODELS_ALL = ["160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
MODEL_LABELS = {
    "160m": "160M", "410m": "410M", "1b": "1B",
    "1.4b": "1.4B", "2.8b": "2.8B", "6.9b": "6.9B",
}
MODEL_PARAMS = {
    "160m": 1.6e8, "410m": 4.1e8, "1b": 1e9,
    "1.4b": 1.4e9, "2.8b": 2.8e9, "6.9b": 6.9e9,
}

def _available_models():
    """Only use models that have all mandela checkpoint data cached."""
    available = []
    for m in MODELS_ALL:
        mandela_count = len(list(RESULTS_DIR.glob(f"mandela_{m}_*.jsonl")))
        a3_count = len(list(SCALING_DIR.glob(f"a3_training_{m}_*.jsonl")))
        if mandela_count >= len(CHECKPOINTS) and a3_count >= len(CHECKPOINTS):
            available.append(m)
    return available

MODELS = _available_models() or MODELS_ALL[:4]

STEP_NUMBERS = [int(c.replace("step", "")) for c in CHECKPOINTS]


# ===================================================================
# Stability metrics
# ===================================================================

def checkpoint_stability(confs: list[float]) -> dict:
    """Compute stability metrics from a sequence of confidence values
    across training checkpoints."""
    c = np.array(confs)
    if len(c) < 3:
        return {}
    diffs = np.diff(c)
    signs = np.sign(diffs)

    variance = float(np.var(c))
    n_reversals = int(np.sum(np.diff(signs[signs != 0]) != 0)) if np.any(signs != 0) else 0
    mean_abs_delta = float(np.mean(np.abs(diffs)))
    max_jump = float(np.max(np.abs(diffs)))

    # Monotonicity: Spearman correlation of confidence with checkpoint index
    rho, _ = stats.spearmanr(np.arange(len(c)), c)
    monotonicity = float(rho) if not np.isnan(rho) else 0.0

    # Late-training stability (last 5 checkpoints)
    late_var = float(np.var(c[-5:])) if len(c) >= 5 else float(np.var(c))

    # Convergence ratio
    convergence_ratio = late_var / (variance + 1e-10)

    return {
        "variance": variance,
        "n_reversals": n_reversals,
        "mean_abs_delta": mean_abs_delta,
        "max_jump": max_jump,
        "monotonicity": monotonicity,
        "late_variance": late_var,
        "convergence_ratio": convergence_ratio,
    }


# ===================================================================
# Data loading: Regime 1 (existing A3 probes)
# ===================================================================

def load_regime1(model_key: str) -> dict[str, list[float]]:
    """Load existing A3 checkpoint data for truth/false probes.
    Returns {probe_label: [conf_at_step1, conf_at_step8, ...]}"""
    trajectories = defaultdict(list)

    for ckpt in CHECKPOINTS:
        path = SCALING_DIR / f"a3_training_{model_key}_{ckpt}.jsonl"
        if not path.exists():
            return {}
        records = load_records(path)
        for r in records:
            # Only use true_fact and false_statement probes
            if r.category in ("true_fact", "false_statement"):
                trajectories[r.label].append(r.mean_top1_prob)

    return dict(trajectories)


# ===================================================================
# Data generation: Regime 2 (Mandela across checkpoints)
# ===================================================================

def get_mandela_checkpoint_path(model_key: str, checkpoint: str) -> Path:
    return RESULTS_DIR / f"mandela_{model_key}_{checkpoint}.jsonl"


def run_mandela_checkpoints(model_key: str, force: bool = False) -> dict[str, list[float]]:
    """Run mandela items across all checkpoints for one model.
    Returns {item_label: [conf_at_step1, conf_at_step8, ...]}
    where label encodes both item_id and version (wrong/correct)."""
    spec = MODEL_REGISTRY[model_key]
    model_name = spec["name"]
    dtype = spec["dtype"]

    trajectories = defaultdict(list)

    for ckpt in CHECKPOINTS:
        cache_path = get_mandela_checkpoint_path(model_key, ckpt)

        if cache_path.exists() and not force:
            records = load_records(cache_path)
        else:
            records = []
            for item in LINGUISTIC_ITEMS:
                # Raw framing only (clean comparison)
                for version, text in [("wrong", item["wrong"]),
                                      ("correct", item["correct"])]:
                    label = f"{item['id']}_{version}"
                    rec = analyze_fixed_text(
                        text,
                        category=f"mandela_{version}",
                        label=label,
                        model_name=model_name,
                        revision=ckpt,
                        dtype=dtype,
                    )
                    rec.metadata = {
                        "item_id": item["id"],
                        "version": version,
                        "checkpoint": ckpt,
                        "step": int(ckpt.replace("step", "")),
                        "human_ratio": item["human_ratio"],
                        "model_size": model_key,
                    }
                    records.append(rec)

            unload_model()

            if cache_path.exists():
                cache_path.unlink()
            save_records(records, cache_path)

        for r in records:
            trajectories[r.label].append(r.mean_top1_prob)

    return dict(trajectories)


# ===================================================================
# Analysis
# ===================================================================

def analyze_regime(trajectories: dict[str, list[float]], regime_name: str) -> list[dict]:
    """Compute stability metrics for all items in a regime."""
    results = []
    for label, confs in trajectories.items():
        if len(confs) != len(CHECKPOINTS):
            continue
        metrics = checkpoint_stability(confs)
        metrics["label"] = label
        metrics["regime"] = regime_name
        metrics["trajectory"] = confs
        metrics["final_conf"] = confs[-1]
        metrics["initial_conf"] = confs[0]
        metrics["total_change"] = confs[-1] - confs[0]
        results.append(metrics)
    return results


# ===================================================================
# Visualization
# ===================================================================

def plot_variance_comparison(r1_all: dict, r2_all: dict):
    """Box plot: regime 1 vs 2 variance across model sizes."""
    sns.set_theme(style="whitegrid", palette="muted")

    available = [m for m in MODELS if m in r1_all and m in r2_all]
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 5),
                             sharey=True)
    if len(available) == 1:
        axes = [axes]

    for ax, model_key in zip(axes, available):
        r1 = r1_all[model_key]
        r2 = r2_all[model_key]
        r1_var = [r["variance"] for r in r1]
        r2_var = [r["variance"] for r in r2]

        data = [r1_var, r2_var]
        bp = ax.boxplot(data, labels=["R1\nFactual", "R2\nMandela"],
                        patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor("#2196F3")
        bp["boxes"][1].set_facecolor("#F44336")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_alpha(0.7)

        # Mann-Whitney
        if r1_var and r2_var:
            u, p = stats.mannwhitneyu(r1_var, r2_var, alternative="two-sided")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(0.5, 0.95, f"p={p:.3f} {sig}", transform=ax.transAxes,
                    ha="center", fontsize=9)

        ax.set_title(f"Pythia {MODEL_LABELS[model_key]}")
        if ax == axes[0]:
            ax.set_ylabel("Variance Across Checkpoints")

    fig.suptitle("Checkpoint Stability: Factual vs Mandela Items", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "variance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_example_trajectories(r1_results: list[dict], r2_results: list[dict],
                              model_key: str):
    """3-4 items from each regime across checkpoints."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Regime 1: pick 3 with most spread
    r1_sorted = sorted(r1_results, key=lambda r: r["variance"], reverse=True)
    r1_examples = r1_sorted[:3] + [r1_sorted[-1]] if len(r1_sorted) > 3 else r1_sorted

    ax = axes[0]
    for item in r1_examples:
        ax.plot(STEP_NUMBERS, item["trajectory"], "o-", markersize=3,
                linewidth=1.5, alpha=0.8, label=item["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Confidence")
    ax.set_title(f"Regime 1: Factual Probes\n(Pythia {MODEL_LABELS[model_key]})")
    ax.legend(fontsize=7, loc="best")

    # Regime 2: pick 3 most oscillatory + 1 stable
    r2_sorted = sorted(r2_results, key=lambda r: r["n_reversals"], reverse=True)
    r2_examples = r2_sorted[:3] + [r2_sorted[-1]] if len(r2_sorted) > 3 else r2_sorted

    ax = axes[1]
    for item in r2_examples:
        ax.plot(STEP_NUMBERS, item["trajectory"], "o-", markersize=3,
                linewidth=1.5, alpha=0.8, label=item["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Confidence")
    ax.set_title(f"Regime 2: Mandela Items\n(Pythia {MODEL_LABELS[model_key]})")
    ax.legend(fontsize=7, loc="best")

    fig.suptitle("Training Trajectories: Factual vs Mandela", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "example_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_oscillation_by_scale(r1_all: dict, r2_all: dict):
    """Does regime 2 instability grow with model size?"""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    available = [m for m in MODELS if m in r1_all and m in r2_all]

    x_params = [MODEL_PARAMS[m] for m in available]
    x_labels = [MODEL_LABELS[m] for m in available]

    metrics_to_plot = [
        ("variance", "Variance", axes[0]),
        ("n_reversals", "Direction Reversals", axes[1]),
        ("convergence_ratio", "Convergence Ratio\n(late var / total var)", axes[2]),
    ]

    for metric_key, ylabel, ax in metrics_to_plot:
        r1_means, r1_sems = [], []
        r2_means, r2_sems = [], []

        for m in available:
            r1_vals = [r[metric_key] for r in r1_all[m]]
            r2_vals = [r[metric_key] for r in r2_all[m]]
            r1_means.append(np.mean(r1_vals))
            r2_means.append(np.mean(r2_vals))
            r1_sems.append(np.std(r1_vals) / np.sqrt(len(r1_vals)) if r1_vals else 0)
            r2_sems.append(np.std(r2_vals) / np.sqrt(len(r2_vals)) if r2_vals else 0)

        ax.errorbar(x_params, r1_means, yerr=r1_sems, fmt="o-",
                    color="#2196F3", linewidth=2, markersize=8, capsize=4,
                    label="R1: Factual")
        ax.errorbar(x_params, r2_means, yerr=r2_sems, fmt="s-",
                    color="#F44336", linewidth=2, markersize=8, capsize=4,
                    label="R2: Mandela")
        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_params)
        ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
        ax.legend(fontsize=9)

    fig.suptitle("Does Regime 2 Instability Grow with Scale?", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "oscillation_by_scale.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_distribution(r1_results: list[dict], r2_results: list[dict],
                                  model_key: str):
    """Convergence ratio distribution for 6.9B."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 5))

    r1_cr = [r["convergence_ratio"] for r in r1_results]
    r2_cr = [r["convergence_ratio"] for r in r2_results]

    bins = np.linspace(0, max(max(r1_cr, default=1), max(r2_cr, default=1)) * 1.1, 20)
    ax.hist(r1_cr, bins=bins, alpha=0.6, color="#2196F3",
            label=f"R1: Factual (n={len(r1_cr)})", edgecolor="white")
    ax.hist(r2_cr, bins=bins, alpha=0.6, color="#F44336",
            label=f"R2: Mandela (n={len(r2_cr)})", edgecolor="white")

    ax.axvline(x=np.median(r1_cr), color="#1565C0", linestyle="--",
               label=f"R1 median: {np.median(r1_cr):.3f}")
    ax.axvline(x=np.median(r2_cr), color="#B71C1C", linestyle="--",
               label=f"R2 median: {np.median(r2_cr):.3f}")

    ax.set_xlabel("Convergence Ratio (late var / total var)")
    ax.set_ylabel("Count")
    ax.set_title(f"Has Training Converged? — Pythia {MODEL_LABELS[model_key]}\n"
                 f"Low = converged, High = still oscillating")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "convergence_ratio_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def run_experiment(force: bool = False):
    total_start = time.time()

    print("=" * 70)
    print("B14: REGIME STABILITY ACROSS TRAINING CHECKPOINTS")
    print("=" * 70)
    print(f"Models: {', '.join(MODEL_LABELS[m] for m in MODELS)}")
    print(f"Checkpoints: {len(CHECKPOINTS)}")
    print(f"Mandela items: {len(LINGUISTIC_ITEMS)} × 2 versions = "
          f"{len(LINGUISTIC_ITEMS) * 2} texts per checkpoint")
    print(f"New inference needed: {len(LINGUISTIC_ITEMS) * 2 * len(CHECKPOINTS) * len(MODELS)} runs")
    print()

    r1_all = {}  # model -> list of stability results
    r2_all = {}

    for model_key in MODELS:
        label = MODEL_LABELS[model_key]
        print(f"\n{'=' * 50}")
        print(f"  [{label}]")
        print(f"{'=' * 50}")

        # Regime 1: load existing A3 data
        print(f"  Loading R1 (existing A3 probes)...")
        r1_traj = load_regime1(model_key)
        if r1_traj:
            r1_results = analyze_regime(r1_traj, "regime1")
            r1_all[model_key] = r1_results
            print(f"    R1: {len(r1_results)} items loaded")
        else:
            print(f"    R1: No A3 data found, skipping")
            continue

        # Regime 2: run mandela across checkpoints
        print(f"  Running R2 (Mandela items across {len(CHECKPOINTS)} checkpoints)...")
        r2_traj = run_mandela_checkpoints(model_key, force=force)
        if r2_traj:
            r2_results = analyze_regime(r2_traj, "regime2")
            r2_all[model_key] = r2_results
            print(f"    R2: {len(r2_results)} items complete")

        # Quick summary
        if model_key in r1_all and model_key in r2_all:
            r1r = r1_all[model_key]
            r2r = r2_all[model_key]
            r1_var = np.mean([r["variance"] for r in r1r])
            r2_var = np.mean([r["variance"] for r in r2r])
            r1_rev = np.mean([r["n_reversals"] for r in r1r])
            r2_rev = np.mean([r["n_reversals"] for r in r2r])
            print(f"    R1 mean variance: {r1_var:.6f}, reversals: {r1_rev:.1f}")
            print(f"    R2 mean variance: {r2_var:.6f}, reversals: {r2_rev:.1f}")

    # ===================================================================
    # Detailed Report
    # ===================================================================
    focus = MODELS[-1]  # largest available model
    if focus in r1_all and focus in r2_all:
        print(f"\n{'=' * 70}")
        print(f"DETAILED REPORT — PYTHIA {MODEL_LABELS[focus]}")
        print(f"{'=' * 70}")

        r1r = r1_all[focus]
        r2r = r2_all[focus]

        metric_keys = ["variance", "n_reversals", "mean_abs_delta",
                       "max_jump", "monotonicity", "late_variance",
                       "convergence_ratio"]

        print(f"\n  {'Metric':<22s} {'R1 mean':<12s} {'R2 mean':<12s} "
              f"{'U-stat':<10s} {'p-value':<10s} {'Effect':<10s} {'Winner'}")
        print("  " + "-" * 86)

        for mk in metric_keys:
            r1_vals = [r[mk] for r in r1r]
            r2_vals = [r[mk] for r in r2r]

            r1m = np.mean(r1_vals)
            r2m = np.mean(r2_vals)

            if len(r1_vals) >= 2 and len(r2_vals) >= 2:
                u, p = stats.mannwhitneyu(r1_vals, r2_vals, alternative="two-sided")
                n1, n2 = len(r1_vals), len(r2_vals)
                rbc = 1 - (2 * u) / (n1 * n2)  # rank-biserial
            else:
                u, p, rbc = 0, 1, 0

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            # Determine which regime "wins" (higher instability)
            if mk == "monotonicity":
                winner = "R2 less monotonic" if r2m < r1m else "R1 less monotonic"
            else:
                winner = "R2 more unstable" if r2m > r1m else "R1 more unstable"

            print(f"  {mk:<22s} {r1m:<12.6f} {r2m:<12.6f} "
                  f"{u:<10.0f} {p:<10.4f} {sig:<10s} {winner}")

        # Per-item details
        print(f"\n  REGIME 1 (Factual) — Per Item:")
        print(f"  {'Label':<20s} {'Var':<10s} {'Rev':<5s} {'Mono':<8s} "
              f"{'Conv':<8s} {'Start':<8s} {'End':<8s}")
        print("  " + "-" * 67)
        for r in sorted(r1r, key=lambda x: x["variance"], reverse=True):
            print(f"  {r['label']:<20s} {r['variance']:<10.6f} "
                  f"{r['n_reversals']:<5d} {r['monotonicity']:<8.3f} "
                  f"{r['convergence_ratio']:<8.3f} "
                  f"{r['initial_conf']:<8.4f} {r['final_conf']:<8.4f}")

        print(f"\n  REGIME 2 (Mandela) — Per Item:")
        print(f"  {'Label':<20s} {'Var':<10s} {'Rev':<5s} {'Mono':<8s} "
              f"{'Conv':<8s} {'Start':<8s} {'End':<8s}")
        print("  " + "-" * 67)
        for r in sorted(r2r, key=lambda x: x["variance"], reverse=True):
            print(f"  {r['label']:<20s} {r['variance']:<10.6f} "
                  f"{r['n_reversals']:<5d} {r['monotonicity']:<8.3f} "
                  f"{r['convergence_ratio']:<8.3f} "
                  f"{r['initial_conf']:<8.4f} {r['final_conf']:<8.4f}")

    # ===================================================================
    # Scaling Summary
    # ===================================================================
    print(f"\n{'=' * 70}")
    print(f"SCALING SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Model':<7s} {'R1 var':<10s} {'R2 var':<10s} {'Ratio':<8s} "
          f"{'R1 rev':<8s} {'R2 rev':<8s} {'U p-val':<10s} {'Sig'}")
    print("  " + "-" * 63)

    scaling_results = {}
    for model_key in MODELS:
        if model_key not in r1_all or model_key not in r2_all:
            continue
        r1r = r1_all[model_key]
        r2r = r2_all[model_key]

        r1_var = np.mean([r["variance"] for r in r1r])
        r2_var = np.mean([r["variance"] for r in r2r])
        r1_rev = np.mean([r["n_reversals"] for r in r1r])
        r2_rev = np.mean([r["n_reversals"] for r in r2r])

        r1v_list = [r["variance"] for r in r1r]
        r2v_list = [r["variance"] for r in r2r]
        u, p = stats.mannwhitneyu(r1v_list, r2v_list, alternative="two-sided") \
            if len(r1v_list) >= 2 and len(r2v_list) >= 2 else (0, 1)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        ratio = r2_var / (r1_var + 1e-10)

        print(f"  {MODEL_LABELS[model_key]:<7s} {r1_var:<10.6f} {r2_var:<10.6f} "
              f"{ratio:<8.1f}x {r1_rev:<8.1f} {r2_rev:<8.1f} "
              f"{p:<10.4f} {sig}")

        scaling_results[model_key] = {
            "model": MODEL_LABELS[model_key],
            "params": MODEL_PARAMS[model_key],
            "r1_mean_variance": float(r1_var),
            "r2_mean_variance": float(r2_var),
            "variance_ratio": float(ratio),
            "r1_mean_reversals": float(r1_rev),
            "r2_mean_reversals": float(r2_rev),
            "u_statistic": float(u),
            "p_value": float(p),
        }

    # ===================================================================
    # Interpretation
    # ===================================================================
    if scaling_results:
        print(f"\n{'=' * 70}")
        print("INTERPRETATION")
        print(f"{'=' * 70}")

        ratios = [v["variance_ratio"] for v in scaling_results.values()]
        p_vals = [v["p_value"] for v in scaling_results.values()]

        if all(r > 1.5 for r in ratios) and any(p < 0.05 for p in p_vals):
            print("\n  → REGIME 2 CONSISTENTLY MORE UNSTABLE")
            print("    Direct evidence for the instability hypothesis.")
            # Check if it grows with scale
            sizes = list(scaling_results.keys())
            if len(sizes) >= 3:
                late_ratios = [scaling_results[s]["variance_ratio"]
                               for s in sizes[-3:]]
                early_ratios = [scaling_results[s]["variance_ratio"]
                                for s in sizes[:3]]
                if np.mean(late_ratios) > np.mean(early_ratios) * 1.3:
                    print("    AND instability GROWS with scale — strongest result.")
                else:
                    print("    Scale effect is unclear — instability present but not clearly growing.")
        elif np.mean(ratios) > 1:
            print("\n  → REGIME 2 TRENDING MORE UNSTABLE (not all significant)")
            print("    Partial evidence for instability hypothesis.")
        else:
            print("\n  → NO CLEAR DIFFERENCE IN STABILITY")
            print("    Instability hypothesis is not supported.")

    # ===================================================================
    # Plots
    # ===================================================================
    print(f"\n  Generating plots...")

    plot_variance_comparison(r1_all, r2_all)
    print(f"    [1/4] variance_comparison.png")

    if focus in r1_all and focus in r2_all:
        plot_example_trajectories(r1_all[focus], r2_all[focus], focus)
        print(f"    [2/4] example_trajectories.png")

        plot_convergence_distribution(r1_all[focus], r2_all[focus], focus)
        print(f"    [4/4] convergence_ratio_distribution.png")

    plot_oscillation_by_scale(r1_all, r2_all)
    print(f"    [3/4] oscillation_by_scale.png")

    # ===================================================================
    # Save
    # ===================================================================
    for regime_name, regime_data in [("regime1", r1_all), ("regime2", r2_all)]:
        save_obj = {}
        for model_key, results in regime_data.items():
            save_obj[model_key] = [
                {k: v for k, v in r.items() if k != "trajectory"}
                for r in results
            ]
            # Save trajectories separately (useful for reanalysis)
            save_obj[f"{model_key}_trajectories"] = {
                r["label"]: r["trajectory"] for r in results
            }
        with open(RESULTS_DIR / f"{regime_name}_stability.json", "w") as f:
            json.dump(save_obj, f, indent=2)

    with open(RESULTS_DIR / "comparison_stats.json", "w") as f:
        json.dump(scaling_results, f, indent=2)

    total_time = time.time() - total_start
    fig_count = len(list(FIGURES_DIR.glob("*.png")))

    print(f"\n{'=' * 70}")
    print(f"COMPLETE")
    print(f"{'=' * 70}")
    print(f"  New inference: mandela items × {len(CHECKPOINTS)} checkpoints × "
          f"{len(MODELS)} models")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_experiment(force=args.force)
