"""
Experiment A3: Training Dynamics Scaling
========================================
For each model size, load checkpoints and track truth-false confidence gap
evolution.

"Phase diagram" — X-axis = training step, Y-axis = truth-false gap,
one curve per model size. This visualizes how the truth signal develops
across both training time and model scale simultaneously.

Key questions:
- Does the gap emerge earlier in larger models?
- Is the maximum gap larger?
- Do all models converge to the same final gap or do larger models
  achieve more separation?

NOTE: This is the heaviest experiment. 8 probes × 14 checkpoints × 6 sizes
= 672 analyses. Estimated 4-8 hours total. Uses incremental saves so
it can be restarted safely.
"""

import sys
import time
import json
import shutil
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
from src.scaling import (
    MODEL_REGISTRY, SCALING_MODELS, PARAM_COUNTS,
)
from src.scaling_viz import MODEL_COLORS, model_display_name
from src.utils import SCALING_RESULTS_DIR, SCALING_FIGURES_DIR

# Import exact probes and checkpoints from Phase 1
from src.experiments.exp4_training import PROBES, CHECKPOINTS


# ---------------------------------------------------------------------------
# Checkpoint step numbers (for plotting)
# ---------------------------------------------------------------------------

def step_number(revision: str) -> int:
    """Extract numeric step from revision string like 'step64000'."""
    return int(revision.replace("step", ""))


STEP_NUMBERS = [step_number(c) for c in CHECKPOINTS]


# ---------------------------------------------------------------------------
# Per-model, per-checkpoint analysis
# ---------------------------------------------------------------------------

def get_checkpoint_path(size: str, checkpoint: str) -> Path:
    """Path for a single (size, checkpoint) result file."""
    return SCALING_RESULTS_DIR / f"a3_training_{size}_{checkpoint}.jsonl"


def run_single_checkpoint(size: str, checkpoint: str, force: bool = False) -> list[ConfidenceRecord]:
    """Run all probes at one (model_size, checkpoint)."""
    output_path = get_checkpoint_path(size, checkpoint)

    if output_path.exists() and not force:
        return load_records(output_path)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]

    records = []
    for probe in PROBES:
        rec = analyze_fixed_text(
            probe["text"],
            category=probe["category"],
            label=probe["label"],
            model_name=model_name,
            revision=checkpoint,
            dtype=dtype,
        )
        rec.metadata = {
            "model_size": size,
            "checkpoint": checkpoint,
            "step_number": step_number(checkpoint),
            "probe_category": probe["category"],
        }
        records.append(rec)

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return records


def _cleanup_hf_blobs(size: str):
    """Delete HF cache blobs for a model to free disk space between checkpoints."""
    import gc
    gc.collect()
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = hf_cache / f"models--EleutherAI--pythia-{size}"
    blobs_dir = model_cache / "blobs"
    snapshots_dir = model_cache / "snapshots"
    try:
        if blobs_dir.exists():
            blob_size = sum(f.stat().st_size for f in blobs_dir.rglob("*") if f.is_file()) / 1e9
            shutil.rmtree(blobs_dir, ignore_errors=True)
            blobs_dir.mkdir(exist_ok=True)
            if blob_size > 0.1:
                print(f"      [cleanup] Freed {blob_size:.1f}GB blobs")
        if snapshots_dir.exists():
            shutil.rmtree(snapshots_dir, ignore_errors=True)
            snapshots_dir.mkdir(exist_ok=True)
    except Exception as e:
        print(f"      [cleanup] Warning: {e}")


def run_single_model(size: str, force: bool = False) -> dict:
    """Run all checkpoints for one model size. Returns {checkpoint: [records]}."""
    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]

    print(f"\n  [{size}] Processing {len(CHECKPOINTS)} checkpoints × {len(PROBES)} probes...")
    start = time.time()

    all_records = {}
    for i, ckpt in enumerate(CHECKPOINTS):
        cached = get_checkpoint_path(size, ckpt).exists() and not force
        records = run_single_checkpoint(size, ckpt, force=force)
        all_records[ckpt] = records
        unload_model()  # Free memory between checkpoints

        # Clean up HF blobs after each checkpoint to prevent disk filling
        if not cached:
            _cleanup_hf_blobs(size)
            print(f"    [{size}/{ckpt}] computed ({i+1}/{len(CHECKPOINTS)})")
        else:
            if (i + 1) % 5 == 0:  # Periodic status for cached
                print(f"    [{size}] {i+1}/{len(CHECKPOINTS)} loaded from cache")

    elapsed = time.time() - start
    print(f"  [{size}] Done in {elapsed:.1f}s ({len(CHECKPOINTS)} checkpoints)")

    return all_records


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_training_metrics(all_records: dict) -> dict:
    """Compute truth-false gap at each checkpoint.

    all_records: {checkpoint: [records]}
    Returns: {checkpoint: {metric: value}}
    """
    metrics_by_ckpt = {}

    for ckpt, records in all_records.items():
        true_probs = [r.mean_top1_prob for r in records if r.category == "true_fact"]
        false_probs = [r.mean_top1_prob for r in records if r.category == "false_statement"]
        contested_probs = [r.mean_top1_prob for r in records if r.category == "contested"]

        true_ents = [r.mean_entropy for r in records if r.category == "true_fact"]
        false_ents = [r.mean_entropy for r in records if r.category == "false_statement"]

        gap = np.mean(true_probs) - np.mean(false_probs) if true_probs and false_probs else 0
        entropy_gap = np.mean(false_ents) - np.mean(true_ents) if true_ents and false_ents else 0

        metrics_by_ckpt[ckpt] = {
            "step": step_number(ckpt),
            "mean_true_prob": float(np.mean(true_probs)) if true_probs else 0,
            "mean_false_prob": float(np.mean(false_probs)) if false_probs else 0,
            "mean_contested_prob": float(np.mean(contested_probs)) if contested_probs else 0,
            "truth_false_gap": float(gap),
            "entropy_gap": float(entropy_gap),
            "mean_true_entropy": float(np.mean(true_ents)) if true_ents else 0,
            "mean_false_entropy": float(np.mean(false_ents)) if false_ents else 0,
        }

    return metrics_by_ckpt


def find_emergence_step(metrics_by_ckpt: dict, threshold: float = 0.02) -> int:
    """Find the first training step where truth-false gap exceeds threshold."""
    for ckpt in CHECKPOINTS:
        if ckpt in metrics_by_ckpt:
            if metrics_by_ckpt[ckpt]["truth_false_gap"] > threshold:
                return metrics_by_ckpt[ckpt]["step"]
    return -1  # Never emerged


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_phase_diagram(all_model_metrics: dict, save_path: Path):
    """Phase diagram: training step (x) vs truth-false gap (y), one curve per model."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for size, metrics_by_ckpt in all_model_metrics.items():
        steps = [metrics_by_ckpt[c]["step"] for c in CHECKPOINTS if c in metrics_by_ckpt]
        gaps = [metrics_by_ckpt[c]["truth_false_gap"] for c in CHECKPOINTS if c in metrics_by_ckpt]

        color = MODEL_COLORS.get(size, "#999999")
        ax.plot(steps, gaps, marker="o", markersize=4, linewidth=2,
                color=color, label=model_display_name(size), alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Truth - Falsehood Confidence Gap", fontsize=12)
    ax.set_title("Phase Diagram: Truth Signal Emergence Across Scale", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_category_evolution(all_model_metrics: dict, save_path: Path):
    """3-panel: true, false, contested confidence evolution per model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for panel_idx, (metric_key, title) in enumerate([
        ("mean_true_prob", "True Facts: Mean Confidence"),
        ("mean_false_prob", "False Statements: Mean Confidence"),
        ("mean_contested_prob", "Contested: Mean Confidence"),
    ]):
        ax = axes[panel_idx]
        for size, metrics_by_ckpt in all_model_metrics.items():
            steps = [metrics_by_ckpt[c]["step"] for c in CHECKPOINTS if c in metrics_by_ckpt]
            vals = [metrics_by_ckpt[c][metric_key] for c in CHECKPOINTS if c in metrics_by_ckpt]

            color = MODEL_COLORS.get(size, "#999999")
            ax.plot(steps, vals, marker="o", markersize=3, linewidth=1.5,
                    color=color, label=model_display_name(size), alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean P(actual token)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle("Confidence Evolution by Category Across Scale", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_evolution(all_model_metrics: dict, save_path: Path):
    """Entropy gap (false - true) evolution per model."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for size, metrics_by_ckpt in all_model_metrics.items():
        steps = [metrics_by_ckpt[c]["step"] for c in CHECKPOINTS if c in metrics_by_ckpt]
        gaps = [metrics_by_ckpt[c]["entropy_gap"] for c in CHECKPOINTS if c in metrics_by_ckpt]

        color = MODEL_COLORS.get(size, "#999999")
        ax.plot(steps, gaps, marker="o", markersize=4, linewidth=2,
                color=color, label=model_display_name(size), alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Entropy Gap (False − True)", fontsize=12)
    ax.set_title("Entropy Differentiation Across Scale", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_emergence_scaling(emergence_data: dict, save_path: Path):
    """Bar chart: training step of truth emergence by model size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = [s for s in SCALING_MODELS if s in emergence_data]
    steps = [emergence_data[s]["emergence_step"] for s in sizes]
    max_gaps = [emergence_data[s]["max_gap"] for s in sizes]
    colors = [MODEL_COLORS.get(s, "#999999") for s in sizes]

    # Bar chart of emergence step
    x = np.arange(len(sizes))
    bars = ax.bar(x, steps, color=colors, edgecolor="white", alpha=0.8)

    # Annotate with max gap
    for i, (bar, gap) in enumerate(zip(bars, max_gaps)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"max gap\n{gap:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([model_display_name(s) for s in sizes], fontsize=9)
    ax.set_ylabel("Training Step of Truth Emergence")
    ax.set_title("When Does Truth Signal Emerge? (gap > 0.02)")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(models: list[str] = None, force: bool = False):
    models = models or SCALING_MODELS

    n_total = len(PROBES) * len(CHECKPOINTS) * len(models)

    print("=" * 65)
    print("EXPERIMENT A3: Training Dynamics Scaling")
    print("=" * 65)
    print(f"Models: {', '.join(models)}")
    print(f"Checkpoints: {len(CHECKPOINTS)}")
    print(f"Probes: {len(PROBES)}")
    print(f"Total analyses: {n_total}")
    print(f"NOTE: This is the heaviest experiment. Estimated 4-8 hours.")

    start_time = time.time()
    all_model_metrics = {}
    emergence_data = {}

    for size in models:
        model_records = run_single_model(size, force=force)
        metrics = compute_training_metrics(model_records)
        all_model_metrics[size] = metrics

        # Find emergence
        emerge_step = find_emergence_step(metrics)
        max_gap = max(m["truth_false_gap"] for m in metrics.values())
        final_gap = metrics[CHECKPOINTS[-1]]["truth_false_gap"] if CHECKPOINTS[-1] in metrics else 0

        emergence_data[size] = {
            "emergence_step": emerge_step,
            "max_gap": max_gap,
            "final_gap": final_gap,
        }

        print(f"  [{size}] Emergence at step {emerge_step}, "
              f"max gap: {max_gap:.4f}, final gap: {final_gap:.4f}")

        # Clear HF cache for this model to free disk space
        # (each model's 14 checkpoints can use 20-50GB)
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache = hf_cache / f"models--EleutherAI--pythia-{size}"
        if model_cache.exists():
            cache_size = sum(f.stat().st_size for f in model_cache.rglob("*") if f.is_file()) / 1e9
            shutil.rmtree(model_cache)
            print(f"  [{size}] Cleared HF cache ({cache_size:.1f}GB freed)")

    # ===================================================================
    # Summary
    # ===================================================================
    sizes_done = [s for s in models if s in all_model_metrics]

    print("\n" + "=" * 65)
    print("TRAINING DYNAMICS SCALING SUMMARY")
    print("=" * 65)

    print(f"\n{'Size':<8} {'Params':<12} {'Emerge Step':<14} {'Max Gap':<10} "
          f"{'Final Gap':<10} {'Final True P':<12} {'Final False P':<12}")
    print("-" * 78)
    for size in sizes_done:
        params = PARAM_COUNTS[size]
        e = emergence_data[size]
        final = all_model_metrics[size].get(CHECKPOINTS[-1], {})
        print(f"{size:<8} {params/1e6:>8.0f}M  {e['emergence_step']:<14} "
              f"{e['max_gap']:<10.4f} {e['final_gap']:<10.4f} "
              f"{final.get('mean_true_prob', 0):<12.4f} "
              f"{final.get('mean_false_prob', 0):<12.4f}")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING PLOTS")
    print("=" * 65)
    fig_dir = SCALING_FIGURES_DIR

    if len(sizes_done) >= 2:
        print("\n[1/4] Phase diagram...")
        plot_phase_diagram(all_model_metrics, fig_dir / "a3_phase_diagram.png")

        print("[2/4] Category evolution...")
        plot_category_evolution(all_model_metrics, fig_dir / "a3_category_evolution.png")

        print("[3/4] Entropy evolution...")
        plot_entropy_evolution(all_model_metrics, fig_dir / "a3_entropy_evolution.png")

        print("[4/4] Emergence scaling...")
        plot_emergence_scaling(emergence_data, fig_dir / "a3_emergence_scaling.png")

    # ===================================================================
    # Final
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(fig_dir.glob("a3_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT A3 COMPLETE")
    print("=" * 65)
    print(f"  Models: {len(sizes_done)}")
    print(f"  Total analyses: {len(sizes_done) * len(CHECKPOINTS) * len(PROBES)}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")

    if sizes_done:
        gaps = [emergence_data[s]["max_gap"] for s in sizes_done]
        emerge_steps = [emergence_data[s]["emergence_step"] for s in sizes_done]

        print(f"\n  Max gap range: {min(gaps):.4f} → {max(gaps):.4f}")
        valid_steps = [s for s in emerge_steps if s > 0]
        if valid_steps:
            print(f"  Emergence step range: {min(valid_steps)} → {max(valid_steps)}")

        if max(gaps) > min(gaps) * 1.5:
            print("  → Truth-false gap SCALES with model size")
        else:
            print("  → Truth-false gap is SCALE-INVARIANT")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model sizes to run")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if cached")
    args = parser.parse_args()

    run_experiment(args.models, force=args.force)
