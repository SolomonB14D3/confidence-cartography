"""
Experiment 4: Confidence Shifts Over Training
==============================================
Pythia uniquely released checkpoints at many training steps.
How do confidence patterns evolve as the model learns?

Questions:
  - Do confidence patterns on true statements stabilize before false ones?
  - Is there a phase transition in uncertainty patterns?
  - At what point does the model develop distinct signatures for different knowledge types?
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text, unload_model
from src.schema import save_records, ConfidenceRecord
from src.utils import RESULTS_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Checkpoints to analyze (log-spaced early, then linear)
# Full set available: step0, step1, step2, step4, ..., step143000
# We sample a meaningful subset to keep runtime reasonable
# ---------------------------------------------------------------------------

CHECKPOINTS = [
    "step1",      # Barely initialized
    "step8",      # Very early
    "step64",     # Starting to learn
    "step256",    # Early training
    "step512",    # ~0.4% through training
    "step1000",   # ~0.7%
    "step2000",   # ~1.4%
    "step4000",   # ~2.8%
    "step8000",   # ~5.6%
    "step16000",  # ~11%
    "step32000",  # ~22%
    "step64000",  # ~45%
    "step100000", # ~70%
    "step143000", # Final checkpoint (= main)
]

# Probe statements: small set of diverse texts to track across training
PROBES = [
    # True facts
    {"text": "The capital of France is Paris.",
     "category": "true_fact", "label": "france_paris"},
    {"text": "Water boils at 100 degrees Celsius at sea level.",
     "category": "true_fact", "label": "water_boils"},
    {"text": "The Earth orbits the Sun.",
     "category": "true_fact", "label": "earth_sun"},

    # False statements
    {"text": "The capital of France is banana.",
     "category": "false_statement", "label": "france_banana"},
    {"text": "Water boils at minus forty degrees.",
     "category": "false_statement", "label": "water_neg40"},
    {"text": "The Sun orbits the Earth.",
     "category": "false_statement", "label": "sun_earth"},

    # Contested
    {"text": "Nuclear energy is the safest form of power generation.",
     "category": "contested", "label": "nuclear_safe"},
    {"text": "Artificial intelligence will eventually surpass human intelligence.",
     "category": "contested", "label": "ai_surpass"},
]


def plot_training_curves(data, metric, ylabel, title, save_name):
    """Plot a metric across training steps for each probe, grouped by category.

    data: dict of {label: [(step_num, value), ...]}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    cat_styles = {
        "true_fact": {"color": "#4CAF50", "linestyle": "-"},
        "false_statement": {"color": "#F44336", "linestyle": "--"},
        "contested": {"color": "#FF9800", "linestyle": "-."},
    }

    probe_to_cat = {p["label"]: p["category"] for p in PROBES}

    for label, points in data.items():
        cat = probe_to_cat[label]
        style = cat_styles[cat]
        steps = [p[0] for p in points]
        vals = [p[1] for p in points]
        ax.plot(steps, vals, marker="o", markersize=3, linewidth=1.5,
                label=f"{label} ({cat})", **style, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


def plot_category_evolution(cat_data, save_name):
    """Plot mean confidence by category across training steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"true_fact": "#4CAF50", "false_statement": "#F44336",
              "contested": "#FF9800"}
    line_labels = {"true_fact": "True Facts", "false_statement": "False Statements",
                   "contested": "Contested Claims"}

    for cat, points_by_step in cat_data.items():
        steps = sorted(points_by_step.keys())
        mean_probs = [np.mean(points_by_step[s]["probs"]) for s in steps]
        mean_ents = [np.mean(points_by_step[s]["ents"]) for s in steps]

        ax1.plot(steps, mean_probs, marker="o", markersize=4, linewidth=2,
                 color=colors[cat], label=line_labels[cat])
        ax2.plot(steps, mean_ents, marker="o", markersize=4, linewidth=2,
                 color=colors[cat], label=line_labels[cat])

    ax1.set_xscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Mean P(actual token)")
    ax1.set_title("Confidence Evolution by Category")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Mean Entropy (bits)")
    ax2.set_title("Entropy Evolution by Category")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, (ax1, ax2)


def plot_separation_over_time(cat_data, save_name):
    """Plot the gap between true and false confidence over training."""
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = sorted(list(cat_data["true_fact"].keys()))

    true_means = [np.mean(cat_data["true_fact"][s]["probs"]) for s in steps]
    false_means = [np.mean(cat_data["false_statement"][s]["probs"]) for s in steps]
    gaps = [t - f for t, f in zip(true_means, false_means)]

    ax.plot(steps, gaps, marker="o", markersize=5, linewidth=2.5,
            color="#2196F3", label="True - False confidence gap")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.fill_between(steps, gaps, alpha=0.15, color="#2196F3")

    ax.set_xscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean P(true) - Mean P(false)")
    ax.set_title("Truth-Falsehood Confidence Separation During Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


def run_experiment():
    output_path = RESULTS_DIR / "exp4_training.jsonl"

    print("=" * 65)
    print("EXPERIMENT 4: Confidence Shifts Over Training")
    print("=" * 65)
    print(f"Checkpoints: {len(CHECKPOINTS)}")
    print(f"Probes: {len(PROBES)} statements")
    print(f"Total analyses: {len(CHECKPOINTS) * len(PROBES)}")
    print()

    start_time = time.time()
    all_records = []

    # Data structures for plotting
    probe_curves = defaultdict(list)  # label -> [(step, mean_prob), ...]
    probe_entropy_curves = defaultdict(list)
    cat_data = defaultdict(lambda: defaultdict(lambda: {"probs": [], "ents": []}))

    for ckpt in tqdm(CHECKPOINTS, desc="Checkpoints"):
        step_num = int(ckpt.replace("step", ""))

        # Unload previous model to force reload with new revision
        unload_model()

        for probe in PROBES:
            rec = analyze_fixed_text(
                probe["text"],
                category=probe["category"],
                label=f"{probe['label']}__{ckpt}",
                revision=ckpt,
            )
            rec.metadata = {"checkpoint": ckpt, "step": step_num,
                            "base_label": probe["label"]}
            all_records.append(rec)

            # Collect for plotting
            probe_curves[probe["label"]].append((step_num, rec.mean_top1_prob))
            probe_entropy_curves[probe["label"]].append((step_num, rec.mean_entropy))
            cat_data[probe["category"]][step_num]["probs"].append(rec.mean_top1_prob)
            cat_data[probe["category"]][step_num]["ents"].append(rec.mean_entropy)

    elapsed = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed:.1f}s")

    # Save
    if output_path.exists():
        output_path.unlink()
    save_records(all_records, output_path)

    # ===================================================================
    # Summary Table
    # ===================================================================
    print("\n" + "=" * 65)
    print("CONFIDENCE EVOLUTION SUMMARY")
    print("=" * 65)

    print(f"\n{'Step':<10}", end="")
    for probe in PROBES:
        print(f" {probe['label'][:10]:<11}", end="")
    print()
    print("-" * (10 + 11 * len(PROBES)))

    for ckpt in CHECKPOINTS:
        step_num = int(ckpt.replace("step", ""))
        print(f"{ckpt:<10}", end="")
        for probe in PROBES:
            # Find the record for this probe at this checkpoint
            mp = cat_data[probe["category"]][step_num]["probs"]
            # Get the specific probe's value
            for entry in probe_curves[probe["label"]]:
                if entry[0] == step_num:
                    print(f" {entry[1]:<11.4f}", end="")
                    break
        print()

    # ===================================================================
    # Key Observations
    # ===================================================================
    print("\n" + "=" * 65)
    print("KEY OBSERVATIONS")
    print("=" * 65)

    # When does true > false first appear?
    steps = sorted(list(cat_data["true_fact"].keys()))
    for s in steps:
        true_m = np.mean(cat_data["true_fact"][s]["probs"])
        false_m = np.mean(cat_data["false_statement"][s]["probs"])
        if true_m > false_m:
            print(f"\n  True > False confidence first at step {s}")
            print(f"    True mean P: {true_m:.4f}")
            print(f"    False mean P: {false_m:.4f}")
            break

    # Max separation
    max_gap = 0
    max_gap_step = 0
    for s in steps:
        true_m = np.mean(cat_data["true_fact"][s]["probs"])
        false_m = np.mean(cat_data["false_statement"][s]["probs"])
        gap = true_m - false_m
        if gap > max_gap:
            max_gap = gap
            max_gap_step = s
    print(f"\n  Max truth-falsehood gap: {max_gap:.4f} at step {max_gap_step}")

    # Entropy trend
    early_ent = np.mean(cat_data["true_fact"][steps[0]]["ents"])
    final_ent = np.mean(cat_data["true_fact"][steps[-1]]["ents"])
    print(f"\n  True facts entropy: {early_ent:.2f} bits (step {steps[0]}) "
          f"-> {final_ent:.2f} bits (step {steps[-1]})")

    early_ent_f = np.mean(cat_data["false_statement"][steps[0]]["ents"])
    final_ent_f = np.mean(cat_data["false_statement"][steps[-1]]["ents"])
    print(f"  False stmts entropy: {early_ent_f:.2f} bits (step {steps[0]}) "
          f"-> {final_ent_f:.2f} bits (step {steps[-1]})")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)

    print("\n[1/4] Per-probe confidence curves...")
    plot_training_curves(probe_curves, "mean_top1_prob",
                        "Mean P(actual token)",
                        "Confidence Over Training: Individual Probes",
                        "exp4_probe_confidence.png")
    plt.close("all")

    print("[2/4] Per-probe entropy curves...")
    plot_training_curves(probe_entropy_curves, "mean_entropy",
                        "Mean Entropy (bits)",
                        "Entropy Over Training: Individual Probes",
                        "exp4_probe_entropy.png")
    plt.close("all")

    print("[3/4] Category evolution...")
    plot_category_evolution(cat_data, "exp4_category_evolution.png")
    plt.close("all")

    print("[4/4] Truth-falsehood separation...")
    plot_separation_over_time(cat_data, "exp4_separation.png")
    plt.close("all")

    # ===================================================================
    # Summary
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(FIGURES_DIR.glob("exp4_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 65)
    print(f"  Checkpoints analyzed: {len(CHECKPOINTS)}")
    print(f"  Total analyses: {len(all_records)}")
    print(f"  Max truth-false gap: {max_gap:.4f} at step {max_gap_step}")
    print(f"  Figures: {fig_count} plots saved")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    run_experiment()
