"""
Experiment 6: Anomaly Detection (Context Injection)
====================================================
Can sudden confidence shifts detect emerging information conflicts?

Method:
  - Establish baseline confidence for factual claims
  - Prepend different context (neutral, supporting, contradicting, noisy)
  - Measure how the confidence profile changes
  - Test whether contradictory context creates a distinct signature
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
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text
from src.schema import save_records
from src.utils import RESULTS_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Base claims and their context variations
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "earth_sun",
        "base_claim": "The Earth orbits the Sun.",
        "contexts": {
            "none": "",
            "neutral": "According to basic astronomy textbooks,",
            "supporting": "As confirmed by centuries of astronomical observation,",
            "contradicting": "Despite what many scientists claim, some believe that",
            "noise": "In a surprising turn of events at the annual conference,",
        }
    },
    {
        "id": "water_boiling",
        "base_claim": "Water boils at 100 degrees Celsius.",
        "contexts": {
            "none": "",
            "neutral": "In standard chemistry,",
            "supporting": "It is a well-established fact that",
            "contradicting": "New research suggests that contrary to popular belief,",
            "noise": "After the recent scandal involving laboratory equipment,",
        }
    },
    {
        "id": "gravity",
        "base_claim": "Gravity causes objects to fall toward the ground.",
        "contexts": {
            "none": "",
            "neutral": "In physics class, students learn that",
            "supporting": "As Isaac Newton demonstrated centuries ago,",
            "contradicting": "A controversial new theory claims that actually",
            "noise": "Following the press conference at the ministry of science,",
        }
    },
    {
        "id": "smoking",
        "base_claim": "Smoking increases the risk of lung cancer.",
        "contexts": {
            "none": "",
            "neutral": "Health organizations report that",
            "supporting": "Decades of medical research have conclusively shown that",
            "contradicting": "Industry-funded research from the 1960s argued that",
            "noise": "In the latest quarterly earnings report discussion,",
        }
    },
    {
        "id": "evolution",
        "base_claim": "Evolution explains the diversity of life.",
        "contexts": {
            "none": "",
            "neutral": "Biologists generally agree that",
            "supporting": "The overwhelming evidence from genetics and fossils shows that",
            "contradicting": "Opponents of mainstream science have argued that",
            "noise": "At the annual festival of ideas and innovation,",
        }
    },
    {
        "id": "climate",
        "base_claim": "Human activity is causing climate change.",
        "contexts": {
            "none": "",
            "neutral": "According to climate scientists,",
            "supporting": "The scientific consensus based on thousands of studies is that",
            "contradicting": "Climate skeptics and some industry groups maintain that",
            "noise": "During the heated debate about educational funding,",
        }
    },
]

CONTEXT_TYPES = ["none", "neutral", "supporting", "contradicting", "noise"]
CONTEXT_COLORS = {
    "none": "#9E9E9E",
    "neutral": "#2196F3",
    "supporting": "#4CAF50",
    "contradicting": "#F44336",
    "noise": "#FF9800",
}


def plot_context_effects(results, save_name):
    """Show how each context type shifts confidence relative to baseline."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, scenario_id in enumerate([s["id"] for s in SCENARIOS]):
        ax = axes[idx]
        scenario_results = results[scenario_id]

        base_prob = scenario_results["none"].mean_top1_prob

        for ctx_type in CONTEXT_TYPES:
            rec = scenario_results[ctx_type]
            probs = [t.top1_prob for t in rec.tokens]
            ax.plot(range(len(probs)), probs, marker=".", markersize=3,
                    linewidth=1.2, color=CONTEXT_COLORS[ctx_type],
                    label=ctx_type, alpha=0.8)

        ax.set_title(scenario_id, fontsize=10)
        ax.set_ylabel("P(actual)")
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Confidence Profiles Under Different Contexts", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, axes


def plot_delta_summary(deltas, save_name):
    """Bar chart: mean confidence delta by context type across all scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ctx_types = ["neutral", "supporting", "contradicting", "noise"]
    scenario_ids = [s["id"] for s in SCENARIOS]

    x = np.arange(len(scenario_ids))
    width = 0.2

    for i, ctx in enumerate(ctx_types):
        vals = [deltas[sid][ctx] for sid in scenario_ids]
        ax.bar(x + i * width, vals, width, color=CONTEXT_COLORS[ctx],
               label=ctx, edgecolor="white")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(scenario_ids, rotation=20, ha="right")
    ax.set_ylabel("Delta Mean P (context - baseline)")
    ax.set_title("Confidence Shift by Context Type")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


def plot_contradiction_detection(deltas, save_name):
    """Can we distinguish contradicting context from others?"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Collect deltas for each context type across all scenarios
    by_context = defaultdict(list)
    for sid in [s["id"] for s in SCENARIOS]:
        for ctx in ["neutral", "supporting", "contradicting", "noise"]:
            by_context[ctx].append(deltas[sid][ctx])

    # Violin/strip plot
    import pandas as pd
    data = []
    for ctx, vals in by_context.items():
        for v in vals:
            data.append({"Context": ctx, "Delta P": v})
    df = pd.DataFrame(data)

    order = ["neutral", "supporting", "contradicting", "noise"]
    palette = {k: CONTEXT_COLORS[k] for k in order}

    sns.stripplot(data=df, x="Context", y="Delta P", order=order,
                  palette=palette, ax=ax, size=8, alpha=0.7, jitter=0.1)
    sns.boxplot(data=df, x="Context", y="Delta P", order=order,
                palette=palette, ax=ax, width=0.4, showfliers=False,
                boxprops=dict(alpha=0.3))

    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Context-Induced Confidence Shifts")
    ax.set_ylabel("Delta Mean P (vs no context)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


def run_experiment():
    output_path = RESULTS_DIR / "exp6_anomaly.jsonl"

    print("=" * 65)
    print("EXPERIMENT 6: Anomaly Detection (Context Injection)")
    print("=" * 65)
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Context types: {len(CONTEXT_TYPES)}")
    print(f"Total analyses: {len(SCENARIOS) * len(CONTEXT_TYPES)}")
    print()

    start_time = time.time()
    all_records = []
    results = defaultdict(dict)  # scenario_id -> {ctx_type: record}
    deltas = defaultdict(dict)   # scenario_id -> {ctx_type: delta_mean_prob}

    for scenario in tqdm(SCENARIOS, desc="Scenarios"):
        for ctx_type in CONTEXT_TYPES:
            ctx = scenario["contexts"][ctx_type]
            claim = scenario["base_claim"]
            full_text = f"{ctx} {claim}".strip() if ctx else claim

            rec = analyze_fixed_text(
                full_text,
                category=ctx_type,
                label=f"{scenario['id']}__{ctx_type}",
            )
            rec.metadata = {
                "scenario": scenario["id"],
                "context_type": ctx_type,
                "context": ctx,
                "claim": claim,
            }
            all_records.append(rec)
            results[scenario["id"]][ctx_type] = rec

    # Compute deltas relative to "none" baseline
    for scenario in SCENARIOS:
        sid = scenario["id"]
        base_prob = results[sid]["none"].mean_top1_prob
        for ctx_type in CONTEXT_TYPES:
            if ctx_type == "none":
                continue
            deltas[sid][ctx_type] = (results[sid][ctx_type].mean_top1_prob
                                     - base_prob)

    elapsed = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed:.1f}s")

    if output_path.exists():
        output_path.unlink()
    save_records(all_records, output_path)

    # ===================================================================
    # Summary Table
    # ===================================================================
    print("\n" + "=" * 65)
    print("CONFIDENCE BY CONTEXT TYPE")
    print("=" * 65)

    print(f"\n{'Scenario':<15}", end="")
    for ctx in CONTEXT_TYPES:
        print(f" {ctx:<14}", end="")
    print()
    print("-" * (15 + 14 * len(CONTEXT_TYPES)))

    for scenario in SCENARIOS:
        sid = scenario["id"]
        print(f"{sid:<15}", end="")
        for ctx in CONTEXT_TYPES:
            print(f" {results[sid][ctx].mean_top1_prob:<14.4f}", end="")
        print()

    # Delta table
    print(f"\n{'Scenario':<15}", end="")
    for ctx in ["neutral", "supporting", "contradicting", "noise"]:
        print(f" d_{ctx[:5]:<9}", end="")
    print()
    print("-" * 55)

    for scenario in SCENARIOS:
        sid = scenario["id"]
        print(f"{sid:<15}", end="")
        for ctx in ["neutral", "supporting", "contradicting", "noise"]:
            print(f" {deltas[sid][ctx]:>+10.4f}", end="")
        print()

    # ===================================================================
    # Statistical Analysis
    # ===================================================================
    print("\n" + "=" * 65)
    print("STATISTICAL ANALYSIS")
    print("=" * 65)

    # Does contradicting context reduce confidence more than others?
    contra_deltas = [deltas[s["id"]]["contradicting"] for s in SCENARIOS]
    support_deltas = [deltas[s["id"]]["supporting"] for s in SCENARIOS]
    neutral_deltas = [deltas[s["id"]]["neutral"] for s in SCENARIOS]
    noise_deltas = [deltas[s["id"]]["noise"] for s in SCENARIOS]

    print(f"\n  Mean delta by context type:")
    for name, vals in [("Neutral", neutral_deltas),
                       ("Supporting", support_deltas),
                       ("Contradicting", contra_deltas),
                       ("Noise", noise_deltas)]:
        print(f"    {name:<15} {np.mean(vals):+.4f} +/- {np.std(vals):.4f}")

    # Paired test: contradicting vs supporting
    t_cs, p_cs = stats.ttest_rel(contra_deltas, support_deltas)
    print(f"\n  Paired t-test (contradicting vs supporting):")
    print(f"    t={t_cs:.3f}, p={p_cs:.4f} "
          f"{'***' if p_cs < 0.05 else 'n.s.'}")

    # Paired test: contradicting vs noise
    t_cn, p_cn = stats.ttest_rel(contra_deltas, noise_deltas)
    print(f"  Paired t-test (contradicting vs noise):")
    print(f"    t={t_cn:.3f}, p={p_cn:.4f} "
          f"{'***' if p_cn < 0.05 else 'n.s.'}")

    # Is contradicting the most negative?
    contra_most_neg = sum(1 for s in SCENARIOS
                          if deltas[s["id"]]["contradicting"] ==
                          min(deltas[s["id"]].values()))
    print(f"\n  Contradicting has lowest delta in {contra_most_neg}/{len(SCENARIOS)} scenarios")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)

    print("\n[1/3] Context effects per scenario...")
    plot_context_effects(results, "exp6_context_effects.png")
    plt.close("all")

    print("[2/3] Delta summary bars...")
    plot_delta_summary(deltas, "exp6_delta_summary.png")
    plt.close("all")

    print("[3/3] Contradiction detection...")
    plot_contradiction_detection(deltas, "exp6_contradiction_detection.png")
    plt.close("all")

    # ===================================================================
    # Summary
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(FIGURES_DIR.glob("exp6_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT 6 COMPLETE")
    print("=" * 65)
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"  Mean contradicting delta: {np.mean(contra_deltas):+.4f}")
    print(f"  Mean supporting delta: {np.mean(support_deltas):+.4f}")
    print(f"  Contra vs Support p-value: {p_cs:.4f}")
    print(f"  Figures: {fig_count} plots saved")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    run_experiment()
