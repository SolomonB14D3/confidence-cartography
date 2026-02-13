"""
Experiment C: Mandela Effect
==============================
Test whether models are more confident about popular-but-wrong versions
of statements vs the actual correct versions.

Key metric: Mandela Rate = % of pairs where model prefers wrong-but-popular.
- Rate > 50% → confidence tracks consensus, not truth
- Rate < 50% → model can distinguish truth from popular misconception

Run at 160M first. If signal exists, scale across all Pythia sizes.
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

from src.engine import analyze_fixed_text, unload_model
from src.schema import save_records, load_records, ConfidenceRecord
from src.scaling import (
    MODEL_REGISTRY, SCALING_MODELS, PARAM_COUNTS,
    get_scaling_output_path, print_runtime_estimates,
)
from src.scaling_viz import (
    plot_scaling_law, MODEL_COLORS, model_display_name,
)
from src.utils import MANDELA_RESULTS_DIR, MANDELA_FIGURES_DIR


# ---------------------------------------------------------------------------
# Mandela Effect pairs: popular-but-wrong vs actual-correct
# ---------------------------------------------------------------------------

MANDELA_PAIRS = [
    # --- Movie misquotes ---
    {"popular": "Luke, I am your father.",
     "correct": "No, I am your father.",
     "id": "darth_vader", "domain": "movies"},
    {"popular": "Mirror, mirror on the wall.",
     "correct": "Magic mirror on the wall.",
     "id": "snow_white", "domain": "movies"},
    {"popular": "We're gonna need a bigger boat.",
     "correct": "You're gonna need a bigger boat.",
     "id": "jaws", "domain": "movies"},
    {"popular": "Life is like a box of chocolates.",
     "correct": "Life was like a box of chocolates.",
     "id": "forrest_gump", "domain": "movies"},
    {"popular": "Hello, Clarice.",
     "correct": "Good evening, Clarice.",
     "id": "silence_lambs", "domain": "movies"},

    # --- Proverbs / sayings ---
    {"popular": "Money is the root of all evil.",
     "correct": "The love of money is the root of all evil.",
     "id": "money_evil", "domain": "proverbs"},
    {"popular": "Curiosity killed the cat.",
     "correct": "Care killed the cat.",
     "id": "curiosity_cat", "domain": "proverbs"},
    {"popular": "Play it again, Sam.",
     "correct": "Play it, Sam.",
     "id": "play_it_sam", "domain": "movies"},

    # --- Visual memory ---
    {"popular": "The Berenstein Bears were a popular children's book series.",
     "correct": "The Berenstain Bears were a popular children's book series.",
     "id": "berenstain", "domain": "culture"},
    {"popular": "Curious George has always had a long tail.",
     "correct": "Curious George has never had a tail.",
     "id": "curious_george", "domain": "culture"},
    {"popular": "The Monopoly Man wears a monocle.",
     "correct": "The Monopoly Man does not wear a monocle.",
     "id": "monopoly_man", "domain": "culture"},
    {"popular": "The Fruit of the Loom logo has a cornucopia.",
     "correct": "The Fruit of the Loom logo has no cornucopia.",
     "id": "fruit_loom", "domain": "culture"},

    # --- Historical / factual ---
    {"popular": "Nelson Mandela died in prison in the 1980s.",
     "correct": "Nelson Mandela was released from prison in 1990.",
     "id": "mandela_death", "domain": "history"},
    {"popular": "Chartreuse is a shade of pink or magenta.",
     "correct": "Chartreuse is a shade of yellow-green.",
     "id": "chartreuse", "domain": "language"},
    {"popular": "Oscar Meyer is the correct spelling of the brand.",
     "correct": "Oscar Mayer is the correct spelling of the brand.",
     "id": "oscar_mayer", "domain": "brands"},
]


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_single_model(size: str, force: bool = False) -> list[ConfidenceRecord]:
    """Run Mandela Effect analysis for one model size."""
    output_path = MANDELA_RESULTS_DIR / f"mandela_{size}.jsonl"

    if output_path.exists() and not force:
        print(f"  [{size}] Results exist, loading from cache...")
        return load_records(output_path)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []

    print(f"\n  [{size}] Analyzing {len(MANDELA_PAIRS)} pairs with {model_name} (dtype={dtype})...")
    start = time.time()

    for pair in tqdm(MANDELA_PAIRS, desc=f"  {size}", leave=False):
        pop_rec = analyze_fixed_text(
            pair["popular"], category="popular_false",
            label=f"{pair['id']}_popular",
            model_name=model_name, revision="main", dtype=dtype,
        )
        pop_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                            "version": "popular"}

        cor_rec = analyze_fixed_text(
            pair["correct"], category="actual_correct",
            label=f"{pair['id']}_correct",
            model_name=model_name, revision="main", dtype=dtype,
        )
        cor_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                            "version": "correct"}

        records.extend([pop_rec, cor_rec])

    elapsed = time.time() - start
    print(f"  [{size}] Done in {elapsed:.1f}s ({len(records)} records)")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)

    return records


def compute_mandela_metrics(records: list[ConfidenceRecord]) -> dict:
    """Compute Mandela Effect metrics from records."""
    by_id = defaultdict(dict)
    for r in records:
        pair_id = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
        version = r.metadata.get("version", "popular" if "popular" in r.label else "correct")
        by_id[pair_id][version] = r

    # Compute pair-level metrics
    deltas = []
    pair_results = []

    for pid, versions in by_id.items():
        if "popular" in versions and "correct" in versions:
            pop_p = versions["popular"].mean_top1_prob
            cor_p = versions["correct"].mean_top1_prob
            delta = pop_p - cor_p  # positive = model prefers popular (wrong)
            deltas.append(delta)
            pair_results.append({
                "pair_id": pid,
                "popular_prob": pop_p,
                "correct_prob": cor_p,
                "delta": delta,
                "mandela_wins": pop_p > cor_p,
            })

    deltas = np.array(deltas)
    n_pairs = len(deltas)
    mandela_wins = int(np.sum(deltas > 0))

    mandela_rate = mandela_wins / n_pairs if n_pairs > 0 else 0

    # Statistical test: is delta different from 0?
    if n_pairs > 5:
        t_stat, p_val = stats.ttest_1samp(deltas, 0)
    else:
        t_stat, p_val = 0.0, 1.0

    return {
        "n_pairs": n_pairs,
        "mandela_rate": mandela_rate,
        "mandela_wins": mandela_wins,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "p_value": p_val,
        "t_stat": t_stat,
        "pair_results": pair_results,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_mandela_deltas(pair_results: list[dict], size: str, save_path: Path):
    """Bar chart of per-pair deltas (popular - correct confidence)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    sorted_pairs = sorted(pair_results, key=lambda x: x["delta"], reverse=True)
    ids = [p["pair_id"] for p in sorted_pairs]
    deltas = [p["delta"] for p in sorted_pairs]
    colors = ["#F44336" if d > 0 else "#4CAF50" for d in deltas]

    ax.bar(range(len(ids)), deltas, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Delta Mean P: Popular(wrong) - Correct")
    ax.set_title(f"Mandela Effect at {model_display_name(size)}: "
                 f"Red=model prefers wrong, Green=model prefers correct")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#F44336", label="Mandela wins (prefers popular wrong)"),
        Patch(color="#4CAF50", label="Truth wins (prefers correct)"),
    ], fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(models: list[str] = None, force: bool = False):
    models = models or SCALING_MODELS

    print("=" * 65)
    print("EXPERIMENT C: Mandela Effect")
    print("=" * 65)
    print(f"Models: {', '.join(models)}")
    print(f"Pairs: {len(MANDELA_PAIRS)}")

    start_time = time.time()
    all_metrics = {}

    for size in models:
        records = run_single_model(size, force=force)
        metrics = compute_mandela_metrics(records)
        all_metrics[size] = metrics
        unload_model()

        print(f"  [{size}] Mandela Rate: {metrics['mandela_wins']}/{metrics['n_pairs']} "
              f"({metrics['mandela_rate']:.1%}), "
              f"Mean delta: {metrics['mean_delta']:+.4f}, "
              f"p={metrics['p_value']:.4f}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 65)
    print("MANDELA EFFECT SCALING SUMMARY")
    print("=" * 65)

    sizes_done = [s for s in models if s in all_metrics]
    print(f"\n{'Size':<8} {'Params':<12} {'Mandela%':<10} {'Mean dP':<10} "
          f"{'p-value':<10} {'Verdict':<20}")
    print("-" * 70)
    for size in sizes_done:
        m = all_metrics[size]
        params = PARAM_COUNTS[size]
        verdict = ("CONSENSUS WINS" if m["mandela_rate"] > 0.6
                    else "TRUTH WINS" if m["mandela_rate"] < 0.4
                    else "INCONCLUSIVE")
        print(f"{size:<8} {params/1e6:>8.0f}M  {m['mandela_rate']:<10.1%} "
              f"{m['mean_delta']:<+10.4f} {m['p_value']:<10.4f} {verdict:<20}")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING PLOTS")
    print("=" * 65)

    # Per-model delta bar charts
    for size in sizes_done:
        m = all_metrics[size]
        plot_mandela_deltas(
            m["pair_results"], size,
            MANDELA_FIGURES_DIR / f"c1_mandela_deltas_{size}.png")

    # Scaling curve: Mandela Rate vs model size
    if len(sizes_done) >= 2:
        print("\n  Mandela Rate scaling curve...")
        plot_scaling_law(
            sizes_done,
            {"Mandela Rate": [all_metrics[s]["mandela_rate"] for s in sizes_done]},
            ylabel="Mandela Rate (popular > correct)",
            title="Mandela Effect vs Model Size",
            save_path=MANDELA_FIGURES_DIR / "c3_mandela_scaling.png",
            hline=0.5, hline_label="No bias (50%)",
        )

        # Mean delta scaling
        plot_scaling_law(
            sizes_done,
            {"Mean delta P": [all_metrics[s]["mean_delta"] for s in sizes_done]},
            ylabel="Mean P(popular) - P(correct)",
            title="Mandela Confidence Gap vs Model Size",
            save_path=MANDELA_FIGURES_DIR / "c3_mandela_delta_scaling.png",
            hline=0.0, hline_label="No bias",
        )

    # ===================================================================
    # Final
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(MANDELA_FIGURES_DIR.glob("*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT C COMPLETE")
    print("=" * 65)
    print(f"  Models: {len(sizes_done)}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s")

    if sizes_done:
        rates = [all_metrics[s]["mandela_rate"] for s in sizes_done]
        if max(rates) > 0.6:
            print("\n  FINDING: Model prefers popular-but-wrong versions!")
            print("  → Confidence tracks CONSENSUS, not TRUTH")
        elif min(rates) < 0.4:
            print("\n  FINDING: Model prefers correct versions!")
            print("  → Confidence tracks TRUTH over consensus")
        else:
            print("\n  FINDING: No clear pattern — Mandela Effect inconclusive")


if __name__ == "__main__":
    # Default: run 160M first, then all if signal exists
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all model sizes")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    args = parser.parse_args()

    if args.all:
        run_experiment(SCALING_MODELS, force=args.force)
    else:
        run_experiment(["160m"], force=args.force)
