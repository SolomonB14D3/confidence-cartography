"""
Experiment: Mandela Effect Calibration
========================================
Test whether model confidence ratios correlate with the measured prevalence
of false beliefs in the human population, using YouGov survey data as ground truth.

Core hypothesis: The model isn't just "biased toward misconceptions" — its
confidence tracks the *popularity* of each misconception proportionally.
If true, the model is a calibrated consensus sensor, not just a noisy one.

Ground truth: YouGov August 2022 poll (n=1,000 US adults).
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
)
from src.scaling_viz import (
    plot_scaling_law, MODEL_COLORS, model_display_name,
)
from src.utils import MANDELA_FIGURES_DIR, MANDELA_RESULTS_DIR


# ---------------------------------------------------------------------------
# YouGov ground truth data (August 2022, n=1,000 US adults)
# ---------------------------------------------------------------------------

MANDELA_ITEMS = [
    {
        "id": "star_wars",
        "wrong": "Luke, I am your father",
        "correct": "No, I am your father",
        "wrong_prevalence": 0.62,
        "correct_prevalence": 0.17,
        "context": 'In Star Wars, Darth Vader says "{quote}"',
    },
    {
        "id": "berenstain",
        "wrong": "The Berenstein Bears",
        "correct": "The Berenstain Bears",
        "wrong_prevalence": 0.61,
        "correct_prevalence": 0.28,
        "context": 'The children\'s book series is called "{quote}"',
    },
    {
        "id": "monopoly_monocle",
        "wrong": "The Monopoly Man wears a monocle",
        "correct": "The Monopoly Man does not wear a monocle",
        "wrong_prevalence": 0.58,
        "correct_prevalence": 0.31,
        "context": "{quote}",
    },
    {
        "id": "fruit_of_loom",
        "wrong": "The Fruit of the Loom logo has a cornucopia",
        "correct": "The Fruit of the Loom logo does not have a cornucopia",
        "wrong_prevalence": 0.53,
        "correct_prevalence": 0.33,
        "context": "{quote}",
    },
    {
        "id": "curious_george",
        "wrong": "Curious George has a tail",
        "correct": "Curious George does not have a tail",
        "wrong_prevalence": 0.55,
        "correct_prevalence": 0.30,
        "context": "{quote}",
    },
    {
        "id": "risky_business",
        "wrong": "In Risky Business, Tom Cruise dances in a white button-down shirt and sunglasses",
        "correct": "In Risky Business, Tom Cruise dances in a pink button-down shirt without sunglasses",
        "wrong_prevalence": 0.55,
        "correct_prevalence": 0.16,
        "context": "{quote}",
    },
    {
        "id": "we_are_champions",
        "wrong": 'We Are the Champions ends with "of the world"',
        "correct": 'We Are the Champions does not end with "of the world"',
        "wrong_prevalence": 0.52,
        "correct_prevalence": 0.22,
        "context": "{quote}",
    },
    {
        "id": "froot_loops",
        "wrong": "The cereal is called Fruit Loops",
        "correct": "The cereal is called Froot Loops",
        "wrong_prevalence": 0.44,
        "correct_prevalence": 0.37,
        "context": "{quote}",
    },
    {
        "id": "mandela_death",
        "wrong": "Nelson Mandela died in prison in the 1980s",
        "correct": "Nelson Mandela died in 2013 after serving as president of South Africa",
        "wrong_prevalence": 0.13,
        "correct_prevalence": 0.57,
        "context": "{quote}",
    },
]

# Pre-compute normalized human prevalence ratios
for item in MANDELA_ITEMS:
    item["human_ratio"] = item["wrong_prevalence"] / (
        item["wrong_prevalence"] + item["correct_prevalence"]
    )

# Output subdirectory for this experiment
CALIB_RESULTS_DIR = MANDELA_RESULTS_DIR / "calibration"
CALIB_FIGURES_DIR = MANDELA_FIGURES_DIR / "calibration"
CALIB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CALIB_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Framing variants — robustness check
# ---------------------------------------------------------------------------

def _make_texts(item: dict) -> list[tuple[str, str, str]]:
    """Return list of (framing_name, wrong_text, correct_text) for one item."""
    framings = []

    # 1. Raw text (just the statement)
    framings.append(("raw", item["wrong"], item["correct"]))

    # 2. Context-embedded
    if item["context"] != "{quote}":
        w_ctx = item["context"].format(quote=item["wrong"])
        c_ctx = item["context"].format(quote=item["correct"])
        framings.append(("context", w_ctx, c_ctx))

    return framings


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_single_model(size: str, force: bool = False) -> dict:
    """Run calibration analysis for one model size.

    Returns dict mapping (item_id, framing) -> {wrong_conf, correct_conf, ...}
    """
    output_path = CALIB_RESULTS_DIR / f"calibration_{size}.jsonl"

    if output_path.exists() and not force:
        print(f"  [{size}] Results cached, loading...")
        records = load_records(output_path)
        return _records_to_pairs(records)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []

    n_texts = sum(len(_make_texts(item)) * 2 for item in MANDELA_ITEMS)
    print(f"\n  [{size}] Analyzing {n_texts} texts with {model_name}...")
    start = time.time()

    for item in tqdm(MANDELA_ITEMS, desc=f"  {size}", leave=False):
        for framing_name, wrong_text, correct_text in _make_texts(item):
            w_rec = analyze_fixed_text(
                wrong_text,
                category="mandela_wrong",
                label=f"{item['id']}_{framing_name}_wrong",
                model_name=model_name, revision="main", dtype=dtype,
            )
            w_rec.metadata = {
                "item_id": item["id"],
                "framing": framing_name,
                "version": "wrong",
                "human_ratio": item["human_ratio"],
                "wrong_prevalence": item["wrong_prevalence"],
                "correct_prevalence": item["correct_prevalence"],
            }

            c_rec = analyze_fixed_text(
                correct_text,
                category="mandela_correct",
                label=f"{item['id']}_{framing_name}_correct",
                model_name=model_name, revision="main", dtype=dtype,
            )
            c_rec.metadata = {
                "item_id": item["id"],
                "framing": framing_name,
                "version": "correct",
                "human_ratio": item["human_ratio"],
                "wrong_prevalence": item["wrong_prevalence"],
                "correct_prevalence": item["correct_prevalence"],
            }

            records.extend([w_rec, c_rec])

    elapsed = time.time() - start
    print(f"  [{size}] Done in {elapsed:.1f}s ({len(records)} records)")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)

    return _records_to_pairs(records)


def _records_to_pairs(records: list[ConfidenceRecord]) -> dict:
    """Group records into (item_id, framing) -> {wrong_conf, correct_conf, ...}"""
    by_key = defaultdict(dict)
    for r in records:
        item_id = r.metadata["item_id"]
        framing = r.metadata["framing"]
        version = r.metadata["version"]
        key = (item_id, framing)
        by_key[key][version] = r

    pairs = {}
    for key, versions in by_key.items():
        if "wrong" not in versions or "correct" not in versions:
            continue
        w_conf = versions["wrong"].mean_top1_prob
        c_conf = versions["correct"].mean_top1_prob
        conf_ratio = w_conf / (w_conf + c_conf) if (w_conf + c_conf) > 0 else 0.5
        human_ratio = versions["wrong"].metadata["human_ratio"]

        pairs[key] = {
            "item_id": key[0],
            "framing": key[1],
            "wrong_conf": w_conf,
            "correct_conf": c_conf,
            "confidence_ratio": conf_ratio,
            "human_ratio": human_ratio,
            "wrong_prevalence": versions["wrong"].metadata["wrong_prevalence"],
            "correct_prevalence": versions["wrong"].metadata["correct_prevalence"],
        }
    return pairs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_calibration_scatter(pairs: dict, size: str, save_path: Path):
    """Scatter: X = human prevalence ratio, Y = model confidence ratio.
    Diagonal = perfect calibration."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Filter to raw framing for the primary plot
    raw_pairs = {k: v for k, v in pairs.items() if v["framing"] == "raw"}
    if not raw_pairs:
        raw_pairs = pairs  # fallback

    human_ratios = [v["human_ratio"] for v in raw_pairs.values()]
    model_ratios = [v["confidence_ratio"] for v in raw_pairs.values()]
    labels = [v["item_id"] for v in raw_pairs.values()]

    ax.scatter(human_ratios, model_ratios, s=80, zorder=5, alpha=0.8, color="#2196F3")

    # Label each point
    for x, y, lbl in zip(human_ratios, model_ratios, labels):
        ax.annotate(lbl, (x, y), fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")

    # No-preference line
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.3)

    # Correlation
    r, p = stats.pearsonr(human_ratios, model_ratios)
    rho, rho_p = stats.spearmanr(human_ratios, model_ratios)
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f} (p = {p:.3f})\n"
                         f"Spearman ρ = {rho:.3f} (p = {rho_p:.3f})",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Human Prevalence Ratio (YouGov)", fontsize=12)
    ax.set_ylabel("Model Confidence Ratio", fontsize=12)
    ax.set_title(f"Mandela Calibration — {model_display_name(size)}", fontsize=14)
    ax.set_xlim(0.1, 0.85)
    ax.set_ylim(0.1, 0.85)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_item_comparison(pairs: dict, size: str, save_path: Path):
    """Bar chart: human prevalence ratio vs model confidence ratio, per item."""
    sns.set_theme(style="whitegrid", palette="muted")

    raw_pairs = {k: v for k, v in pairs.items() if v["framing"] == "raw"}
    if not raw_pairs:
        raw_pairs = pairs

    # Sort by human prevalence ratio descending
    sorted_items = sorted(raw_pairs.values(), key=lambda x: x["human_ratio"], reverse=True)

    ids = [p["item_id"] for p in sorted_items]
    human_r = [p["human_ratio"] for p in sorted_items]
    model_r = [p["confidence_ratio"] for p in sorted_items]

    x = np.arange(len(ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, human_r, width, label="Human (YouGov)", color="#FF9800", alpha=0.85)
    ax.bar(x + width / 2, model_r, width, label=f"Model ({model_display_name(size)})",
           color="#2196F3", alpha=0.85)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Wrong / (Wrong + Correct) Ratio", fontsize=11)
    ax.set_title(f"Human vs Model Misconception Ratios — {model_display_name(size)}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_framing_robustness(pairs: dict, size: str, save_path: Path):
    """Check if raw vs context framing gives similar results."""
    raw = {k[0]: v for k, v in pairs.items() if v["framing"] == "raw"}
    ctx = {k[0]: v for k, v in pairs.items() if v["framing"] == "context"}

    common_ids = sorted(set(raw) & set(ctx))
    if len(common_ids) < 2:
        return  # not enough context-framed items to plot

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(7, 7))

    raw_ratios = [raw[i]["confidence_ratio"] for i in common_ids]
    ctx_ratios = [ctx[i]["confidence_ratio"] for i in common_ids]

    ax.scatter(raw_ratios, ctx_ratios, s=80, zorder=5, color="#9C27B0")
    for r_val, c_val, lbl in zip(raw_ratios, ctx_ratios, common_ids):
        ax.annotate(lbl, (r_val, c_val), fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    r, p = stats.pearsonr(raw_ratios, ctx_ratios)
    ax.text(0.05, 0.95, f"r = {r:.3f} (p = {p:.3f})",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lavender", alpha=0.5))

    ax.set_xlabel("Raw Framing Confidence Ratio", fontsize=11)
    ax.set_ylabel("Context Framing Confidence Ratio", fontsize=11)
    ax.set_title(f"Framing Robustness — {model_display_name(size)}", fontsize=13)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_scaling(all_results: dict, save_path: Path):
    """Correlation coefficient vs model size."""
    sizes = [s for s in SCALING_MODELS if s in all_results]
    if len(sizes) < 2:
        return

    pearson_rs = []
    spearman_rhos = []

    for size in sizes:
        pairs = all_results[size]
        raw_pairs = {k: v for k, v in pairs.items() if v["framing"] == "raw"}
        if not raw_pairs:
            raw_pairs = pairs
        h = [v["human_ratio"] for v in raw_pairs.values()]
        m = [v["confidence_ratio"] for v in raw_pairs.values()]
        r, _ = stats.pearsonr(h, m)
        rho, _ = stats.spearmanr(h, m)
        pearson_rs.append(r)
        spearman_rhos.append(rho)

    plot_scaling_law(
        sizes,
        {"Pearson r": pearson_rs, "Spearman ρ": spearman_rhos},
        ylabel="Correlation with Human Prevalence",
        title="Calibration Correlation vs Model Size",
        save_path=save_path,
        hline=0.0, hline_label="No correlation",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(models: list[str] = None, force: bool = False):
    models = models or SCALING_MODELS

    print("=" * 65)
    print("MANDELA CALIBRATION EXPERIMENT")
    print("=" * 65)
    print(f"Models: {', '.join(models)}")
    print(f"Items: {len(MANDELA_ITEMS)}")
    print(f"Human ratio range: {min(i['human_ratio'] for i in MANDELA_ITEMS):.2f} "
          f"– {max(i['human_ratio'] for i in MANDELA_ITEMS):.2f}")

    start_time = time.time()
    all_results = {}

    for size in models:
        pairs = run_single_model(size, force=force)
        all_results[size] = pairs
        unload_model()

        # Quick summary
        raw_pairs = {k: v for k, v in pairs.items() if v["framing"] == "raw"}
        if not raw_pairs:
            raw_pairs = pairs
        h = [v["human_ratio"] for v in raw_pairs.values()]
        m = [v["confidence_ratio"] for v in raw_pairs.values()]
        r, p = stats.pearsonr(h, m)
        rho, rho_p = stats.spearmanr(h, m)
        print(f"  [{size}] Pearson r={r:.3f} (p={p:.3f}), "
              f"Spearman ρ={rho:.3f} (p={rho_p:.3f})")

    # ===================================================================
    # Summary table
    # ===================================================================
    print("\n" + "=" * 65)
    print("CALIBRATION SUMMARY")
    print("=" * 65)

    sizes_done = [s for s in models if s in all_results]
    print(f"\n{'Size':<8} {'Params':<12} {'Pearson r':<12} {'p-value':<10} "
          f"{'Spearman ρ':<12} {'p-value':<10} {'Verdict':<20}")
    print("-" * 84)

    for size in sizes_done:
        pairs = all_results[size]
        raw_pairs = {k: v for k, v in pairs.items() if v["framing"] == "raw"}
        if not raw_pairs:
            raw_pairs = pairs
        h = [v["human_ratio"] for v in raw_pairs.values()]
        m = [v["confidence_ratio"] for v in raw_pairs.values()]
        r, p = stats.pearsonr(h, m)
        rho, rho_p = stats.spearmanr(h, m)
        params = PARAM_COUNTS[size]

        if r > 0.6 and p < 0.05:
            verdict = "CALIBRATED SENSOR"
        elif r > 0.3:
            verdict = "MODERATE SIGNAL"
        elif r < -0.3:
            verdict = "INVERSE SIGNAL"
        else:
            verdict = "NO SIGNAL"

        print(f"{size:<8} {params/1e6:>8.0f}M  {r:<12.3f} {p:<10.3f} "
              f"{rho:<12.3f} {rho_p:<10.3f} {verdict:<20}")

    # Per-item details for largest model
    largest = sizes_done[-1]
    raw_pairs = {k: v for k, v in all_results[largest].items() if v["framing"] == "raw"}
    if not raw_pairs:
        raw_pairs = all_results[largest]

    print(f"\n{'Item':<20} {'Human Ratio':<14} {'Model Ratio':<14} {'Gap':<10}")
    print("-" * 58)
    for v in sorted(raw_pairs.values(), key=lambda x: x["human_ratio"], reverse=True):
        gap = v["confidence_ratio"] - v["human_ratio"]
        print(f"{v['item_id']:<20} {v['human_ratio']:<14.3f} "
              f"{v['confidence_ratio']:<14.3f} {gap:<+10.3f}")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING PLOTS")
    print("=" * 65)

    for size in sizes_done:
        pairs = all_results[size]
        print(f"  Scatter plot for {size}...")
        plot_calibration_scatter(
            pairs, size,
            CALIB_FIGURES_DIR / f"calibration_scatter_{size}.png",
        )
        print(f"  Item comparison for {size}...")
        plot_item_comparison(
            pairs, size,
            CALIB_FIGURES_DIR / f"calibration_items_{size}.png",
        )
        print(f"  Framing robustness for {size}...")
        plot_framing_robustness(
            pairs, size,
            CALIB_FIGURES_DIR / f"calibration_framing_{size}.png",
        )

    if len(sizes_done) >= 2:
        print("  Correlation scaling curve...")
        plot_correlation_scaling(
            all_results,
            CALIB_FIGURES_DIR / "calibration_correlation_scaling.png",
        )

    # ===================================================================
    # Final
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(CALIB_FIGURES_DIR.glob("*.png")))

    print("\n" + "=" * 65)
    print("MANDELA CALIBRATION COMPLETE")
    print("=" * 65)
    print(f"  Models: {len(sizes_done)}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s")

    # Headline finding
    if sizes_done:
        pairs = all_results[sizes_done[-1]]
        raw_pairs = {k: v for k, v in pairs.items() if v["framing"] == "raw"}
        if not raw_pairs:
            raw_pairs = pairs
        h = [v["human_ratio"] for v in raw_pairs.values()]
        m = [v["confidence_ratio"] for v in raw_pairs.values()]
        r, p = stats.pearsonr(h, m)

        if r > 0.6 and p < 0.05:
            print(f"\n  FINDING: Strong correlation (r={r:.3f}, p={p:.3f})")
            print("  → Model is a CALIBRATED CONSENSUS SENSOR")
        elif r > 0.3:
            print(f"\n  FINDING: Moderate correlation (r={r:.3f}, p={p:.3f})")
            print("  → Directionally correct but noisy")
        elif r < -0.3:
            print(f"\n  FINDING: Inverse correlation (r={r:.3f})")
            print("  → Model actively avoids popular misconceptions")
        else:
            print(f"\n  FINDING: No meaningful correlation (r={r:.3f}, p={p:.3f})")
            print("  → Dead end — model confidence doesn't track misconception popularity")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all model sizes")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    args = parser.parse_args()

    if args.all:
        run_experiment(SCALING_MODELS, force=args.force)
    else:
        run_experiment(SCALING_MODELS, force=args.force)
