"""
Mandela Effect Reanalysis: Linguistic vs Visual Split
======================================================
The calibration experiment showed berenstain and curious_george as clear
outliers — both are visual/perceptual Mandela Effects where the false belief
lives in how things *look*, not how things are *written*. The model has no
visual memory, so it correctly prefers the text-accurate version regardless
of human false belief prevalence.

This reanalysis splits items into linguistic (text-footprint) vs visual
(perceptual) subsets and recomputes correlations for each.
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import load_records
from src.scaling import SCALING_MODELS, PARAM_COUNTS
from src.scaling_viz import plot_scaling_law, model_display_name
from src.utils import MANDELA_RESULTS_DIR, MANDELA_FIGURES_DIR


# ---------------------------------------------------------------------------
# Item classification
# ---------------------------------------------------------------------------

LINGUISTIC_ITEMS = [
    "star_wars",        # "Luke, I am your father" is written constantly
    "we_are_champions", # People write "of the world" in quotes
    "risky_business",   # Described in text as white shirt + sunglasses
    "mandela_death",    # "died in prison" was written/discussed
]

VISUAL_ITEMS = [
    "berenstain",       # Spelling — text always has the correct spelling
    "monopoly_monocle", # Visual detail — text descriptions vary
    "fruit_of_loom",    # Visual detail — logo description
    "curious_george",   # Visual detail — tail presence
    "froot_loops",      # Spelling — packaging always has correct spelling
]

CALIB_RESULTS_DIR = MANDELA_RESULTS_DIR / "calibration"
REANALYSIS_FIGURES_DIR = MANDELA_FIGURES_DIR / "reanalysis"
REANALYSIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load and reconstruct pairs from cached calibration results
# ---------------------------------------------------------------------------

def load_calibration_pairs(size: str) -> dict:
    """Load calibration results and reconstruct pair dict."""
    path = CALIB_RESULTS_DIR / f"calibration_{size}.jsonl"
    if not path.exists():
        return {}

    records = load_records(path)
    by_key = defaultdict(dict)
    for r in records:
        item_id = r.metadata["item_id"]
        framing = r.metadata["framing"]
        version = r.metadata["version"]
        by_key[(item_id, framing)][version] = r

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
        }
    return pairs


def filter_raw(pairs: dict, item_ids: list[str]) -> list[dict]:
    """Filter to raw framing and specific item IDs."""
    return [v for k, v in pairs.items()
            if v["framing"] == "raw" and v["item_id"] in item_ids]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_split_scatter(pairs: dict, size: str, save_path: Path):
    """Scatter plot with linguistic and visual items colored differently."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 8))

    ling = filter_raw(pairs, LINGUISTIC_ITEMS)
    vis = filter_raw(pairs, VISUAL_ITEMS)

    # Plot visual items
    if vis:
        vx = [v["human_ratio"] for v in vis]
        vy = [v["confidence_ratio"] for v in vis]
        ax.scatter(vx, vy, s=90, color="#FF9800", alpha=0.85, zorder=5,
                   label="Visual/Perceptual", marker="s")
        for x, y, v in zip(vx, vy, vis):
            ax.annotate(v["item_id"], (x, y), fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points", color="#E65100")

    # Plot linguistic items
    if ling:
        lx = [v["human_ratio"] for v in ling]
        ly = [v["confidence_ratio"] for v in ling]
        ax.scatter(lx, ly, s=90, color="#2196F3", alpha=0.85, zorder=6,
                   label="Linguistic", marker="o")
        for x, y, v in zip(lx, ly, ling):
            ax.annotate(v["item_id"], (x, y), fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points", color="#0D47A1")

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.3)

    # Correlation stats
    lines = []
    if len(ling) >= 3:
        r, p = stats.pearsonr([v["human_ratio"] for v in ling],
                              [v["confidence_ratio"] for v in ling])
        rho, rho_p = stats.spearmanr([v["human_ratio"] for v in ling],
                                     [v["confidence_ratio"] for v in ling])
        lines.append(f"Linguistic: r={r:.3f} (p={p:.3f}), ρ={rho:.3f}")
    if len(vis) >= 3:
        r, p = stats.pearsonr([v["human_ratio"] for v in vis],
                              [v["confidence_ratio"] for v in vis])
        rho, rho_p = stats.spearmanr([v["human_ratio"] for v in vis],
                                     [v["confidence_ratio"] for v in vis])
        lines.append(f"Visual: r={r:.3f} (p={p:.3f}), ρ={rho:.3f}")

    all_items = ling + vis
    if len(all_items) >= 3:
        r, p = stats.pearsonr([v["human_ratio"] for v in all_items],
                              [v["confidence_ratio"] for v in all_items])
        lines.append(f"Combined: r={r:.3f} (p={p:.3f})")

    if lines:
        ax.text(0.05, 0.95, "\n".join(lines),
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Human Prevalence Ratio (YouGov)", fontsize=12)
    ax.set_ylabel("Model Confidence Ratio", fontsize=12)
    ax.set_title(f"Linguistic vs Visual Split — {model_display_name(size)}", fontsize=14)
    ax.set_xlim(0.1, 0.85)
    ax.set_ylim(0.1, 0.85)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subset_scaling(all_results: dict, save_path: Path):
    """Correlation vs model size for linguistic, visual, and combined."""
    sizes = [s for s in SCALING_MODELS if s in all_results]
    if len(sizes) < 2:
        return

    ling_pearson = []
    vis_pearson = []
    combined_pearson = []
    ling_spearman = []
    vis_spearman = []
    combined_spearman = []

    for size in sizes:
        pairs = all_results[size]
        ling = filter_raw(pairs, LINGUISTIC_ITEMS)
        vis = filter_raw(pairs, VISUAL_ITEMS)
        all_items = ling + vis

        if len(ling) >= 3:
            r, _ = stats.pearsonr([v["human_ratio"] for v in ling],
                                  [v["confidence_ratio"] for v in ling])
            rho, _ = stats.spearmanr([v["human_ratio"] for v in ling],
                                     [v["confidence_ratio"] for v in ling])
            ling_pearson.append(r)
            ling_spearman.append(rho)
        else:
            ling_pearson.append(np.nan)
            ling_spearman.append(np.nan)

        if len(vis) >= 3:
            r, _ = stats.pearsonr([v["human_ratio"] for v in vis],
                                  [v["confidence_ratio"] for v in vis])
            rho, _ = stats.spearmanr([v["human_ratio"] for v in vis],
                                     [v["confidence_ratio"] for v in vis])
            vis_pearson.append(r)
            vis_spearman.append(rho)
        else:
            vis_pearson.append(np.nan)
            vis_spearman.append(np.nan)

        if len(all_items) >= 3:
            r, _ = stats.pearsonr([v["human_ratio"] for v in all_items],
                                  [v["confidence_ratio"] for v in all_items])
            rho, _ = stats.spearmanr([v["human_ratio"] for v in all_items],
                                     [v["confidence_ratio"] for v in all_items])
            combined_pearson.append(r)
            combined_spearman.append(rho)
        else:
            combined_pearson.append(np.nan)
            combined_spearman.append(np.nan)

    # Pearson scaling
    sns.set_theme(style="whitegrid", palette="muted")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    params = [PARAM_COUNTS[s] for s in sizes]
    labels = [model_display_name(s) for s in sizes]

    for ax, pearson_data, spearman_data, title_suffix in [
        (ax1, [ling_pearson, vis_pearson, combined_pearson], None, "Pearson r"),
        (ax2, None, [ling_spearman, vis_spearman, combined_spearman], "Spearman ρ"),
    ]:
        data = pearson_data if pearson_data else spearman_data
        names = ["Linguistic", "Visual", "Combined"]
        colors = ["#2196F3", "#FF9800", "#757575"]

        for vals, name, color in zip(data, names, colors):
            valid = [(p, v) for p, v in zip(params, vals) if not np.isnan(v)]
            if valid:
                px, vx = zip(*valid)
                ax.plot(px, vx, "o-", linewidth=2.5, markersize=8,
                        color=color, label=name)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters", fontsize=12)
        ax.set_ylabel(title_suffix, fontsize=12)
        ax.set_title(f"{title_suffix} vs Model Size", fontsize=13)
        ax.set_xticks(params)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.05, 1.05)

    fig.suptitle("Mandela Effect: Linguistic vs Visual Correlation Scaling", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side_bars(pairs: dict, size: str, save_path: Path):
    """Side-by-side bar charts for linguistic and visual subsets."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, item_ids, title, color in [
        (ax1, LINGUISTIC_ITEMS, "Linguistic Items", "#2196F3"),
        (ax2, VISUAL_ITEMS, "Visual/Perceptual Items", "#FF9800"),
    ]:
        items = filter_raw(pairs, item_ids)
        items = sorted(items, key=lambda x: x["human_ratio"], reverse=True)
        if not items:
            continue

        ids = [v["item_id"] for v in items]
        human_r = [v["human_ratio"] for v in items]
        model_r = [v["confidence_ratio"] for v in items]

        x = np.arange(len(ids))
        width = 0.35

        ax.bar(x - width / 2, human_r, width, label="Human (YouGov)",
               color="#FF9800", alpha=0.85)
        ax.bar(x + width / 2, model_r, width,
               label=f"Model ({model_display_name(size)})",
               color=color, alpha=0.85)

        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Wrong / (Wrong + Correct)", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)

    fig.suptitle(f"Human vs Model by Item Type — {model_display_name(size)}",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_reanalysis():
    print("=" * 65)
    print("MANDELA EFFECT REANALYSIS: Linguistic vs Visual Split")
    print("=" * 65)
    print(f"Linguistic items ({len(LINGUISTIC_ITEMS)}): {', '.join(LINGUISTIC_ITEMS)}")
    print(f"Visual items ({len(VISUAL_ITEMS)}): {', '.join(VISUAL_ITEMS)}")

    all_results = {}
    available_sizes = []

    for size in SCALING_MODELS:
        pairs = load_calibration_pairs(size)
        if pairs:
            all_results[size] = pairs
            available_sizes.append(size)

    if not available_sizes:
        print("\nERROR: No calibration results found. Run exp_mandela_calibration.py first.")
        return

    print(f"\nLoaded results for: {', '.join(available_sizes)}")

    # ===================================================================
    # Per-model analysis
    # ===================================================================
    print("\n" + "=" * 65)
    print("CORRELATION BY SUBSET")
    print("=" * 65)

    header = (f"{'Size':<8} {'Subset':<14} {'n':<4} {'Pearson r':<12} {'p-value':<10} "
              f"{'Spearman ρ':<12} {'p-value':<10}")
    print(f"\n{header}")
    print("-" * 70)

    for size in available_sizes:
        pairs = all_results[size]

        for subset_name, item_ids in [("Linguistic", LINGUISTIC_ITEMS),
                                       ("Visual", VISUAL_ITEMS),
                                       ("Combined", LINGUISTIC_ITEMS + VISUAL_ITEMS)]:
            items = filter_raw(pairs, item_ids)
            n = len(items)
            if n < 3:
                print(f"{size:<8} {subset_name:<14} {n:<4} {'(too few)':<12}")
                continue

            h = [v["human_ratio"] for v in items]
            m = [v["confidence_ratio"] for v in items]
            r, p = stats.pearsonr(h, m)
            rho, rho_p = stats.spearmanr(h, m)

            print(f"{size:<8} {subset_name:<14} {n:<4} {r:<12.3f} {p:<10.3f} "
                  f"{rho:<12.3f} {rho_p:<10.3f}")

        print()  # blank line between models

    # ===================================================================
    # Item-level detail for largest model
    # ===================================================================
    largest = available_sizes[-1]
    print("=" * 65)
    print(f"ITEM DETAIL — {model_display_name(largest)}")
    print("=" * 65)

    pairs = all_results[largest]
    all_raw = filter_raw(pairs, LINGUISTIC_ITEMS + VISUAL_ITEMS)
    all_raw.sort(key=lambda x: x["human_ratio"], reverse=True)

    print(f"\n{'Item':<20} {'Type':<12} {'Human':<10} {'Model':<10} {'Gap':<10}")
    print("-" * 62)
    for v in all_raw:
        item_type = "Linguistic" if v["item_id"] in LINGUISTIC_ITEMS else "Visual"
        gap = v["confidence_ratio"] - v["human_ratio"]
        print(f"{v['item_id']:<20} {item_type:<12} {v['human_ratio']:<10.3f} "
              f"{v['confidence_ratio']:<10.3f} {gap:<+10.3f}")

    # ===================================================================
    # Plots
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING PLOTS")
    print("=" * 65)

    for size in available_sizes:
        print(f"  Split scatter for {size}...")
        plot_split_scatter(
            all_results[size], size,
            REANALYSIS_FIGURES_DIR / f"split_scatter_{size}.png",
        )
        print(f"  Side-by-side bars for {size}...")
        plot_side_by_side_bars(
            all_results[size], size,
            REANALYSIS_FIGURES_DIR / f"split_bars_{size}.png",
        )

    if len(available_sizes) >= 2:
        print("  Subset scaling curves...")
        plot_subset_scaling(
            all_results,
            REANALYSIS_FIGURES_DIR / "split_correlation_scaling.png",
        )

    fig_count = len(list(REANALYSIS_FIGURES_DIR.glob("*.png")))
    print(f"\n  {fig_count} figures saved to {REANALYSIS_FIGURES_DIR}")

    # ===================================================================
    # Verdict
    # ===================================================================
    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)

    # Check linguistic subset at largest model
    ling = filter_raw(all_results[largest], LINGUISTIC_ITEMS)
    vis = filter_raw(all_results[largest], VISUAL_ITEMS)

    if len(ling) >= 3:
        r_ling, p_ling = stats.pearsonr([v["human_ratio"] for v in ling],
                                        [v["confidence_ratio"] for v in ling])
        rho_ling, _ = stats.spearmanr([v["human_ratio"] for v in ling],
                                      [v["confidence_ratio"] for v in ling])
    else:
        r_ling, rho_ling = 0, 0

    if len(vis) >= 3:
        r_vis, _ = stats.pearsonr([v["human_ratio"] for v in vis],
                                  [v["confidence_ratio"] for v in vis])
    else:
        r_vis = 0

    all_items = ling + vis
    r_all, _ = stats.pearsonr([v["human_ratio"] for v in all_items],
                              [v["confidence_ratio"] for v in all_items])

    print(f"\n  At {model_display_name(largest)}:")
    print(f"    Combined r = {r_all:.3f}")
    print(f"    Linguistic r = {r_ling:.3f}, ρ = {rho_ling:.3f}")
    print(f"    Visual r = {r_vis:.3f}")

    if r_ling > r_all + 0.1:
        print(f"\n  CONFIRMED: Visual items dilute the correlation.")
        print(f"  Linguistic subset shows {'stronger' if r_ling > 0.3 else 'improved'} signal.")
    else:
        print(f"\n  Split does NOT meaningfully improve correlation.")

    if r_ling > 0.6:
        print("  → Linguistic Mandela Effects: model IS a calibrated consensus sensor.")
    elif r_ling > 0.3:
        print("  → Linguistic Mandela Effects: moderate signal, directionally correct.")
    else:
        print("  → Even linguistic items don't show strong calibration.")


if __name__ == "__main__":
    run_reanalysis()
