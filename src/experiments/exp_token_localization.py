"""
Token-Level Uncertainty Localization
======================================
Does model confidence dip specifically at the token position where a
false claim diverges from truth?

Uses EXISTING per-token data from scaling experiments — no new model runs.

Key metrics:
  1. Win rate at divergence point: % of pairs where true > false confidence
     at the exact divergence token. Baseline ~50% if no signal.
  2. Prefix delta: Mean confidence difference in shared prefix. Should be ~0.
  3. Post-divergence recovery: Does the gap persist or recover?

Runs at multiple scales: 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B.
"""

import sys
import json
import time
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

from src.schema import load_records, ConfidenceRecord
from src.experiments.exp2_truth import PAIRS as TRUTH_PAIRS

# Output dirs
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "token_localization"
FIGURES_DIR = PROJECT_ROOT / "figures" / "token_localization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Scaling data paths
SCALING_DIR = PROJECT_ROOT / "data" / "results" / "scaling"
MODELS = ["160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
MODEL_LABELS = {
    "160m": "160M", "410m": "410M", "1b": "1B", "1.4b": "1.4B",
    "2.8b": "2.8B", "6.9b": "6.9B", "12b": "12B",
}
MODEL_PARAMS = {
    "160m": 1.6e8, "410m": 4.1e8, "1b": 1e9, "1.4b": 1.4e9,
    "2.8b": 2.8e9, "6.9b": 6.9e9, "12b": 1.2e10,
}


# ===================================================================
# Core analysis
# ===================================================================

def find_divergence_point(true_rec: ConfidenceRecord,
                          false_rec: ConfidenceRecord) -> int | None:
    """Find first token position where true and false versions differ."""
    min_len = min(len(true_rec.tokens), len(false_rec.tokens))
    for i in range(min_len):
        if true_rec.tokens[i].token_id != false_rec.tokens[i].token_id:
            return i
    return None


def analyze_pair(true_rec: ConfidenceRecord, false_rec: ConfidenceRecord,
                 pair_id: str) -> dict | None:
    """Full divergence analysis for one true/false pair."""
    div_point = find_divergence_point(true_rec, false_rec)
    if div_point is None:
        return None

    true_probs = [t.top1_prob for t in true_rec.tokens]
    false_probs = [t.top1_prob for t in false_rec.tokens]

    # Confidence at exact divergence point
    true_conf_at_div = true_probs[div_point]
    false_conf_at_div = false_probs[div_point]

    # Window around divergence (±2 tokens)
    window = 2
    start = max(0, div_point - window)
    end_true = min(len(true_probs), div_point + window + 1)
    end_false = min(len(false_probs), div_point + window + 1)
    true_window = np.mean(true_probs[start:end_true])
    false_window = np.mean(false_probs[start:end_false])

    # Prefix: shared tokens BEFORE divergence (should be nearly identical)
    if div_point > 1:
        true_prefix = np.mean(true_probs[:div_point])
        false_prefix = np.mean(false_probs[:div_point])
    else:
        true_prefix = None
        false_prefix = None

    # Suffix: everything AFTER divergence
    if div_point + 1 < min(len(true_probs), len(false_probs)):
        true_suffix = np.mean(true_probs[div_point + 1:])
        false_suffix = np.mean(false_probs[div_point + 1:])
    else:
        true_suffix = None
        false_suffix = None

    # Token strings at divergence
    true_token = true_rec.tokens[div_point].token_str
    false_token = false_rec.tokens[div_point].token_str

    # Entropy at divergence
    true_entropy_at_div = true_rec.tokens[div_point].entropy
    false_entropy_at_div = false_rec.tokens[div_point].entropy

    return {
        "pair_id": pair_id,
        "divergence_point": div_point,
        "n_tokens_true": len(true_probs),
        "n_tokens_false": len(false_probs),
        "true_token": true_token,
        "false_token": false_token,
        # At divergence
        "true_conf_at_div": float(true_conf_at_div),
        "false_conf_at_div": float(false_conf_at_div),
        "div_delta": float(true_conf_at_div - false_conf_at_div),
        "div_ratio": float(true_conf_at_div / (false_conf_at_div + 1e-10)),
        "true_entropy_at_div": float(true_entropy_at_div),
        "false_entropy_at_div": float(false_entropy_at_div),
        # Window
        "true_window": float(true_window),
        "false_window": float(false_window),
        "window_delta": float(true_window - false_window),
        # Prefix (shared tokens)
        "true_prefix": float(true_prefix) if true_prefix is not None else None,
        "false_prefix": float(false_prefix) if false_prefix is not None else None,
        "prefix_delta": float(true_prefix - false_prefix) if true_prefix is not None else None,
        # Suffix (after divergence)
        "true_suffix": float(true_suffix) if true_suffix is not None else None,
        "false_suffix": float(false_suffix) if false_suffix is not None else None,
        "suffix_delta": float(true_suffix - false_suffix) if true_suffix is not None else None,
        # Full token traces for plotting
        "true_probs": [float(p) for p in true_probs],
        "false_probs": [float(p) for p in false_probs],
    }


def analyze_model(model_key: str) -> list[dict]:
    """Load scaling data for one model and analyze all pairs."""
    path = SCALING_DIR / f"a1_truth_{model_key}.jsonl"
    if not path.exists():
        print(f"    [SKIP] {path.name} not found")
        return []

    records = load_records(path)

    # Group by pair ID
    by_pair = defaultdict(dict)
    for r in records:
        # label format: "france_capital_true" or "france_capital_false"
        parts = r.label.rsplit("_", 1)
        if len(parts) == 2:
            pair_id, version = parts
            by_pair[pair_id][version] = r

    results = []
    for pair_id, versions in by_pair.items():
        if "true" in versions and "false" in versions:
            analysis = analyze_pair(versions["true"], versions["false"], pair_id)
            if analysis is not None:
                results.append(analysis)

    return results


# ===================================================================
# Visualization
# ===================================================================

def plot_divergence_delta_distribution(all_results: dict[str, list[dict]]):
    """Histogram of div_delta across pairs, one subplot per model scale."""
    focus_models = ["160m", "1b", "6.9b"]
    available = [m for m in focus_models if m in all_results and all_results[m]]

    if not available:
        return

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, model_key in zip(axes, available):
        results = all_results[model_key]
        deltas = [r["div_delta"] for r in results]
        wins = sum(1 for d in deltas if d > 0)
        win_rate = wins / len(deltas) if deltas else 0

        ax.hist(deltas, bins=20, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Zero (no signal)")
        ax.axvline(x=np.median(deltas), color="green", linestyle="-",
                   alpha=0.7, label=f"Median: {np.median(deltas):.4f}")
        ax.set_xlabel("Δ Confidence (True − False) at Divergence Token")
        ax.set_ylabel("Count")
        ax.set_title(f"Pythia {MODEL_LABELS[model_key]}\n"
                     f"Win: {wins}/{len(deltas)} ({win_rate:.0%})")
        ax.legend(fontsize=8)

    fig.suptitle("Confidence Delta at Exact Divergence Point", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "divergence_delta_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_example_token_traces(all_results: dict[str, list[dict]]):
    """Show 4 example pairs with token-level confidence traces (true vs false)."""
    # Use 6.9B results for examples
    model_key = "6.9b"
    if model_key not in all_results or not all_results[model_key]:
        return

    results = all_results[model_key]

    # Pick 4 interesting pairs: biggest delta, smallest delta, one negative, one median
    sorted_by_delta = sorted(results, key=lambda r: r["div_delta"], reverse=True)

    # Select diverse examples
    examples = []
    if len(sorted_by_delta) >= 4:
        examples.append(sorted_by_delta[0])   # biggest positive
        examples.append(sorted_by_delta[len(sorted_by_delta) // 4])  # 25th percentile
        examples.append(sorted_by_delta[len(sorted_by_delta) // 2])  # median
        examples.append(sorted_by_delta[-1])   # smallest / most negative
    else:
        examples = sorted_by_delta[:4]

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax, ex in zip(axes_flat, examples):
        true_p = ex["true_probs"]
        false_p = ex["false_probs"]
        div = ex["divergence_point"]

        x_true = range(len(true_p))
        x_false = range(len(false_p))

        ax.plot(x_true, true_p, "o-", color="#4CAF50", markersize=4,
                linewidth=1.5, label="True", alpha=0.8)
        ax.plot(x_false, false_p, "s-", color="#F44336", markersize=4,
                linewidth=1.5, label="False", alpha=0.8)
        ax.axvline(x=div, color="#FF9800", linestyle="--", linewidth=2,
                   alpha=0.8, label=f"Divergence (pos {div})")

        # Shade divergence region
        ax.axvspan(div - 0.5, div + 0.5, alpha=0.15, color="#FF9800")

        ax.set_xlabel("Token Position")
        ax.set_ylabel("P(actual token)")
        ax.set_title(f"{ex['pair_id']}\n"
                     f"'{ex['true_token']}' vs '{ex['false_token']}' "
                     f"(Δ={ex['div_delta']:+.4f})",
                     fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.set_ylim(-0.02, min(1.0, max(max(true_p), max(false_p)) * 1.15))

    fig.suptitle("Token-Level Confidence Traces — Pythia 6.9B\n"
                 "Orange = divergence point", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "example_token_traces.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prefix_vs_div_delta(all_results: dict[str, list[dict]]):
    """Sanity check: prefix gap (~0) vs divergence gap (should be large)."""
    model_key = "6.9b"
    if model_key not in all_results or not all_results[model_key]:
        return

    results = all_results[model_key]
    # Filter to pairs with prefix data
    with_prefix = [r for r in results if r["prefix_delta"] is not None]

    if not with_prefix:
        return

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: paired bar chart — prefix delta vs div delta vs suffix delta
    ax = axes[0]
    pair_ids = [r["pair_id"][:15] for r in with_prefix]
    prefix_deltas = [r["prefix_delta"] for r in with_prefix]
    div_deltas = [r["div_delta"] for r in with_prefix]
    suffix_deltas = [r["suffix_delta"] for r in with_prefix if r["suffix_delta"] is not None]

    x = np.arange(len(with_prefix))
    w = 0.25
    ax.bar(x - w, prefix_deltas, w, label="Prefix (shared)", color="#9E9E9E", alpha=0.8)
    ax.bar(x, div_deltas, w, label="At Divergence", color="#F44336", alpha=0.8)
    if len(suffix_deltas) == len(with_prefix):
        ax.bar(x + w, suffix_deltas, w, label="Post-Divergence", color="#2196F3", alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_ids, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Δ Confidence (True − False)")
    ax.set_title("Per-Pair: Prefix vs Divergence vs Suffix")
    ax.legend(fontsize=8)

    # Right: scatter — prefix delta (x) vs div delta (y)
    ax = axes[1]
    ax.scatter(prefix_deltas, div_deltas, s=50, alpha=0.7, color="#7a0177", zorder=5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Prefix Δ (should be ~0)")
    ax.set_ylabel("Divergence Δ (should be > 0)")
    ax.set_title("Prefix Gap vs Divergence Gap")

    # Annotate quadrants
    ax.text(0.05, 0.95, "Signal without\nleakage ✓",
            transform=ax.transAxes, fontsize=8, va="top", color="green",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3))

    r_val, p_val = stats.pearsonr(prefix_deltas, div_deltas) if len(prefix_deltas) > 3 else (0, 1)
    ax.text(0.95, 0.05,
            f"r={r_val:.3f} (p={p_val:.3f})\nCorrelation = {'low' if abs(r_val) < 0.3 else 'moderate' if abs(r_val) < 0.6 else 'high'}",
            transform=ax.transAxes, fontsize=9, ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Sanity Check: Is the Signal Localized or Leaking? — Pythia 6.9B",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prefix_vs_div_delta.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_localization(all_results: dict[str, list[dict]]):
    """How does localization quality change with model scale?"""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models_available = [m for m in MODELS if m in all_results and all_results[m]]
    x_params = [MODEL_PARAMS[m] for m in models_available]
    x_labels = [MODEL_LABELS[m] for m in models_available]

    # Metric 1: Win rate at divergence
    ax = axes[0]
    win_rates = []
    for m in models_available:
        results = all_results[m]
        wins = sum(1 for r in results if r["div_delta"] > 0)
        win_rates.append(wins / len(results) if results else 0.5)
    ax.plot(x_params, win_rates, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Chance (50%)")
    ax.set_xscale("log")
    ax.set_ylabel("Win Rate at Divergence")
    ax.set_xlabel("Parameters")
    ax.set_title("Does Localization Improve\nwith Scale?")
    ax.set_xticks(x_params)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax.set_ylim(0.35, 1.0)
    ax.legend(fontsize=8)

    # Metric 2: Mean prefix delta (should stay near 0 = no leakage)
    ax = axes[1]
    prefix_means = []
    prefix_stds = []
    for m in models_available:
        results = all_results[m]
        pd_vals = [r["prefix_delta"] for r in results if r["prefix_delta"] is not None]
        prefix_means.append(np.mean(pd_vals) if pd_vals else 0)
        prefix_stds.append(np.std(pd_vals) if pd_vals else 0)
    ax.errorbar(x_params, prefix_means, yerr=prefix_stds,
                fmt="o-", color="#4CAF50", linewidth=2, markersize=8, capsize=4)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_ylabel("Mean Prefix Δ (noise floor)")
    ax.set_xlabel("Parameters")
    ax.set_title("Prefix Leakage\n(should be ~0)")
    ax.set_xticks(x_params)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=8)

    # Metric 3: Suffix delta (recovery vs persistence)
    ax = axes[2]
    suffix_means = []
    div_means = []
    for m in models_available:
        results = all_results[m]
        sd_vals = [r["suffix_delta"] for r in results if r["suffix_delta"] is not None]
        dd_vals = [r["div_delta"] for r in results]
        suffix_means.append(np.mean(sd_vals) if sd_vals else 0)
        div_means.append(np.mean(dd_vals) if dd_vals else 0)
    ax.plot(x_params, div_means, "o-", color="#F44336", linewidth=2,
            markersize=8, label="At Divergence")
    ax.plot(x_params, suffix_means, "s-", color="#2196F3", linewidth=2,
            markersize=8, label="Post-Divergence")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_ylabel("Mean Δ Confidence")
    ax.set_xlabel("Parameters")
    ax.set_title("Recovery After Divergence\n(gap persists or recovers?)")
    ax.set_xticks(x_params)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax.legend(fontsize=8)

    fig.suptitle("Token-Level Uncertainty Localization Across Scales",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scaling_localization.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def run_experiment():
    total_start = time.time()

    print("=" * 70)
    print("TOKEN-LEVEL UNCERTAINTY LOCALIZATION")
    print("=" * 70)
    print(f"Models: {', '.join(MODEL_LABELS[m] for m in MODELS)}")
    print(f"Truth pairs: {len(TRUTH_PAIRS)}")
    print(f"Data source: existing scaling JSONL (no new model runs)")
    print()

    # Analyze all scales
    all_results = {}
    for model_key in MODELS:
        print(f"\n  [{MODEL_LABELS[model_key]}] Analyzing divergence points...")
        results = analyze_model(model_key)
        if results:
            all_results[model_key] = results
            wins = sum(1 for r in results if r["div_delta"] > 0)
            print(f"    Pairs analyzed: {len(results)}")
            print(f"    Win rate at divergence: {wins}/{len(results)} "
                  f"({wins/len(results):.1%})")
            prefix_deltas = [r["prefix_delta"] for r in results
                             if r["prefix_delta"] is not None]
            if prefix_deltas:
                print(f"    Mean prefix delta: {np.mean(prefix_deltas):+.6f} "
                      f"(noise floor)")
            suffix_deltas = [r["suffix_delta"] for r in results
                             if r["suffix_delta"] is not None]
            if suffix_deltas:
                print(f"    Mean suffix delta: {np.mean(suffix_deltas):+.6f} "
                      f"(post-div)")

    # ===================================================================
    # Detailed Report — 6.9B focus
    # ===================================================================
    focus = "6.9b"
    if focus in all_results:
        results = all_results[focus]
        print(f"\n{'=' * 70}")
        print(f"DETAILED REPORT — PYTHIA 6.9B")
        print(f"{'=' * 70}")

        div_deltas = [r["div_delta"] for r in results]
        prefix_deltas = [r["prefix_delta"] for r in results
                         if r["prefix_delta"] is not None]
        suffix_deltas = [r["suffix_delta"] for r in results
                         if r["suffix_delta"] is not None]

        # 1. Win rate at divergence
        wins = sum(1 for d in div_deltas if d > 0)
        win_rate = wins / len(div_deltas)
        print(f"\n  1. WIN RATE AT DIVERGENCE: {wins}/{len(div_deltas)} "
              f"({win_rate:.1%})")
        binom = stats.binomtest(wins, len(div_deltas), 0.5)
        print(f"     Binomial test vs 50%: p={binom.pvalue:.6f}")
        if win_rate > 0.65:
            print(f"     → SIGNAL EXISTS at divergence point")
        else:
            print(f"     → Weak or no signal at divergence point")

        # 2. Prefix delta
        if prefix_deltas:
            mean_pd = np.mean(prefix_deltas)
            t_pd, p_pd = stats.ttest_1samp(prefix_deltas, 0)
            print(f"\n  2. PREFIX DELTA (noise floor): {mean_pd:+.6f}")
            print(f"     t-test vs 0: t={t_pd:.3f}, p={p_pd:.4f}")
            if abs(mean_pd) < 0.01 and p_pd > 0.05:
                print(f"     → Prefix is clean (no leakage) ✓")
            else:
                print(f"     → WARNING: Some signal leaking into prefix")

        # 3. Post-divergence recovery
        if suffix_deltas:
            mean_sd = np.mean(suffix_deltas)
            mean_dd = np.mean(div_deltas)
            recovery = 1 - (mean_sd / mean_dd) if mean_dd != 0 else 0
            print(f"\n  3. POST-DIVERGENCE:")
            print(f"     Mean divergence delta: {mean_dd:+.6f}")
            print(f"     Mean suffix delta:     {mean_sd:+.6f}")
            print(f"     Recovery: {recovery:.1%}")
            if abs(mean_sd) < abs(mean_dd) * 0.3:
                print(f"     → Uncertainty is a LOCAL DIP (recovers after)")
            elif mean_sd > 0 and abs(mean_sd) > abs(mean_dd) * 0.5:
                print(f"     → Uncertainty CASCADES (persists after divergence)")
            else:
                print(f"     → Partial recovery")

        # Per-pair table
        print(f"\n  {'Pair ID':<20s} {'Div':<5s} {'True Tok':<12s} "
              f"{'False Tok':<12s} {'P(true)':<10s} {'P(false)':<10s} "
              f"{'Δ div':<10s} {'Δ prefix':<10s} {'Δ suffix':<10s}")
        print("  " + "-" * 99)

        sorted_results = sorted(results, key=lambda r: r["div_delta"], reverse=True)
        for r in sorted_results:
            pd = f"{r['prefix_delta']:+.5f}" if r["prefix_delta"] is not None else "N/A"
            sd = f"{r['suffix_delta']:+.5f}" if r["suffix_delta"] is not None else "N/A"
            print(f"  {r['pair_id']:<20s} {r['divergence_point']:<5d} "
                  f"{repr(r['true_token']):<12s} {repr(r['false_token']):<12s} "
                  f"{r['true_conf_at_div']:<10.5f} {r['false_conf_at_div']:<10.5f} "
                  f"{r['div_delta']:+<10.5f} {pd:<10s} {sd:<10s}")

    # ===================================================================
    # Scaling Summary Table
    # ===================================================================
    print(f"\n{'=' * 70}")
    print(f"SCALING SUMMARY")
    print(f"{'=' * 70}")

    print(f"\n  {'Model':<10s} {'N pairs':<8s} {'Win%':<8s} {'Mean Δ div':<12s} "
          f"{'Mean Δ pfx':<12s} {'Mean Δ sfx':<12s} {'Localized?':<12s}")
    print("  " + "-" * 72)

    scaling_summary = {}
    for model_key in MODELS:
        if model_key not in all_results:
            continue
        results = all_results[model_key]
        dd = [r["div_delta"] for r in results]
        pd_vals = [r["prefix_delta"] for r in results if r["prefix_delta"] is not None]
        sd_vals = [r["suffix_delta"] for r in results if r["suffix_delta"] is not None]

        wins = sum(1 for d in dd if d > 0)
        win_rate = wins / len(dd)
        mean_dd = np.mean(dd)
        mean_pd = np.mean(pd_vals) if pd_vals else 0
        mean_sd = np.mean(sd_vals) if sd_vals else 0

        # Is it localized? High div delta, low prefix delta
        localized = "YES" if (win_rate > 0.6 and abs(mean_pd) < abs(mean_dd) * 0.3) else "NO"

        print(f"  {MODEL_LABELS[model_key]:<10s} {len(results):<8d} "
              f"{win_rate:<8.1%} {mean_dd:<+12.6f} {mean_pd:<+12.6f} "
              f"{mean_sd:<+12.6f} {localized:<12s}")

        scaling_summary[model_key] = {
            "model": MODEL_LABELS[model_key],
            "params": MODEL_PARAMS[model_key],
            "n_pairs": len(results),
            "win_rate": win_rate,
            "mean_div_delta": float(mean_dd),
            "mean_prefix_delta": float(mean_pd),
            "mean_suffix_delta": float(mean_sd),
            "localized": localized == "YES",
        }

    # ===================================================================
    # Plots
    # ===================================================================
    print(f"\n  Generating plots...")

    plot_divergence_delta_distribution(all_results)
    print(f"    [1/4] divergence_delta_distribution.png")

    plot_example_token_traces(all_results)
    print(f"    [2/4] example_token_traces.png")

    plot_prefix_vs_div_delta(all_results)
    print(f"    [3/4] prefix_vs_div_delta.png")

    plot_scaling_localization(all_results)
    print(f"    [4/4] scaling_localization.png")

    # ===================================================================
    # Save JSON results
    # ===================================================================

    # Detailed per-pair analysis (without token traces — too large)
    for model_key, results in all_results.items():
        clean = []
        for r in results:
            c = {k: v for k, v in r.items() if k not in ("true_probs", "false_probs")}
            clean.append(c)
        with open(RESULTS_DIR / f"divergence_{model_key}.json", "w") as f:
            json.dump(clean, f, indent=2)

    # Summary stats
    with open(RESULTS_DIR / "summary_stats.json", "w") as f:
        json.dump(scaling_summary, f, indent=2)

    total_time = time.time() - total_start
    fig_count = len(list(FIGURES_DIR.glob("*.png")))

    print(f"\n{'=' * 70}")
    print(f"COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    run_experiment()
