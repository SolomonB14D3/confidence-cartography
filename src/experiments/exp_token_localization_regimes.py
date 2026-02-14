"""
Token-Level Divergence: Regime Comparison
==========================================
Regime 1: Factual pairs (truth/false + medical) — shared prefix, swap at one point
Regime 2: Mandela linguistic items — often entirely different phrasing

Key question: Does token-level localization work the same way when the
wrong version is the *popular* version (Mandela) vs a fabricated falsehood?

Uses existing per-token data — no new model runs.
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

# Output
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "token_localization"
FIGURES_DIR = PROJECT_ROOT / "figures" / "token_localization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
MODEL_LABELS = {
    "160m": "160M", "410m": "410M", "1b": "1B", "1.4b": "1.4B",
    "2.8b": "2.8B", "6.9b": "6.9B", "12b": "12B",
}
MODEL_PARAMS = {
    "160m": 1.6e8, "410m": 4.1e8, "1b": 1e9, "1.4b": 1.4e9,
    "2.8b": 2.8e9, "6.9b": 6.9e9, "12b": 1.2e10,
}

# Data paths
TRUTH_PATH = PROJECT_ROOT / "data" / "results" / "scaling"         # a1_truth_{model}.jsonl
MEDICAL_PATH = PROJECT_ROOT / "data" / "results" / "exp9"          # medical_pairs_{model}.jsonl
MANDELA_ORIG_PATH = PROJECT_ROOT / "data" / "results" / "mandela"  # mandela_{model}.jsonl
MANDELA_EXP_PATH = PROJECT_ROOT / "data" / "results" / "mandela" / "expanded"  # expanded_{model}.jsonl


# ===================================================================
# Core analysis (reused from exp_token_localization)
# ===================================================================

def find_divergence_point(rec_a: ConfidenceRecord,
                          rec_b: ConfidenceRecord) -> int | None:
    """First token position where two records differ."""
    min_len = min(len(rec_a.tokens), len(rec_b.tokens))
    for i in range(min_len):
        if rec_a.tokens[i].token_id != rec_b.tokens[i].token_id:
            return i
    return None


def analyze_pair(true_rec: ConfidenceRecord, false_rec: ConfidenceRecord,
                 pair_id: str) -> dict | None:
    """Divergence analysis for one pair."""
    div_point = find_divergence_point(true_rec, false_rec)
    if div_point is None:
        return None

    true_probs = [t.top1_prob for t in true_rec.tokens]
    false_probs = [t.top1_prob for t in false_rec.tokens]

    true_conf = true_probs[div_point]
    false_conf = false_probs[div_point]

    # Prefix
    if div_point > 1:
        true_prefix = np.mean(true_probs[:div_point])
        false_prefix = np.mean(false_probs[:div_point])
    else:
        true_prefix = None
        false_prefix = None

    # Suffix
    if div_point + 1 < min(len(true_probs), len(false_probs)):
        true_suffix = np.mean(true_probs[div_point + 1:])
        false_suffix = np.mean(false_probs[div_point + 1:])
    else:
        true_suffix = None
        false_suffix = None

    return {
        "pair_id": pair_id,
        "divergence_point": div_point,
        "n_prefix_tokens": div_point,
        "true_token": true_rec.tokens[div_point].token_str,
        "false_token": false_rec.tokens[div_point].token_str,
        "true_conf_at_div": float(true_conf),
        "false_conf_at_div": float(false_conf),
        "div_delta": float(true_conf - false_conf),
        "prefix_delta": float(true_prefix - false_prefix) if true_prefix is not None else None,
        "suffix_delta": float(true_suffix - false_suffix) if true_suffix is not None else None,
    }


# ===================================================================
# Data loaders
# ===================================================================

def load_truth_pairs(model_key: str) -> list[dict]:
    """Load truth pair divergence data."""
    path = TRUTH_PATH / f"a1_truth_{model_key}.jsonl"
    if not path.exists():
        return []
    records = load_records(path)
    by_pair = defaultdict(dict)
    for r in records:
        parts = r.label.rsplit("_", 1)
        if len(parts) == 2:
            by_pair[parts[0]][parts[1]] = r
    results = []
    for pid, vs in by_pair.items():
        if "true" in vs and "false" in vs:
            a = analyze_pair(vs["true"], vs["false"], pid)
            if a:
                a["source"] = "truth"
                results.append(a)
    return results


def load_medical_pairs(model_key: str) -> list[dict]:
    """Load medical pair divergence data."""
    path = MEDICAL_PATH / f"medical_pairs_{model_key}.jsonl"
    if not path.exists():
        return []
    records = load_records(path)
    by_pair = defaultdict(dict)
    for r in records:
        version = r.metadata.get("version", "")
        pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
        by_pair[pid][version] = r
    results = []
    for pid, vs in by_pair.items():
        if "true" in vs and "false" in vs:
            a = analyze_pair(vs["true"], vs["false"], pid)
            if a:
                a["source"] = "medical"
                results.append(a)
    return results


def load_mandela_original(model_key: str) -> list[dict]:
    """Load original mandela divergence data (popular_false vs actual_correct)."""
    path = MANDELA_ORIG_PATH / f"mandela_{model_key}.jsonl"
    if not path.exists():
        return []
    records = load_records(path)
    by_pair = defaultdict(dict)
    for r in records:
        version = r.metadata.get("version", "")
        pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
        # Map: "correct" = true side, "popular" = false side
        if version in ("correct", "actual"):
            by_pair[pid]["correct"] = r
        elif version in ("popular", "wrong"):
            by_pair[pid]["wrong"] = r
    results = []
    for pid, vs in by_pair.items():
        if "correct" in vs and "wrong" in vs:
            # correct = ground truth, wrong = popular misconception
            a = analyze_pair(vs["correct"], vs["wrong"], pid)
            if a:
                a["source"] = "mandela_orig"
                results.append(a)
    return results


def load_mandela_expanded(model_key: str) -> list[dict]:
    """Load expanded mandela divergence data (raw framings only for clean comparison)."""
    path = MANDELA_EXP_PATH / f"expanded_{model_key}.jsonl"
    if not path.exists():
        return []
    records = load_records(path)
    by_pair = defaultdict(lambda: defaultdict(dict))
    for r in records:
        item_id = r.metadata.get("item_id", "")
        framing = r.metadata.get("framing", "raw")
        version = r.metadata.get("version", "")
        by_pair[item_id][framing][version] = r

    results = []
    for item_id, framings in by_pair.items():
        for framing_name, vs in framings.items():
            if "correct" in vs and "wrong" in vs:
                pid = f"{item_id}_{framing_name}"
                a = analyze_pair(vs["correct"], vs["wrong"], pid)
                if a:
                    a["source"] = f"mandela_exp_{framing_name}"
                    results.append(a)
    return results


# ===================================================================
# Main
# ===================================================================

def run_experiment():
    total_start = time.time()

    print("=" * 70)
    print("TOKEN DIVERGENCE: REGIME COMPARISON")
    print("=" * 70)
    print("Regime 1: Factual pairs (truth/false + medical)")
    print("  → Shared prefix, diverge at a specific substitution point")
    print("Regime 2: Mandela linguistic items (original + expanded)")
    print("  → Often entirely different phrasing (wrong = popular version)")
    print()

    # Collect results per model, per regime
    regime1_by_model = {}  # model_key -> list of pair analyses
    regime2_by_model = {}

    for model_key in MODELS:
        label = MODEL_LABELS[model_key]

        # Regime 1: truth + medical
        truth = load_truth_pairs(model_key)
        medical = load_medical_pairs(model_key)
        r1 = truth + medical
        if r1:
            regime1_by_model[model_key] = r1

        # Regime 2: mandela original + expanded
        mandela_orig = load_mandela_original(model_key)
        mandela_exp = load_mandela_expanded(model_key)
        r2 = mandela_orig + mandela_exp
        if r2:
            regime2_by_model[model_key] = r2

        # Quick per-model summary
        r1_wins = sum(1 for r in r1 if r["div_delta"] > 0) if r1 else 0
        r2_wins = sum(1 for r in r2 if r["div_delta"] > 0) if r2 else 0
        r1_n = len(r1)
        r2_n = len(r2)

        print(f"  [{label:>4s}]  R1: {r1_wins:>2d}/{r1_n:<2d} "
              f"({r1_wins/r1_n:.0%})  "
              f"[{len(truth)}t + {len(medical)}m]"
              f"    R2: {r2_wins:>2d}/{r2_n:<2d} "
              f"({r2_wins/r2_n:.0%} )" if r2_n > 0 else
              f"  [{label:>4s}]  R1: {r1_wins:>2d}/{r1_n:<2d} "
              f"({r1_wins/r1_n:.0%})  "
              f"[{len(truth)}t + {len(medical)}m]"
              f"    R2: no data")

    # ===================================================================
    # Detailed Report
    # ===================================================================
    print(f"\n{'=' * 70}")
    print("DETAILED COMPARISON")
    print(f"{'=' * 70}")

    print(f"\n  {'Model':<7s} {'R1 win%':<9s} {'R1 Δdiv':<11s} {'R1 Δpfx':<11s} "
          f"{'R1 Δsfx':<11s} {'R2 win%':<9s} {'R2 Δdiv':<11s} {'R2 Δpfx':<11s} "
          f"{'R2 Δsfx':<11s}")
    print("  " + "-" * 87)

    scaling_data = []

    for model_key in MODELS:
        label = MODEL_LABELS[model_key]
        row = {"model": label, "params": MODEL_PARAMS[model_key]}

        for regime_name, by_model in [("R1", regime1_by_model), ("R2", regime2_by_model)]:
            if model_key not in by_model:
                row[f"{regime_name}_win"] = None
                row[f"{regime_name}_div"] = None
                row[f"{regime_name}_pfx"] = None
                row[f"{regime_name}_sfx"] = None
                continue

            results = by_model[model_key]
            wins = sum(1 for r in results if r["div_delta"] > 0)
            n = len(results)
            dd = [r["div_delta"] for r in results]
            pd = [r["prefix_delta"] for r in results if r["prefix_delta"] is not None]
            sd = [r["suffix_delta"] for r in results if r["suffix_delta"] is not None]

            row[f"{regime_name}_n"] = n
            row[f"{regime_name}_win"] = wins / n
            row[f"{regime_name}_div"] = np.mean(dd)
            row[f"{regime_name}_pfx"] = np.mean(pd) if pd else None
            row[f"{regime_name}_sfx"] = np.mean(sd) if sd else None

        scaling_data.append(row)

        # Print row
        def fmt(v, width=9):
            if v is None:
                return "—".ljust(width)
            return f"{v:+.5f}".ljust(width) if abs(v) < 1 else f"{v:.1%}".ljust(width)

        r1w = f"{row.get('R1_win', 0):.0%}" if row.get('R1_win') is not None else "—"
        r2w = f"{row.get('R2_win', 0):.0%}" if row.get('R2_win') is not None else "—"
        r1d = f"{row.get('R1_div', 0):+.5f}" if row.get('R1_div') is not None else "—"
        r2d = f"{row.get('R2_div', 0):+.5f}" if row.get('R2_div') is not None else "—"
        r1p = f"{row.get('R1_pfx', 0):+.5f}" if row.get('R1_pfx') is not None else "—"
        r2p = f"{row.get('R2_pfx', 0):+.5f}" if row.get('R2_pfx') is not None else "—"
        r1s = f"{row.get('R1_sfx', 0):+.5f}" if row.get('R1_sfx') is not None else "—"
        r2s = f"{row.get('R2_sfx', 0):+.5f}" if row.get('R2_sfx') is not None else "—"

        print(f"  {label:<7s} {r1w:<9s} {r1d:<11s} {r1p:<11s} {r1s:<11s} "
              f"{r2w:<9s} {r2d:<11s} {r2p:<11s} {r2s:<11s}")

    # ===================================================================
    # Focus: 6.9B head-to-head
    # ===================================================================
    focus = "6.9b"
    print(f"\n{'=' * 70}")
    print(f"HEAD-TO-HEAD: PYTHIA 6.9B")
    print(f"{'=' * 70}")

    for regime_name, by_model, desc in [
        ("REGIME 1", regime1_by_model, "Factual Pairs (Truth + Medical)"),
        ("REGIME 2", regime2_by_model, "Mandela Linguistic Items"),
    ]:
        if focus not in by_model:
            print(f"\n  {regime_name} ({desc}): No data")
            continue

        results = by_model[focus]
        dd = [r["div_delta"] for r in results]
        pd_vals = [r["prefix_delta"] for r in results if r["prefix_delta"] is not None]
        sd_vals = [r["suffix_delta"] for r in results if r["suffix_delta"] is not None]
        div0 = [r for r in results if r["divergence_point"] == 0]

        wins = sum(1 for d in dd if d > 0)
        n = len(dd)
        binom = stats.binomtest(wins, n, 0.5)

        print(f"\n  {regime_name}: {desc}")
        print(f"  {'—' * 50}")
        print(f"    N pairs:              {n}")
        print(f"    Diverge at pos 0:     {len(div0)} ({len(div0)/n:.0%})")
        print(f"    Win rate at div:      {wins}/{n} ({wins/n:.1%})  "
              f"p={binom.pvalue:.6f}")
        print(f"    Mean Δ at divergence: {np.mean(dd):+.6f}")
        print(f"    Median Δ:             {np.median(dd):+.6f}")
        if pd_vals:
            t_pd, p_pd = stats.ttest_1samp(pd_vals, 0) if len(pd_vals) > 2 else (0, 1)
            print(f"    Mean prefix Δ:        {np.mean(pd_vals):+.6f}  "
                  f"(p={p_pd:.4f})")
        else:
            print(f"    Mean prefix Δ:        N/A (most pairs diverge at pos 0)")
        if sd_vals:
            print(f"    Mean suffix Δ:        {np.mean(sd_vals):+.6f}")
            recovery = 1 - (np.mean(sd_vals) / np.mean(dd)) if np.mean(dd) != 0 else 0
            print(f"    Recovery:             {recovery:.1%}")

        # Per-pair detail
        print(f"\n    {'Pair':<30s} {'Div':<5s} {'True tok':<12s} {'False tok':<12s} "
              f"{'Δ div':<12s} {'Δ pfx':<12s} {'src'}")
        print(f"    {'—' * 95}")
        sorted_r = sorted(results, key=lambda r: r["div_delta"], reverse=True)
        for r in sorted_r:
            pfx = f"{r['prefix_delta']:+.5f}" if r["prefix_delta"] is not None else "—"
            print(f"    {r['pair_id']:<30s} {r['divergence_point']:<5d} "
                  f"{repr(r['true_token']):<12s} {repr(r['false_token']):<12s} "
                  f"{r['div_delta']:+.5f}     {pfx:<12s} {r['source']}")

    # ===================================================================
    # Statistical comparison between regimes at 6.9B
    # ===================================================================
    if focus in regime1_by_model and focus in regime2_by_model:
        print(f"\n{'=' * 70}")
        print(f"STATISTICAL COMPARISON (6.9B)")
        print(f"{'=' * 70}")

        r1_dd = [r["div_delta"] for r in regime1_by_model[focus]]
        r2_dd = [r["div_delta"] for r in regime2_by_model[focus]]

        r1_wins = sum(1 for d in r1_dd if d > 0)
        r2_wins = sum(1 for d in r2_dd if d > 0)

        # Mann-Whitney U: are the div_delta distributions different?
        u_stat, u_p = stats.mannwhitneyu(r1_dd, r2_dd, alternative="two-sided")
        # Effect size (rank-biserial)
        n1, n2 = len(r1_dd), len(r2_dd)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        # Win rate comparison (Fisher's exact or chi-squared)
        r1_losses = len(r1_dd) - r1_wins
        r2_losses = len(r2_dd) - r2_wins
        table = [[r1_wins, r1_losses], [r2_wins, r2_losses]]
        fisher_or, fisher_p = stats.fisher_exact(table)

        print(f"\n  Regime 1 (factual):  win={r1_wins}/{len(r1_dd)} ({r1_wins/len(r1_dd):.1%}), "
              f"mean Δ={np.mean(r1_dd):+.5f}")
        print(f"  Regime 2 (mandela):  win={r2_wins}/{len(r2_dd)} ({r2_wins/len(r2_dd):.1%}), "
              f"mean Δ={np.mean(r2_dd):+.5f}")
        print(f"\n  Win rate difference (Fisher exact): OR={fisher_or:.2f}, p={fisher_p:.4f}")
        print(f"  Delta distribution (Mann-Whitney):  U={u_stat:.1f}, p={u_p:.4f}, "
              f"rank-biserial r={rank_biserial:.3f}")

        if u_p < 0.05:
            if np.mean(r1_dd) > np.mean(r2_dd):
                print(f"\n  → Regimes are SIGNIFICANTLY different. "
                      f"Factual pairs show stronger localization.")
            else:
                print(f"\n  → Regimes are SIGNIFICANTLY different. "
                      f"Mandela items show stronger localization.")
        else:
            print(f"\n  → No significant difference between regimes (p={u_p:.4f})")

        # Structural difference: prefix length
        r1_divs = [r["divergence_point"] for r in regime1_by_model[focus]]
        r2_divs = [r["divergence_point"] for r in regime2_by_model[focus]]
        print(f"\n  Structural difference:")
        print(f"    R1 mean divergence position: {np.mean(r1_divs):.1f} "
              f"(median {np.median(r1_divs):.0f})")
        print(f"    R2 mean divergence position: {np.mean(r2_divs):.1f} "
              f"(median {np.median(r2_divs):.0f})")
        r1_at0 = sum(1 for d in r1_divs if d == 0)
        r2_at0 = sum(1 for d in r2_divs if d == 0)
        print(f"    R1 diverge at pos 0: {r1_at0}/{len(r1_divs)} ({r1_at0/len(r1_divs):.0%})")
        print(f"    R2 diverge at pos 0: {r2_at0}/{len(r2_divs)} ({r2_at0/len(r2_divs):.0%})")

    # ===================================================================
    # Visualization
    # ===================================================================
    print(f"\n  Generating plots...")

    sns.set_theme(style="whitegrid", palette="muted")

    # --- Plot 1: Scaling comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    available_r1 = [m for m in MODELS if m in regime1_by_model]
    available_r2 = [m for m in MODELS if m in regime2_by_model]
    all_available = sorted(set(available_r1) | set(available_r2),
                           key=lambda m: MODELS.index(m))

    x_params = [MODEL_PARAMS[m] for m in all_available]
    x_labels = [MODEL_LABELS[m] for m in all_available]

    # Panel 1: Win rates
    ax = axes[0]
    r1_wr = [sum(1 for r in regime1_by_model[m] if r["div_delta"] > 0) / len(regime1_by_model[m])
             if m in regime1_by_model else None for m in all_available]
    r2_wr = [sum(1 for r in regime2_by_model[m] if r["div_delta"] > 0) / len(regime2_by_model[m])
             if m in regime2_by_model else None for m in all_available]

    x1 = [MODEL_PARAMS[m] for m, w in zip(all_available, r1_wr) if w is not None]
    y1 = [w for w in r1_wr if w is not None]
    x2 = [MODEL_PARAMS[m] for m, w in zip(all_available, r2_wr) if w is not None]
    y2 = [w for w in r2_wr if w is not None]

    ax.plot(x1, y1, "o-", color="#2196F3", linewidth=2, markersize=8,
            label="R1: Factual")
    ax.plot(x2, y2, "s-", color="#F44336", linewidth=2, markersize=8,
            label="R2: Mandela")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xscale("log")
    ax.set_ylabel("Win Rate at Divergence")
    ax.set_xlabel("Parameters")
    ax.set_title("Win Rate by Regime")
    ax.set_xticks(x_params)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=9)

    # Panel 2: Mean divergence delta
    ax = axes[1]
    r1_dd_means = [np.mean([r["div_delta"] for r in regime1_by_model[m]])
                   if m in regime1_by_model else None for m in all_available]
    r2_dd_means = [np.mean([r["div_delta"] for r in regime2_by_model[m]])
                   if m in regime2_by_model else None for m in all_available]

    x1 = [MODEL_PARAMS[m] for m, v in zip(all_available, r1_dd_means) if v is not None]
    y1 = [v for v in r1_dd_means if v is not None]
    x2 = [MODEL_PARAMS[m] for m, v in zip(all_available, r2_dd_means) if v is not None]
    y2 = [v for v in r2_dd_means if v is not None]

    ax.plot(x1, y1, "o-", color="#2196F3", linewidth=2, markersize=8,
            label="R1: Factual")
    ax.plot(x2, y2, "s-", color="#F44336", linewidth=2, markersize=8,
            label="R2: Mandela")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_ylabel("Mean Δ Confidence at Divergence")
    ax.set_xlabel("Parameters")
    ax.set_title("Divergence Delta by Regime")
    ax.set_xticks(x_params)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax.legend(fontsize=9)

    # Panel 3: Suffix delta (recovery)
    ax = axes[2]
    r1_sd_means = []
    r2_sd_means = []
    x1_sd, x2_sd = [], []
    for m in all_available:
        if m in regime1_by_model:
            sd = [r["suffix_delta"] for r in regime1_by_model[m] if r["suffix_delta"] is not None]
            if sd:
                r1_sd_means.append(np.mean(sd))
                x1_sd.append(MODEL_PARAMS[m])
        if m in regime2_by_model:
            sd = [r["suffix_delta"] for r in regime2_by_model[m] if r["suffix_delta"] is not None]
            if sd:
                r2_sd_means.append(np.mean(sd))
                x2_sd.append(MODEL_PARAMS[m])

    ax.plot(x1_sd, r1_sd_means, "o-", color="#2196F3", linewidth=2, markersize=8,
            label="R1: Factual")
    ax.plot(x2_sd, r2_sd_means, "s-", color="#F44336", linewidth=2, markersize=8,
            label="R2: Mandela")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_ylabel("Mean Δ Post-Divergence")
    ax.set_xlabel("Parameters")
    ax.set_title("Post-Divergence Gap\n(cascade vs recovery)")
    ax.set_xticks(x_params)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax.legend(fontsize=9)

    fig.suptitle("Token-Level Divergence: Factual Pairs vs Mandela Items",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regime_comparison_scaling.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [1/2] regime_comparison_scaling.png")

    # --- Plot 2: 6.9B delta distributions side-by-side ---
    if focus in regime1_by_model and focus in regime2_by_model:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        r1_dd = [r["div_delta"] for r in regime1_by_model[focus]]
        r2_dd = [r["div_delta"] for r in regime2_by_model[focus]]

        for ax, data, label, color in [
            (axes[0], r1_dd, "Regime 1: Factual", "#2196F3"),
            (axes[1], r2_dd, "Regime 2: Mandela", "#F44336"),
        ]:
            wins = sum(1 for d in data if d > 0)
            ax.hist(data, bins=20, color=color, alpha=0.7, edgecolor="white")
            ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
            ax.axvline(x=np.median(data), color="green", linestyle="-", alpha=0.7,
                       label=f"Median: {np.median(data):+.4f}")
            ax.set_xlabel("Δ Confidence (Correct − Wrong)")
            ax.set_ylabel("Count")
            ax.set_title(f"{label}\n"
                         f"Win: {wins}/{len(data)} ({wins/len(data):.0%}), "
                         f"mean={np.mean(data):+.4f}")
            ax.legend(fontsize=9)

        fig.suptitle("Divergence Delta Distribution — Pythia 6.9B",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "regime_comparison_histograms.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    [2/2] regime_comparison_histograms.png")

    # ===================================================================
    # Save
    # ===================================================================
    save_data = {"scaling": scaling_data}
    if focus in regime1_by_model:
        save_data["regime1_6.9b"] = [
            {k: v for k, v in r.items()} for r in regime1_by_model[focus]
        ]
    if focus in regime2_by_model:
        save_data["regime2_6.9b"] = [
            {k: v for k, v in r.items()} for r in regime2_by_model[focus]
        ]

    with open(RESULTS_DIR / "regime_comparison.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE ({total_time:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
