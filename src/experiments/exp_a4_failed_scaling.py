"""
Experiment A4: Failed Experiments at Scale
==========================================
Run Exp 5 (Consensus Detection) and Exp 6 (Context Anomaly) across all
Pythia model sizes. These produced no signal at 160M — can larger models
rescue them?

A flat line at zero = fundamental limitation of confidence-as-sensor.
A rising curve = capacity-gated capability. Both are valuable findings.
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
    get_scaling_output_path, print_runtime_estimates,
)
from src.scaling_viz import (
    plot_scaling_law, plot_pvalue_cascade, MODEL_COLORS, model_display_name,
)
from src.utils import SCALING_FIGURES_DIR

# Import the exact prompts from Phase 1 experiments
from src.experiments.exp5_consensus import STATEMENTS
from src.experiments.exp6_anomaly import SCENARIOS, CONTEXT_TYPES


# ---------------------------------------------------------------------------
# Exp 5 (Consensus) at scale
# ---------------------------------------------------------------------------

def run_consensus_model(size: str, force: bool = False) -> list[ConfidenceRecord]:
    """Run consensus analysis for one model size."""
    output_path = get_scaling_output_path("a4_consensus", size)

    if output_path.exists() and not force:
        print(f"  [consensus/{size}] Loading from cache...")
        return load_records(output_path)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []

    print(f"\n  [consensus/{size}] Analyzing {len(STATEMENTS)} statements with {model_name}...")
    start = time.time()

    for s in tqdm(STATEMENTS, desc=f"  cons/{size}", leave=False):
        rec = analyze_fixed_text(
            s["text"], category=s["domain"], label=s["label"],
            model_name=model_name, revision="main", dtype=dtype,
        )
        rec.metadata = {"consensus": s["consensus"], "domain": s["domain"]}
        records.append(rec)

    elapsed = time.time() - start
    print(f"  [consensus/{size}] Done in {elapsed:.1f}s")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return records


def compute_consensus_metrics(records: list[ConfidenceRecord], statements: list[dict]) -> dict:
    """Compute consensus correlation metrics."""
    consensus = np.array([s["consensus"] for s in statements])
    mean_probs = np.array([r.mean_top1_prob for r in records])
    mean_ents = np.array([r.mean_entropy for r in records])

    # Pearson
    r_p, p_p = stats.pearsonr(consensus, mean_probs)
    r_e, p_e = stats.pearsonr(consensus, mean_ents)

    # Spearman
    rs_p, ps_p = stats.spearmanr(consensus, mean_probs)
    rs_e, ps_e = stats.spearmanr(consensus, mean_ents)

    return {
        "pearson_r": r_p, "pearson_p": p_p,
        "pearson_r_entropy": r_e, "pearson_p_entropy": p_e,
        "spearman_rho": rs_p, "spearman_p": ps_p,
        "spearman_rho_entropy": rs_e, "spearman_p_entropy": ps_e,
    }


# ---------------------------------------------------------------------------
# Exp 6 (Anomaly) at scale
# ---------------------------------------------------------------------------

def run_anomaly_model(size: str, force: bool = False) -> list[ConfidenceRecord]:
    """Run anomaly (context injection) analysis for one model size."""
    output_path = get_scaling_output_path("a4_anomaly", size)

    if output_path.exists() and not force:
        print(f"  [anomaly/{size}] Loading from cache...")
        return load_records(output_path)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []

    n_total = len(SCENARIOS) * len(CONTEXT_TYPES)
    print(f"\n  [anomaly/{size}] Analyzing {n_total} texts with {model_name}...")
    start = time.time()

    for scenario in tqdm(SCENARIOS, desc=f"  anom/{size}", leave=False):
        for ctx_type in CONTEXT_TYPES:
            ctx = scenario["contexts"][ctx_type]
            claim = scenario["base_claim"]
            full_text = f"{ctx} {claim}".strip() if ctx else claim

            rec = analyze_fixed_text(
                full_text, category=ctx_type,
                label=f"{scenario['id']}__{ctx_type}",
                model_name=model_name, revision="main", dtype=dtype,
            )
            rec.metadata = {
                "scenario": scenario["id"],
                "context_type": ctx_type,
            }
            records.append(rec)

    elapsed = time.time() - start
    print(f"  [anomaly/{size}] Done in {elapsed:.1f}s")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return records


def compute_anomaly_metrics(records: list[ConfidenceRecord]) -> dict:
    """Compute anomaly detection metrics from records."""
    # Organize by scenario and context
    results = defaultdict(dict)
    for r in records:
        sid = r.metadata.get("scenario", r.label.split("__")[0])
        ctx = r.metadata.get("context_type", r.label.split("__")[1] if "__" in r.label else r.category)
        results[sid][ctx] = r

    # Compute deltas relative to "none" baseline
    deltas = defaultdict(dict)
    for sid, ctx_recs in results.items():
        if "none" not in ctx_recs:
            continue
        base_prob = ctx_recs["none"].mean_top1_prob
        for ctx_type in ["neutral", "supporting", "contradicting", "noise"]:
            if ctx_type in ctx_recs:
                deltas[sid][ctx_type] = ctx_recs[ctx_type].mean_top1_prob - base_prob

    # Aggregate
    contra_deltas = [deltas[sid]["contradicting"] for sid in deltas if "contradicting" in deltas[sid]]
    support_deltas = [deltas[sid]["supporting"] for sid in deltas if "supporting" in deltas[sid]]

    if len(contra_deltas) >= 2 and len(support_deltas) >= 2:
        t_cs, p_cs = stats.ttest_rel(contra_deltas, support_deltas)
    else:
        t_cs, p_cs = 0.0, 1.0

    mean_contra = float(np.mean(contra_deltas)) if contra_deltas else 0.0
    mean_support = float(np.mean(support_deltas)) if support_deltas else 0.0

    return {
        "mean_contra_delta": mean_contra,
        "mean_support_delta": mean_support,
        "contra_vs_support_t": t_cs,
        "contra_vs_support_p": p_cs,
        "effect_size": mean_contra - mean_support,
        "n_scenarios": len(contra_deltas),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(models: list[str] = None, force: bool = False):
    models = models or SCALING_MODELS

    n_texts = len(STATEMENTS) + len(SCENARIOS) * len(CONTEXT_TYPES)

    print("=" * 65)
    print("EXPERIMENT A4: Failed Experiments at Scale")
    print("=" * 65)
    print(f"Models: {', '.join(models)}")
    print(f"Exp 5 (Consensus): {len(STATEMENTS)} statements")
    print(f"Exp 6 (Anomaly): {len(SCENARIOS)} scenarios × {len(CONTEXT_TYPES)} contexts = {len(SCENARIOS) * len(CONTEXT_TYPES)}")
    print(f"Total per model: {n_texts} analyses")
    print_runtime_estimates(n_texts)

    start_time = time.time()
    consensus_metrics = {}
    anomaly_metrics = {}

    for size in models:
        # --- Consensus ---
        cons_records = run_consensus_model(size, force=force)
        cons_m = compute_consensus_metrics(cons_records, STATEMENTS)
        consensus_metrics[size] = cons_m
        print(f"  [consensus/{size}] Pearson r={cons_m['pearson_r']:+.3f} "
              f"(p={cons_m['pearson_p']:.4f}), "
              f"Spearman ρ={cons_m['spearman_rho']:+.3f} "
              f"(p={cons_m['spearman_p']:.4f})")

        # --- Anomaly ---
        anom_records = run_anomaly_model(size, force=force)
        anom_m = compute_anomaly_metrics(anom_records)
        anomaly_metrics[size] = anom_m
        print(f"  [anomaly/{size}] Contra-Support effect={anom_m['effect_size']:+.4f} "
              f"(p={anom_m['contra_vs_support_p']:.4f})")

        unload_model()

    # ===================================================================
    # Summary Tables
    # ===================================================================
    sizes_done = [s for s in models if s in consensus_metrics]

    print("\n" + "=" * 65)
    print("CONSENSUS DETECTION SCALING")
    print("=" * 65)
    print(f"\n{'Size':<8} {'Params':<12} {'Pearson r':<12} {'p-val':<10} "
          f"{'Spearman ρ':<12} {'p-val':<10}")
    print("-" * 64)
    for size in sizes_done:
        m = consensus_metrics[size]
        params = PARAM_COUNTS[size]
        print(f"{size:<8} {params/1e6:>8.0f}M  {m['pearson_r']:+10.3f} "
              f"{m['pearson_p']:<10.4f} {m['spearman_rho']:+10.3f} "
              f"{m['spearman_p']:<10.4f}")

    print("\n" + "=" * 65)
    print("ANOMALY DETECTION SCALING")
    print("=" * 65)
    print(f"\n{'Size':<8} {'Params':<12} {'Contra Δ':<10} {'Support Δ':<10} "
          f"{'Effect':<10} {'p-val':<10}")
    print("-" * 60)
    for size in sizes_done:
        m = anomaly_metrics[size]
        params = PARAM_COUNTS[size]
        print(f"{size:<8} {params/1e6:>8.0f}M  {m['mean_contra_delta']:+8.4f} "
              f"{m['mean_support_delta']:+8.4f} {m['effect_size']:+8.4f} "
              f"{m['contra_vs_support_p']:<10.4f}")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING SCALING PLOTS")
    print("=" * 65)
    fig_dir = SCALING_FIGURES_DIR

    if len(sizes_done) >= 2:
        # 1. Consensus: Pearson r vs size
        print("\n[1/5] Consensus Pearson r scaling...")
        plot_scaling_law(
            sizes_done,
            {
                "Pearson r (confidence)": [consensus_metrics[s]["pearson_r"] for s in sizes_done],
                "Pearson r (entropy)": [consensus_metrics[s]["pearson_r_entropy"] for s in sizes_done],
            },
            ylabel="Pearson Correlation",
            title="Consensus Detection vs Model Size",
            save_path=fig_dir / "a4_consensus_r_scaling.png",
            hline=0.0, hline_label="No correlation",
        )

        # 2. Consensus: Spearman rho scaling
        print("[2/5] Consensus Spearman ρ scaling...")
        plot_scaling_law(
            sizes_done,
            {"Spearman ρ": [consensus_metrics[s]["spearman_rho"] for s in sizes_done]},
            ylabel="Spearman ρ",
            title="Consensus Rank Correlation vs Model Size",
            save_path=fig_dir / "a4_consensus_rho_scaling.png",
            hline=0.0, hline_label="No correlation",
        )

        # 3. Consensus p-value cascade
        print("[3/5] Consensus p-value cascade...")
        plot_pvalue_cascade(
            sizes_done,
            {
                "Pearson": [consensus_metrics[s]["pearson_p"] for s in sizes_done],
                "Spearman": [consensus_metrics[s]["spearman_p"] for s in sizes_done],
            },
            title="Consensus Detection: Statistical Significance vs Size",
            save_path=fig_dir / "a4_consensus_pvalue.png",
        )

        # 4. Anomaly: effect size scaling
        print("[4/5] Anomaly effect size scaling...")
        plot_scaling_law(
            sizes_done,
            {
                "Contradiction delta": [anomaly_metrics[s]["mean_contra_delta"] for s in sizes_done],
                "Support delta": [anomaly_metrics[s]["mean_support_delta"] for s in sizes_done],
                "Effect (contra-support)": [anomaly_metrics[s]["effect_size"] for s in sizes_done],
            },
            ylabel="Confidence Delta",
            title="Context Anomaly Detection vs Model Size",
            save_path=fig_dir / "a4_anomaly_effect_scaling.png",
            hline=0.0, hline_label="No effect",
        )

        # 5. Anomaly p-value cascade
        print("[5/5] Anomaly p-value cascade...")
        plot_pvalue_cascade(
            sizes_done,
            {"Contra vs Support": [anomaly_metrics[s]["contra_vs_support_p"] for s in sizes_done]},
            title="Anomaly Detection: Statistical Significance vs Size",
            save_path=fig_dir / "a4_anomaly_pvalue.png",
        )

    # ===================================================================
    # Final Summary
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(fig_dir.glob("a4_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT A4 COMPLETE")
    print("=" * 65)
    print(f"  Models: {len(sizes_done)}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s")

    # Verdicts
    if sizes_done:
        # Consensus verdict
        first_r = consensus_metrics[sizes_done[0]]["pearson_r"]
        last_r = consensus_metrics[sizes_done[-1]]["pearson_r"]
        last_p = consensus_metrics[sizes_done[-1]]["pearson_p"]
        print(f"\n  CONSENSUS: r={first_r:+.3f} ({sizes_done[0]}) → "
              f"r={last_r:+.3f} ({sizes_done[-1]})")
        if last_p < 0.05:
            print("  → CONSENSUS DETECTION EMERGES AT SCALE ✓")
        elif abs(last_r) > abs(first_r) + 0.1:
            print("  → TRENDING toward signal, larger models may reach significance")
        else:
            print("  → FUNDAMENTAL LIMITATION: consensus not encoded in confidence")

        # Anomaly verdict
        first_e = anomaly_metrics[sizes_done[0]]["effect_size"]
        last_e = anomaly_metrics[sizes_done[-1]]["effect_size"]
        last_ap = anomaly_metrics[sizes_done[-1]]["contra_vs_support_p"]
        print(f"\n  ANOMALY: effect={first_e:+.4f} ({sizes_done[0]}) → "
              f"effect={last_e:+.4f} ({sizes_done[-1]})")
        if last_ap < 0.05:
            print("  → ANOMALY DETECTION EMERGES AT SCALE ✓")
        elif abs(last_e) > abs(first_e) * 2:
            print("  → TRENDING toward signal, in-context reasoning developing")
        else:
            print("  → FUNDAMENTAL LIMITATION: context anomaly not detectable via confidence")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_experiment(args.models, force=args.force)
