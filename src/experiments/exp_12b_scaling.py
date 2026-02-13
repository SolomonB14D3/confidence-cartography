"""
Experiment: Pythia 12B — 7th Scaling Data Point
=================================================
Run all scaling experiments through Pythia 12B to complete the 7-point
scaling curves.

Key target: Does anomaly detection (A4) cross p < 0.05?
  - 6.9B: effect=-0.069, p=0.094 (trending)
  - Linear extrapolation: ~0.085-0.095 effect, p~0.05-0.07

Also runs:
  1. Truth detection (A1) — does win rate reach 92-95%?
  2. Settled/contested (A2) — does Spearman ρ continue strengthening?
  3. Anomaly detection (A4) — THE HEADLINE TARGET
  4. Medical validation (Exp 9) — does 88% at 6.9B improve?
  5. Mandela expanded (13 items) — does ρ > 0.70?

This gives complete 7-point scaling curves for everything.
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
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text, unload_model
from src.schema import save_records, load_records, ConfidenceRecord
from src.scaling import (
    MODEL_REGISTRY, SCALING_MODELS_ALL, PARAM_COUNTS,
    get_scaling_output_path,
)
from src.scaling_viz import (
    plot_scaling_law, plot_pvalue_cascade, plot_roc_overlay,
    MODEL_COLORS, model_display_name,
)
from src.utils import SCALING_FIGURES_DIR

# Import prompts from each experiment
from src.experiments.exp2_truth import PAIRS as TRUTH_PAIRS
from src.experiments.exp3_contested import PROMPTS as CONTESTED_PROMPTS, CATEGORY_ORDER
from src.experiments.exp5_consensus import STATEMENTS as CONSENSUS_STATEMENTS
from src.experiments.exp6_anomaly import SCENARIOS, CONTEXT_TYPES
from src.experiments.exp9_medical_validation import (
    MEDICAL_PAIRS, MEDICAL_MANDELA,
)
from src.experiments.exp_mandela_expanded import (
    LINGUISTIC_ITEMS, _make_texts, filter_raw, _records_to_pairs,
)

# Reuse metric computation from existing experiments
from src.experiments.exp_a1_truth_scaling import compute_metrics as compute_truth_metrics, compute_roc
from src.experiments.exp_a2_contested_scaling import compute_metrics as compute_contested_metrics
from src.experiments.exp_a4_failed_scaling import (
    compute_consensus_metrics, compute_anomaly_metrics,
)

# Output directories
FIGURES_12B = SCALING_FIGURES_DIR / "12b"
FIGURES_12B.mkdir(parents=True, exist_ok=True)
FIGURES_UPDATED = SCALING_FIGURES_DIR / "updated"
FIGURES_UPDATED.mkdir(parents=True, exist_ok=True)

SIZE = "12b"
SPEC = MODEL_REGISTRY[SIZE]
MODEL_NAME = SPEC["name"]
DTYPE = SPEC["dtype"]


# ===================================================================
# Experiment runners (12B only — reuse existing functions' patterns)
# ===================================================================

def run_truth_12b(force: bool = False) -> dict:
    """A1: Truth vs Falsehood at 12B."""
    output_path = get_scaling_output_path("a1_truth", SIZE)

    if output_path.exists() and not force:
        print(f"  [truth/12b] Loading from cache...")
        records = load_records(output_path)
    else:
        records = []
        print(f"\n  [truth/12b] Analyzing {len(TRUTH_PAIRS)} pairs...")
        start = time.time()

        for pair in tqdm(TRUTH_PAIRS, desc="  truth/12b", leave=False):
            true_rec = analyze_fixed_text(
                pair["true"], category="true", label=f"{pair['id']}_true",
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            false_rec = analyze_fixed_text(
                pair["false"], category="false", label=f"{pair['id']}_false",
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            records.extend([true_rec, false_rec])

        elapsed = time.time() - start
        print(f"  [truth/12b] Done in {elapsed:.1f}s ({len(records)} records)")

        if output_path.exists():
            output_path.unlink()
        save_records(records, output_path)

    metrics = compute_truth_metrics(records)
    print(f"  [truth/12b] Win rate: {metrics['wins']}/{metrics['n_pairs']} "
          f"({metrics['win_rate']:.1%}), AUC: {metrics['auc']:.3f}, "
          f"p={metrics['p_value']:.4f}")
    return metrics


def run_contested_12b(force: bool = False) -> dict:
    """A2: Settled vs Contested at 12B."""
    output_path = get_scaling_output_path("a2_contested", SIZE)

    if output_path.exists() and not force:
        print(f"  [contested/12b] Loading from cache...")
        records = load_records(output_path)
    else:
        records = []
        print(f"\n  [contested/12b] Analyzing {len(CONTESTED_PROMPTS)} prompts...")
        start = time.time()

        for p in tqdm(CONTESTED_PROMPTS, desc="  contested/12b", leave=False):
            rec = analyze_fixed_text(
                p["text"], category=p["category"], label=p["label"],
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            rec.metadata = {"knowledge_level": p["category"]}
            records.append(rec)

        elapsed = time.time() - start
        print(f"  [contested/12b] Done in {elapsed:.1f}s")

        if output_path.exists():
            output_path.unlink()
        save_records(records, output_path)

    metrics = compute_contested_metrics(records)
    print(f"  [contested/12b] Spearman ρ={metrics['spearman_rho']:+.3f} "
          f"(p={metrics['spearman_p']:.4f}), "
          f"2-way: {metrics['acc_2way']:.1%}, 4-way: {metrics['acc_4way']:.1%}")
    return metrics


def run_anomaly_12b(force: bool = False) -> dict:
    """A4: Anomaly (context injection) at 12B — THE HEADLINE."""
    output_path = get_scaling_output_path("a4_anomaly", SIZE)

    if output_path.exists() and not force:
        print(f"  [anomaly/12b] Loading from cache...")
        records = load_records(output_path)
    else:
        records = []
        n_total = len(SCENARIOS) * len(CONTEXT_TYPES)
        print(f"\n  [anomaly/12b] Analyzing {n_total} texts...")
        start = time.time()

        for scenario in tqdm(SCENARIOS, desc="  anomaly/12b", leave=False):
            for ctx_type in CONTEXT_TYPES:
                ctx = scenario["contexts"][ctx_type]
                claim = scenario["base_claim"]
                full_text = f"{ctx} {claim}".strip() if ctx else claim

                rec = analyze_fixed_text(
                    full_text, category=ctx_type,
                    label=f"{scenario['id']}__{ctx_type}",
                    model_name=MODEL_NAME, revision="main", dtype=DTYPE,
                )
                rec.metadata = {
                    "scenario": scenario["id"],
                    "context_type": ctx_type,
                }
                records.append(rec)

        elapsed = time.time() - start
        print(f"  [anomaly/12b] Done in {elapsed:.1f}s")

        if output_path.exists():
            output_path.unlink()
        save_records(records, output_path)

    metrics = compute_anomaly_metrics(records)
    print(f"  [anomaly/12b] Contra-Support effect={metrics['effect_size']:+.4f} "
          f"(p={metrics['contra_vs_support_p']:.4f})")
    return metrics


def run_consensus_12b(force: bool = False) -> dict:
    """A4 consensus: Consensus detection at 12B."""
    output_path = get_scaling_output_path("a4_consensus", SIZE)

    if output_path.exists() and not force:
        print(f"  [consensus/12b] Loading from cache...")
        records = load_records(output_path)
    else:
        records = []
        print(f"\n  [consensus/12b] Analyzing {len(CONSENSUS_STATEMENTS)} statements...")
        start = time.time()

        for s in tqdm(CONSENSUS_STATEMENTS, desc="  consensus/12b", leave=False):
            rec = analyze_fixed_text(
                s["text"], category=s["domain"], label=s["label"],
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            rec.metadata = {"consensus": s["consensus"], "domain": s["domain"]}
            records.append(rec)

        elapsed = time.time() - start
        print(f"  [consensus/12b] Done in {elapsed:.1f}s")

        if output_path.exists():
            output_path.unlink()
        save_records(records, output_path)

    metrics = compute_consensus_metrics(records, CONSENSUS_STATEMENTS)
    print(f"  [consensus/12b] Pearson r={metrics['pearson_r']:+.3f} "
          f"(p={metrics['pearson_p']:.4f})")
    return metrics


def run_medical_12b(force: bool = False) -> dict:
    """Exp 9: Medical validation at 12B."""
    from src.utils import PROJECT_ROOT as _PR
    results_dir = _PR / "data" / "results" / "exp9"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"medical_pairs_{SIZE}.jsonl"

    if output_path.exists() and not force:
        print(f"  [medical/12b] Loading from cache...")
        records = load_records(output_path)
    else:
        records = []
        print(f"\n  [medical/12b] Analyzing {len(MEDICAL_PAIRS)} pairs...")
        start = time.time()

        for pair in tqdm(MEDICAL_PAIRS, desc="  medical/12b", leave=False):
            true_rec = analyze_fixed_text(
                pair["true"], category="medical_true",
                label=f"{pair['id']}_true",
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            true_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                                 "version": "true", "source": "curated"}

            false_rec = analyze_fixed_text(
                pair["false"], category="medical_false",
                label=f"{pair['id']}_false",
                model_name=MODEL_NAME, revision="main", dtype=DTYPE,
            )
            false_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                                  "version": "false", "source": "curated"}

            records.extend([true_rec, false_rec])

        elapsed = time.time() - start
        print(f"  [medical/12b] Done in {elapsed:.1f}s")

        if output_path.exists():
            output_path.unlink()
        save_records(records, output_path)

    # Compute metrics
    by_id = defaultdict(dict)
    for r in records:
        pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
        version = r.metadata.get("version", "true" if "true" in r.label else "false")
        by_id[pid][version] = r

    pair_results = []
    for pid, versions in by_id.items():
        if "true" in versions and "false" in versions:
            delta = versions["true"].mean_top1_prob - versions["false"].mean_top1_prob
            pair_results.append({
                "pair_id": pid,
                "true_conf": versions["true"].mean_top1_prob,
                "false_conf": versions["false"].mean_top1_prob,
                "delta": delta,
                "true_wins": delta > 0,
            })

    wins = sum(1 for r in pair_results if r["true_wins"])
    n = len(pair_results)
    win_rate = wins / n if n > 0 else 0
    deltas = [r["delta"] for r in pair_results]
    t_stat, p_val = stats.ttest_1samp(deltas, 0) if len(deltas) > 1 else (0, 1)
    d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0

    metrics = {
        "win_rate": win_rate,
        "wins": wins,
        "n": n,
        "p_value": p_val,
        "cohens_d": d,
    }

    print(f"  [medical/12b] Win rate: {wins}/{n} ({win_rate:.1%}), "
          f"p={p_val:.4f}, d={d:.3f}")
    return metrics


def run_mandela_expanded_12b(force: bool = False) -> dict:
    """Expanded linguistic Mandela at 12B."""
    from src.utils import MANDELA_RESULTS_DIR
    results_dir = MANDELA_RESULTS_DIR / "expanded"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"expanded_{SIZE}.jsonl"

    if output_path.exists() and not force:
        print(f"  [mandela/12b] Loading from cache...")
        records = load_records(output_path)
    else:
        records = []
        n_texts = sum(len(_make_texts(item)) * 2 for item in LINGUISTIC_ITEMS)
        print(f"\n  [mandela/12b] Analyzing {n_texts} texts...")
        start = time.time()

        for item in tqdm(LINGUISTIC_ITEMS, desc="  mandela/12b", leave=False):
            for framing_name, wrong_text, correct_text in _make_texts(item):
                w_rec = analyze_fixed_text(
                    wrong_text,
                    category="mandela_wrong",
                    label=f"{item['id']}_{framing_name}_wrong",
                    model_name=MODEL_NAME, revision="main", dtype=DTYPE,
                )
                w_rec.metadata = {
                    "item_id": item["id"],
                    "framing": framing_name,
                    "version": "wrong",
                    "human_ratio": item["human_ratio"],
                    "human_wrong_pct": item["human_wrong_pct"],
                    "human_correct_pct": item["human_correct_pct"],
                    "source": item["source"],
                }

                c_rec = analyze_fixed_text(
                    correct_text,
                    category="mandela_correct",
                    label=f"{item['id']}_{framing_name}_correct",
                    model_name=MODEL_NAME, revision="main", dtype=DTYPE,
                )
                c_rec.metadata = {
                    "item_id": item["id"],
                    "framing": framing_name,
                    "version": "correct",
                    "human_ratio": item["human_ratio"],
                    "human_wrong_pct": item["human_wrong_pct"],
                    "human_correct_pct": item["human_correct_pct"],
                    "source": item["source"],
                }

                records.extend([w_rec, c_rec])

        elapsed = time.time() - start
        print(f"  [mandela/12b] Done in {elapsed:.1f}s ({len(records)} records)")

        if output_path.exists():
            output_path.unlink()
        save_records(records, output_path)

    # Compute correlation metrics
    pairs = _records_to_pairs(records)
    raw = filter_raw(pairs)

    h = [v["human_ratio"] for v in raw]
    m = [v["confidence_ratio"] for v in raw]
    r, p = stats.pearsonr(h, m)
    rho, rho_p = stats.spearmanr(h, m)

    metrics = {
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "n_items": len(raw),
        "pairs": pairs,
    }

    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  [mandela/12b] r={r:.3f} (p={p:.4f}){sig}  ρ={rho:.3f} (p={rho_p:.4f})")
    return metrics


# ===================================================================
# Updated 7-point scaling plots
# ===================================================================

def generate_updated_plots(
    truth_12b: dict,
    contested_12b: dict,
    anomaly_12b: dict,
    consensus_12b: dict,
    medical_12b: dict,
    mandela_12b: dict,
):
    """Regenerate all scaling plots with the 7th data point."""
    print("\n" + "=" * 65)
    print("GENERATING UPDATED 7-POINT SCALING PLOTS")
    print("=" * 65)

    all_sizes = SCALING_MODELS_ALL

    # Load existing metrics from cached results
    print("\n  Loading existing 6 model results for comparison...")

    # === A1: Truth Detection ===
    print("\n  [1/6] Truth detection 7-point curve...")
    truth_metrics = {}
    for size in SCALING_MODELS_ALL:
        path = get_scaling_output_path("a1_truth", size)
        if path.exists():
            records = load_records(path)
            truth_metrics[size] = compute_truth_metrics(records)

    sizes_t = [s for s in all_sizes if s in truth_metrics]
    if len(sizes_t) >= 2:
        plot_scaling_law(
            sizes_t,
            {"AUC": [truth_metrics[s]["auc"] for s in sizes_t]},
            ylabel="AUC",
            title="Truth Detection AUC vs Model Size (7 points)",
            save_path=FIGURES_UPDATED / "a1_auc_scaling_7pt.png",
            hline=0.5, hline_label="Random",
        )
        plot_scaling_law(
            sizes_t,
            {
                "Win Rate": [truth_metrics[s]["win_rate"] for s in sizes_t],
                "Cohen's d": [truth_metrics[s]["cohens_d"] for s in sizes_t],
            },
            ylabel="Value",
            title="Truth Detection Metrics vs Model Size (7 points)",
            save_path=FIGURES_UPDATED / "a1_metrics_scaling_7pt.png",
            hline=0.5, hline_label="Chance",
        )
        plot_pvalue_cascade(
            sizes_t,
            {"Wilcoxon (true > false)": [truth_metrics[s]["p_value"] for s in sizes_t]},
            title="Truth Detection: Significance (7 points)",
            save_path=FIGURES_UPDATED / "a1_pvalue_7pt.png",
        )

    # === A2: Settled vs Contested ===
    print("  [2/6] Settled vs contested 7-point curve...")
    contested_metrics = {}
    for size in SCALING_MODELS_ALL:
        path = get_scaling_output_path("a2_contested", size)
        if path.exists():
            records = load_records(path)
            contested_metrics[size] = compute_contested_metrics(records)

    sizes_c = [s for s in all_sizes if s in contested_metrics]
    if len(sizes_c) >= 2:
        plot_scaling_law(
            sizes_c,
            {
                "Spearman ρ (confidence)": [contested_metrics[s]["spearman_rho"] for s in sizes_c],
                "Spearman ρ (entropy)": [contested_metrics[s]["spearman_rho_entropy"] for s in sizes_c],
            },
            ylabel="Spearman ρ",
            title="Knowledge Level Correlation (7 points)",
            save_path=FIGURES_UPDATED / "a2_spearman_scaling_7pt.png",
            hline=0.0, hline_label="No correlation",
        )
        plot_pvalue_cascade(
            sizes_c,
            {
                "Spearman": [contested_metrics[s]["spearman_p"] for s in sizes_c],
                "Mann-Whitney": [contested_metrics[s]["mann_whitney_p"] for s in sizes_c],
            },
            title="Knowledge Level: Significance (7 points)",
            save_path=FIGURES_UPDATED / "a2_pvalue_7pt.png",
        )

    # === A4: Anomaly Detection (HEADLINE) ===
    print("  [3/6] ANOMALY DETECTION 7-point curve (THE HEADLINE)...")
    anomaly_metrics = {}
    for size in SCALING_MODELS_ALL:
        path = get_scaling_output_path("a4_anomaly", size)
        if path.exists():
            records = load_records(path)
            anomaly_metrics[size] = compute_anomaly_metrics(records)

    sizes_a = [s for s in all_sizes if s in anomaly_metrics]
    if len(sizes_a) >= 2:
        plot_scaling_law(
            sizes_a,
            {
                "Contradiction delta": [anomaly_metrics[s]["mean_contra_delta"] for s in sizes_a],
                "Support delta": [anomaly_metrics[s]["mean_support_delta"] for s in sizes_a],
                "Effect (contra-support)": [anomaly_metrics[s]["effect_size"] for s in sizes_a],
            },
            ylabel="Confidence Delta",
            title="Context Anomaly Detection vs Model Size (7 points)",
            save_path=FIGURES_UPDATED / "a4_anomaly_effect_7pt.png",
            hline=0.0, hline_label="No effect",
        )
        plot_pvalue_cascade(
            sizes_a,
            {"Contra vs Support": [anomaly_metrics[s]["contra_vs_support_p"] for s in sizes_a]},
            title="Anomaly Detection: Significance (7 points)",
            save_path=FIGURES_UPDATED / "a4_anomaly_pvalue_7pt.png",
        )

    # === Medical ===
    print("  [4/6] Medical validation scaling...")
    from src.utils import PROJECT_ROOT as _PR
    medical_wins = {}
    for size in SCALING_MODELS_ALL:
        path = _PR / "data" / "results" / "exp9" / f"medical_pairs_{size}.jsonl"
        if path.exists():
            records = load_records(path)
            by_id = defaultdict(dict)
            for r in records:
                pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
                version = r.metadata.get("version", "true" if "true" in r.label else "false")
                by_id[pid][version] = r
            wins = sum(1 for pid, v in by_id.items()
                      if "true" in v and "false" in v
                      and v["true"].mean_top1_prob > v["false"].mean_top1_prob)
            total = sum(1 for pid, v in by_id.items()
                       if "true" in v and "false" in v)
            medical_wins[size] = wins / total if total > 0 else 0

    sizes_m = [s for s in all_sizes if s in medical_wins]
    if len(sizes_m) >= 2:
        plot_scaling_law(
            sizes_m,
            {"Medical Win Rate": [medical_wins[s] for s in sizes_m]},
            ylabel="Win Rate",
            title="Medical Domain Validation (7 points)",
            save_path=FIGURES_UPDATED / "medical_scaling_7pt.png",
            hline=0.5, hline_label="Chance",
        )

    # === Mandela Expanded ===
    print("  [5/6] Mandela expanded scaling...")
    from src.utils import MANDELA_RESULTS_DIR
    mandela_correlations = {}
    for size in SCALING_MODELS_ALL:
        path = MANDELA_RESULTS_DIR / "expanded" / f"expanded_{size}.jsonl"
        if path.exists():
            records = load_records(path)
            pairs = _records_to_pairs(records)
            raw = filter_raw(pairs)
            if raw:
                h = [v["human_ratio"] for v in raw]
                m = [v["confidence_ratio"] for v in raw]
                r, p = stats.pearsonr(h, m)
                rho, rho_p = stats.spearmanr(h, m)
                mandela_correlations[size] = {"r": r, "p": p, "rho": rho, "rho_p": rho_p}

    sizes_md = [s for s in all_sizes if s in mandela_correlations]
    if len(sizes_md) >= 2:
        plot_scaling_law(
            sizes_md,
            {
                "Pearson r": [mandela_correlations[s]["r"] for s in sizes_md],
                "Spearman ρ": [mandela_correlations[s]["rho"] for s in sizes_md],
            },
            ylabel="Correlation with Human Prevalence",
            title="Mandela Calibration — Correlation Scaling (7 points)",
            save_path=FIGURES_UPDATED / "mandela_correlation_7pt.png",
            hline=0.0, hline_label="No correlation",
        )

    # === Grand Summary Dashboard ===
    print("  [6/6] Grand summary dashboard...")
    from src.scaling_viz import plot_scaling_dashboard
    panels = []

    if len(sizes_t) >= 2:
        panels.append({
            "sizes": sizes_t,
            "metrics": {"Win Rate": [truth_metrics[s]["win_rate"] for s in sizes_t]},
            "ylabel": "Win Rate",
            "subtitle": "A1: Truth Detection",
            "hline": 0.5,
        })

    if len(sizes_c) >= 2:
        panels.append({
            "sizes": sizes_c,
            "metrics": {"Spearman ρ": [contested_metrics[s]["spearman_rho"] for s in sizes_c]},
            "ylabel": "Spearman ρ",
            "subtitle": "A2: Settled vs Contested",
            "hline": 0.0,
        })

    if len(sizes_a) >= 2:
        panels.append({
            "sizes": sizes_a,
            "metrics": {"Effect Size": [anomaly_metrics[s]["effect_size"] for s in sizes_a]},
            "ylabel": "Effect Size",
            "subtitle": "A4: Anomaly Detection ★",
            "hline": 0.0,
        })

    if len(sizes_m) >= 2:
        panels.append({
            "sizes": sizes_m,
            "metrics": {"Win Rate": [medical_wins[s] for s in sizes_m]},
            "ylabel": "Win Rate",
            "subtitle": "Exp9: Medical Domain",
            "hline": 0.5,
        })

    if len(sizes_md) >= 2:
        panels.append({
            "sizes": sizes_md,
            "metrics": {"Spearman ρ": [mandela_correlations[s]["rho"] for s in sizes_md]},
            "ylabel": "Spearman ρ",
            "subtitle": "Mandela Calibration",
            "hline": 0.0,
        })

    if len(panels) >= 2:
        plot_scaling_dashboard(
            panels,
            title="Confidence Cartography — Complete 7-Point Scaling Laws",
            save_path=FIGURES_UPDATED / "grand_dashboard_7pt.png",
            ncols=3,
        )


# ===================================================================
# Main
# ===================================================================

def run_experiment(force: bool = False):
    total_start = time.time()

    print("=" * 70)
    print("PYTHIA 12B — 7th SCALING DATA POINT")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Parameters: {SPEC['params']/1e9:.1f}B")
    print(f"Dtype: {DTYPE}")
    print(f"Device: M3 Ultra 96GB unified memory")
    print()
    print("Experiments to run:")
    print(f"  1. Truth detection (A1): {len(TRUTH_PAIRS)} pairs = {len(TRUTH_PAIRS)*2} texts")
    print(f"  2. Settled/contested (A2): {len(CONTESTED_PROMPTS)} prompts")
    print(f"  3. Consensus (A4 part 1): {len(CONSENSUS_STATEMENTS)} statements")
    print(f"  4. Anomaly detection (A4 part 2): {len(SCENARIOS)}×{len(CONTEXT_TYPES)} = {len(SCENARIOS)*len(CONTEXT_TYPES)} texts")
    print(f"  5. Medical validation (Exp9): {len(MEDICAL_PAIRS)} pairs = {len(MEDICAL_PAIRS)*2} texts")
    n_mandela_texts = sum(len(_make_texts(item)) * 2 for item in LINGUISTIC_ITEMS)
    print(f"  6. Mandela expanded: {len(LINGUISTIC_ITEMS)} items = {n_mandela_texts} texts")
    total_texts = (len(TRUTH_PAIRS)*2 + len(CONTESTED_PROMPTS) +
                   len(CONSENSUS_STATEMENTS) + len(SCENARIOS)*len(CONTEXT_TYPES) +
                   len(MEDICAL_PAIRS)*2 + n_mandela_texts)
    print(f"\n  TOTAL: {total_texts} forward passes through 12B model")
    print(f"  Estimated time: ~{total_texts * 0.3 * 42 / 60:.0f} min")

    # ---------------------------------------------------------------
    # Run all experiments (model stays loaded between them)
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1: RUNNING EXPERIMENTS")
    print("=" * 70)

    truth_m = run_truth_12b(force=force)
    contested_m = run_contested_12b(force=force)
    consensus_m = run_consensus_12b(force=force)
    anomaly_m = run_anomaly_12b(force=force)

    # Unload and reload for medical/mandela (same model, just clearing state)
    medical_m = run_medical_12b(force=force)
    mandela_m = run_mandela_expanded_12b(force=force)

    # Free memory before plotting
    unload_model()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("12B RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  {'Experiment':<30} {'Metric':<25} {'6.9B':<15} {'12B':<15} {'Target':<15}")
    print("  " + "-" * 100)
    print(f"  {'A1: Truth detection':<30} {'Win rate':<25} {'90%':<15} "
          f"{truth_m['win_rate']:.1%}{'':<10} {'92%+':<15}")
    print(f"  {'A2: Settled/contested':<30} {'Spearman ρ':<25} {'-0.618':<15} "
          f"{contested_m['spearman_rho']:+.3f}{'':<10} {'< -0.65':<15}")
    print(f"  {'A4: Anomaly detection ★':<30} {'p-value':<25} {'0.094':<15} "
          f"{anomaly_m['contra_vs_support_p']:.4f}{'':<10} {'< 0.05':<15}")
    print(f"  {'A4: Anomaly effect':<30} {'Effect size':<25} {'-0.069':<15} "
          f"{anomaly_m['effect_size']:+.4f}{'':<10} {'~0.085+':<15}")
    print(f"  {'Exp9: Medical':<30} {'Win rate':<25} {'88%':<15} "
          f"{medical_m['win_rate']:.1%}{'':<10} {'90%+':<15}")
    print(f"  {'Mandela expanded':<30} {'Spearman ρ':<25} {'0.652':<15} "
          f"{mandela_m['spearman_rho']:.3f}{'':<10} {'> 0.70':<15}")

    # ---------------------------------------------------------------
    # HEADLINE VERDICT
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    anomaly_p = anomaly_m["contra_vs_support_p"]
    if anomaly_p < 0.05:
        print("★★★ HEADLINE: ANOMALY DETECTION CROSSES SIGNIFICANCE AT 12B ★★★")
        print(f"    p = {anomaly_p:.4f} < 0.05")
        print(f"    Effect size: {anomaly_m['effect_size']:+.4f}")
        print("    → Context-sensitivity EMERGES with sufficient model capacity")
        print("    → Add emergence finding to paper!")
    elif anomaly_p < 0.10:
        print("★ ANOMALY DETECTION: STILL TRENDING, NOT SIGNIFICANT")
        print(f"    p = {anomaly_p:.4f} (was 0.094 at 6.9B)")
        print(f"    Effect size: {anomaly_m['effect_size']:+.4f} (was -0.069 at 6.9B)")
        print("    → Report trend line with prediction for threshold model size")
    else:
        print("ANOMALY DETECTION: NO IMPROVEMENT AT 12B")
        print(f"    p = {anomaly_p:.4f}")
        print("    → May be fundamental limitation, not just capacity-gated")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Generate updated 7-point plots
    # ---------------------------------------------------------------
    generate_updated_plots(
        truth_m, contested_m, anomaly_m, consensus_m, medical_m, mandela_m,
    )

    # ---------------------------------------------------------------
    # Save complete results JSON
    # ---------------------------------------------------------------
    results_json = {
        "model": MODEL_NAME,
        "size": SIZE,
        "params": SPEC["params"],
        "truth": {k: v for k, v in truth_m.items() if k != "roc"},
        "contested": contested_m,
        "anomaly": anomaly_m,
        "consensus": consensus_m,
        "medical": medical_m,
        "mandela": {k: v for k, v in mandela_m.items() if k != "pairs"},
    }

    results_path = FIGURES_UPDATED / "12b_results_summary.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  Results summary saved to: {results_path}")

    total_time = time.time() - total_start
    print(f"\n  Total experiment time: {total_time/60:.1f} min")
    print(f"  Figures directory: {FIGURES_UPDATED}")

    fig_count = len(list(FIGURES_UPDATED.glob("*.png")))
    print(f"  New figures: {fig_count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if cached results exist")
    args = parser.parse_args()

    run_experiment(force=args.force)
