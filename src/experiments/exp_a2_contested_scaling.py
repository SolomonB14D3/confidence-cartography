"""
Experiment A2: Settled vs Contested Knowledge Scaling
=====================================================
Run the same 32 prompts from Exp 3 across all Pythia model sizes.

Key question: Does the 4-way classifier ever become viable?
At 160M it was 43.8% (chance=25%). If it reaches 60%+ at larger scales,
the model is encoding fine-grained knowledge-type information.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
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
    plot_scaling_law, plot_pvalue_cascade, plot_scaling_heatmap,
    MODEL_COLORS, model_display_name,
)
from src.utils import SCALING_FIGURES_DIR

# Import exact same prompts from Phase 1
from src.experiments.exp3_contested import PROMPTS, CATEGORY_ORDER


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_single_model(size: str, force: bool = False) -> list[ConfidenceRecord]:
    """Run settled/contested analysis for one model size."""
    output_path = get_scaling_output_path("a2_contested", size)

    if output_path.exists() and not force:
        print(f"  [{size}] Loading from cache...")
        return load_records(output_path)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []

    print(f"\n  [{size}] Analyzing {len(PROMPTS)} prompts with {model_name} (dtype={dtype})...")
    start = time.time()

    for p in tqdm(PROMPTS, desc=f"  {size}", leave=False):
        rec = analyze_fixed_text(
            p["text"], category=p["category"], label=p["label"],
            model_name=model_name, revision="main", dtype=dtype,
        )
        rec.metadata = {"knowledge_level": p["category"]}
        records.append(rec)

    elapsed = time.time() - start
    print(f"  [{size}] Done in {elapsed:.1f}s ({len(records)} records)")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return records


def compute_metrics(records: list[ConfidenceRecord]) -> dict:
    """Compute settled/contested classification metrics."""
    # Group by category
    by_cat = defaultdict(list)
    for r in records:
        cat = r.metadata.get("knowledge_level", r.category)
        by_cat[cat].append(r)

    # Spearman: knowledge level ordinal score vs mean confidence
    level_map = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    levels = []
    probs = []
    ents = []
    for r in records:
        cat = r.metadata.get("knowledge_level", r.category)
        if cat in level_map:
            levels.append(level_map[cat])
            probs.append(r.mean_top1_prob)
            ents.append(r.mean_entropy)

    levels = np.array(levels)
    probs = np.array(probs)
    ents = np.array(ents)

    rho_p, p_rho_p = stats.spearmanr(levels, probs)
    rho_e, p_rho_e = stats.spearmanr(levels, ents)

    # Mann-Whitney: settled vs contested
    settled_probs = [r.mean_top1_prob for r in by_cat.get("settled", [])]
    contested_probs = [r.mean_top1_prob for r in by_cat.get("contested", [])]
    if len(settled_probs) >= 2 and len(contested_probs) >= 2:
        mw_stat, mw_p = stats.mannwhitneyu(settled_probs, contested_probs, alternative="two-sided")
    else:
        mw_stat, mw_p = 0.0, 1.0

    # 2-way classifier: settled vs contested
    acc_2way = 0.0
    sc_records = [r for r in records if r.category in ("settled", "contested")]
    if len(sc_records) >= 6:
        X_2 = np.array([[r.mean_top1_prob, r.mean_entropy] for r in sc_records])
        y_2 = np.array([0 if r.category == "settled" else 1 for r in sc_records])
        scaler = StandardScaler()
        X_2s = scaler.fit_transform(X_2)
        n_cv = min(5, len(y_2) // 2)
        if n_cv >= 2:
            scores = cross_val_score(
                LogisticRegression(max_iter=1000), X_2s, y_2, cv=n_cv, scoring="accuracy"
            )
            acc_2way = float(np.mean(scores))

    # 4-way classifier: all categories
    acc_4way = 0.0
    cat_records = [r for r in records if r.category in CATEGORY_ORDER]
    if len(cat_records) >= 8:
        X_4 = np.array([[r.mean_top1_prob, r.mean_entropy] for r in cat_records])
        y_4 = np.array([level_map[r.category] for r in cat_records])
        scaler = StandardScaler()
        X_4s = scaler.fit_transform(X_4)
        n_cv = min(5, min(np.bincount(y_4)))
        if n_cv >= 2:
            scores = cross_val_score(
                LogisticRegression(max_iter=1000),
                X_4s, y_4, cv=n_cv, scoring="accuracy"
            )
            acc_4way = float(np.mean(scores))

    # Per-category mean confidence
    cat_means = {}
    for cat in CATEGORY_ORDER:
        if cat in by_cat:
            cat_means[cat] = float(np.mean([r.mean_top1_prob for r in by_cat[cat]]))

    return {
        "spearman_rho": rho_p,
        "spearman_p": p_rho_p,
        "spearman_rho_entropy": rho_e,
        "spearman_p_entropy": p_rho_e,
        "mann_whitney_stat": mw_stat,
        "mann_whitney_p": mw_p,
        "acc_2way": acc_2way,
        "acc_4way": acc_4way,
        "cat_means": cat_means,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(models: list[str] = None, force: bool = False):
    models = models or SCALING_MODELS

    print("=" * 65)
    print("EXPERIMENT A2: Settled vs Contested Knowledge Scaling")
    print("=" * 65)
    print(f"Models: {', '.join(models)}")
    print(f"Prompts per model: {len(PROMPTS)}")
    print(f"Categories: {', '.join(CATEGORY_ORDER)}")
    print_runtime_estimates(len(PROMPTS))

    start_time = time.time()
    all_metrics = {}

    for size in models:
        records = run_single_model(size, force=force)
        metrics = compute_metrics(records)
        all_metrics[size] = metrics
        unload_model()

        print(f"  [{size}] Spearman ρ={metrics['spearman_rho']:+.3f} "
              f"(p={metrics['spearman_p']:.4f}), "
              f"2-way: {metrics['acc_2way']:.1%}, "
              f"4-way: {metrics['acc_4way']:.1%}")

    # ===================================================================
    # Summary
    # ===================================================================
    sizes_done = [s for s in models if s in all_metrics]

    print("\n" + "=" * 65)
    print("SCALING SUMMARY")
    print("=" * 65)
    print(f"\n{'Size':<8} {'Params':<12} {'Spear ρ':<10} {'p-val':<10} "
          f"{'MW p-val':<10} {'2-way':<8} {'4-way':<8}")
    print("-" * 66)
    for size in sizes_done:
        m = all_metrics[size]
        params = PARAM_COUNTS[size]
        print(f"{size:<8} {params/1e6:>8.0f}M  {m['spearman_rho']:+8.3f} "
              f"{m['spearman_p']:<10.4f} {m['mann_whitney_p']:<10.4f} "
              f"{m['acc_2way']:<8.1%} {m['acc_4way']:<8.1%}")

    # Per-category confidence by size
    print(f"\n{'Size':<8}", end="")
    for cat in CATEGORY_ORDER:
        print(f" {cat[:10]:<12}", end="")
    print()
    print("-" * (8 + 12 * len(CATEGORY_ORDER)))
    for size in sizes_done:
        m = all_metrics[size]
        print(f"{size:<8}", end="")
        for cat in CATEGORY_ORDER:
            val = m["cat_means"].get(cat, 0)
            print(f" {val:<12.4f}", end="")
        print()

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING SCALING PLOTS")
    print("=" * 65)
    fig_dir = SCALING_FIGURES_DIR

    if len(sizes_done) >= 2:
        # 1. Spearman rho scaling
        print("\n[1/4] Spearman ρ scaling...")
        plot_scaling_law(
            sizes_done,
            {
                "Spearman ρ (confidence)": [all_metrics[s]["spearman_rho"] for s in sizes_done],
                "Spearman ρ (entropy)": [all_metrics[s]["spearman_rho_entropy"] for s in sizes_done],
            },
            ylabel="Spearman ρ",
            title="Knowledge Level Correlation vs Model Size",
            save_path=fig_dir / "a2_spearman_scaling.png",
            hline=0.0, hline_label="No correlation",
        )

        # 2. Classifier accuracy scaling
        print("[2/4] Classifier accuracy scaling...")
        plot_scaling_law(
            sizes_done,
            {
                "2-way accuracy": [all_metrics[s]["acc_2way"] for s in sizes_done],
                "4-way accuracy": [all_metrics[s]["acc_4way"] for s in sizes_done],
            },
            ylabel="Cross-validated Accuracy",
            title="Knowledge Level Classification vs Model Size",
            save_path=fig_dir / "a2_classifier_scaling.png",
            hline=0.5, hline_label="Chance (2-way)",
        )

        # 3. Per-category confidence heatmap
        print("[3/4] Per-category confidence heatmap...")
        heatmap_data = []
        for size in sizes_done:
            row = [all_metrics[size]["cat_means"].get(cat, 0) for cat in CATEGORY_ORDER]
            heatmap_data.append(row)
        heatmap_data = np.array(heatmap_data)
        plot_scaling_heatmap(
            heatmap_data,
            x_labels=CATEGORY_ORDER,
            y_labels=[model_display_name(s) for s in sizes_done],
            title="Mean Confidence by Knowledge Level & Model Size",
            save_path=fig_dir / "a2_category_heatmap.png",
        )

        # 4. P-value cascade
        print("[4/4] P-value cascade...")
        plot_pvalue_cascade(
            sizes_done,
            {
                "Spearman": [all_metrics[s]["spearman_p"] for s in sizes_done],
                "Mann-Whitney": [all_metrics[s]["mann_whitney_p"] for s in sizes_done],
            },
            title="Knowledge Level Detection: Significance vs Size",
            save_path=fig_dir / "a2_pvalue_cascade.png",
        )

    # ===================================================================
    # Final
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(fig_dir.glob("a2_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT A2 COMPLETE")
    print("=" * 65)
    print(f"  Models: {len(sizes_done)}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s")

    if sizes_done:
        first_4way = all_metrics[sizes_done[0]]["acc_4way"]
        last_4way = all_metrics[sizes_done[-1]]["acc_4way"]
        print(f"\n  4-way accuracy: {first_4way:.1%} ({sizes_done[0]}) → "
              f"{last_4way:.1%} ({sizes_done[-1]})")
        if last_4way >= 0.60:
            print("  → FINE-GRAINED KNOWLEDGE ENCODING EMERGES ✓")
        elif last_4way > first_4way + 0.1:
            print("  → IMPROVING with scale, may reach viability at larger sizes")
        else:
            print("  → Knowledge level classification not improving with scale")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_experiment(args.models, force=args.force)
