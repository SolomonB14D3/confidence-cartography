"""
Experiment 3: Settled vs Contested Knowledge
=============================================
Can confidence patterns distinguish topics where human knowledge is settled
from topics where it's contested?

Spectrum: Settled → Mostly Settled → Actively Contested → Unknown/Speculative

Key question: Does the entropy distribution widen for contested topics?
Can we build a classifier from confidence features alone?
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text
from src.schema import save_records
from src.viz import plot_confidence_landscape, plot_confidence_heatmap
from src.utils import RESULTS_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Prompts across the settled → contested spectrum
# ---------------------------------------------------------------------------

PROMPTS = [
    # --- SETTLED: Universally agreed facts ---
    {"text": "The speed of light in a vacuum is constant.",
     "category": "settled", "label": "lightspeed_const"},
    {"text": "Water is composed of hydrogen and oxygen atoms.",
     "category": "settled", "label": "water_h2o"},
    {"text": "The Earth revolves around the Sun once per year.",
     "category": "settled", "label": "earth_orbit_year"},
    {"text": "DNA carries genetic information in living organisms.",
     "category": "settled", "label": "dna_genetic"},
    {"text": "Gravity causes objects to fall toward the Earth.",
     "category": "settled", "label": "gravity_fall"},
    {"text": "The boiling point of water at sea level is 100 degrees Celsius.",
     "category": "settled", "label": "boiling_point"},
    {"text": "Antibiotics are effective against bacterial infections.",
     "category": "settled", "label": "antibiotics"},
    {"text": "The human genome contains approximately 20,000 genes.",
     "category": "settled", "label": "genome_genes"},

    # --- MOSTLY SETTLED: Scientific consensus with minor debate ---
    {"text": "Human activity is the primary cause of recent climate change.",
     "category": "mostly_settled", "label": "climate_human"},
    {"text": "Evolution by natural selection explains the diversity of life.",
     "category": "mostly_settled", "label": "evolution"},
    {"text": "The universe began with the Big Bang approximately 13.8 billion years ago.",
     "category": "mostly_settled", "label": "big_bang"},
    {"text": "Vaccines are safe and effective for preventing infectious diseases.",
     "category": "mostly_settled", "label": "vaccines"},
    {"text": "Smoking tobacco significantly increases the risk of lung cancer.",
     "category": "mostly_settled", "label": "smoking_cancer"},
    {"text": "Plate tectonics explains the movement of continents over time.",
     "category": "mostly_settled", "label": "plate_tectonics"},
    {"text": "Regular exercise has significant benefits for mental health.",
     "category": "mostly_settled", "label": "exercise_mental"},
    {"text": "The extinction of the dinosaurs was caused by an asteroid impact.",
     "category": "mostly_settled", "label": "dinosaur_asteroid"},

    # --- ACTIVELY CONTESTED: Genuine scientific or policy debate ---
    {"text": "The most effective approach to reducing poverty is direct cash transfers.",
     "category": "contested", "label": "poverty_cash"},
    {"text": "Artificial general intelligence will be achieved within the next fifty years.",
     "category": "contested", "label": "agi_timeline"},
    {"text": "Nuclear energy should be the primary solution to climate change.",
     "category": "contested", "label": "nuclear_climate"},
    {"text": "Free will is an illusion created by deterministic brain processes.",
     "category": "contested", "label": "free_will"},
    {"text": "Universal basic income would improve societal outcomes overall.",
     "category": "contested", "label": "ubi"},
    {"text": "Consciousness arises from quantum processes in the brain.",
     "category": "contested", "label": "quantum_consciousness"},
    {"text": "Social media has been a net negative for democratic societies.",
     "category": "contested", "label": "social_media_democracy"},
    {"text": "Genetic engineering of human embryos should be permitted for disease prevention.",
     "category": "contested", "label": "gene_editing"},

    # --- UNKNOWN/SPECULATIVE: Frontier questions with no consensus ---
    {"text": "The solution to the dark matter problem involves undiscovered particles.",
     "category": "unknown", "label": "dark_matter"},
    {"text": "Life exists on other planets in our galaxy.",
     "category": "unknown", "label": "alien_life"},
    {"text": "The fundamental nature of consciousness can be explained by physics.",
     "category": "unknown", "label": "consciousness_physics"},
    {"text": "There are additional spatial dimensions beyond the three we observe.",
     "category": "unknown", "label": "extra_dimensions"},
    {"text": "The universe will eventually end in a heat death.",
     "category": "unknown", "label": "heat_death"},
    {"text": "Quantum computers will eventually break all current encryption methods.",
     "category": "unknown", "label": "quantum_encryption"},
    {"text": "Human civilization will establish permanent colonies on Mars.",
     "category": "unknown", "label": "mars_colonies"},
    {"text": "A unified theory of physics will be discovered in this century.",
     "category": "unknown", "label": "unified_theory"},
]

CATEGORY_ORDER = ["settled", "mostly_settled", "contested", "unknown"]
CATEGORY_LABELS = {
    "settled": "Settled Science",
    "mostly_settled": "Mostly Settled",
    "contested": "Actively Contested",
    "unknown": "Unknown/Speculative",
}


def plot_entropy_by_category(records_by_cat, save_name):
    """Box plot of per-token entropy distributions across categories."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    for cat in CATEGORY_ORDER:
        for rec in records_by_cat[cat]:
            for ta in rec.tokens:
                data.append(ta.entropy)
                labels.append(CATEGORY_LABELS[cat])

    import pandas as pd
    df = pd.DataFrame({"Entropy (bits)": data, "Category": labels})

    cat_order = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER]
    palette = {"Settled Science": "#4CAF50", "Mostly Settled": "#8BC34A",
               "Actively Contested": "#FF9800", "Unknown/Speculative": "#F44336"}

    sns.boxplot(data=df, x="Category", y="Entropy (bits)", order=cat_order,
                palette=palette, ax=ax, showfliers=False)
    sns.stripplot(data=df, x="Category", y="Entropy (bits)", order=cat_order,
                  palette=palette, ax=ax, alpha=0.1, size=2, jitter=True)

    ax.set_title("Per-Token Entropy Distribution by Knowledge Category")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


def plot_confidence_spectrum(records_by_cat, save_name):
    """Scatter plot: mean P vs mean entropy, colored by category."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"settled": "#4CAF50", "mostly_settled": "#8BC34A",
              "contested": "#FF9800", "unknown": "#F44336"}

    for cat in CATEGORY_ORDER:
        recs = records_by_cat[cat]
        x = [r.mean_top1_prob for r in recs]
        y = [r.mean_entropy for r in recs]
        ax.scatter(x, y, c=colors[cat], s=80, alpha=0.7,
                   label=CATEGORY_LABELS[cat], edgecolors="white", linewidth=0.5)
        for r in recs:
            ax.annotate(r.label.split("_")[0], (r.mean_top1_prob, r.mean_entropy),
                        fontsize=5, alpha=0.5, xytext=(3, 3),
                        textcoords="offset points")

    ax.set_xlabel("Mean P(actual token)")
    ax.set_ylabel("Mean Entropy (bits)")
    ax.set_title("Knowledge Spectrum: Confidence vs Entropy")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


def plot_confidence_variance(records_by_cat, save_name):
    """Compare within-sentence confidence variance across categories."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"settled": "#4CAF50", "mostly_settled": "#8BC34A",
              "contested": "#FF9800", "unknown": "#F44336"}

    # Left: std of top1_prob within each sentence
    ax = axes[0]
    for i, cat in enumerate(CATEGORY_ORDER):
        vals = [r.std_top1_prob for r in records_by_cat[cat]]
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   c=colors[cat], s=50, alpha=0.7)
    ax.boxplot([[r.std_top1_prob for r in records_by_cat[cat]]
                for cat in CATEGORY_ORDER],
               positions=range(4), widths=0.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CATEGORY_LABELS[c][:12] for c in CATEGORY_ORDER],
                        fontsize=8)
    ax.set_ylabel("Std of P(actual) within sentence")
    ax.set_title("Confidence Variance")

    # Right: std of entropy within each sentence
    ax = axes[1]
    for i, cat in enumerate(CATEGORY_ORDER):
        vals = [r.std_entropy for r in records_by_cat[cat]]
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   c=colors[cat], s=50, alpha=0.7)
    ax.boxplot([[r.std_entropy for r in records_by_cat[cat]]
                for cat in CATEGORY_ORDER],
               positions=range(4), widths=0.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CATEGORY_LABELS[c][:12] for c in CATEGORY_ORDER],
                        fontsize=8)
    ax.set_ylabel("Std of Entropy within sentence")
    ax.set_title("Entropy Variance")

    fig.suptitle("Within-Sentence Variability by Knowledge Category", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, axes


def run_experiment():
    output_path = RESULTS_DIR / "exp3_contested.jsonl"

    print("=" * 65)
    print("EXPERIMENT 3: Settled vs Contested Knowledge")
    print("=" * 65)
    print(f"Analyzing {len(PROMPTS)} prompts across 4 knowledge levels...")
    print()

    start_time = time.time()
    records = []

    for p in tqdm(PROMPTS, desc="Analyzing"):
        rec = analyze_fixed_text(p["text"], p["category"], p["label"])
        records.append(rec)

    elapsed = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed:.1f}s")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)

    # Group by category
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.category].append(r)

    # ===================================================================
    # Summary Statistics
    # ===================================================================
    print("\n" + "=" * 65)
    print("SUMMARY BY KNOWLEDGE LEVEL")
    print("=" * 65)

    print(f"\n{'Category':<20} {'N':<4} {'Mean P':<10} {'Mean Ent':<10} "
          f"{'Std P(intra)':<14} {'Std Ent(intra)':<14}")
    print("-" * 72)
    for cat in CATEGORY_ORDER:
        recs = by_cat[cat]
        mp = np.mean([r.mean_top1_prob for r in recs])
        me = np.mean([r.mean_entropy for r in recs])
        sp = np.mean([r.std_top1_prob for r in recs])
        se = np.mean([r.std_entropy for r in recs])
        print(f"{CATEGORY_LABELS[cat]:<20} {len(recs):<4} {mp:<10.4f} "
              f"{me:<10.2f} {sp:<14.4f} {se:<14.2f}")

    # ===================================================================
    # Statistical Tests: Trend across the spectrum
    # ===================================================================
    print("\n" + "=" * 65)
    print("STATISTICAL TESTS")
    print("=" * 65)

    # Assign numeric levels: settled=0, mostly_settled=1, contested=2, unknown=3
    level_map = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    levels = [level_map[r.category] for r in records]
    mean_probs = [r.mean_top1_prob for r in records]
    mean_ents = [r.mean_entropy for r in records]

    # Spearman correlation: does confidence decrease as knowledge becomes less settled?
    rho_p, p_p = stats.spearmanr(levels, mean_probs)
    rho_e, p_e = stats.spearmanr(levels, mean_ents)
    print(f"\n  Spearman correlation (level vs mean P):")
    print(f"    rho={rho_p:.3f}, p={p_p:.4f} "
          f"{'***' if p_p < 0.05 else 'n.s.'}")
    print(f"  Spearman correlation (level vs mean entropy):")
    print(f"    rho={rho_e:.3f}, p={p_e:.4f} "
          f"{'***' if p_e < 0.05 else 'n.s.'}")

    # Kruskal-Wallis: are the distributions different across categories?
    groups_p = [np.array([r.mean_top1_prob for r in by_cat[c]])
                for c in CATEGORY_ORDER]
    groups_e = [np.array([r.mean_entropy for r in by_cat[c]])
                for c in CATEGORY_ORDER]

    h_p, p_kw_p = stats.kruskal(*groups_p)
    h_e, p_kw_e = stats.kruskal(*groups_e)
    print(f"\n  Kruskal-Wallis (mean P across categories):")
    print(f"    H={h_p:.3f}, p={p_kw_p:.4f} "
          f"{'***' if p_kw_p < 0.05 else 'n.s.'}")
    print(f"  Kruskal-Wallis (mean entropy across categories):")
    print(f"    H={h_e:.3f}, p={p_kw_e:.4f} "
          f"{'***' if p_kw_e < 0.05 else 'n.s.'}")

    # Pairwise: settled vs contested
    t_sc, p_sc = stats.mannwhitneyu(
        [r.mean_top1_prob for r in by_cat["settled"]],
        [r.mean_top1_prob for r in by_cat["contested"]],
        alternative="greater")
    print(f"\n  Mann-Whitney (settled > contested, mean P):")
    print(f"    U={t_sc:.1f}, p={p_sc:.4f} "
          f"{'***' if p_sc < 0.05 else 'n.s.'}")

    # ===================================================================
    # Classifier: can we predict the category from confidence features?
    # ===================================================================
    print("\n" + "=" * 65)
    print("CLASSIFIER: Settled vs Non-Settled")
    print("=" * 65)

    X = np.array([[r.mean_top1_prob, r.mean_entropy, r.std_top1_prob,
                    r.std_entropy, r.min_confidence_value] for r in records])
    # Binary: settled (0,1) vs contested (2,3)
    y_binary = np.array([0 if level_map[r.category] <= 1 else 1
                         for r in records])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(clf, X_scaled, y_binary, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {scores.mean():.1%} +/- {scores.std():.1%}")
    print(f"  Baseline (majority class): {max(y_binary.mean(), 1-y_binary.mean()):.1%}")

    # Also try 4-way classification
    y_multi = np.array([level_map[r.category] for r in records])
    scores_multi = cross_val_score(clf, X_scaled, y_multi, cv=4, scoring="accuracy")
    print(f"\n  4-way classification (4-fold CV): {scores_multi.mean():.1%} +/- {scores_multi.std():.1%}")
    print(f"  Baseline (random): 25.0%")

    # ===================================================================
    # Per-prompt details
    # ===================================================================
    print("\n" + "=" * 65)
    print("PER-PROMPT DETAILS")
    print("=" * 65)

    print(f"\n{'Label':<24} {'Category':<16} {'Mean P':<8} {'Mean Ent':<9} "
          f"{'Min Token':<14} {'Min P':<10}")
    print("-" * 81)
    for cat in CATEGORY_ORDER:
        for r in by_cat[cat]:
            print(f"{r.label:<24} {cat:<16} {r.mean_top1_prob:<8.4f} "
                  f"{r.mean_entropy:<9.2f} {repr(r.min_confidence_token):<14} "
                  f"{r.min_confidence_value:<10.6f}")
        print()

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)

    print("\n[1/5] Confidence spectrum scatter...")
    plot_confidence_spectrum(by_cat, "exp3_confidence_spectrum.png")
    plt.close("all")

    print("[2/5] Entropy box plots by category...")
    plot_entropy_by_category(by_cat, "exp3_entropy_boxplot.png")
    plt.close("all")

    print("[3/5] Within-sentence variance...")
    plot_confidence_variance(by_cat, "exp3_variance.png")
    plt.close("all")

    print("[4/5] Per-category heatmaps...")
    for cat in CATEGORY_ORDER:
        plot_confidence_heatmap(
            by_cat[cat], title=f"Exp 3: {CATEGORY_LABELS[cat]}",
            save_name=f"exp3_heatmap_{cat}.png")
        plt.close("all")

    print("[5/5] Selected landscapes...")
    for cat in CATEGORY_ORDER:
        rec = by_cat[cat][0]
        plot_confidence_landscape(rec, save_name=f"exp3_landscape_{rec.label}.png")
        plt.close("all")

    # ===================================================================
    # Summary
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(FIGURES_DIR.glob("exp3_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 65)
    print(f"  Prompts analyzed: {len(records)}")
    print(f"  Spearman rho (level vs P): {rho_p:.3f} (p={p_p:.4f})")
    print(f"  Binary classifier: {scores.mean():.1%}")
    print(f"  4-way classifier: {scores_multi.mean():.1%}")
    print(f"  Results: {output_path}")
    print(f"  Figures: {fig_count} plots saved")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    run_experiment()
