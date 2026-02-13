"""
Experiment 5: Confidence as Consensus Detector
===============================================
Test the hypothesis that model confidence reflects the degree of human
consensus in the training data.

Method: Assign consensus scores to topics (manually, based on known
polling data and public debate), then correlate with model confidence.

If this works, we've built a consensus sensor — a tool that maps how
settled or contested any given claim is in collective human knowledge.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text
from src.schema import save_records
from src.utils import RESULTS_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Statements with estimated human consensus scores (0-100)
# 100 = universal agreement, 0 = maximally split
# Sources: scientific surveys, polling data, common knowledge
# ---------------------------------------------------------------------------

STATEMENTS = [
    # Near-universal consensus (90-100)
    {"text": "The Earth is round.",
     "consensus": 97, "label": "earth_round", "domain": "science"},
    {"text": "Smoking causes cancer.",
     "consensus": 95, "label": "smoking_cancer", "domain": "health"},
    {"text": "Humans need water to survive.",
     "consensus": 99, "label": "water_survive", "domain": "biology"},
    {"text": "The Holocaust happened during World War II.",
     "consensus": 98, "label": "holocaust", "domain": "history"},
    {"text": "Antibiotics can treat bacterial infections.",
     "consensus": 95, "label": "antibiotics", "domain": "health"},

    # Strong consensus (75-89)
    {"text": "Climate change is primarily caused by human activity.",
     "consensus": 85, "label": "climate_human", "domain": "science"},
    {"text": "Evolution explains the diversity of life on Earth.",
     "consensus": 80, "label": "evolution", "domain": "science"},
    {"text": "Vaccines are generally safe for most people.",
     "consensus": 82, "label": "vaccines_safe", "domain": "health"},
    {"text": "Regular exercise improves physical and mental health.",
     "consensus": 88, "label": "exercise_health", "domain": "health"},
    {"text": "Education is important for economic success.",
     "consensus": 85, "label": "education_success", "domain": "social"},

    # Moderate consensus (55-74)
    {"text": "Organic food is healthier than conventional food.",
     "consensus": 55, "label": "organic_healthy", "domain": "health"},
    {"text": "Immigration generally benefits the economy.",
     "consensus": 60, "label": "immigration_economy", "domain": "economics"},
    {"text": "Social media has been mostly harmful to teenagers.",
     "consensus": 65, "label": "social_media_teens", "domain": "social"},
    {"text": "A college degree is worth the cost for most people.",
     "consensus": 58, "label": "college_worth", "domain": "social"},
    {"text": "Renewable energy can fully replace fossil fuels.",
     "consensus": 60, "label": "renewables_replace", "domain": "energy"},

    # Divided/contested (35-54)
    {"text": "Capitalism is the best economic system.",
     "consensus": 45, "label": "capitalism_best", "domain": "economics"},
    {"text": "Gun control reduces violent crime.",
     "consensus": 50, "label": "gun_control", "domain": "policy"},
    {"text": "The death penalty is an effective deterrent to crime.",
     "consensus": 40, "label": "death_penalty", "domain": "policy"},
    {"text": "Universal basic income would be good for society.",
     "consensus": 45, "label": "ubi_good", "domain": "economics"},
    {"text": "Nuclear energy is the best solution to climate change.",
     "consensus": 42, "label": "nuclear_best", "domain": "energy"},

    # Highly contested (15-34)
    {"text": "Artificial intelligence poses an existential risk to humanity.",
     "consensus": 30, "label": "ai_existential", "domain": "technology"},
    {"text": "Cryptocurrency will replace traditional currencies.",
     "consensus": 20, "label": "crypto_replace", "domain": "economics"},
    {"text": "Colonizing Mars should be a priority for humanity.",
     "consensus": 25, "label": "mars_priority", "domain": "technology"},
    {"text": "Free will is an illusion.",
     "consensus": 25, "label": "free_will_illusion", "domain": "philosophy"},
    {"text": "Consciousness can be fully explained by neuroscience.",
     "consensus": 30, "label": "consciousness_neuro", "domain": "philosophy"},
]


def plot_consensus_correlation(records, statements, save_name):
    """Scatter plot: consensus score vs model confidence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    consensus_scores = [s["consensus"] for s in statements]
    mean_probs = [r.mean_top1_prob for r in records]
    mean_ents = [r.mean_entropy for r in records]
    domains = [s["domain"] for s in statements]

    domain_colors = {
        "science": "#4CAF50", "health": "#2196F3", "biology": "#8BC34A",
        "history": "#FF9800", "social": "#9C27B0", "economics": "#F44336",
        "energy": "#795548", "policy": "#607D8B", "technology": "#00BCD4",
        "philosophy": "#E91E63",
    }

    # Left: consensus vs mean P
    ax = axes[0]
    for i, (c, p, d) in enumerate(zip(consensus_scores, mean_probs, domains)):
        ax.scatter(c, p, c=domain_colors.get(d, "#999"), s=60, alpha=0.7,
                   edgecolors="white", linewidth=0.5)
        ax.annotate(statements[i]["label"][:8], (c, p), fontsize=5,
                    alpha=0.5, xytext=(3, 3), textcoords="offset points")

    # Regression line
    slope, intercept, r_val, p_val, std_err = stats.linregress(
        consensus_scores, mean_probs)
    x_line = np.linspace(15, 100, 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.5,
            label=f"r={r_val:.3f}, p={p_val:.4f}")

    ax.set_xlabel("Human Consensus Score (0-100)")
    ax.set_ylabel("Mean P(actual token)")
    ax.set_title("Consensus vs Model Confidence")
    ax.legend(fontsize=9)

    # Right: consensus vs mean entropy
    ax = axes[1]
    for i, (c, e, d) in enumerate(zip(consensus_scores, mean_ents, domains)):
        ax.scatter(c, e, c=domain_colors.get(d, "#999"), s=60, alpha=0.7,
                   edgecolors="white", linewidth=0.5)
        ax.annotate(statements[i]["label"][:8], (c, e), fontsize=5,
                    alpha=0.5, xytext=(3, 3), textcoords="offset points")

    slope_e, intercept_e, r_val_e, p_val_e, _ = stats.linregress(
        consensus_scores, mean_ents)
    ax.plot(x_line, slope_e * x_line + intercept_e, "k--", alpha=0.5,
            label=f"r={r_val_e:.3f}, p={p_val_e:.4f}")

    ax.set_xlabel("Human Consensus Score (0-100)")
    ax.set_ylabel("Mean Entropy (bits)")
    ax.set_title("Consensus vs Model Entropy")
    ax.legend(fontsize=9)

    # Legend for domains
    from matplotlib.patches import Patch
    patches = [Patch(color=c, label=d) for d, c in domain_colors.items()
               if d in set(domains)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Does Model Confidence Reflect Human Consensus?", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, axes


def plot_consensus_bins(records, statements, save_name):
    """Group by consensus bins and show distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = [(90, 100, "Near-universal\n(90-100)"),
            (75, 89, "Strong\n(75-89)"),
            (55, 74, "Moderate\n(55-74)"),
            (35, 54, "Divided\n(35-54)"),
            (15, 34, "Contested\n(15-34)")]

    colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]

    positions = []
    data_groups = []
    xlabels = []

    for i, (lo, hi, label) in enumerate(bins):
        probs = [r.mean_top1_prob for r, s in zip(records, statements)
                 if lo <= s["consensus"] <= hi]
        data_groups.append(probs)
        positions.append(i)
        xlabels.append(label)

    bp = ax.boxplot(data_groups, positions=positions, widths=0.5, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, probs in enumerate(data_groups):
        jitter = np.random.normal(0, 0.05, len(probs))
        ax.scatter(np.full(len(probs), i) + jitter, probs,
                   c=colors[i], s=30, alpha=0.7, zorder=5,
                   edgecolors="white", linewidth=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Mean P(actual token)")
    ax.set_title("Model Confidence by Human Consensus Level")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


def run_experiment():
    output_path = RESULTS_DIR / "exp5_consensus.jsonl"

    print("=" * 65)
    print("EXPERIMENT 5: Confidence as Consensus Detector")
    print("=" * 65)
    print(f"Analyzing {len(STATEMENTS)} statements with consensus scores...")
    print()

    start_time = time.time()
    records = []

    for s in tqdm(STATEMENTS, desc="Analyzing"):
        rec = analyze_fixed_text(s["text"], category=s["domain"], label=s["label"])
        rec.metadata = {"consensus": s["consensus"], "domain": s["domain"]}
        records.append(rec)

    elapsed = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed:.1f}s")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)

    # ===================================================================
    # Correlation Analysis
    # ===================================================================
    print("\n" + "=" * 65)
    print("CORRELATION ANALYSIS")
    print("=" * 65)

    consensus = np.array([s["consensus"] for s in STATEMENTS])
    mean_probs = np.array([r.mean_top1_prob for r in records])
    mean_ents = np.array([r.mean_entropy for r in records])
    min_probs = np.array([r.min_confidence_value for r in records])

    # Pearson
    r_p, p_p = stats.pearsonr(consensus, mean_probs)
    r_e, p_e = stats.pearsonr(consensus, mean_ents)
    r_m, p_m = stats.pearsonr(consensus, min_probs)
    print(f"\n  Pearson correlations:")
    print(f"    Consensus vs Mean P:      r={r_p:+.3f}, p={p_p:.4f} "
          f"{'***' if p_p < 0.05 else 'n.s.'}")
    print(f"    Consensus vs Mean Entropy: r={r_e:+.3f}, p={p_e:.4f} "
          f"{'***' if p_e < 0.05 else 'n.s.'}")
    print(f"    Consensus vs Min P:       r={r_m:+.3f}, p={p_m:.4f} "
          f"{'***' if p_m < 0.05 else 'n.s.'}")

    # Spearman (rank-based, more robust)
    rs_p, ps_p = stats.spearmanr(consensus, mean_probs)
    rs_e, ps_e = stats.spearmanr(consensus, mean_ents)
    print(f"\n  Spearman correlations:")
    print(f"    Consensus vs Mean P:      rho={rs_p:+.3f}, p={ps_p:.4f} "
          f"{'***' if ps_p < 0.05 else 'n.s.'}")
    print(f"    Consensus vs Mean Entropy: rho={rs_e:+.3f}, p={ps_e:.4f} "
          f"{'***' if ps_e < 0.05 else 'n.s.'}")

    # ===================================================================
    # Per-statement details
    # ===================================================================
    print("\n" + "=" * 65)
    print("PER-STATEMENT DETAILS (sorted by consensus)")
    print("=" * 65)

    sorted_idx = np.argsort(consensus)[::-1]
    print(f"\n{'Label':<22} {'Cons':<6} {'Mean P':<9} {'Mean Ent':<9} "
          f"{'Domain':<12}")
    print("-" * 58)
    for i in sorted_idx:
        s = STATEMENTS[i]
        r = records[i]
        print(f"{s['label']:<22} {s['consensus']:<6} {r.mean_top1_prob:<9.4f} "
              f"{r.mean_entropy:<9.2f} {s['domain']:<12}")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)

    print("\n[1/2] Consensus correlation scatter...")
    plot_consensus_correlation(records, STATEMENTS, "exp5_consensus_correlation.png")
    plt.close("all")

    print("[2/2] Consensus bins boxplot...")
    plot_consensus_bins(records, STATEMENTS, "exp5_consensus_bins.png")
    plt.close("all")

    # ===================================================================
    # Summary
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(FIGURES_DIR.glob("exp5_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 65)
    print(f"  Statements analyzed: {len(records)}")
    print(f"  Pearson r (consensus vs P): {r_p:+.3f} (p={p_p:.4f})")
    print(f"  Spearman rho (consensus vs P): {rs_p:+.3f} (p={ps_p:.4f})")
    print(f"  Figures: {fig_count} plots saved")
    print(f"  Total time: {total_time:.1f}s")
    print(f"\n  Verdict: {'CONSENSUS SENSOR WORKS' if ps_p < 0.05 else 'NO CLEAR SIGNAL'}")


if __name__ == "__main__":
    run_experiment()
