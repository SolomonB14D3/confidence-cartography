"""
Experiment 2: Truth vs Falsehood
=================================
Can we detect a confidence fingerprint that distinguishes true from false claims?

Method: Create paired statements — identical structure, one true, one false.
Measure confidence on the critical differing tokens and surrounding context.
Scale up to ~40 pairs across diverse domains for statistical power.

Analysis:
  - Per-pair: confidence delta at the critical (differing) token
  - Aggregate: can a simple threshold classifier separate true from false?
  - ROC/AUC analysis
  - Effect sizes with confidence intervals
  - Where exactly does confidence drop in false statements?
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text
from src.schema import save_records, ConfidenceRecord
from src.viz import (
    plot_confidence_landscape,
    plot_confidence_heatmap,
    plot_false_statement_zoom,
)
from src.utils import RESULTS_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Paired true/false statements
# Each pair: identical structure, one true, one false.
# "critical_pos_hint" helps us find the divergence point (approximate).
# ---------------------------------------------------------------------------

PAIRS = [
    # --- Geography ---
    {"true": "The capital of France is Paris.",
     "false": "The capital of France is Berlin.",
     "domain": "geography", "id": "france_capital"},
    {"true": "The capital of Japan is Tokyo.",
     "false": "The capital of Japan is Beijing.",
     "domain": "geography", "id": "japan_capital"},
    {"true": "The capital of Australia is Canberra.",
     "false": "The capital of Australia is Sydney.",
     "domain": "geography", "id": "australia_capital"},
    {"true": "The largest ocean is the Pacific Ocean.",
     "false": "The largest ocean is the Atlantic Ocean.",
     "domain": "geography", "id": "largest_ocean"},
    {"true": "The longest river in the world is the Nile.",
     "false": "The longest river in the world is the Thames.",
     "domain": "geography", "id": "longest_river"},
    {"true": "Mount Everest is the tallest mountain on Earth.",
     "false": "Mount Kilimanjaro is the tallest mountain on Earth.",
     "domain": "geography", "id": "tallest_mountain"},

    # --- Science ---
    {"true": "Water boils at 100 degrees Celsius at sea level.",
     "false": "Water boils at 50 degrees Celsius at sea level.",
     "domain": "science", "id": "water_boiling"},
    {"true": "The Earth orbits the Sun.",
     "false": "The Sun orbits the Earth.",
     "domain": "science", "id": "earth_orbit"},
    {"true": "Light travels faster than sound.",
     "false": "Sound travels faster than light.",
     "domain": "science", "id": "light_vs_sound"},
    {"true": "Diamonds are made of carbon.",
     "false": "Diamonds are made of silicon.",
     "domain": "science", "id": "diamond_composition"},
    {"true": "The chemical symbol for gold is Au.",
     "false": "The chemical symbol for gold is Ag.",
     "domain": "science", "id": "gold_symbol"},
    {"true": "Humans have 23 pairs of chromosomes.",
     "false": "Humans have 30 pairs of chromosomes.",
     "domain": "science", "id": "chromosomes"},
    {"true": "The speed of light is approximately 300,000 kilometers per second.",
     "false": "The speed of light is approximately 300,000 miles per second.",
     "domain": "science", "id": "speed_of_light"},
    {"true": "Oxygen is the most abundant element in the Earth's crust.",
     "false": "Iron is the most abundant element in the Earth's crust.",
     "domain": "science", "id": "abundant_element"},

    # --- History ---
    {"true": "World War II ended in 1945.",
     "false": "World War II ended in 1952.",
     "domain": "history", "id": "ww2_end"},
    {"true": "The Berlin Wall fell in 1989.",
     "false": "The Berlin Wall fell in 1975.",
     "domain": "history", "id": "berlin_wall"},
    {"true": "Shakespeare wrote Hamlet.",
     "false": "Shakespeare wrote The Odyssey.",
     "domain": "history", "id": "shakespeare"},
    {"true": "The first moon landing was in 1969.",
     "false": "The first moon landing was in 1959.",
     "domain": "history", "id": "moon_landing"},
    {"true": "The Roman Empire fell in 476 AD.",
     "false": "The Roman Empire fell in 276 AD.",
     "domain": "history", "id": "roman_fall"},
    {"true": "The Declaration of Independence was signed in 1776.",
     "false": "The Declaration of Independence was signed in 1676.",
     "domain": "history", "id": "declaration"},

    # --- Biology ---
    {"true": "Humans have two lungs.",
     "false": "Humans have three lungs.",
     "domain": "biology", "id": "human_lungs"},
    {"true": "The heart has four chambers.",
     "false": "The heart has six chambers.",
     "domain": "biology", "id": "heart_chambers"},
    {"true": "Dolphins are mammals.",
     "false": "Dolphins are fish.",
     "domain": "biology", "id": "dolphins"},
    {"true": "Spiders have eight legs.",
     "false": "Spiders have six legs.",
     "domain": "biology", "id": "spider_legs"},
    {"true": "Plants produce oxygen through photosynthesis.",
     "false": "Plants produce nitrogen through photosynthesis.",
     "domain": "biology", "id": "photosynthesis"},
    {"true": "The largest organ in the human body is the skin.",
     "false": "The largest organ in the human body is the liver.",
     "domain": "biology", "id": "largest_organ"},

    # --- Math/Logic ---
    {"true": "The square root of 144 is 12.",
     "false": "The square root of 144 is 14.",
     "domain": "math", "id": "sqrt_144"},
    {"true": "A triangle has three sides.",
     "false": "A triangle has four sides.",
     "domain": "math", "id": "triangle_sides"},
    {"true": "Pi is approximately 3.14159.",
     "false": "Pi is approximately 4.14159.",
     "domain": "math", "id": "pi_value"},
    {"true": "There are 360 degrees in a circle.",
     "false": "There are 400 degrees in a circle.",
     "domain": "math", "id": "circle_degrees"},

    # --- Culture/Common Knowledge ---
    {"true": "The Mona Lisa was painted by Leonardo da Vinci.",
     "false": "The Mona Lisa was painted by Michelangelo.",
     "domain": "culture", "id": "mona_lisa"},
    {"true": "The Great Wall of China is in China.",
     "false": "The Great Wall of China is in Japan.",
     "domain": "culture", "id": "great_wall"},
    {"true": "Coffee contains caffeine.",
     "false": "Coffee contains nicotine.",
     "domain": "culture", "id": "coffee"},
    {"true": "The Olympic Games are held every four years.",
     "false": "The Olympic Games are held every three years.",
     "domain": "culture", "id": "olympics"},
    {"true": "The currency of the United States is the dollar.",
     "false": "The currency of the United States is the pound.",
     "domain": "culture", "id": "us_currency"},
    {"true": "The Statue of Liberty is in New York.",
     "false": "The Statue of Liberty is in Chicago.",
     "domain": "culture", "id": "statue_liberty"},

    # --- Astronomy ---
    {"true": "Jupiter is the largest planet in our solar system.",
     "false": "Mars is the largest planet in our solar system.",
     "domain": "astronomy", "id": "largest_planet"},
    {"true": "The Moon orbits the Earth.",
     "false": "The Moon orbits Mars.",
     "domain": "astronomy", "id": "moon_orbit"},
    {"true": "There are eight planets in our solar system.",
     "false": "There are twelve planets in our solar system.",
     "domain": "astronomy", "id": "num_planets"},
    {"true": "The Sun is a star.",
     "false": "The Sun is a planet.",
     "domain": "astronomy", "id": "sun_type"},
]


@dataclass
class PairResult:
    """Analysis of one true/false pair."""
    pair_id: str
    domain: str
    true_text: str
    false_text: str
    true_record: ConfidenceRecord
    false_record: ConfidenceRecord
    # Aggregate deltas
    delta_mean_prob: float       # true - false mean P
    delta_mean_entropy: float    # true - false mean entropy
    # Critical token analysis
    false_min_prob: float        # lowest P in false statement
    false_min_token: str
    false_min_rank: int          # rank of the false token at its position
    true_prob_at_same_pos: float # P in true statement at same position


def find_divergence_point(true_rec: ConfidenceRecord,
                          false_rec: ConfidenceRecord) -> int:
    """Find the first token position where true and false statements differ."""
    min_len = min(len(true_rec.tokens), len(false_rec.tokens))
    for i in range(min_len):
        if true_rec.tokens[i].token_id != false_rec.tokens[i].token_id:
            return i
    return min_len - 1


def analyze_pair(pair: dict) -> PairResult:
    """Analyze one true/false pair."""
    true_rec = analyze_fixed_text(
        pair["true"], category="true", label=f"{pair['id']}_true")
    false_rec = analyze_fixed_text(
        pair["false"], category="false", label=f"{pair['id']}_false")

    # Find where the false statement has minimum confidence
    false_min_pos = false_rec.min_confidence_pos
    false_min_tok = false_rec.tokens[false_min_pos]

    # Get true statement's confidence at the same position (if it exists)
    true_at_pos = 0.0
    if false_min_pos < len(true_rec.tokens):
        true_at_pos = true_rec.tokens[false_min_pos].top1_prob

    return PairResult(
        pair_id=pair["id"],
        domain=pair["domain"],
        true_text=pair["true"],
        false_text=pair["false"],
        true_record=true_rec,
        false_record=false_rec,
        delta_mean_prob=true_rec.mean_top1_prob - false_rec.mean_top1_prob,
        delta_mean_entropy=true_rec.mean_entropy - false_rec.mean_entropy,
        false_min_prob=false_rec.min_confidence_value,
        false_min_token=false_rec.min_confidence_token,
        false_min_rank=false_min_tok.top1_rank,
        true_prob_at_same_pos=true_at_pos,
    )


# ---------------------------------------------------------------------------
# Classifier: can we separate true from false using confidence features?
# ---------------------------------------------------------------------------

def build_classifier_features(results: list[PairResult]) -> tuple:
    """Extract features for ROC analysis.

    For each statement (true and false), compute:
      - mean_top1_prob
      - mean_entropy
      - min_confidence
      - std_top1_prob
    Then measure how well these separate true from false.
    """
    features = []   # (mean_prob, mean_entropy, min_prob, std_prob)
    labels = []     # 1 = true, 0 = false

    for r in results:
        # True statement features
        features.append([
            r.true_record.mean_top1_prob,
            r.true_record.mean_entropy,
            r.true_record.min_confidence_value,
            r.true_record.std_top1_prob,
        ])
        labels.append(1)

        # False statement features
        features.append([
            r.false_record.mean_top1_prob,
            r.false_record.mean_entropy,
            r.false_record.min_confidence_value,
            r.false_record.std_top1_prob,
        ])
        labels.append(0)

    return np.array(features), np.array(labels)


def compute_roc(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """Compute ROC curve and AUC for a single score dimension."""
    # Sort by score descending (higher score → predict true)
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    tp = 0
    fp = 0
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    tpr_list = [0.0]
    fpr_list = [0.0]

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # AUC via trapezoidal rule
    auc = np.trapezoid(tpr_list, fpr_list)
    return np.array(fpr_list), np.array(tpr_list), auc


# ---------------------------------------------------------------------------
# Visualization helpers specific to Exp 2
# ---------------------------------------------------------------------------

def plot_paired_deltas(results: list[PairResult], save_name: str):
    """Bar chart of delta_mean_prob for each pair, colored by domain."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort by delta
    sorted_results = sorted(results, key=lambda r: r.delta_mean_prob, reverse=True)
    ids = [r.pair_id for r in sorted_results]
    deltas = [r.delta_mean_prob for r in sorted_results]
    domains = [r.domain for r in sorted_results]

    # Color by domain
    domain_colors = {
        "geography": "#2196F3", "science": "#4CAF50",
        "history": "#FF9800", "biology": "#9C27B0",
        "math": "#F44336", "culture": "#00BCD4",
        "astronomy": "#795548",
    }
    colors = [domain_colors.get(d, "#999999") for d in domains]

    bars = ax.bar(range(len(ids)), deltas, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Delta Mean P(actual): True - False")
    ax.set_title("Confidence Gap: True vs False Statements")

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=d) for d, c in domain_colors.items()
                      if d in set(domains)]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


def plot_true_false_distributions(results: list[PairResult], save_name: str):
    """Violin/box plot comparing P(actual) distributions for true vs false."""
    true_probs = [r.true_record.mean_top1_prob for r in results]
    false_probs = [r.false_record.mean_top1_prob for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Mean P(actual) distributions
    ax = axes[0]
    data = [true_probs, false_probs]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
    parts["bodies"][0].set_facecolor("#4CAF50")
    parts["bodies"][1].set_facecolor("#F44336")
    parts["bodies"][0].set_alpha(0.7)
    parts["bodies"][1].set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["True", "False"])
    ax.set_ylabel("Mean P(actual token)")
    ax.set_title("Distribution of Mean Confidence")

    # Overlay individual points
    ax.scatter(np.zeros(len(true_probs)) + np.random.normal(0, 0.02, len(true_probs)),
               true_probs, color="#2E7D32", alpha=0.5, s=20, zorder=5)
    ax.scatter(np.ones(len(false_probs)) + np.random.normal(0, 0.02, len(false_probs)),
               false_probs, color="#B71C1C", alpha=0.5, s=20, zorder=5)

    # Right: Mean entropy distributions
    ax = axes[1]
    true_ents = [r.true_record.mean_entropy for r in results]
    false_ents = [r.false_record.mean_entropy for r in results]
    data = [true_ents, false_ents]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
    parts["bodies"][0].set_facecolor("#4CAF50")
    parts["bodies"][1].set_facecolor("#F44336")
    parts["bodies"][0].set_alpha(0.7)
    parts["bodies"][1].set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["True", "False"])
    ax.set_ylabel("Mean Entropy (bits)")
    ax.set_title("Distribution of Mean Entropy")

    ax.scatter(np.zeros(len(true_ents)) + np.random.normal(0, 0.02, len(true_ents)),
               true_ents, color="#2E7D32", alpha=0.5, s=20, zorder=5)
    ax.scatter(np.ones(len(false_ents)) + np.random.normal(0, 0.02, len(false_ents)),
               false_ents, color="#B71C1C", alpha=0.5, s=20, zorder=5)

    fig.suptitle(f"True vs False: {len(results)} Pairs", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, axes


def plot_roc_curves(features: np.ndarray, labels: np.ndarray, save_name: str):
    """ROC curves for each feature dimension as a truth classifier."""
    feature_names = ["Mean P(actual)", "Mean Entropy (inv)", "Min P(actual)", "Std P(actual)"]

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    for i, (name, color) in enumerate(zip(feature_names, colors)):
        scores = features[:, i]
        # For entropy, higher = less likely true, so invert
        if "Entropy" in name:
            scores = -scores
        fpr, tpr, auc = compute_roc(scores, labels)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={auc:.3f})")

    # Combined score: mean_prob - 0.5*mean_entropy (simple weighted)
    combined = features[:, 0] - 0.1 * features[:, 1] + features[:, 2]
    fpr, tpr, auc = compute_roc(combined, labels)
    ax.plot(fpr, tpr, color="black", linewidth=2.5, linestyle="--",
            label=f"Combined (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Can Confidence Predict Truth?")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


def plot_divergence_analysis(results: list[PairResult], save_name: str):
    """Scatter: P(true token) vs P(false token) at the divergence point."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for r in results:
        div_pos = find_divergence_point(r.true_record, r.false_record)
        if div_pos < len(r.true_record.tokens) and div_pos < len(r.false_record.tokens):
            true_p = r.true_record.tokens[div_pos].top1_prob
            false_p = r.false_record.tokens[div_pos].top1_prob
            ax.scatter(true_p, false_p, s=40, alpha=0.7, zorder=5)
            ax.annotate(r.pair_id, (true_p, false_p),
                        fontsize=5, alpha=0.6, xytext=(3, 3),
                        textcoords="offset points")

    # Diagonal line (equal confidence)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Equal confidence")
    ax.set_xlabel("P(actual token) — TRUE statement")
    ax.set_ylabel("P(actual token) — FALSE statement")
    ax.set_title("Confidence at Divergence Point: True vs False")
    ax.legend()
    ax.set_xlim(-0.02, max(0.5, ax.get_xlim()[1]))
    ax.set_ylim(-0.02, max(0.5, ax.get_ylim()[1]))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment():
    output_path = RESULTS_DIR / "exp2_truth.jsonl"

    print("=" * 65)
    print("EXPERIMENT 2: Truth vs Falsehood")
    print("=" * 65)
    print(f"Analyzing {len(PAIRS)} true/false pairs across "
          f"{len(set(p['domain'] for p in PAIRS))} domains...")
    print()

    start_time = time.time()
    results = []

    for pair in tqdm(PAIRS, desc="Analyzing pairs"):
        result = analyze_pair(pair)
        results.append(result)

    elapsed = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed:.1f}s")

    # Save all records
    if output_path.exists():
        output_path.unlink()
    all_records = []
    for r in results:
        all_records.extend([r.true_record, r.false_record])
    save_records(all_records, output_path)
    print(f"Results saved to {output_path}")

    # ===================================================================
    # Statistical Summary
    # ===================================================================
    print("\n" + "=" * 65)
    print("STATISTICAL SUMMARY")
    print("=" * 65)

    delta_probs = [r.delta_mean_prob for r in results]
    delta_ents = [r.delta_mean_entropy for r in results]

    print(f"\nPairs analyzed: {len(results)}")
    print(f"\nDelta Mean P (true - false):")
    print(f"  Mean:   {np.mean(delta_probs):+.4f}")
    print(f"  Median: {np.median(delta_probs):+.4f}")
    print(f"  Std:    {np.std(delta_probs):.4f}")
    print(f"  Range:  [{min(delta_probs):+.4f}, {max(delta_probs):+.4f}]")

    # Paired t-test: is delta_prob significantly > 0?
    t_stat, p_val = stats.ttest_1samp(delta_probs, 0)
    cohens_d = np.mean(delta_probs) / np.std(delta_probs) if np.std(delta_probs) > 0 else 0
    print(f"\n  Paired t-test (H0: delta=0):")
    print(f"    t={t_stat:.3f}, p={p_val:.6f}, Cohen's d={cohens_d:.3f}")
    print(f"    {'*** SIGNIFICANT ***' if p_val < 0.05 else 'Not significant'}")

    # Win rate: how often does true have higher mean P?
    wins = sum(1 for d in delta_probs if d > 0)
    print(f"\n  Win rate (true > false): {wins}/{len(results)} "
          f"({wins/len(results):.1%})")

    # Sign test
    sign_stat = stats.binomtest(wins, len(results), 0.5)
    print(f"  Sign test p-value: {sign_stat.pvalue:.6f}")

    # ===================================================================
    # Per-domain breakdown
    # ===================================================================
    print("\n" + "=" * 65)
    print("PER-DOMAIN BREAKDOWN")
    print("=" * 65)

    by_domain = defaultdict(list)
    for r in results:
        by_domain[r.domain].append(r)

    print(f"\n{'Domain':<14} {'N':<4} {'Mean dP':<10} {'Win%':<8} "
          f"{'Mean True P':<12} {'Mean False P':<12}")
    print("-" * 60)
    for domain in sorted(by_domain.keys()):
        dr = by_domain[domain]
        dp = [r.delta_mean_prob for r in dr]
        w = sum(1 for d in dp if d > 0)
        mt = np.mean([r.true_record.mean_top1_prob for r in dr])
        mf = np.mean([r.false_record.mean_top1_prob for r in dr])
        print(f"{domain:<14} {len(dr):<4} {np.mean(dp):+<10.4f} "
              f"{w/len(dr):<8.0%} {mt:<12.4f} {mf:<12.4f}")

    # ===================================================================
    # Per-pair details
    # ===================================================================
    print("\n" + "=" * 65)
    print("PER-PAIR DETAILS (sorted by delta)")
    print("=" * 65)

    sorted_results = sorted(results, key=lambda r: r.delta_mean_prob, reverse=True)
    print(f"\n{'Pair ID':<20} {'Domain':<12} {'True P':<9} {'False P':<9} "
          f"{'Delta':<9} {'False Min Token':<16} {'Min P':<10}")
    print("-" * 85)
    for r in sorted_results:
        print(f"{r.pair_id:<20} {r.domain:<12} "
              f"{r.true_record.mean_top1_prob:<9.4f} "
              f"{r.false_record.mean_top1_prob:<9.4f} "
              f"{r.delta_mean_prob:>+9.4f} "
              f"{repr(r.false_min_token):<16} "
              f"{r.false_min_prob:<10.6f}")

    # ===================================================================
    # Divergence Point Analysis
    # ===================================================================
    print("\n" + "=" * 65)
    print("DIVERGENCE POINT ANALYSIS")
    print("=" * 65)
    print("Where the true/false statements first differ:")

    print(f"\n{'Pair ID':<20} {'Div Pos':<8} {'True Token':<14} "
          f"{'P(true)':<10} {'False Token':<14} {'P(false)':<10} {'Ratio':<8}")
    print("-" * 84)

    div_ratios = []
    for r in sorted_results:
        div_pos = find_divergence_point(r.true_record, r.false_record)
        if div_pos < len(r.true_record.tokens) and div_pos < len(r.false_record.tokens):
            tt = r.true_record.tokens[div_pos]
            ft = r.false_record.tokens[div_pos]
            ratio = tt.top1_prob / (ft.top1_prob + 1e-10)
            div_ratios.append(ratio)
            print(f"{r.pair_id:<20} {div_pos:<8} "
                  f"{repr(tt.token_str):<14} {tt.top1_prob:<10.6f} "
                  f"{repr(ft.token_str):<14} {ft.top1_prob:<10.6f} "
                  f"{ratio:<8.1f}x")

    if div_ratios:
        print(f"\n  Median confidence ratio (true/false) at divergence: "
              f"{np.median(div_ratios):.1f}x")
        print(f"  Mean: {np.mean(div_ratios):.1f}x, "
              f"Range: [{min(div_ratios):.1f}x, {max(div_ratios):.1f}x]")

    # ===================================================================
    # Classifier Analysis
    # ===================================================================
    print("\n" + "=" * 65)
    print("CLASSIFIER ANALYSIS: Can confidence predict truth?")
    print("=" * 65)

    features, labels = build_classifier_features(results)

    feature_names = ["Mean P(actual)", "Mean Entropy", "Min P(actual)", "Std P(actual)"]
    for i, name in enumerate(feature_names):
        scores = features[:, i]
        if "Entropy" in name:
            scores = -scores
        _, _, auc = compute_roc(scores, labels)
        print(f"  {name:<20} AUC = {auc:.3f}")

    combined = features[:, 0] - 0.1 * features[:, 1] + features[:, 2]
    _, _, auc = compute_roc(combined, labels)
    print(f"  {'Combined':<20} AUC = {auc:.3f}")

    # Simple threshold classifier on mean_prob
    best_acc = 0
    best_thresh = 0
    for thresh in np.arange(0, 0.3, 0.005):
        preds = (features[:, 0] > thresh).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    print(f"\n  Best threshold classifier (mean P > {best_thresh:.3f}): "
          f"accuracy = {best_acc:.1%}")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)

    print("\n[1/5] Paired delta bar chart...")
    plot_paired_deltas(results, "exp2_paired_deltas.png")
    plt.close("all")

    print("[2/5] True vs False distributions...")
    plot_true_false_distributions(results, "exp2_distributions.png")
    plt.close("all")

    print("[3/5] ROC curves...")
    plot_roc_curves(features, labels, "exp2_roc.png")
    plt.close("all")

    print("[4/5] Divergence point scatter...")
    plot_divergence_analysis(results, "exp2_divergence_scatter.png")
    plt.close("all")

    print("[5/5] Selected false vs true overlays...")
    # Pick 5 most interesting pairs (biggest deltas)
    for r in sorted_results[:5]:
        plot_false_statement_zoom(
            r.false_record, r.true_record,
            save_name=f"exp2_overlay_{r.pair_id}.png")
        plt.close("all")

    # ===================================================================
    # Summary
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(FIGURES_DIR.glob("exp2_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 65)
    print(f"  Pairs analyzed: {len(results)}")
    print(f"  Win rate (true > false confidence): {wins}/{len(results)} "
          f"({wins/len(results):.1%})")
    print(f"  Mean delta P: {np.mean(delta_probs):+.4f} "
          f"(p={p_val:.6f}, d={cohens_d:.3f})")
    print(f"  Best single-feature AUC: "
          f"{max(compute_roc(features[:, i] if i != 1 else -features[:, i], labels)[2] for i in range(4)):.3f}")
    print(f"  Results: {output_path}")
    print(f"  Figures: {fig_count} plots saved")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    run_experiment()
