"""
Experiment 1: Baseline Confidence Fingerprints
===============================================
Map what "normal" confidence looks like across 5 categories of knowledge.
Feed fixed text through Pythia 160M, measure the model's surprise at each token.

Categories:
  1. Simple facts       - highly predictable, well-known
  2. Complex facts      - specific numbers/names, less predictable
  3. Common opinions    - culturally common, moderately predictable
  4. Controversial      - contested claims, potentially higher uncertainty
  5. False statements   - factually wrong, expect confidence drop at false word

Run: python src/experiments/exp1_baselines.py
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text
from src.schema import save_records, load_records
from src.viz import (
    plot_confidence_landscape,
    plot_confidence_heatmap,
    plot_transition_detector,
    plot_distribution_snapshot,
    plot_category_comparison,
    plot_false_statement_zoom,
)
from src.utils import RESULTS_DIR, FIGURES_DIR


# ---------------------------------------------------------------------------
# Prompt definitions: 20 prompts, 5 categories, 4 each
# ---------------------------------------------------------------------------

PROMPTS = [
    # --- Simple Facts ---
    {"text": "The capital of France is Paris.",
     "category": "simple_fact", "label": "capital_france_true"},
    {"text": "Water freezes at zero degrees Celsius.",
     "category": "simple_fact", "label": "water_freezes"},
    {"text": "The Earth has one moon.",
     "category": "simple_fact", "label": "earth_moon"},
    {"text": "Humans have 206 bones in their body.",
     "category": "simple_fact", "label": "human_bones"},

    # --- Complex Facts ---
    {"text": "The half-life of carbon-14 is approximately 5,730 years.",
     "category": "complex_fact", "label": "carbon14_halflife"},
    {"text": "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
     "category": "complex_fact", "label": "speed_of_light"},
    {"text": "DNA was first identified by Friedrich Miescher in 1869.",
     "category": "complex_fact", "label": "dna_discovery"},
    {"text": "The Mariana Trench reaches a depth of about 36,000 feet.",
     "category": "complex_fact", "label": "mariana_trench"},

    # --- Common Opinions/Cultural ---
    {"text": "Pizza is one of the most popular foods in the world.",
     "category": "common_opinion", "label": "pizza_popular"},
    {"text": "Dogs are often considered to be loyal companions.",
     "category": "common_opinion", "label": "dogs_loyal"},
    {"text": "Music can have a powerful effect on human emotions.",
     "category": "common_opinion", "label": "music_emotions"},
    {"text": "Summer is a popular time for vacations.",
     "category": "common_opinion", "label": "summer_vacations"},

    # --- Controversial/Contested ---
    {"text": "Nuclear energy is the safest form of power generation.",
     "category": "controversial", "label": "nuclear_safest"},
    {"text": "The best economic system is free market capitalism.",
     "category": "controversial", "label": "free_market_best"},
    {"text": "Social media has been mostly harmful to society.",
     "category": "controversial", "label": "social_media_harmful"},
    {"text": "Artificial intelligence will eventually surpass human intelligence.",
     "category": "controversial", "label": "ai_surpass_human"},

    # --- False/Nonsense ---
    {"text": "The capital of France is banana.",
     "category": "false_statement", "label": "capital_france_false"},
    {"text": "Water boils at minus forty degrees.",
     "category": "false_statement", "label": "water_boils_neg40"},
    {"text": "The Sun orbits around the Earth.",
     "category": "false_statement", "label": "sun_orbits_earth"},
    {"text": "Humans have three hearts.",
     "category": "false_statement", "label": "humans_three_hearts"},
]

# Category display order
CATEGORY_ORDER = [
    "simple_fact", "complex_fact", "common_opinion",
    "controversial", "false_statement",
]


def run_experiment():
    """Run the full Experiment 1 pipeline."""
    output_path = RESULTS_DIR / "exp1_baselines.jsonl"

    # ===================================================================
    # Phase 1: Analyze all prompts
    # ===================================================================
    print("=" * 65)
    print("EXPERIMENT 1: Baseline Confidence Fingerprints")
    print("=" * 65)
    print(f"Analyzing {len(PROMPTS)} prompts across 5 categories...")
    print()

    records = []
    start_time = time.time()

    for prompt_info in tqdm(PROMPTS, desc="Analyzing prompts"):
        record = analyze_fixed_text(
            text=prompt_info["text"],
            category=prompt_info["category"],
            label=prompt_info["label"],
        )
        records.append(record)

    elapsed = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed:.1f}s")

    # Save results (overwrite for reproducibility)
    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    print(f"Results saved to {output_path}")

    # ===================================================================
    # Phase 2: Summary Statistics
    # ===================================================================
    print("\n" + "=" * 65)
    print("SUMMARY STATISTICS BY CATEGORY")
    print("=" * 65)

    by_category = defaultdict(list)
    for r in records:
        by_category[r.category].append(r)

    print(f"\n{'Category':<20} {'Mean P(actual)':<16} {'Mean Entropy':<14} "
          f"{'Std P':<10} {'Std Ent':<10}")
    print("-" * 70)
    for cat in CATEGORY_ORDER:
        recs = by_category[cat]
        mean_p = np.mean([r.mean_top1_prob for r in recs])
        mean_e = np.mean([r.mean_entropy for r in recs])
        std_p = np.mean([r.std_top1_prob for r in recs])
        std_e = np.mean([r.std_entropy for r in recs])
        print(f"{cat:<20} {mean_p:<16.4f} {mean_e:<14.2f} "
              f"{std_p:<10.4f} {std_e:<10.2f}")

    # ===================================================================
    # Phase 3: Per-prompt details
    # ===================================================================
    print(f"\n{'Label':<28} {'Cat':<18} {'Toks':<6} {'MeanP':<8} "
          f"{'MinP':<10} {'MinTok':<16} {'MeanEnt':<8}")
    print("-" * 95)
    for r in records:
        print(f"{r.label:<28} {r.category:<18} {r.num_tokens:<6} "
              f"{r.mean_top1_prob:<8.4f} {r.min_confidence_value:<10.4f} "
              f"{repr(r.min_confidence_token):<16} {r.mean_entropy:<8.2f}")

    # ===================================================================
    # Phase 4: Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)

    # 4a. Individual confidence landscapes
    print("\n[1/6] Individual confidence landscapes...")
    for r in tqdm(records, desc="  Landscapes"):
        plot_confidence_landscape(r, save_name=f"exp1_landscape_{r.label}.png")
        plt.close("all")

    # 4b. Full heatmap (all 20 prompts)
    print("[2/6] Full heatmap (all 20 prompts)...")
    plot_confidence_heatmap(
        records, title="Exp 1: All Prompts — Confidence Heatmap",
        save_name="exp1_heatmap_all.png")
    plt.close("all")

    # 4c. Per-category heatmaps
    print("[3/6] Per-category heatmaps...")
    for cat in CATEGORY_ORDER:
        recs = by_category[cat]
        plot_confidence_heatmap(
            recs, title=f"Exp 1: {cat}",
            save_name=f"exp1_heatmap_{cat}.png")
        plt.close("all")

    # 4d. Transition detection for false statements
    print("[4/6] Transition detection (false statements)...")
    for r in by_category["false_statement"]:
        plot_transition_detector(
            r, threshold=0.15,
            save_name=f"exp1_transitions_{r.label}.png")
        plt.close("all")

    # 4e. False vs True comparisons
    print("[5/6] False vs True comparisons...")
    false_true_pairs = [
        ("capital_france_false", "capital_france_true"),
    ]
    label_to_record = {r.label: r for r in records}

    for false_label, true_label in false_true_pairs:
        if false_label in label_to_record and true_label in label_to_record:
            plot_false_statement_zoom(
                label_to_record[false_label],
                label_to_record[true_label],
                save_name=f"exp1_false_vs_true_{false_label}.png",
            )
            plt.close("all")

    # Also plot each false statement solo
    for r in by_category["false_statement"]:
        plot_false_statement_zoom(
            r, save_name=f"exp1_false_zoom_{r.label}.png")
        plt.close("all")

    # 4f. Category comparison bar chart
    print("[6/6] Category comparison...")
    # Reorder for display
    ordered_by_cat = {cat: by_category[cat] for cat in CATEGORY_ORDER}
    plot_category_comparison(
        ordered_by_cat, save_name="exp1_category_comparison.png")
    plt.close("all")

    # ===================================================================
    # Phase 5: False Statement Deep Dive
    # ===================================================================
    print("\n" + "=" * 65)
    print("FALSE STATEMENT DEEP DIVE")
    print("=" * 65)

    for r in by_category["false_statement"]:
        print(f'\n--- {r.label}: "{r.text}" ---')
        print(f"  Overall mean P(actual): {r.mean_top1_prob:.4f}")
        print(f"  Min confidence at position {r.min_confidence_pos}: "
              f"token={repr(r.min_confidence_token)}, "
              f"P={r.min_confidence_value:.6f}")

        # What the model wanted to say at the low-confidence position
        ta = r.tokens[r.min_confidence_pos]
        print(f"  Model's top 5 predictions at that position:")
        for i, (tok, prob) in enumerate(zip(ta.top5_tokens, ta.top5_probs)):
            marker = " <-- actual" if tok == ta.token_str else ""
            print(f"    {i+1}. {repr(tok):>14}  P={prob:.4f}{marker}")

        # Distribution snapshot
        plot_distribution_snapshot(
            r, r.min_confidence_pos,
            save_name=f"exp1_snapshot_{r.label}_pos{r.min_confidence_pos}.png")
        plt.close("all")

    # ===================================================================
    # Phase 6: True vs False Direct Comparison
    # ===================================================================
    print("\n" + "=" * 65)
    print("TRUE vs FALSE: Side-by-Side")
    print("=" * 65)

    # Compare France/Paris vs France/banana token by token
    if "capital_france_true" in label_to_record and "capital_france_false" in label_to_record:
        true_r = label_to_record["capital_france_true"]
        false_r = label_to_record["capital_france_false"]

        min_len = min(len(true_r.tokens), len(false_r.tokens))
        print(f"\n  Position-by-position (first {min_len} positions):")
        print(f"  {'Pos':<5} {'True Token':<14} {'P(true)':<10} "
              f"{'False Token':<14} {'P(false)':<10} {'Delta':<10}")
        print("  " + "-" * 63)
        for i in range(min_len):
            tt = true_r.tokens[i]
            ft = false_r.tokens[i]
            delta = tt.top1_prob - ft.top1_prob
            print(f"  {i:<5} {repr(tt.token_str):<14} {tt.top1_prob:<10.4f} "
                  f"{repr(ft.token_str):<14} {ft.top1_prob:<10.4f} "
                  f"{delta:>+10.4f}")

    # ===================================================================
    # Summary
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(FIGURES_DIR.glob("exp1_*.png")))

    print("\n" + "=" * 65)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 65)
    print(f"  Results: {output_path}")
    print(f"  Figures: {fig_count} plots saved to {FIGURES_DIR}/")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    run_experiment()
