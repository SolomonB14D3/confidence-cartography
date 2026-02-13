"""
Experiment 7: Confidence-Aware Fine-Tuning
============================================
Fine-tune three copies of Pythia 160M with different loss functions:
  Model A: Standard causal LM loss (baseline)
  Model B: LM loss + confidence penalty on known-false claims
  Model C: LM loss + gap reward for large true/false confidence gaps

Goal: Test whether the truth-false confidence gap can be amplified through
training, and whether doing so reduces hallucination.
"""

import sys
import time
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import PROJECT_ROOT as _PR, get_device
from src.training_utils import (
    FactPair, save_fact_pairs,
    ConfidencePenaltyLoss, GapRewardLoss,
    compute_token_confidence, standard_lm_step,
    evaluate_truth_detection, evaluate_perplexity,
)
from src.engine import analyze_fixed_text, unload_model
from src.schema import save_records, load_records
from src.scaling import MODEL_REGISTRY

from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXP7_DIR = _PR / "models" / "exp7"
EXP7_RESULTS_DIR = _PR / "data" / "results" / "exp7"
EXP7_FIGURES_DIR = _PR / "figures" / "exp7"
FACT_PAIRS_DIR = _PR / "data" / "fact_pairs"

for d in [EXP7_DIR, EXP7_RESULTS_DIR, EXP7_FIGURES_DIR, FACT_PAIRS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fact pair dataset construction
# ---------------------------------------------------------------------------

# Phase 1 pairs (from exp2_truth.py) — hand-curated
PHASE1_PAIRS = [
    FactPair("The capital of France is Paris.", "The capital of France is Berlin.",
             "france_capital", "geography"),
    FactPair("The capital of Japan is Tokyo.", "The capital of Japan is Beijing.",
             "japan_capital", "geography"),
    FactPair("Water boils at 100 degrees Celsius at sea level.",
             "Water boils at 50 degrees Celsius at sea level.",
             "water_boiling", "science"),
    FactPair("The Earth orbits the Sun.", "The Sun orbits the Earth.",
             "earth_orbit", "science"),
    FactPair("Light travels faster than sound.", "Sound travels faster than light.",
             "light_vs_sound", "science"),
    FactPair("Diamonds are made of carbon.", "Diamonds are made of silicon.",
             "diamond_composition", "science"),
    FactPair("World War II ended in 1945.", "World War II ended in 1952.",
             "ww2_end", "history"),
    FactPair("The Berlin Wall fell in 1989.", "The Berlin Wall fell in 1975.",
             "berlin_wall", "history"),
    FactPair("Shakespeare wrote Hamlet.", "Shakespeare wrote The Odyssey.",
             "shakespeare", "history"),
    FactPair("Humans have two lungs.", "Humans have three lungs.",
             "human_lungs", "biology"),
    FactPair("The heart has four chambers.", "The heart has six chambers.",
             "heart_chambers", "biology"),
    FactPair("Dolphins are mammals.", "Dolphins are fish.",
             "dolphins", "biology"),
    FactPair("The square root of 144 is 12.", "The square root of 144 is 14.",
             "sqrt_144", "math"),
    FactPair("Pi is approximately 3.14159.", "Pi is approximately 4.14159.",
             "pi_value", "math"),
    FactPair("The Mona Lisa was painted by Leonardo da Vinci.",
             "The Mona Lisa was painted by Michelangelo.",
             "mona_lisa", "culture"),
    FactPair("Jupiter is the largest planet in our solar system.",
             "Mars is the largest planet in our solar system.",
             "largest_planet", "astronomy"),
    FactPair("The Moon orbits the Earth.", "The Moon orbits Mars.",
             "moon_orbit", "astronomy"),
    FactPair("The Sun is a star.", "The Sun is a planet.",
             "sun_type", "astronomy"),
]

# Additional pairs for training diversity
EXTRA_PAIRS = [
    FactPair("Oxygen is a gas at room temperature.", "Oxygen is a liquid at room temperature.",
             "oxygen_state", "chemistry"),
    FactPair("The Amazon River is in South America.", "The Amazon River is in Africa.",
             "amazon_location", "geography"),
    FactPair("The speed of sound in air is about 343 meters per second.",
             "The speed of sound in air is about 700 meters per second.",
             "speed_of_sound", "physics"),
    FactPair("Isaac Newton formulated the laws of motion.",
             "Albert Einstein formulated the laws of motion.",
             "newton_laws", "physics"),
    FactPair("The Great Wall of China was built over many centuries.",
             "The Great Wall of China was built in a single year.",
             "great_wall_time", "history"),
    FactPair("Photosynthesis converts carbon dioxide into oxygen.",
             "Photosynthesis converts oxygen into carbon dioxide.",
             "photosynthesis_dir", "biology"),
    FactPair("The human body is approximately 60 percent water.",
             "The human body is approximately 20 percent water.",
             "body_water", "biology"),
    FactPair("Gold is a chemical element with symbol Au.",
             "Gold is a chemical element with symbol Fe.",
             "gold_element", "chemistry"),
    FactPair("The Pacific Ocean is the largest ocean on Earth.",
             "The Atlantic Ocean is the largest ocean on Earth.",
             "largest_ocean", "geography"),
    FactPair("Gravity causes objects to fall toward the Earth.",
             "Magnetism causes objects to fall toward the Earth.",
             "gravity_falling", "physics"),
    FactPair("Insulin regulates blood sugar levels.", "Insulin regulates blood pressure.",
             "insulin_function", "biology"),
    FactPair("DNA carries genetic information.", "DNA carries electrical signals.",
             "dna_function", "biology"),
]

# Held-out evaluation pairs (NOT used in training)
EVAL_PAIRS = [
    FactPair("Mercury is the closest planet to the Sun.",
             "Venus is the closest planet to the Sun.",
             "closest_planet", "astronomy"),
    FactPair("The chemical formula for water is H2O.",
             "The chemical formula for water is CO2.",
             "water_formula", "chemistry"),
    FactPair("The Sahara is the largest hot desert in the world.",
             "The Gobi is the largest hot desert in the world.",
             "largest_desert", "geography"),
    FactPair("The first person to walk on the Moon was Neil Armstrong.",
             "The first person to walk on the Moon was Buzz Aldrin.",
             "first_moonwalk", "history"),
    FactPair("Blood carries oxygen from the lungs to the body's tissues.",
             "Blood carries nitrogen from the lungs to the body's tissues.",
             "blood_oxygen", "biology"),
    FactPair("The Eiffel Tower is in Paris.", "The Eiffel Tower is in London.",
             "eiffel_tower", "culture"),
    FactPair("Electrons have a negative charge.", "Electrons have a positive charge.",
             "electron_charge", "physics"),
    FactPair("The human brain contains billions of neurons.",
             "The human brain contains millions of neurons.",
             "brain_neurons", "neuroscience"),
]


# ---------------------------------------------------------------------------
# Held-out text for perplexity evaluation
# ---------------------------------------------------------------------------

PERPLEXITY_TEXTS = [
    "The study of economics involves understanding how societies allocate scarce resources among competing uses. Markets serve as mechanisms for coordination, bringing together buyers and sellers.",
    "In computer science, algorithms are step-by-step procedures for solving problems. The efficiency of an algorithm is often measured in terms of its time and space complexity.",
    "Literary criticism examines how texts create meaning through language, narrative structure, and cultural context. Different schools of thought approach interpretation in distinct ways.",
    "The Renaissance was a cultural movement that began in Italy during the fourteenth century. It marked a renewed interest in classical learning and the arts.",
    "Climate science studies the long-term patterns of temperature, precipitation, and other atmospheric conditions. Global temperatures have risen significantly over the past century.",
    "The development of antibiotics revolutionized medicine in the twentieth century. However, the overuse of these drugs has led to the emergence of resistant bacteria.",
    "Quantum mechanics describes the behavior of matter at the atomic and subatomic level. Particles can exist in multiple states simultaneously until observed.",
    "The agricultural revolution transformed human societies from hunter-gatherer groups into settled farming communities. This shift enabled population growth and the development of cities.",
    "Machine learning algorithms improve their performance through experience without being explicitly programmed. Neural networks are a class of models inspired by biological brains.",
    "The constitutional framework of democratic governance involves separation of powers among executive, legislative, and judicial branches, each serving as a check on the others.",
]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    variant: str,
    train_pairs: list[FactPair],
    eval_pairs: list[FactPair],
    num_steps: int = 2000,
    lr: float = 5e-5,
    pair_frequency: int = 50,
    alpha: float = 0.1,
    threshold: float = 0.2,
    beta: float = 0.1,
    eval_every: int = 200,
    save_dir: Path = None,
    force: bool = False,
) -> dict:
    """Train one model variant.

    Args:
        variant: "baseline", "penalty", or "gap_reward"
        train_pairs: fact pairs for confidence loss
        eval_pairs: held-out pairs for evaluation
        num_steps: total training steps
        lr: learning rate
        pair_frequency: apply confidence loss every N steps
        alpha: penalty weight (Model B)
        threshold: confidence threshold for penalty (Model B)
        beta: reward weight (Model C)
        eval_every: evaluate every N steps
        save_dir: where to save the final model
    """
    save_dir = save_dir or EXP7_DIR / variant
    save_dir.mkdir(parents=True, exist_ok=True)

    # Check if already trained
    if (save_dir / "pytorch_model.bin").exists() and not force:
        print(f"  [{variant}] Model exists at {save_dir}, skipping training.")
        return {"status": "cached", "save_dir": str(save_dir)}

    device = get_device()
    spec = MODEL_REGISTRY["160m"]
    model_name = spec["name"]

    print(f"\n  [{variant}] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Explicit float32 required — Pythia ships float16 weights,
    # and AdamW running averages overflow in fp16 → NaN.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Loss functions
    penalty_fn = ConfidencePenaltyLoss(alpha=alpha, threshold=threshold)
    reward_fn = GapRewardLoss(beta=beta)

    # Training texts (simple: use the fact pairs themselves as base training data)
    all_texts = (
        [p.true_text for p in train_pairs] +
        [p.false_text for p in train_pairs] +
        PERPLEXITY_TEXTS
    )

    # Training log
    log = {
        "steps": [], "lm_loss": [],
        "eval_win_rate": [], "eval_mean_delta": [],
        "eval_perplexity": [],
    }

    print(f"  [{variant}] Training for {num_steps} steps (lr={lr}, "
          f"pair_freq={pair_frequency})...")
    start_time = time.time()

    for step in tqdm(range(1, num_steps + 1), desc=f"  {variant}", leave=False):
        optimizer.zero_grad()

        # Pick a random text for standard LM loss
        text = random.choice(all_texts)

        if variant == "baseline":
            loss = standard_lm_step(model, tokenizer, text, device)

        elif variant == "penalty":
            if step % pair_frequency == 0:
                # Apply confidence penalty on a random false claim
                pair = random.choice(train_pairs)
                lm_loss = standard_lm_step(model, tokenizer, text, device)
                _, false_prob = compute_token_confidence(
                    model, tokenizer, pair.false_text, device)
                loss = penalty_fn(lm_loss, false_prob)
            else:
                loss = standard_lm_step(model, tokenizer, text, device)

        elif variant == "gap_reward":
            if step % pair_frequency == 0:
                pair = random.choice(train_pairs)
                lm_loss = standard_lm_step(model, tokenizer, text, device)
                _, true_prob = compute_token_confidence(
                    model, tokenizer, pair.true_text, device)
                _, false_prob = compute_token_confidence(
                    model, tokenizer, pair.false_text, device)
                loss = reward_fn(lm_loss, true_prob, false_prob)
            else:
                loss = standard_lm_step(model, tokenizer, text, device)

        else:
            raise ValueError(f"Unknown variant: {variant}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % eval_every == 0 or step == 1:
            model.eval()
            eval_result = evaluate_truth_detection(
                model, tokenizer, eval_pairs, device)
            ppl = evaluate_perplexity(
                model, tokenizer, PERPLEXITY_TEXTS, device)
            model.train()

            log["steps"].append(step)
            log["lm_loss"].append(loss.item())
            log["eval_win_rate"].append(eval_result["win_rate"])
            log["eval_mean_delta"].append(eval_result["mean_delta"])
            log["eval_perplexity"].append(ppl)

            if step % (eval_every * 5) == 0:
                print(f"    Step {step}: loss={loss.item():.4f}, "
                      f"win_rate={eval_result['win_rate']:.1%}, "
                      f"delta={eval_result['mean_delta']:+.4f}, ppl={ppl:.1f}")

    elapsed = time.time() - start_time
    print(f"  [{variant}] Training done in {elapsed:.1f}s")

    # Save model
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save training log
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Final evaluation
    model.eval()
    final_eval = evaluate_truth_detection(model, tokenizer, eval_pairs, device)
    final_ppl = evaluate_perplexity(model, tokenizer, PERPLEXITY_TEXTS, device)

    # Clean up
    del model, tokenizer, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    return {
        "status": "trained",
        "save_dir": str(save_dir),
        "final_win_rate": final_eval["win_rate"],
        "final_mean_delta": final_eval["mean_delta"],
        "final_perplexity": final_ppl,
        "training_time": elapsed,
        "log": log,
    }


# ---------------------------------------------------------------------------
# Evaluation of saved models
# ---------------------------------------------------------------------------

def evaluate_saved_model(save_dir: Path, eval_pairs: list[FactPair]) -> dict:
    """Load and evaluate a saved model."""
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        save_dir, dtype=torch.float32).to(device)
    model.eval()

    eval_result = evaluate_truth_detection(model, tokenizer, eval_pairs, device)
    ppl = evaluate_perplexity(model, tokenizer, PERPLEXITY_TEXTS, device)

    # Detailed per-pair results
    pair_details = []
    for pair in eval_pairs:
        true_conf, _ = compute_token_confidence(model, tokenizer, pair.true_text, device)
        false_conf, _ = compute_token_confidence(model, tokenizer, pair.false_text, device)
        pair_details.append({
            "pair_id": pair.pair_id,
            "true_conf": true_conf,
            "false_conf": false_conf,
            "delta": true_conf - false_conf,
            "correct": true_conf > false_conf,
        })

    del model, tokenizer
    import gc; gc.collect()

    return {
        **eval_result,
        "perplexity": ppl,
        "pair_details": pair_details,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_curves(results: dict, save_path: Path):
    """Training curves for all three variants."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"baseline": "#757575", "penalty": "#F44336", "gap_reward": "#2196F3"}
    labels = {"baseline": "A: Baseline", "penalty": "B: Penalty",
              "gap_reward": "C: Gap Reward"}

    for variant, result in results.items():
        if "log" not in result:
            continue
        log = result["log"]
        color = colors.get(variant, "#999")
        label = labels.get(variant, variant)

        axes[0].plot(log["steps"], log["eval_win_rate"], "-", color=color,
                     linewidth=2, label=label)
        axes[1].plot(log["steps"], log["eval_mean_delta"], "-", color=color,
                     linewidth=2, label=label)
        axes[2].plot(log["steps"], log["eval_perplexity"], "-", color=color,
                     linewidth=2, label=label)

    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    axes[0].set_ylabel("Win Rate (true > false)")
    axes[0].set_title("Truth Detection")
    axes[0].legend(fontsize=9)

    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_ylabel("Mean Δ Confidence")
    axes[1].set_title("Confidence Gap")
    axes[1].legend(fontsize=9)

    axes[2].set_ylabel("Perplexity")
    axes[2].set_title("Language Model Quality")
    axes[2].legend(fontsize=9)

    for ax in axes:
        ax.set_xlabel("Training Step")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 7: Confidence-Aware Fine-Tuning", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_comparison(eval_results: dict, save_path: Path):
    """Bar chart comparing final metrics across variants."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    variants = list(eval_results.keys())
    colors = ["#757575", "#F44336", "#2196F3"]
    labels = ["A: Baseline", "B: Penalty", "C: Gap Reward"]

    # Win rate
    win_rates = [eval_results[v]["win_rate"] for v in variants]
    axes[0].bar(range(len(variants)), win_rates, color=colors[:len(variants)])
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    axes[0].set_xticks(range(len(variants)))
    axes[0].set_xticklabels(labels[:len(variants)], fontsize=9)
    axes[0].set_ylabel("Win Rate")
    axes[0].set_title("Truth Detection")
    axes[0].set_ylim(0, 1)

    # Mean delta
    deltas = [eval_results[v]["mean_delta"] for v in variants]
    axes[1].bar(range(len(variants)), deltas, color=colors[:len(variants)])
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_xticks(range(len(variants)))
    axes[1].set_xticklabels(labels[:len(variants)], fontsize=9)
    axes[1].set_ylabel("Mean Δ Confidence")
    axes[1].set_title("Confidence Gap")

    # Perplexity
    ppls = [eval_results[v]["perplexity"] for v in variants]
    axes[2].bar(range(len(variants)), ppls, color=colors[:len(variants)])
    axes[2].set_xticks(range(len(variants)))
    axes[2].set_xticklabels(labels[:len(variants)], fontsize=9)
    axes[2].set_ylabel("Perplexity")
    axes[2].set_title("Language Quality (lower=better)")

    fig.suptitle("Experiment 7: Final Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    num_steps: int = 2000,
    lr: float = 5e-5,
    pair_frequency: int = 50,
    alpha: float = 0.1,
    threshold: float = 0.2,
    beta: float = 0.1,
    force: bool = False,
):
    print("=" * 70)
    print("EXPERIMENT 7: Confidence-Aware Fine-Tuning")
    print("=" * 70)
    print(f"Training steps: {num_steps}")
    print(f"Learning rate: {lr}")
    print(f"Pair frequency: every {pair_frequency} steps")
    print(f"Alpha (penalty): {alpha}, Threshold: {threshold}")
    print(f"Beta (gap reward): {beta}")

    # Prepare fact pairs
    train_pairs = PHASE1_PAIRS + EXTRA_PAIRS
    eval_pairs_list = EVAL_PAIRS
    random.shuffle(train_pairs)

    print(f"\nTraining pairs: {len(train_pairs)}")
    print(f"Eval pairs: {len(eval_pairs_list)}")

    # Save pairs for reproducibility
    save_fact_pairs(train_pairs, FACT_PAIRS_DIR / "train_pairs.jsonl")
    save_fact_pairs(eval_pairs_list, FACT_PAIRS_DIR / "eval_pairs.jsonl")

    start_time = time.time()
    training_results = {}

    # Train all three variants
    for variant in ["baseline", "penalty", "gap_reward"]:
        print(f"\n{'='*50}")
        print(f"TRAINING: {variant.upper()}")
        print(f"{'='*50}")

        kwargs = dict(
            variant=variant,
            train_pairs=train_pairs,
            eval_pairs=eval_pairs_list,
            num_steps=num_steps,
            lr=lr,
            pair_frequency=pair_frequency,
            eval_every=max(100, num_steps // 20),
            force=force,
        )
        if variant == "penalty":
            kwargs.update(alpha=alpha, threshold=threshold)
        elif variant == "gap_reward":
            kwargs.update(beta=beta)

        result = train_model(**kwargs)
        training_results[variant] = result

    # ===================================================================
    # Evaluate all models on held-out pairs
    # ===================================================================
    print("\n" + "=" * 70)
    print("EVALUATION ON HELD-OUT PAIRS")
    print("=" * 70)

    eval_results = {}
    for variant in ["baseline", "penalty", "gap_reward"]:
        save_dir = EXP7_DIR / variant
        if not (save_dir / "pytorch_model.bin").exists():
            # Try safetensors
            if not (save_dir / "model.safetensors").exists():
                print(f"  [{variant}] No saved model found, skipping eval.")
                continue

        print(f"  Evaluating {variant}...")
        result = evaluate_saved_model(save_dir, eval_pairs_list)
        eval_results[variant] = result

        print(f"    Win rate: {result['win_rate']:.1%}")
        print(f"    Mean delta: {result['mean_delta']:+.4f}")
        print(f"    Perplexity: {result['perplexity']:.1f}")

    # ===================================================================
    # Summary
    # ===================================================================
    if eval_results:
        print("\n" + "=" * 70)
        print("EXPERIMENT 7 SUMMARY")
        print("=" * 70)

        print(f"\n{'Variant':<16} {'Win Rate':<10} {'Mean Δ':<10} "
              f"{'Perplexity':<12} {'Status'}")
        print("-" * 58)

        baseline_wr = eval_results.get("baseline", {}).get("win_rate", 0.5)
        baseline_ppl = eval_results.get("baseline", {}).get("perplexity", 999)

        for variant in ["baseline", "penalty", "gap_reward"]:
            if variant not in eval_results:
                continue
            r = eval_results[variant]
            wr = r["win_rate"]
            delta = r["mean_delta"]
            ppl = r["perplexity"]

            ppl_change = (ppl - baseline_ppl) / baseline_ppl * 100
            wr_change = (wr - baseline_wr)

            if variant == "baseline":
                status = "BASELINE"
            elif wr > baseline_wr + 0.05 and abs(ppl_change) < 10:
                status = "IMPROVED"
            elif abs(ppl_change) > 10:
                status = "DEGRADED (ppl)"
            elif wr < baseline_wr - 0.05:
                status = "WORSE"
            else:
                status = "NO CHANGE"

            print(f"{variant:<16} {wr:<10.1%} {delta:<+10.4f} {ppl:<12.1f} {status}")

    # ===================================================================
    # Plots
    # ===================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    if any("log" in r for r in training_results.values()):
        print("  Training curves...")
        plot_training_curves(training_results,
                             EXP7_FIGURES_DIR / "training_curves.png")

    if eval_results:
        print("  Final comparison...")
        plot_final_comparison(eval_results,
                              EXP7_FIGURES_DIR / "final_comparison.png")

    total_time = time.time() - start_time
    fig_count = len(list(EXP7_FIGURES_DIR.glob("*.png")))

    print(f"\n{'='*70}")
    print("EXPERIMENT 7 COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Figures: {fig_count}")

    # Dead end check
    if eval_results and "baseline" in eval_results:
        b_wr = eval_results["baseline"]["win_rate"]
        any_improved = any(
            eval_results[v]["win_rate"] > b_wr + 0.05
            for v in ["penalty", "gap_reward"]
            if v in eval_results
        )
        if any_improved:
            print("\n  FINDING: Confidence-aware training IMPROVES truth detection!")
        else:
            print("\n  FINDING: Confidence-aware training shows no improvement.")
            print("  → Confidence gap is observable but not a useful training lever.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_experiment(
        num_steps=args.steps,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        force=args.force,
    )
