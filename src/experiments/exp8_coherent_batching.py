"""
Experiment 8: Concept-Coherent Batching
=========================================
Fine-tune two copies of Pythia 160M:
  Model D: Random batching (standard shuffled training)
  Model E: Concept-coherent batching (topical clusters, round-robin)

Goal: Test whether training on related concepts together produces
cleaner internal representations, as measured by confidence patterns.
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
    FactPair, standard_lm_step,
    evaluate_truth_detection, evaluate_perplexity,
    compute_token_confidence,
)
from src.scaling import MODEL_REGISTRY

from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXP8_DIR = _PR / "models" / "exp8"
EXP8_RESULTS_DIR = _PR / "data" / "results" / "exp8"
EXP8_FIGURES_DIR = _PR / "figures" / "exp8"

for d in [EXP8_DIR, EXP8_RESULTS_DIR, EXP8_FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Topic-organized training data
# ---------------------------------------------------------------------------

TOPIC_DATA = {
    "geography": [
        "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower. The country borders Germany, Spain, Italy, and Belgium.",
        "Japan is an island nation in East Asia. Its capital is Tokyo, one of the most populous cities in the world. Japan consists of four main islands.",
        "Brazil is the largest country in South America. Its capital is Brasilia, though São Paulo is the largest city. The Amazon rainforest covers much of the country.",
        "Australia is both a country and a continent. Its capital is Canberra, not Sydney as many people think. The Great Barrier Reef is located off its coast.",
        "Egypt is located in northeastern Africa. Its capital is Cairo, which sits on the Nile River. The Great Pyramids of Giza are among the oldest structures on Earth.",
        "Canada is the second largest country in the world by total area. Its capital is Ottawa, and it borders the United States to the south.",
        "India is a country in South Asia. Its capital is New Delhi, and it is the second most populous country in the world after China.",
        "Germany is a country in Central Europe. Its capital is Berlin, and it has the largest economy in Europe.",
    ],
    "biology": [
        "Cells are the basic building blocks of all living organisms. They contain DNA, which carries genetic instructions. Cells reproduce through division.",
        "Photosynthesis is the process by which plants convert sunlight into energy. This process takes place in chloroplasts and produces oxygen as a byproduct.",
        "The human heart is a muscular organ that pumps blood throughout the body. It has four chambers and beats approximately 100,000 times per day.",
        "Evolution is the process of change in living organisms over successive generations. Natural selection is the mechanism by which beneficial traits become more common.",
        "The immune system protects the body against disease. White blood cells identify and destroy pathogens including bacteria and viruses.",
        "DNA is a molecule that carries genetic information in all living organisms. It has a double helix structure and is composed of four nucleotide bases.",
        "Ecosystems are communities of living organisms interacting with their environment. Energy flows through ecosystems from producers to consumers to decomposers.",
        "The nervous system transmits signals between different parts of the body. The brain and spinal cord form the central nervous system.",
    ],
    "physics": [
        "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, gravity accelerates objects downward at approximately 9.8 meters per second squared.",
        "Light travels at approximately 300,000 kilometers per second in a vacuum. This is the fastest speed possible according to Einstein's theory of relativity.",
        "Energy cannot be created or destroyed, only converted from one form to another. This principle is known as the first law of thermodynamics.",
        "Atoms are composed of protons, neutrons, and electrons. The number of protons determines the element, while the arrangement of electrons determines chemical properties.",
        "Electromagnetic waves include radio waves, microwaves, infrared radiation, visible light, ultraviolet radiation, X-rays, and gamma rays.",
        "Newton's third law states that for every action there is an equal and opposite reaction. This principle explains how rockets propel themselves through space.",
        "The speed of sound in air at room temperature is approximately 343 meters per second. Sound travels faster in water and even faster in solid materials.",
        "Quantum mechanics describes the behavior of particles at the atomic and subatomic level. Particles can exhibit both wave-like and particle-like properties.",
    ],
    "history": [
        "The Roman Empire was one of the largest empires in history. It was founded in 27 BC and the Western Empire fell in 476 AD.",
        "World War II lasted from 1939 to 1945. It involved most of the world's nations and resulted in an estimated 70 to 85 million fatalities.",
        "The Renaissance was a cultural movement that began in Italy in the fourteenth century. It brought renewed interest in art, science, and classical learning.",
        "The Industrial Revolution began in Britain in the late eighteenth century. It transformed manufacturing from hand production to machine production.",
        "The French Revolution began in 1789 and had a lasting impact on European politics. It led to the rise of Napoleon Bonaparte.",
        "Ancient Egypt developed along the Nile River around 3100 BC. The pyramids were built as tombs for the pharaohs during the Old Kingdom period.",
        "The Cold War was a period of geopolitical tension between the United States and the Soviet Union from 1947 to 1991.",
        "The printing press was invented by Johannes Gutenberg around 1440. It revolutionized the spread of information and played a key role in the Reformation.",
    ],
    "chemistry": [
        "Water is a chemical compound with the formula H2O. Each molecule consists of two hydrogen atoms bonded to one oxygen atom.",
        "The periodic table organizes elements by their atomic number and chemical properties. Elements in the same column share similar characteristics.",
        "Chemical reactions involve the breaking and forming of chemical bonds. Reactants are transformed into products through these processes.",
        "Acids have a pH less than 7 and donate hydrogen ions in solution. Bases have a pH greater than 7 and accept hydrogen ions.",
        "Carbon is the basis of organic chemistry. It can form four covalent bonds, allowing it to create diverse molecular structures.",
        "Catalysts speed up chemical reactions without being consumed. Enzymes are biological catalysts that facilitate reactions in living organisms.",
        "Oxidation is the loss of electrons, while reduction is the gain of electrons. These processes always occur together in redox reactions.",
        "The states of matter include solid, liquid, gas, and plasma. Transitions between states occur when energy is added or removed.",
    ],
}


# Evaluation fact pairs (held-out)
EVAL_PAIRS = [
    FactPair("The capital of France is Paris.", "The capital of France is Berlin.",
             "france_capital", "geography"),
    FactPair("The heart has four chambers.", "The heart has six chambers.",
             "heart_chambers", "biology"),
    FactPair("Light travels faster than sound.", "Sound travels faster than light.",
             "light_vs_sound", "physics"),
    FactPair("World War II ended in 1945.", "World War II ended in 1952.",
             "ww2_end", "history"),
    FactPair("Water is composed of hydrogen and oxygen.",
             "Water is composed of hydrogen and nitrogen.",
             "water_composition", "chemistry"),
    FactPair("The Earth orbits the Sun.", "The Sun orbits the Earth.",
             "earth_orbit", "physics"),
    FactPair("DNA has a double helix structure.", "DNA has a triple helix structure.",
             "dna_helix", "biology"),
    FactPair("The Amazon River is in South America.",
             "The Amazon River is in Africa.",
             "amazon_location", "geography"),
]

PERPLEXITY_TEXTS = [
    "The study of economics involves understanding how societies allocate scarce resources among competing uses.",
    "In computer science, algorithms are step-by-step procedures for solving problems.",
    "Literary criticism examines how texts create meaning through language and narrative structure.",
    "The Renaissance was a cultural movement that began in Italy during the fourteenth century.",
    "Climate science studies the long-term patterns of temperature and atmospheric conditions.",
]


# ---------------------------------------------------------------------------
# Batch generators
# ---------------------------------------------------------------------------

def random_batch_generator(all_texts: list[str]):
    """Model D: random shuffled batching."""
    texts = list(all_texts)
    while True:
        random.shuffle(texts)
        for text in texts:
            yield text


def coherent_batch_generator(topic_data: dict[str, list[str]]):
    """Model E: round-robin topic-coherent batching."""
    topics = list(topic_data.keys())
    while True:
        for topic in topics:
            texts = list(topic_data[topic])
            random.shuffle(texts)
            for text in texts:
                yield text


def mixed_coherent_generator(topic_data: dict[str, list[str]],
                              coherence_ratio: float = 0.7):
    """Option 3: mixed coherent — 70% one topic, 30% random."""
    topics = list(topic_data.keys())
    all_texts = [t for texts in topic_data.values() for t in texts]

    while True:
        for topic in topics:
            texts = list(topic_data[topic])
            random.shuffle(texts)
            for text in texts:
                if random.random() < coherence_ratio:
                    yield text
                else:
                    yield random.choice(all_texts)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    variant: str,
    topic_data: dict,
    eval_pairs: list[FactPair],
    num_steps: int = 2000,
    lr: float = 5e-5,
    eval_every: int = 200,
    force: bool = False,
) -> dict:
    save_dir = EXP8_DIR / variant
    save_dir.mkdir(parents=True, exist_ok=True)

    if ((save_dir / "pytorch_model.bin").exists() or
        (save_dir / "model.safetensors").exists()) and not force:
        print(f"  [{variant}] Model exists, skipping training.")
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

    # Set up batch generator
    all_texts = [t for texts in topic_data.values() for t in texts]
    if variant == "random":
        gen = random_batch_generator(all_texts)
    elif variant == "coherent":
        gen = coherent_batch_generator(topic_data)
    elif variant == "mixed_coherent":
        gen = mixed_coherent_generator(topic_data, coherence_ratio=0.7)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    log = {
        "steps": [], "lm_loss": [],
        "eval_win_rate": [], "eval_mean_delta": [],
        "eval_perplexity": [],
        "per_topic_confidence": [],
    }

    print(f"  [{variant}] Training for {num_steps} steps...")
    start_time = time.time()

    for step in tqdm(range(1, num_steps + 1), desc=f"  {variant}", leave=False):
        optimizer.zero_grad()
        text = next(gen)
        loss = standard_lm_step(model, tokenizer, text, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            model.eval()
            eval_result = evaluate_truth_detection(
                model, tokenizer, eval_pairs, device)
            ppl = evaluate_perplexity(
                model, tokenizer, PERPLEXITY_TEXTS, device)

            # Per-topic confidence
            topic_confs = {}
            for topic, texts in topic_data.items():
                confs = []
                for t in texts[:3]:
                    c, _ = compute_token_confidence(model, tokenizer, t, device)
                    confs.append(c)
                topic_confs[topic] = float(np.mean(confs))

            model.train()

            log["steps"].append(step)
            log["lm_loss"].append(loss.item())
            log["eval_win_rate"].append(eval_result["win_rate"])
            log["eval_mean_delta"].append(eval_result["mean_delta"])
            log["eval_perplexity"].append(ppl)
            log["per_topic_confidence"].append(topic_confs)

            if step % (eval_every * 5) == 0:
                print(f"    Step {step}: loss={loss.item():.4f}, "
                      f"win_rate={eval_result['win_rate']:.1%}, ppl={ppl:.1f}")

    elapsed = time.time() - start_time
    print(f"  [{variant}] Done in {elapsed:.1f}s")

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    model.eval()
    final_eval = evaluate_truth_detection(model, tokenizer, eval_pairs, device)
    final_ppl = evaluate_perplexity(model, tokenizer, PERPLEXITY_TEXTS, device)

    del model, tokenizer, optimizer
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
# Hidden state geometry analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    layer: int = -1,
) -> np.ndarray:
    """Extract mean hidden states from a specified layer for a list of texts."""
    model.eval()
    states = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=128).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states is tuple of (n_layers + 1, batch, seq, hidden)
        hs = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
        mean_hs = hs[0].mean(dim=0).cpu().numpy()  # (hidden_dim,)
        states.append(mean_hs)
    return np.array(states)


def compute_topic_geometry(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    topic_data: dict,
    device: torch.device,
) -> dict:
    """Compute within-topic and between-topic cosine similarities."""
    topic_states = {}
    for topic, texts in topic_data.items():
        states = extract_hidden_states(model, tokenizer, texts, device)
        topic_states[topic] = states

    # Within-topic similarity
    within_sims = {}
    for topic, states in topic_states.items():
        if len(states) < 2:
            continue
        norms = states / (np.linalg.norm(states, axis=1, keepdims=True) + 1e-10)
        sim_matrix = norms @ norms.T
        # Upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        within_sims[topic] = float(sim_matrix[mask].mean())

    # Between-topic similarity
    topics = list(topic_states.keys())
    between_sims = []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            s_i = topic_states[topics[i]]
            s_j = topic_states[topics[j]]
            n_i = s_i / (np.linalg.norm(s_i, axis=1, keepdims=True) + 1e-10)
            n_j = s_j / (np.linalg.norm(s_j, axis=1, keepdims=True) + 1e-10)
            cross_sim = (n_i @ n_j.T).mean()
            between_sims.append(float(cross_sim))

    return {
        "within_topic_sims": within_sims,
        "mean_within": float(np.mean(list(within_sims.values()))),
        "mean_between": float(np.mean(between_sims)),
        "separation": float(np.mean(list(within_sims.values())) - np.mean(between_sims)),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_comparison(results: dict, save_path: Path):
    """Training curves for random vs coherent batching."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"random": "#757575", "coherent": "#4CAF50", "mixed_coherent": "#2196F3"}
    labels = {"random": "D: Random", "coherent": "E: Coherent",
              "mixed_coherent": "E': Mixed Coherent"}

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
    axes[0].set_ylabel("Win Rate"); axes[0].set_title("Truth Detection")
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    axes[1].set_ylabel("Mean Δ Confidence"); axes[1].set_title("Confidence Gap")
    axes[2].set_ylabel("Perplexity"); axes[2].set_title("Language Quality")

    for ax in axes:
        ax.set_xlabel("Training Step")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 8: Concept-Coherent Batching", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_topic_confidence(results: dict, save_path: Path):
    """Per-topic confidence evolution during training."""
    sns.set_theme(style="whitegrid", palette="muted")
    topics = list(TOPIC_DATA.keys())

    for variant, result in results.items():
        if "log" not in result:
            continue
        log = result["log"]
        if not log.get("per_topic_confidence"):
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        steps = log["steps"]

        for topic in topics:
            confs = [tc.get(topic, 0) for tc in log["per_topic_confidence"]]
            ax.plot(steps, confs, "-", linewidth=1.5, label=topic)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Confidence")
        ax.set_title(f"Per-Topic Confidence — {variant}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path.parent / f"topic_conf_{variant}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_geometry_comparison(geo_results: dict, save_path: Path):
    """Bar chart comparing within vs between topic similarity."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6))

    variants = list(geo_results.keys())
    x = np.arange(len(variants))
    width = 0.3

    within = [geo_results[v]["mean_within"] for v in variants]
    between = [geo_results[v]["mean_between"] for v in variants]
    separation = [geo_results[v]["separation"] for v in variants]

    ax.bar(x - width, within, width, label="Within-topic similarity", color="#4CAF50")
    ax.bar(x, between, width, label="Between-topic similarity", color="#F44336")
    ax.bar(x + width, separation, width, label="Separation (within - between)", color="#2196F3")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Hidden State Geometry: Topic Clustering")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(num_steps: int = 2000, lr: float = 5e-5, force: bool = False):
    print("=" * 70)
    print("EXPERIMENT 8: Concept-Coherent Batching")
    print("=" * 70)
    print(f"Training steps: {num_steps}")
    print(f"Topics: {', '.join(TOPIC_DATA.keys())}")
    total_texts = sum(len(v) for v in TOPIC_DATA.values())
    print(f"Total training texts: {total_texts}")

    start_time = time.time()
    training_results = {}

    for variant in ["random", "coherent"]:
        print(f"\n{'='*50}")
        print(f"TRAINING: {variant.upper()}")
        print(f"{'='*50}")

        result = train_model(
            variant=variant,
            topic_data=TOPIC_DATA,
            eval_pairs=EVAL_PAIRS,
            num_steps=num_steps,
            lr=lr,
            eval_every=max(100, num_steps // 20),
            force=force,
        )
        training_results[variant] = result

    # ===================================================================
    # Hidden state geometry analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("HIDDEN STATE GEOMETRY ANALYSIS")
    print("=" * 70)

    device = get_device()
    geo_results = {}

    for variant in ["random", "coherent"]:
        save_dir = EXP8_DIR / variant
        if not (save_dir / "pytorch_model.bin").exists() and \
           not (save_dir / "model.safetensors").exists():
            continue

        print(f"  Analyzing {variant}...")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            save_dir, dtype=torch.float32).to(device)

        geo = compute_topic_geometry(model, tokenizer, TOPIC_DATA, device)
        geo_results[variant] = geo

        print(f"    Within-topic: {geo['mean_within']:.4f}")
        print(f"    Between-topic: {geo['mean_between']:.4f}")
        print(f"    Separation: {geo['separation']:.4f}")

        del model, tokenizer
        import gc; gc.collect()

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 8 SUMMARY")
    print("=" * 70)

    print(f"\n{'Variant':<16} {'Win Rate':<10} {'Mean Δ':<10} "
          f"{'Perplexity':<12} {'Separation':<12}")
    print("-" * 60)

    for variant in ["random", "coherent"]:
        r = training_results.get(variant, {})
        geo = geo_results.get(variant, {})
        wr = r.get("final_win_rate", 0)
        delta = r.get("final_mean_delta", 0)
        ppl = r.get("final_perplexity", 0)
        sep = geo.get("separation", 0)
        print(f"{variant:<16} {wr:<10.1%} {delta:<+10.4f} {ppl:<12.1f} {sep:<12.4f}")

    # ===================================================================
    # Plots
    # ===================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    if any("log" in r for r in training_results.values()):
        print("  Training comparison...")
        plot_training_comparison(training_results,
                                 EXP8_FIGURES_DIR / "training_comparison.png")
        print("  Topic confidence evolution...")
        plot_topic_confidence(training_results,
                               EXP8_FIGURES_DIR / "topic_confidence.png")

    if geo_results:
        print("  Geometry comparison...")
        plot_geometry_comparison(geo_results,
                                 EXP8_FIGURES_DIR / "geometry_comparison.png")

    total_time = time.time() - start_time
    fig_count = len(list(EXP8_FIGURES_DIR.glob("*.png")))

    print(f"\n{'='*70}")
    print("EXPERIMENT 8 COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Figures: {fig_count}")

    # Verdict
    if "random" in training_results and "coherent" in training_results:
        r_wr = training_results["random"].get("final_win_rate", 0)
        c_wr = training_results["coherent"].get("final_win_rate", 0)
        r_sep = geo_results.get("random", {}).get("separation", 0)
        c_sep = geo_results.get("coherent", {}).get("separation", 0)

        if c_wr > r_wr + 0.05 or c_sep > r_sep + 0.01:
            print("\n  FINDING: Coherent batching IMPROVES confidence patterns!")
            if c_sep > r_sep + 0.01:
                print("  → Tighter topic clusters in hidden state space.")
        else:
            print("\n  FINDING: No meaningful difference from coherent batching.")
            print("  → Data ordering doesn't affect internal knowledge structure at this scale.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_experiment(num_steps=args.steps, lr=args.lr, force=args.force)
