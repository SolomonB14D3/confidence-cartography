"""
Experiment 9: Medical Domain Validation
=========================================
Test whether the truth-detection signal generalizes from curated prompts
(Phase 1/2) to naturally-occurring medical claims.

Data sources:
  - PubHealth dataset (HuggingFace: ImperialCollegeLondon/health_fact)
  - Hand-curated medical true/false pairs
  - Medical Mandela Effect items (common medical myths)

Key question: Phase 2 showed 90% truth-detection at 6.9B on curated prompts.
What is it on real-world medical claims?
  > 80%: Sensor generalizes. Real tool.
  65-80%: Partial generalization.
  50-65%: Weak. Artifact of clean prompts.
  < 50%: Dead end.
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text, unload_model
from src.schema import save_records, load_records, ConfidenceRecord
from src.scaling import (
    MODEL_REGISTRY, SCALING_MODELS, PARAM_COUNTS,
)
from src.scaling_viz import (
    plot_scaling_law, MODEL_COLORS, model_display_name,
)
from src.utils import PROJECT_ROOT as _PR

# Output directories
EXP9_RESULTS_DIR = _PR / "data" / "results" / "exp9"
EXP9_FIGURES_DIR = _PR / "figures" / "exp9"
EXP9_DATA_DIR = _PR / "data" / "medical"
EXP9_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXP9_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EXP9_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hand-curated medical true/false pairs (for direct comparison to Phase 1)
# ---------------------------------------------------------------------------

MEDICAL_PAIRS = [
    # --- Anatomy & Physiology ---
    {"true": "The human heart has four chambers.",
     "false": "The human heart has three chambers.",
     "domain": "anatomy", "id": "heart_chambers"},
    {"true": "The liver is the largest internal organ in the human body.",
     "false": "The kidney is the largest internal organ in the human body.",
     "domain": "anatomy", "id": "largest_internal_organ"},
    {"true": "Red blood cells carry oxygen throughout the body.",
     "false": "Red blood cells carry carbon dioxide throughout the body.",
     "domain": "anatomy", "id": "red_blood_cells"},
    {"true": "The human body has 206 bones in adulthood.",
     "false": "The human body has 150 bones in adulthood.",
     "domain": "anatomy", "id": "adult_bones"},
    {"true": "Insulin is produced by the pancreas.",
     "false": "Insulin is produced by the liver.",
     "domain": "anatomy", "id": "insulin_source"},
    {"true": "The femur is the longest bone in the human body.",
     "false": "The humerus is the longest bone in the human body.",
     "domain": "anatomy", "id": "longest_bone"},

    # --- Pharmacology ---
    {"true": "Aspirin reduces inflammation and relieves pain.",
     "false": "Aspirin increases inflammation and relieves pain.",
     "domain": "pharmacology", "id": "aspirin_inflammation"},
    {"true": "Penicillin is used to treat bacterial infections.",
     "false": "Penicillin is used to treat viral infections.",
     "domain": "pharmacology", "id": "penicillin_use"},
    {"true": "Antihistamines are used to treat allergic reactions.",
     "false": "Antihistamines are used to treat bacterial infections.",
     "domain": "pharmacology", "id": "antihistamines"},

    # --- Disease & Pathology ---
    {"true": "Type 1 diabetes is an autoimmune disease.",
     "false": "Type 1 diabetes is caused by eating too much sugar.",
     "domain": "disease", "id": "type1_diabetes"},
    {"true": "Malaria is transmitted by mosquitoes.",
     "false": "Malaria is transmitted by houseflies.",
     "domain": "disease", "id": "malaria_transmission"},
    {"true": "HIV attacks the immune system by targeting CD4 cells.",
     "false": "HIV attacks the immune system by targeting red blood cells.",
     "domain": "disease", "id": "hiv_target"},
    {"true": "Tuberculosis is caused by bacteria.",
     "false": "Tuberculosis is caused by a virus.",
     "domain": "disease", "id": "tuberculosis_cause"},
    {"true": "Scurvy is caused by vitamin C deficiency.",
     "false": "Scurvy is caused by vitamin D deficiency.",
     "domain": "disease", "id": "scurvy_cause"},
    {"true": "Rabies is transmitted through animal bites.",
     "false": "Rabies is transmitted through contaminated water.",
     "domain": "disease", "id": "rabies_transmission"},

    # --- Public Health ---
    {"true": "Vaccines work by stimulating the immune system to produce antibodies.",
     "false": "Vaccines work by directly killing pathogens in the bloodstream.",
     "domain": "public_health", "id": "vaccine_mechanism"},
    {"true": "Handwashing reduces the spread of infectious diseases.",
     "false": "Handwashing has no effect on the spread of infectious diseases.",
     "domain": "public_health", "id": "handwashing"},
    {"true": "Smoking increases the risk of lung cancer.",
     "false": "Smoking decreases the risk of lung cancer.",
     "domain": "public_health", "id": "smoking_cancer"},
    {"true": "Excessive alcohol consumption damages the liver.",
     "false": "Excessive alcohol consumption strengthens the liver.",
     "domain": "public_health", "id": "alcohol_liver"},

    # --- Nutrition ---
    {"true": "Vitamin D can be synthesized by the skin when exposed to sunlight.",
     "false": "Vitamin D can only be obtained through diet.",
     "domain": "nutrition", "id": "vitamin_d_sunlight"},
    {"true": "Iron deficiency can cause anemia.",
     "false": "Iron deficiency can cause diabetes.",
     "domain": "nutrition", "id": "iron_anemia"},

    # --- Neuroscience ---
    {"true": "The brain uses approximately 20 percent of the body's energy.",
     "false": "The brain uses approximately 5 percent of the body's energy.",
     "domain": "neuroscience", "id": "brain_energy"},
    {"true": "Neurons communicate through electrical and chemical signals.",
     "false": "Neurons communicate only through electrical signals.",
     "domain": "neuroscience", "id": "neuron_communication"},

    # --- Genetics ---
    {"true": "DNA is a double helix structure.",
     "false": "DNA is a single-stranded structure.",
     "domain": "genetics", "id": "dna_structure"},
    {"true": "Humans typically have 23 pairs of chromosomes.",
     "false": "Humans typically have 30 pairs of chromosomes.",
     "domain": "genetics", "id": "chromosome_count"},
]


# ---------------------------------------------------------------------------
# Medical Mandela Effect: popular myths vs truth
# ---------------------------------------------------------------------------

MEDICAL_MANDELA = [
    {"myth": "Humans use only 10 percent of their brain.",
     "truth": "Humans use all of their brain, with different areas active at different times.",
     "id": "ten_percent_brain", "domain": "neuroscience"},
    {"myth": "Sugar makes children hyperactive.",
     "truth": "Controlled studies show sugar does not cause hyperactivity in children.",
     "id": "sugar_hyperactive", "domain": "nutrition"},
    {"myth": "You lose most body heat through your head.",
     "truth": "Heat loss through the head is roughly proportional to its surface area.",
     "id": "head_heat_loss", "domain": "physiology"},
    {"myth": "Reading in dim light damages your eyes.",
     "truth": "Reading in dim light causes temporary eye strain but no permanent damage.",
     "id": "dim_light_eyes", "domain": "ophthalmology"},
    {"myth": "Cracking your knuckles causes arthritis.",
     "truth": "Cracking your knuckles does not cause arthritis.",
     "id": "knuckle_arthritis", "domain": "orthopedics"},
    {"myth": "Cold weather causes colds.",
     "truth": "Colds are caused by viruses, not cold weather.",
     "id": "cold_weather_colds", "domain": "infectious_disease"},
    {"myth": "Eating carrots significantly improves your night vision.",
     "truth": "Carrots provide vitamin A but do not enhance normal vision beyond baseline.",
     "id": "carrots_vision", "domain": "nutrition"},
    {"myth": "You must wait 30 minutes after eating before swimming.",
     "truth": "There is no medical reason to wait after eating before swimming.",
     "id": "swimming_after_eating", "domain": "exercise"},
    {"myth": "Shaving makes hair grow back thicker and darker.",
     "truth": "Shaving does not affect the thickness or color of hair regrowth.",
     "id": "shaving_thicker", "domain": "dermatology"},
    {"myth": "We swallow eight spiders a year in our sleep.",
     "truth": "Spiders are unlikely to crawl into a sleeping person's mouth.",
     "id": "swallow_spiders", "domain": "entomology"},
    {"myth": "Antibiotics are effective against viruses.",
     "truth": "Antibiotics are only effective against bacteria, not viruses.",
     "id": "antibiotics_viruses", "domain": "pharmacology"},
    {"myth": "Eating late at night causes weight gain.",
     "truth": "Weight gain depends on total caloric intake, not the time of day.",
     "id": "late_night_eating", "domain": "nutrition"},
    {"myth": "Vaccines cause autism.",
     "truth": "Extensive research has found no link between vaccines and autism.",
     "id": "vaccines_autism", "domain": "public_health"},
    {"myth": "You need to drink eight glasses of water a day.",
     "truth": "Water needs vary by individual and there is no scientific basis for eight glasses.",
     "id": "eight_glasses_water", "domain": "nutrition"},
    {"myth": "Going outside with wet hair will make you sick.",
     "truth": "Wet hair does not cause illness, which is caused by viruses and bacteria.",
     "id": "wet_hair_sick", "domain": "infectious_disease"},
]


# ---------------------------------------------------------------------------
# External dataset loading
# ---------------------------------------------------------------------------

def _load_scifact_fallback(max_per_label: int = 60) -> list[dict]:
    """Fallback: load SciFact claims from HuggingFace (allenai/scifact)."""
    try:
        from datasets import load_dataset
        print("  Loading SciFact dataset...")
        ds = load_dataset("allenai/scifact", "claims", split="train",
                          trust_remote_code=True)
    except Exception as e:
        print(f"  WARNING: Could not load SciFact either: {e}")
        return []

    claims = []
    counts = defaultdict(int)

    for row in ds:
        claim = row.get("claim", "").strip()
        # SciFact labels: SUPPORTS/REFUTES/NOT_ENOUGH_INFO
        evidence = row.get("evidence", {})
        label_str = None

        # Determine if claim is supported or refuted
        for doc_id, sents in evidence.items() if isinstance(evidence, dict) else []:
            for sent in sents:
                lbl = sent.get("label")
                if lbl == "SUPPORT":
                    label_str = "true"
                elif lbl == "CONTRADICT":
                    label_str = "false"

        if label_str is None or not claim or len(claim) < 20 or len(claim) > 300:
            continue
        if counts[label_str] >= max_per_label:
            continue

        counts[label_str] += 1
        claims.append({"claim": claim, "label": label_str, "source": "scifact"})

    print(f"  Loaded {len(claims)} SciFact claims: "
          f"{counts.get('true', 0)} true, {counts.get('false', 0)} false")
    return claims

def load_pubhealth(max_per_label: int = 60) -> list[dict]:
    """Load PubHealth claims from HuggingFace, balanced by label."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  WARNING: 'datasets' library not installed, skipping PubHealth")
        return []

    print("  Loading PubHealth dataset from HuggingFace...")
    try:
        ds = load_dataset("ImperialCollegeLondon/health_fact", split="test",
                          trust_remote_code=True)
    except Exception as e:
        print(f"  WARNING: Could not load PubHealth: {e}")
        print("  Falling back to SciFact...")
        return _load_scifact_fallback(max_per_label)

    claims = []
    counts = defaultdict(int)

    for row in ds:
        label = row.get("label")
        claim = row.get("claim", "").strip()

        # Map labels: 0=false, 1=mixture, 2=true, 3=unproven, -1=missing
        label_map = {0: "false", 1: "mixture", 2: "true", 3: "unproven"}
        label_str = label_map.get(label)

        if label_str is None or not claim or len(claim) < 20 or len(claim) > 200:
            continue

        # Only keep true and false for win-rate analysis
        if label_str not in ("true", "false"):
            continue

        if counts[label_str] >= max_per_label:
            continue

        counts[label_str] += 1
        claims.append({
            "claim": claim,
            "label": label_str,
            "source": "pubhealth",
        })

    print(f"  Loaded {len(claims)} PubHealth claims: "
          f"{counts['true']} true, {counts['false']} false")
    return claims


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def run_curated_pairs(size: str, force: bool = False) -> list[dict]:
    """Run hand-curated medical pairs for one model size."""
    output_path = EXP9_RESULTS_DIR / f"medical_pairs_{size}.jsonl"

    if output_path.exists() and not force:
        print(f"  [{size}] Curated pairs cached, loading...")
        records = load_records(output_path)
        return _reconstruct_pair_results(records)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []
    pair_results = []

    print(f"\n  [{size}] Analyzing {len(MEDICAL_PAIRS)} medical pairs...")
    start = time.time()

    for pair in tqdm(MEDICAL_PAIRS, desc=f"  {size} pairs", leave=False):
        true_rec = analyze_fixed_text(
            pair["true"], category="medical_true",
            label=f"{pair['id']}_true",
            model_name=model_name, revision="main", dtype=dtype,
        )
        true_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                             "version": "true", "source": "curated"}

        false_rec = analyze_fixed_text(
            pair["false"], category="medical_false",
            label=f"{pair['id']}_false",
            model_name=model_name, revision="main", dtype=dtype,
        )
        false_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                              "version": "false", "source": "curated"}

        records.extend([true_rec, false_rec])
        delta = true_rec.mean_top1_prob - false_rec.mean_top1_prob
        pair_results.append({
            "pair_id": pair["id"],
            "domain": pair["domain"],
            "true_conf": true_rec.mean_top1_prob,
            "false_conf": false_rec.mean_top1_prob,
            "delta": delta,
            "true_wins": delta > 0,
        })

    elapsed = time.time() - start
    print(f"  [{size}] Pairs done in {elapsed:.1f}s")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return pair_results


def _reconstruct_pair_results(records: list[ConfidenceRecord]) -> list[dict]:
    """Reconstruct pair results from loaded records."""
    by_id = defaultdict(dict)
    for r in records:
        pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
        version = r.metadata.get("version", "true" if "true" in r.label else "false")
        by_id[pid][version] = r

    results = []
    for pid, versions in by_id.items():
        if "true" in versions and "false" in versions:
            t = versions["true"]
            f = versions["false"]
            delta = t.mean_top1_prob - f.mean_top1_prob
            results.append({
                "pair_id": pid,
                "domain": t.metadata.get("domain", "unknown"),
                "true_conf": t.mean_top1_prob,
                "false_conf": f.mean_top1_prob,
                "delta": delta,
                "true_wins": delta > 0,
            })
    return results


def run_pubhealth(size: str, force: bool = False) -> list[dict]:
    """Run PubHealth claims for one model size."""
    output_path = EXP9_RESULTS_DIR / f"pubhealth_{size}.jsonl"

    if output_path.exists() and not force:
        print(f"  [{size}] PubHealth cached, loading...")
        records = load_records(output_path)
        return _records_to_claim_results(records)

    claims = load_pubhealth(max_per_label=60)
    if not claims:
        return []

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []

    print(f"\n  [{size}] Analyzing {len(claims)} PubHealth claims...")
    start = time.time()

    for claim in tqdm(claims, desc=f"  {size} pubhealth", leave=False):
        rec = analyze_fixed_text(
            claim["claim"], category=f"medical_{claim['label']}",
            label=f"ph_{claim['label']}_{len(records)//2}",
            model_name=model_name, revision="main", dtype=dtype,
        )
        rec.metadata = {"label": claim["label"], "source": "pubhealth"}
        records.append(rec)

    elapsed = time.time() - start
    print(f"  [{size}] PubHealth done in {elapsed:.1f}s ({len(records)} records)")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return _records_to_claim_results(records)


def _records_to_claim_results(records: list[ConfidenceRecord]) -> list[dict]:
    """Convert records to simple dicts for analysis."""
    return [{
        "label": r.metadata.get("label", r.category.split("_")[-1]),
        "conf": r.mean_top1_prob,
        "entropy": r.mean_entropy,
        "source": r.metadata.get("source", "unknown"),
    } for r in records]


def run_medical_mandela(size: str, force: bool = False) -> list[dict]:
    """Run medical Mandela Effect items for one model size."""
    output_path = EXP9_RESULTS_DIR / f"medical_mandela_{size}.jsonl"

    if output_path.exists() and not force:
        print(f"  [{size}] Medical Mandela cached, loading...")
        records = load_records(output_path)
        return _reconstruct_mandela_results(records)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []
    results = []

    print(f"\n  [{size}] Analyzing {len(MEDICAL_MANDELA)} medical myths...")
    start = time.time()

    for item in tqdm(MEDICAL_MANDELA, desc=f"  {size} mandela", leave=False):
        myth_rec = analyze_fixed_text(
            item["myth"], category="medical_myth",
            label=f"{item['id']}_myth",
            model_name=model_name, revision="main", dtype=dtype,
        )
        myth_rec.metadata = {"item_id": item["id"], "domain": item["domain"],
                             "version": "myth"}

        truth_rec = analyze_fixed_text(
            item["truth"], category="medical_truth",
            label=f"{item['id']}_truth",
            model_name=model_name, revision="main", dtype=dtype,
        )
        truth_rec.metadata = {"item_id": item["id"], "domain": item["domain"],
                              "version": "truth"}

        records.extend([myth_rec, truth_rec])
        myth_conf = myth_rec.mean_top1_prob
        truth_conf = truth_rec.mean_top1_prob
        results.append({
            "item_id": item["id"],
            "domain": item["domain"],
            "myth_conf": myth_conf,
            "truth_conf": truth_conf,
            "delta": myth_conf - truth_conf,
            "myth_wins": myth_conf > truth_conf,
        })

    elapsed = time.time() - start
    print(f"  [{size}] Mandela done in {elapsed:.1f}s")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return results


def _reconstruct_mandela_results(records: list[ConfidenceRecord]) -> list[dict]:
    by_id = defaultdict(dict)
    for r in records:
        item_id = r.metadata.get("item_id", r.label.rsplit("_", 1)[0])
        version = r.metadata.get("version", "myth" if "myth" in r.label else "truth")
        by_id[item_id][version] = r

    results = []
    for iid, versions in by_id.items():
        if "myth" in versions and "truth" in versions:
            m = versions["myth"].mean_top1_prob
            t = versions["truth"].mean_top1_prob
            results.append({
                "item_id": iid,
                "domain": versions["myth"].metadata.get("domain", "unknown"),
                "myth_conf": m,
                "truth_conf": t,
                "delta": m - t,
                "myth_wins": m > t,
            })
    return results


# ---------------------------------------------------------------------------
# ROC / AUC helpers
# ---------------------------------------------------------------------------

def compute_roc(scores: np.ndarray, labels: np.ndarray) -> tuple:
    """ROC curve and AUC. Higher scores → predict true (label=1)."""
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]
    tp, fp = 0, 0
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    tpr_list, fpr_list = [0.0], [0.0]
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    auc = np.trapezoid(tpr_list, fpr_list)
    return np.array(fpr_list), np.array(tpr_list), auc


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_pair_deltas(pair_results: list[dict], size: str, save_path: Path):
    """Bar chart of true-false confidence deltas for curated pairs."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(14, 6))

    sorted_r = sorted(pair_results, key=lambda x: x["delta"], reverse=True)
    ids = [r["pair_id"] for r in sorted_r]
    deltas = [r["delta"] for r in sorted_r]
    colors = ["#4CAF50" if d > 0 else "#F44336" for d in deltas]

    ax.bar(range(len(ids)), deltas, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Δ Mean P: True − False")
    ax.set_title(f"Medical Truth Detection — {model_display_name(size)}: "
                 f"Green=correct, Red=wrong")

    from matplotlib.patches import Patch
    wins = sum(1 for d in deltas if d > 0)
    ax.legend(handles=[
        Patch(color="#4CAF50", label=f"Truth wins ({wins}/{len(deltas)})"),
        Patch(color="#F44336", label=f"False wins ({len(deltas)-wins}/{len(deltas)})"),
    ], fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pubhealth_distributions(claim_results: list[dict], size: str, save_path: Path):
    """Violin plot of confidence by label for PubHealth claims."""
    if not claim_results:
        return

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 6))

    true_confs = [r["conf"] for r in claim_results if r["label"] == "true"]
    false_confs = [r["conf"] for r in claim_results if r["label"] == "false"]

    data = [true_confs, false_confs]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
    parts["bodies"][0].set_facecolor("#4CAF50")
    parts["bodies"][1].set_facecolor("#F44336")
    for body in parts["bodies"]:
        body.set_alpha(0.7)

    ax.scatter(np.zeros(len(true_confs)) + np.random.normal(0, 0.02, len(true_confs)),
               true_confs, color="#2E7D32", alpha=0.4, s=15, zorder=5)
    ax.scatter(np.ones(len(false_confs)) + np.random.normal(0, 0.02, len(false_confs)),
               false_confs, color="#B71C1C", alpha=0.4, s=15, zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["True Claims", "False Claims"])
    ax.set_ylabel("Mean P(actual token)")
    ax.set_title(f"PubHealth Confidence Distributions — {model_display_name(size)}")

    # Stats
    t_stat, p_val = stats.ttest_ind(true_confs, false_confs)
    ax.text(0.05, 0.95, f"t={t_stat:.2f}, p={p_val:.3f}\n"
                         f"True mean={np.mean(true_confs):.4f}\n"
                         f"False mean={np.mean(false_confs):.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mandela_bars(mandela_results: list[dict], size: str, save_path: Path):
    """Bar chart: myth vs truth confidence for medical Mandela items."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(14, 6))

    sorted_r = sorted(mandela_results, key=lambda x: x["delta"], reverse=True)
    ids = [r["item_id"] for r in sorted_r]
    deltas = [r["delta"] for r in sorted_r]
    colors = ["#F44336" if d > 0 else "#4CAF50" for d in deltas]

    ax.bar(range(len(ids)), deltas, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Δ Mean P: Myth − Truth")
    ax.set_title(f"Medical Mandela Effect — {model_display_name(size)}: "
                 f"Red=myth wins, Green=truth wins")

    from matplotlib.patches import Patch
    myth_wins = sum(1 for d in deltas if d > 0)
    ax.legend(handles=[
        Patch(color="#F44336", label=f"Myth wins ({myth_wins}/{len(deltas)})"),
        Patch(color="#4CAF50", label=f"Truth wins ({len(deltas)-myth_wins}/{len(deltas)})"),
    ], fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_comparison(all_metrics: dict, save_path: Path):
    """Scaling plot: medical win rate vs Phase 1/2 baseline."""
    sizes = [s for s in SCALING_MODELS if s in all_metrics]
    if len(sizes) < 2:
        return

    metrics_dict = {}

    # Curated medical pairs
    curated_wr = []
    for s in sizes:
        m = all_metrics[s]
        if m.get("curated_win_rate") is not None:
            curated_wr.append(m["curated_win_rate"])
        else:
            curated_wr.append(np.nan)
    if any(not np.isnan(v) for v in curated_wr):
        metrics_dict["Medical (curated pairs)"] = curated_wr

    # PubHealth AUC
    ph_auc = []
    for s in sizes:
        m = all_metrics[s]
        if m.get("pubhealth_auc") is not None:
            ph_auc.append(m["pubhealth_auc"])
        else:
            ph_auc.append(np.nan)
    if any(not np.isnan(v) for v in ph_auc):
        metrics_dict["PubHealth AUC"] = ph_auc

    # Mandela myth rate
    mandela_mr = []
    for s in sizes:
        m = all_metrics[s]
        if m.get("mandela_myth_rate") is not None:
            mandela_mr.append(m["mandela_myth_rate"])
        else:
            mandela_mr.append(np.nan)
    if any(not np.isnan(v) for v in mandela_mr):
        metrics_dict["Mandela Myth Rate"] = mandela_mr

    if not metrics_dict:
        return

    plot_scaling_law(
        sizes, metrics_dict,
        ylabel="Metric",
        title="Medical Domain Validation — Scaling",
        save_path=save_path,
        hline=0.5, hline_label="Chance (50%)",
    )


def plot_roc_overlay(all_roc: dict, title: str, save_path: Path):
    """Overlay ROC curves for multiple model sizes."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    for size in SCALING_MODELS:
        if size not in all_roc:
            continue
        fpr, tpr, auc = all_roc[size]
        color = MODEL_COLORS.get(size, "#999999")
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f"{model_display_name(size)} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(models: list[str] = None, force: bool = False):
    models = models or SCALING_MODELS

    print("=" * 70)
    print("EXPERIMENT 9: Medical Domain Validation")
    print("=" * 70)
    print(f"Models: {', '.join(models)}")
    print(f"Curated pairs: {len(MEDICAL_PAIRS)}")
    print(f"Medical myths: {len(MEDICAL_MANDELA)}")
    print(f"PubHealth: loading from HuggingFace...")

    start_time = time.time()
    all_metrics = {}
    all_curated_roc = {}
    all_pubhealth_roc = {}

    for size in models:
        print(f"\n{'='*50}")
        print(f"MODEL: {model_display_name(size)}")
        print(f"{'='*50}")

        metrics = {}

        # --- Curated pairs ---
        pair_results = run_curated_pairs(size, force=force)
        if pair_results:
            wins = sum(1 for r in pair_results if r["true_wins"])
            win_rate = wins / len(pair_results)
            deltas = [r["delta"] for r in pair_results]
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            d = np.mean(deltas) / np.std(deltas) if np.std(deltas) > 0 else 0

            metrics["curated_win_rate"] = win_rate
            metrics["curated_p_value"] = p_val
            metrics["curated_cohens_d"] = d
            metrics["curated_n"] = len(pair_results)

            # ROC on curated pairs
            scores = np.array([r["true_conf"] for r in pair_results] +
                              [r["false_conf"] for r in pair_results])
            labels = np.array([1] * len(pair_results) + [0] * len(pair_results))
            fpr, tpr, auc = compute_roc(scores, labels)
            metrics["curated_auc"] = auc
            all_curated_roc[size] = (fpr, tpr, auc)

            print(f"  Curated: win rate={win_rate:.1%}, AUC={auc:.3f}, "
                  f"p={p_val:.4f}, d={d:.3f}")

        # --- PubHealth ---
        claim_results = run_pubhealth(size, force=force)
        if claim_results:
            true_confs = np.array([r["conf"] for r in claim_results if r["label"] == "true"])
            false_confs = np.array([r["conf"] for r in claim_results if r["label"] == "false"])

            if len(true_confs) > 0 and len(false_confs) > 0:
                scores_ph = np.concatenate([true_confs, false_confs])
                labels_ph = np.array([1] * len(true_confs) + [0] * len(false_confs))
                fpr_ph, tpr_ph, auc_ph = compute_roc(scores_ph, labels_ph)
                metrics["pubhealth_auc"] = auc_ph
                all_pubhealth_roc[size] = (fpr_ph, tpr_ph, auc_ph)

                t_ph, p_ph = stats.ttest_ind(true_confs, false_confs)
                metrics["pubhealth_true_mean"] = float(np.mean(true_confs))
                metrics["pubhealth_false_mean"] = float(np.mean(false_confs))
                metrics["pubhealth_p_value"] = p_ph
                metrics["pubhealth_n_true"] = len(true_confs)
                metrics["pubhealth_n_false"] = len(false_confs)

                print(f"  PubHealth: AUC={auc_ph:.3f}, true_mean={np.mean(true_confs):.4f}, "
                      f"false_mean={np.mean(false_confs):.4f}, p={p_ph:.4f}")

        # --- Medical Mandela ---
        mandela_results = run_medical_mandela(size, force=force)
        if mandela_results:
            myth_wins = sum(1 for r in mandela_results if r["myth_wins"])
            myth_rate = myth_wins / len(mandela_results)
            metrics["mandela_myth_rate"] = myth_rate
            metrics["mandela_n"] = len(mandela_results)

            print(f"  Mandela: myth wins {myth_wins}/{len(mandela_results)} ({myth_rate:.1%})")

        all_metrics[size] = metrics
        unload_model()

    # ===================================================================
    # Summary Tables
    # ===================================================================
    sizes_done = [s for s in models if s in all_metrics]

    print("\n" + "=" * 70)
    print("CURATED MEDICAL PAIRS — SCALING SUMMARY")
    print("=" * 70)
    print(f"\n{'Size':<8} {'Params':<12} {'Win Rate':<10} {'AUC':<8} "
          f"{'p-value':<10} {'Cohen d':<10} {'Verdict':<20}")
    print("-" * 78)

    for size in sizes_done:
        m = all_metrics[size]
        params = PARAM_COUNTS[size]
        wr = m.get("curated_win_rate", 0)
        auc = m.get("curated_auc", 0.5)
        p = m.get("curated_p_value", 1)
        d = m.get("curated_cohens_d", 0)

        if wr > 0.8:
            verdict = "STRONG SENSOR"
        elif wr > 0.65:
            verdict = "PARTIAL SIGNAL"
        elif wr > 0.55:
            verdict = "WEAK SIGNAL"
        else:
            verdict = "NO SIGNAL"

        print(f"{size:<8} {params/1e6:>8.0f}M  {wr:<10.1%} {auc:<8.3f} "
              f"{p:<10.4f} {d:<10.3f} {verdict:<20}")

    if any(m.get("pubhealth_auc") for m in all_metrics.values()):
        print("\n" + "=" * 70)
        print("PUBHEALTH (REAL-WORLD CLAIMS) — SCALING SUMMARY")
        print("=" * 70)
        print(f"\n{'Size':<8} {'Params':<12} {'AUC':<8} {'True Mean':<12} "
              f"{'False Mean':<12} {'p-value':<10}")
        print("-" * 62)

        for size in sizes_done:
            m = all_metrics[size]
            if m.get("pubhealth_auc") is None:
                continue
            params = PARAM_COUNTS[size]
            print(f"{size:<8} {params/1e6:>8.0f}M  {m['pubhealth_auc']:<8.3f} "
                  f"{m.get('pubhealth_true_mean',0):<12.4f} "
                  f"{m.get('pubhealth_false_mean',0):<12.4f} "
                  f"{m.get('pubhealth_p_value',1):<10.4f}")

    if any(m.get("mandela_myth_rate") is not None for m in all_metrics.values()):
        print("\n" + "=" * 70)
        print("MEDICAL MANDELA EFFECT — SCALING SUMMARY")
        print("=" * 70)
        print(f"\n{'Size':<8} {'Params':<12} {'Myth Rate':<12} {'Verdict':<20}")
        print("-" * 52)

        for size in sizes_done:
            m = all_metrics[size]
            if m.get("mandela_myth_rate") is None:
                continue
            params = PARAM_COUNTS[size]
            mr = m["mandela_myth_rate"]
            verdict = ("MYTH WINS" if mr > 0.6
                       else "TRUTH WINS" if mr < 0.4
                       else "INCONCLUSIVE")
            print(f"{size:<8} {params/1e6:>8.0f}M  {mr:<12.1%} {verdict:<20}")

    # ===================================================================
    # Plots
    # ===================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    for size in sizes_done:
        # Curated pair deltas
        pair_results = run_curated_pairs(size, force=False)
        if pair_results:
            print(f"  Curated delta bars for {size}...")
            plot_pair_deltas(pair_results, size,
                             EXP9_FIGURES_DIR / f"medical_pairs_{size}.png")

        # PubHealth distributions
        claim_results = run_pubhealth(size, force=False)
        if claim_results:
            print(f"  PubHealth violin for {size}...")
            plot_pubhealth_distributions(claim_results, size,
                                         EXP9_FIGURES_DIR / f"pubhealth_dist_{size}.png")

        # Medical Mandela bars
        mandela_results = run_medical_mandela(size, force=False)
        if mandela_results:
            print(f"  Mandela bars for {size}...")
            plot_mandela_bars(mandela_results, size,
                              EXP9_FIGURES_DIR / f"medical_mandela_{size}.png")

    # Scaling curves
    if len(sizes_done) >= 2:
        print("  Scaling comparison curve...")
        plot_scaling_comparison(all_metrics,
                                EXP9_FIGURES_DIR / "medical_scaling.png")

    # ROC overlays
    if len(all_curated_roc) >= 2:
        print("  Curated ROC overlay...")
        plot_roc_overlay(all_curated_roc, "Medical Pairs — Truth Detection ROC",
                         EXP9_FIGURES_DIR / "medical_pairs_roc.png")

    if len(all_pubhealth_roc) >= 2:
        print("  PubHealth ROC overlay...")
        plot_roc_overlay(all_pubhealth_roc, "PubHealth — Truth Detection ROC",
                         EXP9_FIGURES_DIR / "pubhealth_roc.png")

    # ===================================================================
    # Final
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(EXP9_FIGURES_DIR.glob("*.png")))

    print("\n" + "=" * 70)
    print("EXPERIMENT 9 COMPLETE")
    print("=" * 70)
    print(f"  Models: {len(sizes_done)}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s")

    # Headline
    if sizes_done:
        largest = sizes_done[-1]
        m = all_metrics[largest]
        wr = m.get("curated_win_rate", 0)
        ph_auc = m.get("pubhealth_auc")

        print(f"\n  At {model_display_name(largest)}:")
        print(f"    Curated medical pairs: win rate = {wr:.1%}")
        if ph_auc:
            print(f"    PubHealth AUC: {ph_auc:.3f}")

        if wr > 0.8:
            print("\n  FINDING: SENSOR GENERALIZES to medical domain!")
        elif wr > 0.65:
            print("\n  FINDING: Partial generalization — signal exists but weaker.")
        elif wr > 0.55:
            print("\n  FINDING: Weak signal — marginally above chance.")
        else:
            print("\n  FINDING: Dead end — no generalization to medical claims.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all model sizes")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    parser.add_argument("--skip-pubhealth", action="store_true",
                        help="Skip PubHealth dataset (if datasets library unavailable)")
    args = parser.parse_args()

    run_experiment(SCALING_MODELS, force=args.force)
