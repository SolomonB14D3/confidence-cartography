"""
Experiment B9: RLHF Effect on Regime Classification
=====================================================
Does RLHF/instruction tuning systematically worsen regime 2 errors
and degrade the token-level uncertainty diagnostic?

Core hypothesis: RLHF suppresses the diagnostic signal without fixing
the underlying error. It makes models *harder to catch being wrong*.

Design:
  - Run Qwen2.5-7B base and Qwen2.5-7B-Instruct on the same item set
    used for Pythia 6.9B regime classification.
  - Compare confidence patterns split by regime (R1 vs R2).
  - Analyze token-level divergence, regime transitions, domain effects.

Predictions:
  P1: RLHF amplifies regime 2 confidence (more confidently wrong)
  P2: RLHF preserves or improves regime 1 (correct answer is preferred)
  P3: RLHF degrades the token-level diagnostic (R2 diagnostic worsens)
  P4: Some R1 items flip to R2 after RLHF (boundary expansion)
  P5: Cultural domain shows largest RLHF amplification (from B15)

Models:
  Base:     Qwen/Qwen2.5-7B           (cached in ~/.cache/huggingface/hub/)
  Instruct: Qwen/Qwen2.5-7B-Instruct  (cached in ~/.cache/huggingface/hub/)
"""

import sys
import os
import time
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import torch

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import analyze_fixed_text, unload_model
from src.schema import save_records, load_records, ConfidenceRecord

# Import item definitions
from src.experiments.exp2_truth import PAIRS as TRUTH_PAIRS
from src.experiments.exp9_medical_validation import MEDICAL_PAIRS
from src.experiments.exp_c_mandela import MANDELA_PAIRS
from src.experiments.exp_mandela_expanded import (
    LINGUISTIC_ITEMS, _make_texts,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Both models already cached in ~/.cache/huggingface/hub/ (~14GB each)
BASE_MODEL = "Qwen/Qwen2.5-7B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = torch.float16  # ~14GB on MPS; fits easily in 96GB M3 Ultra

# Optional second pair for cross-architecture comparison
# Both also already cached (6GB + 14GB)
MISTRAL_BASE = "mistralai/Mistral-7B-v0.3"
MISTRAL_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.3"

# Directories
BASE_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "qwen7b_base"
INSTRUCT_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "qwen7b_instruct"
COMPARISON_DIR = PROJECT_ROOT / "data" / "results" / "rlhf_comparison"
FIGURES_DIR = PROJECT_ROOT / "figures" / "rlhf_comparison"

for d in [BASE_RESULTS_DIR, INSTRUCT_RESULTS_DIR, COMPARISON_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Regime labels
REGIME_PATH = PROJECT_ROOT / "data" / "results" / "token_localization" / "regime_comparison.json"


# ===================================================================
# Phase 1: Run instruct model inference
# ===================================================================

def run_model_truth(model_name: str, results_dir: Path, label_prefix: str,
                    force: bool = False) -> list[ConfidenceRecord]:
    """Run truth pairs through a model."""
    output_path = results_dir / f"truth_{label_prefix}.jsonl"

    if output_path.exists() and not force:
        print(f"  [truth/{label_prefix}] Loading from cache...")
        return load_records(output_path)

    records = []
    print(f"\n  [truth/{label_prefix}] Analyzing {len(TRUTH_PAIRS)} pairs ({len(TRUTH_PAIRS)*2} texts)...")
    start = time.time()

    for pair in tqdm(TRUTH_PAIRS, desc=f"  truth/{label_prefix}", leave=False):
        true_rec = analyze_fixed_text(
            pair["true"], category="true", label=f"{pair['id']}_true",
            model_name=model_name, revision="main", dtype=DTYPE,
        )
        false_rec = analyze_fixed_text(
            pair["false"], category="false", label=f"{pair['id']}_false",
            model_name=model_name, revision="main", dtype=DTYPE,
        )
        records.extend([true_rec, false_rec])

    elapsed = time.time() - start
    print(f"  [truth/{label_prefix}] Done in {elapsed:.1f}s ({elapsed/len(TRUTH_PAIRS):.1f}s/pair)")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return records


def run_model_medical(model_name: str, results_dir: Path, label_prefix: str,
                      force: bool = False) -> list[ConfidenceRecord]:
    """Run medical pairs through a model."""
    output_path = results_dir / f"medical_{label_prefix}.jsonl"

    if output_path.exists() and not force:
        print(f"  [medical/{label_prefix}] Loading from cache...")
        return load_records(output_path)

    records = []
    print(f"\n  [medical/{label_prefix}] Analyzing {len(MEDICAL_PAIRS)} pairs ({len(MEDICAL_PAIRS)*2} texts)...")
    start = time.time()

    for pair in tqdm(MEDICAL_PAIRS, desc=f"  medical/{label_prefix}", leave=False):
        true_rec = analyze_fixed_text(
            pair["true"], category="medical_true",
            label=f"{pair['id']}_true",
            model_name=model_name, revision="main", dtype=DTYPE,
        )
        true_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                             "version": "true"}

        false_rec = analyze_fixed_text(
            pair["false"], category="medical_false",
            label=f"{pair['id']}_false",
            model_name=model_name, revision="main", dtype=DTYPE,
        )
        false_rec.metadata = {"pair_id": pair["id"], "domain": pair["domain"],
                              "version": "false"}

        records.extend([true_rec, false_rec])

    elapsed = time.time() - start
    print(f"  [medical/{label_prefix}] Done in {elapsed:.1f}s")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return records


def run_model_mandela(model_name: str, results_dir: Path, label_prefix: str,
                      force: bool = False) -> list[ConfidenceRecord]:
    """Run Mandela expanded items through a model."""
    output_path = results_dir / f"mandela_{label_prefix}.jsonl"

    if output_path.exists() and not force:
        print(f"  [mandela/{label_prefix}] Loading from cache...")
        return load_records(output_path)

    records = []
    n_texts = sum(len(_make_texts(item)) * 2 for item in LINGUISTIC_ITEMS)
    print(f"\n  [mandela/{label_prefix}] Analyzing {len(LINGUISTIC_ITEMS)} items ({n_texts} texts)...")
    start = time.time()

    for item in tqdm(LINGUISTIC_ITEMS, desc=f"  mandela/{label_prefix}", leave=False):
        for framing_name, wrong_text, correct_text in _make_texts(item):
            w_rec = analyze_fixed_text(
                wrong_text,
                category="mandela_wrong",
                label=f"{item['id']}_{framing_name}_wrong",
                model_name=model_name, revision="main", dtype=DTYPE,
            )
            w_rec.metadata = {
                "item_id": item["id"],
                "framing": framing_name,
                "version": "wrong",
                "human_ratio": item["human_ratio"],
            }

            c_rec = analyze_fixed_text(
                correct_text,
                category="mandela_correct",
                label=f"{item['id']}_{framing_name}_correct",
                model_name=model_name, revision="main", dtype=DTYPE,
            )
            c_rec.metadata = {
                "item_id": item["id"],
                "framing": framing_name,
                "version": "correct",
                "human_ratio": item["human_ratio"],
            }

            records.extend([w_rec, c_rec])

    elapsed = time.time() - start
    print(f"  [mandela/{label_prefix}] Done in {elapsed:.1f}s ({len(records)} records)")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return records


def run_model_mandela_original(model_name: str, results_dir: Path, label_prefix: str,
                               force: bool = False) -> list[ConfidenceRecord]:
    """Run original Mandela pairs (exp_c) through a model.

    These are separate from the expanded set — they use a different
    sentence structure. We include them because some regime labels
    reference 'mandela_orig' source.
    """
    output_path = results_dir / f"mandela_orig_{label_prefix}.jsonl"

    if output_path.exists() and not force:
        print(f"  [mandela_orig/{label_prefix}] Loading from cache...")
        return load_records(output_path)

    records = []
    print(f"\n  [mandela_orig/{label_prefix}] Analyzing {len(MANDELA_PAIRS)} original Mandela pairs...")
    start = time.time()

    for pair in tqdm(MANDELA_PAIRS, desc=f"  mandela_orig/{label_prefix}", leave=False):
        wrong_rec = analyze_fixed_text(
            pair["popular"],
            category="mandela_orig_wrong",
            label=f"mandela_orig_{pair['id']}_wrong",
            model_name=model_name, revision="main", dtype=DTYPE,
        )
        wrong_rec.metadata = {"pair_id": pair["id"], "version": "wrong"}

        correct_rec = analyze_fixed_text(
            pair["correct"],
            category="mandela_orig_correct",
            label=f"mandela_orig_{pair['id']}_correct",
            model_name=model_name, revision="main", dtype=DTYPE,
        )
        correct_rec.metadata = {"pair_id": pair["id"], "version": "correct"}

        records.extend([wrong_rec, correct_rec])

    elapsed = time.time() - start
    print(f"  [mandela_orig/{label_prefix}] Done in {elapsed:.1f}s")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)
    return records


def run_all_inference(force: bool = False):
    """Run all item sets through BOTH base and instruct models."""
    # --- Base model ---
    print("=" * 70)
    print("PHASE 1a: BASE MODEL INFERENCE")
    print("=" * 70)
    print(f"Model: {BASE_MODEL}")
    print(f"Dtype: {DTYPE}")
    print()

    run_model_truth(BASE_MODEL, BASE_RESULTS_DIR, "base", force=force)
    run_model_medical(BASE_MODEL, BASE_RESULTS_DIR, "base", force=force)
    run_model_mandela(BASE_MODEL, BASE_RESULTS_DIR, "base", force=force)
    run_model_mandela_original(BASE_MODEL, BASE_RESULTS_DIR, "base", force=force)
    unload_model()

    # --- Instruct model ---
    print("\n" + "=" * 70)
    print("PHASE 1b: INSTRUCT MODEL INFERENCE")
    print("=" * 70)
    print(f"Model: {INSTRUCT_MODEL}")
    print(f"Dtype: {DTYPE}")
    print()

    run_model_truth(INSTRUCT_MODEL, INSTRUCT_RESULTS_DIR, "instruct", force=force)
    run_model_medical(INSTRUCT_MODEL, INSTRUCT_RESULTS_DIR, "instruct", force=force)
    run_model_mandela(INSTRUCT_MODEL, INSTRUCT_RESULTS_DIR, "instruct", force=force)
    run_model_mandela_original(INSTRUCT_MODEL, INSTRUCT_RESULTS_DIR, "instruct", force=force)
    unload_model()

    print(f"\n  Inference complete for both models.")


# ===================================================================
# Phase 2: Load all data and build comparison structures
# ===================================================================

def build_pair_lookup(records: list[ConfidenceRecord], source_type: str) -> dict:
    """Build {pair_id: {"true": record, "false": record}} from records.

    Handles different labeling conventions across item types.
    """
    by_id = defaultdict(dict)

    for r in records:
        if source_type == "truth":
            # Labels like "france_capital_true" / "france_capital_false"
            base_id = r.label.rsplit("_", 1)[0]
            version = "true" if r.category == "true" else "false"
            by_id[base_id][version] = r

        elif source_type == "medical":
            pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
            version = r.metadata.get("version", "true" if "true" in r.label else "false")
            by_id[pid][version] = r

        elif source_type == "mandela_expanded":
            item_id = r.metadata.get("item_id", "")
            framing = r.metadata.get("framing", "raw")
            version = r.metadata.get("version", "wrong" if "wrong" in r.label else "correct")
            key = f"{item_id}_{framing}"
            # Map wrong→false, correct→true for consistency
            mapped = "false" if version == "wrong" else "true"
            by_id[key][mapped] = r

        elif source_type == "mandela_orig":
            pid = r.metadata.get("pair_id", "")
            version = r.metadata.get("version", "wrong" if "wrong" in r.label else "correct")
            mapped = "false" if version == "wrong" else "true"
            by_id[pid][mapped] = r

    return dict(by_id)


def load_all_data():
    """Load base and instruct results, build comparison structures."""
    print("\n" + "=" * 70)
    print("LOADING DATA FOR COMPARISON")
    print("=" * 70)

    # --- Load base model results ---
    base_truth = load_records(BASE_RESULTS_DIR / "truth_base.jsonl")
    base_medical = load_records(BASE_RESULTS_DIR / "medical_base.jsonl")
    base_mandela = load_records(BASE_RESULTS_DIR / "mandela_base.jsonl")

    base_mandela_orig_path = BASE_RESULTS_DIR / "mandela_orig_base.jsonl"
    if base_mandela_orig_path.exists():
        base_mandela_orig = load_records(base_mandela_orig_path)
    else:
        base_mandela_orig = []

    # --- Load instruct model results ---
    inst_truth = load_records(INSTRUCT_RESULTS_DIR / "truth_instruct.jsonl")
    inst_medical = load_records(INSTRUCT_RESULTS_DIR / "medical_instruct.jsonl")
    inst_mandela = load_records(INSTRUCT_RESULTS_DIR / "mandela_instruct.jsonl")

    inst_mandela_orig_path = INSTRUCT_RESULTS_DIR / "mandela_orig_instruct.jsonl"
    if inst_mandela_orig_path.exists():
        inst_mandela_orig = load_records(inst_mandela_orig_path)
    else:
        inst_mandela_orig = []

    # --- Build pair lookups ---
    base_pairs = {}
    base_pairs.update(build_pair_lookup(base_truth, "truth"))
    base_pairs.update(build_pair_lookup(base_medical, "medical"))
    base_pairs.update(build_pair_lookup(base_mandela, "mandela_expanded"))
    if base_mandela_orig:
        base_pairs.update(build_pair_lookup(base_mandela_orig, "mandela_orig"))

    inst_pairs = {}
    inst_pairs.update(build_pair_lookup(inst_truth, "truth"))
    inst_pairs.update(build_pair_lookup(inst_medical, "medical"))
    inst_pairs.update(build_pair_lookup(inst_mandela, "mandela_expanded"))
    if inst_mandela_orig:
        inst_pairs.update(build_pair_lookup(inst_mandela_orig, "mandela_orig"))

    # --- Load regime labels ---
    with open(REGIME_PATH) as f:
        regime_data = json.load(f)

    r1_items = regime_data["regime1_6.9b"]  # 65 items
    r2_items = regime_data["regime2_6.9b"]  # 34 items

    # Build regime lookup: pair_id → regime
    regime_lookup = {}
    r1_pair_ids = set()
    r2_pair_ids = set()

    for item in r1_items:
        regime_lookup[item["pair_id"]] = 1
        r1_pair_ids.add(item["pair_id"])

    for item in r2_items:
        regime_lookup[item["pair_id"]] = 2
        r2_pair_ids.add(item["pair_id"])

    print(f"  Base model pairs: {len(base_pairs)}")
    print(f"  Instruct model pairs: {len(inst_pairs)}")
    print(f"  Regime labels: {len(r1_pair_ids)} R1, {len(r2_pair_ids)} R2")

    return base_pairs, inst_pairs, regime_lookup, r1_pair_ids, r2_pair_ids, r1_items, r2_items


def match_regime_items(base_pairs, inst_pairs, regime_lookup, r1_pair_ids, r2_pair_ids):
    """Match regime-labeled items to both model results.

    The tricky part: regime pair_ids may not exactly match the pair lookup
    keys (e.g., "france_capital" vs "france_capital", or
    "star_wars_raw" vs "star_wars_raw").

    Returns lists of matched items with both base and instruct data.
    """
    matched = []
    missing_base = []
    missing_inst = []

    all_regime_ids = list(r1_pair_ids | r2_pair_ids)

    for pair_id in all_regime_ids:
        regime = regime_lookup[pair_id]

        # Try exact match first, then with prefixes
        b_pair = base_pairs.get(pair_id)
        i_pair = inst_pairs.get(pair_id)

        if b_pair is None:
            missing_base.append(pair_id)
            continue
        if i_pair is None:
            missing_inst.append(pair_id)
            continue

        if "true" not in b_pair or "false" not in b_pair:
            missing_base.append(pair_id)
            continue
        if "true" not in i_pair or "false" not in i_pair:
            missing_inst.append(pair_id)
            continue

        matched.append({
            "pair_id": pair_id,
            "regime": regime,
            "base_true": b_pair["true"],
            "base_false": b_pair["false"],
            "inst_true": i_pair["true"],
            "inst_false": i_pair["false"],
        })

    if missing_base:
        print(f"\n  WARNING: {len(missing_base)} regime items not found in base results:")
        for pid in missing_base[:10]:
            print(f"    - {pid}")
    if missing_inst:
        print(f"\n  WARNING: {len(missing_inst)} regime items not found in instruct results:")
        for pid in missing_inst[:10]:
            print(f"    - {pid}")

    n_r1 = sum(1 for m in matched if m["regime"] == 1)
    n_r2 = sum(1 for m in matched if m["regime"] == 2)
    print(f"\n  Matched items: {len(matched)} ({n_r1} R1, {n_r2} R2)")

    return matched


# ===================================================================
# Phase 3: Analysis
# ===================================================================

def compute_confidence_delta(true_rec: ConfidenceRecord, false_rec: ConfidenceRecord) -> float:
    """Compute confidence delta: true - false. Positive = model prefers truth."""
    return true_rec.mean_top1_prob - false_rec.mean_top1_prob


def compute_divergence_point(true_rec: ConfidenceRecord, false_rec: ConfidenceRecord) -> dict:
    """Find where true/false versions diverge and compare token-level confidence.

    Returns divergence point analysis dict.
    """
    true_tokens = true_rec.tokens
    false_tokens = false_rec.tokens

    min_len = min(len(true_tokens), len(false_tokens))
    if min_len == 0:
        return None

    # Find divergence point: first position where token IDs differ
    div_point = None
    for i in range(min_len):
        if true_tokens[i].token_id != false_tokens[i].token_id:
            div_point = i
            break

    if div_point is None:
        # Texts may differ only in length or not at all
        return None

    # At divergence point, compare confidence
    # In the TRUE text, the model's confidence for the true token
    true_conf_at_div = true_tokens[div_point].top1_prob
    # In the FALSE text, the model's confidence for the false token
    false_conf_at_div = false_tokens[div_point].top1_prob

    return {
        "divergence_point": div_point,
        "true_token": true_tokens[div_point].token_str,
        "false_token": false_tokens[div_point].token_str,
        "true_conf": true_conf_at_div,
        "false_conf": false_conf_at_div,
        "true_wins": true_conf_at_div > false_conf_at_div,
        "margin": true_conf_at_div - false_conf_at_div,
    }


def analyze_overall_confidence(matched_items):
    """Test P1 and P2: Compare overall confidence deltas by regime."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: OVERALL CONFIDENCE (P1 & P2)")
    print("=" * 70)

    r1_base_deltas = []
    r1_inst_deltas = []
    r2_base_deltas = []
    r2_inst_deltas = []

    for item in matched_items:
        base_delta = compute_confidence_delta(item["base_true"], item["base_false"])
        inst_delta = compute_confidence_delta(item["inst_true"], item["inst_false"])

        if item["regime"] == 1:
            r1_base_deltas.append(base_delta)
            r1_inst_deltas.append(inst_delta)
        else:
            r2_base_deltas.append(base_delta)
            r2_inst_deltas.append(inst_delta)

    r1_base = np.array(r1_base_deltas)
    r1_inst = np.array(r1_inst_deltas)
    r2_base = np.array(r2_base_deltas)
    r2_inst = np.array(r2_inst_deltas)

    # P1: RLHF amplifies R2 confidence (makes R2 deltas more negative)
    print(f"\n  --- P1: RLHF effect on Regime 2 ---")
    print(f"  Base R2 mean delta:    {r2_base.mean():+.6f}")
    print(f"  Instruct R2 mean delta: {r2_inst.mean():+.6f}")
    print(f"  Shift: {r2_inst.mean() - r2_base.mean():+.6f}")

    if len(r2_base) >= 5:
        stat, p = stats.wilcoxon(r2_base, r2_inst)
        print(f"  Wilcoxon: stat={stat:.1f}, p={p:.6f}")
    else:
        stat, p = 0, 1
        print(f"  Not enough R2 items for Wilcoxon test")

    r2_base_wins = int(np.sum(r2_base > 0))
    r2_inst_wins = int(np.sum(r2_inst > 0))
    print(f"  Base R2 'truth wins': {r2_base_wins}/{len(r2_base)} ({r2_base_wins/len(r2_base):.1%})")
    print(f"  Inst R2 'truth wins': {r2_inst_wins}/{len(r2_inst)} ({r2_inst_wins/len(r2_inst):.1%})")

    # P2: RLHF preserves R1
    print(f"\n  --- P2: RLHF effect on Regime 1 ---")
    print(f"  Base R1 mean delta:    {r1_base.mean():+.6f}")
    print(f"  Instruct R1 mean delta: {r1_inst.mean():+.6f}")
    print(f"  Shift: {r1_inst.mean() - r1_base.mean():+.6f}")

    if len(r1_base) >= 5:
        stat1, p1 = stats.wilcoxon(r1_base, r1_inst)
        print(f"  Wilcoxon: stat={stat1:.1f}, p={p1:.6f}")
    else:
        stat1, p1 = 0, 1

    r1_base_wins = int(np.sum(r1_base > 0))
    r1_inst_wins = int(np.sum(r1_inst > 0))
    print(f"  Base R1 'truth wins': {r1_base_wins}/{len(r1_base)} ({r1_base_wins/len(r1_base):.1%})")
    print(f"  Inst R1 'truth wins': {r1_inst_wins}/{len(r1_inst)} ({r1_inst_wins/len(r1_inst):.1%})")

    # Also compute confidence on FALSE versions specifically
    r2_base_false_conf = np.array([item["base_false"].mean_top1_prob for item in matched_items if item["regime"] == 2])
    r2_inst_false_conf = np.array([item["inst_false"].mean_top1_prob for item in matched_items if item["regime"] == 2])

    print(f"\n  --- R2 false-version confidence (raw) ---")
    print(f"  Base: {r2_base_false_conf.mean():.6f}")
    print(f"  Inst: {r2_inst_false_conf.mean():.6f}")
    print(f"  Shift: {r2_inst_false_conf.mean() - r2_base_false_conf.mean():+.6f}")

    results = {
        "r1_base_mean_delta": float(r1_base.mean()),
        "r1_inst_mean_delta": float(r1_inst.mean()),
        "r1_shift": float(r1_inst.mean() - r1_base.mean()),
        "r1_base_win_rate": float(r1_base_wins / len(r1_base)),
        "r1_inst_win_rate": float(r1_inst_wins / len(r1_inst)),
        "r1_wilcoxon_p": float(p1),
        "r2_base_mean_delta": float(r2_base.mean()),
        "r2_inst_mean_delta": float(r2_inst.mean()),
        "r2_shift": float(r2_inst.mean() - r2_base.mean()),
        "r2_base_win_rate": float(r2_base_wins / len(r2_base)),
        "r2_inst_win_rate": float(r2_inst_wins / len(r2_inst)),
        "r2_wilcoxon_p": float(p),
        "r2_base_false_conf": float(r2_base_false_conf.mean()),
        "r2_inst_false_conf": float(r2_inst_false_conf.mean()),
        "n_r1": len(r1_base),
        "n_r2": len(r2_base),
    }

    return results, r1_base, r1_inst, r2_base, r2_inst


def analyze_divergence_points(matched_items):
    """Test P3: Compare token-level divergence analysis by regime."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: TOKEN-LEVEL DIVERGENCE (P3)")
    print("=" * 70)

    r1_results = []
    r2_results = []

    for item in matched_items:
        # Base model divergence
        base_div = compute_divergence_point(item["base_true"], item["base_false"])
        # Instruct model divergence
        inst_div = compute_divergence_point(item["inst_true"], item["inst_false"])

        if base_div is None or inst_div is None:
            continue

        entry = {
            "pair_id": item["pair_id"],
            "regime": item["regime"],
            "base": base_div,
            "inst": inst_div,
        }

        if item["regime"] == 1:
            r1_results.append(entry)
        else:
            r2_results.append(entry)

    # R1 win rates
    r1_base_wins = sum(1 for r in r1_results if r["base"]["true_wins"])
    r1_inst_wins = sum(1 for r in r1_results if r["inst"]["true_wins"])

    print(f"\n  --- R1 divergence-point win rate ---")
    print(f"  Base: {r1_base_wins}/{len(r1_results)} ({r1_base_wins/len(r1_results):.1%})")
    print(f"  Inst: {r1_inst_wins}/{len(r1_results)} ({r1_inst_wins/len(r1_results):.1%})")

    # R2 win rates
    r2_base_wins = sum(1 for r in r2_results if r["base"]["true_wins"])
    r2_inst_wins = sum(1 for r in r2_results if r["inst"]["true_wins"])

    print(f"\n  --- R2 divergence-point win rate ---")
    print(f"  Base: {r2_base_wins}/{len(r2_results)} ({r2_base_wins/len(r2_results):.1%})")
    print(f"  Inst: {r2_inst_wins}/{len(r2_results)} ({r2_inst_wins/len(r2_results):.1%})")

    # Margin comparison
    r1_base_margins = [r["base"]["margin"] for r in r1_results]
    r1_inst_margins = [r["inst"]["margin"] for r in r1_results]
    r2_base_margins = [r["base"]["margin"] for r in r2_results]
    r2_inst_margins = [r["inst"]["margin"] for r in r2_results]

    print(f"\n  --- R1 margin (true_conf - false_conf at div point) ---")
    print(f"  Base mean: {np.mean(r1_base_margins):+.6f}")
    print(f"  Inst mean: {np.mean(r1_inst_margins):+.6f}")

    print(f"\n  --- R2 margin ---")
    print(f"  Base mean: {np.mean(r2_base_margins):+.6f}")
    print(f"  Inst mean: {np.mean(r2_inst_margins):+.6f}")

    # Diagnostic separation: difference between R1 and R2 win rates
    r1_wr_base = r1_base_wins / len(r1_results)
    r1_wr_inst = r1_inst_wins / len(r1_results)
    r2_wr_base = r2_base_wins / len(r2_results) if r2_results else 0
    r2_wr_inst = r2_inst_wins / len(r2_results) if r2_results else 0

    print(f"\n  --- Diagnostic separation ---")
    print(f"  Base: R1 win rate - R2 win rate = {r1_wr_base - r2_wr_base:+.3f}")
    print(f"  Inst: R1 win rate - R2 win rate = {r1_wr_inst - r2_wr_inst:+.3f}")
    prediction = "WIDENED" if (r1_wr_inst - r2_wr_inst) > (r1_wr_base - r2_wr_base) else "NARROWED"
    print(f"  Prediction (should WIDEN): {prediction}")

    results = {
        "r1_base_win_rate": float(r1_wr_base),
        "r1_inst_win_rate": float(r1_wr_inst),
        "r2_base_win_rate": float(r2_wr_base),
        "r2_inst_win_rate": float(r2_wr_inst),
        "r1_base_mean_margin": float(np.mean(r1_base_margins)),
        "r1_inst_mean_margin": float(np.mean(r1_inst_margins)),
        "r2_base_mean_margin": float(np.mean(r2_base_margins)),
        "r2_inst_mean_margin": float(np.mean(r2_inst_margins)),
        "base_diagnostic_separation": float(r1_wr_base - r2_wr_base),
        "inst_diagnostic_separation": float(r1_wr_inst - r2_wr_inst),
        "separation_change": prediction,
        "n_r1": len(r1_results),
        "n_r2": len(r2_results),
    }

    return results, r1_results, r2_results


def analyze_regime_transitions(matched_items):
    """Test P4: Do items change regime after RLHF?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: REGIME TRANSITIONS (P4)")
    print("=" * 70)

    transitions = Counter()
    transition_items = []

    for item in matched_items:
        base_div = compute_divergence_point(item["base_true"], item["base_false"])
        inst_div = compute_divergence_point(item["inst_true"], item["inst_false"])

        if base_div is None or inst_div is None:
            continue

        # Classify regime by divergence behavior in each model
        base_regime = "R1" if base_div["true_wins"] else "R2"
        inst_regime = "R1" if inst_div["true_wins"] else "R2"
        orig_regime = f"R{item['regime']}"  # From Pythia 6.9B labels

        transition = f"{base_regime}→{inst_regime}"
        transitions[transition] += 1

        if base_regime != inst_regime:
            transition_items.append({
                "pair_id": item["pair_id"],
                "original_regime": orig_regime,
                "base_regime": base_regime,
                "inst_regime": inst_regime,
                "base_margin": base_div["margin"],
                "inst_margin": inst_div["margin"],
            })

    print(f"\n  Transition matrix (Qwen base → Qwen instruct):")
    for trans, count in sorted(transitions.items()):
        print(f"    {trans}: {count}")

    r1_to_r2 = transitions.get("R1→R2", 0)
    r2_to_r1 = transitions.get("R2→R1", 0)
    total = sum(transitions.values())

    print(f"\n  R1→R2 (RLHF broke it): {r1_to_r2}")
    print(f"  R2→R1 (RLHF fixed it): {r2_to_r1}")
    print(f"  Asymmetry: {r1_to_r2 - r2_to_r1} net items moved to R2")

    if r1_to_r2 + r2_to_r1 > 0:
        # Fisher's exact test on asymmetry
        table = np.array([
            [transitions.get("R1→R1", 0), r1_to_r2],
            [r2_to_r1, transitions.get("R2→R2", 0)],
        ])
        odds, fisher_p = stats.fisher_exact(table)
        print(f"  Fisher's exact: OR={odds:.3f}, p={fisher_p:.6f}")
    else:
        fisher_p = 1.0

    if transition_items:
        print(f"\n  Items that changed regime:")
        for t in transition_items:
            print(f"    {t['pair_id']}: {t['base_regime']}→{t['inst_regime']} "
                  f"(base margin={t['base_margin']:+.4f}, inst margin={t['inst_margin']:+.4f})")

    results = {
        "transitions": dict(transitions),
        "r1_to_r2": r1_to_r2,
        "r2_to_r1": r2_to_r1,
        "net_to_r2": r1_to_r2 - r2_to_r1,
        "fisher_p": float(fisher_p),
        "total_items": total,
        "transition_items": transition_items,
    }

    return results


def analyze_domain_interaction(matched_items, r1_items, r2_items):
    """Test P5: Domain-specific RLHF effects (connects to B15)."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: DOMAIN INTERACTION (P5)")
    print("=" * 70)

    # Build domain lookup from regime items
    domain_lookup = {}
    for item in r1_items + r2_items:
        pid = item["pair_id"]
        source = item.get("source", "unknown")
        if "truth" in source:
            domain_lookup[pid] = "general_facts"
        elif "medical" in source:
            domain_lookup[pid] = "medical"
        elif "mandela_orig" in source:
            domain_lookup[pid] = "cultural"
        elif "mandela_exp" in source:
            domain_lookup[pid] = "cultural"
        else:
            domain_lookup[pid] = "other"

    # Group matched items by domain
    by_domain = defaultdict(list)
    for item in matched_items:
        domain = domain_lookup.get(item["pair_id"], "unknown")
        base_delta = compute_confidence_delta(item["base_true"], item["base_false"])
        inst_delta = compute_confidence_delta(item["inst_true"], item["inst_false"])
        by_domain[domain].append({
            "pair_id": item["pair_id"],
            "regime": item["regime"],
            "base_delta": base_delta,
            "inst_delta": inst_delta,
            "shift": inst_delta - base_delta,
        })

    results = {}
    for domain in sorted(by_domain.keys()):
        items = by_domain[domain]
        shifts = [it["shift"] for it in items]
        base_deltas = [it["base_delta"] for it in items]
        inst_deltas = [it["inst_delta"] for it in items]

        n_r1 = sum(1 for it in items if it["regime"] == 1)
        n_r2 = sum(1 for it in items if it["regime"] == 2)

        mean_shift = np.mean(shifts)
        print(f"\n  {domain} (n={len(items)}, R1={n_r1}, R2={n_r2}):")
        print(f"    Base mean delta:    {np.mean(base_deltas):+.6f}")
        print(f"    Instruct mean delta: {np.mean(inst_deltas):+.6f}")
        print(f"    Mean shift:          {mean_shift:+.6f}")

        if len(items) >= 5:
            stat, p = stats.wilcoxon(base_deltas, inst_deltas)
            print(f"    Wilcoxon: stat={stat:.1f}, p={p:.6f}")
        else:
            p = 1.0

        results[domain] = {
            "n": len(items),
            "n_r1": n_r1,
            "n_r2": n_r2,
            "base_mean_delta": float(np.mean(base_deltas)),
            "inst_mean_delta": float(np.mean(inst_deltas)),
            "mean_shift": float(mean_shift),
            "wilcoxon_p": float(p),
        }

    return results


# ===================================================================
# Phase 4: Visualization
# ===================================================================

def plot_confidence_shift_by_regime(r1_base, r1_inst, r2_base, r2_inst, overall_results):
    """Paired plot: base vs instruct confidence deltas, split by regime."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # R1 panel
    ax = axes[0]
    ax.scatter(r1_base, r1_inst, color="#2196F3", alpha=0.6, s=60, edgecolors="white", linewidths=0.5)
    lim = max(abs(r1_base).max(), abs(r1_inst).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="No change")
    ax.axhline(0, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)
    ax.set_xlabel("Base model Δ (true − false)", fontsize=11)
    ax.set_ylabel("Instruct model Δ (true − false)", fontsize=11)
    ax.set_title(f"Regime 1 (n={len(r1_base)})\nScaling-reducible", fontsize=12)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Add annotation
    r1_wr_base = overall_results["r1_base_win_rate"]
    r1_wr_inst = overall_results["r1_inst_win_rate"]
    ax.text(0.05, 0.95,
            f"Base win rate: {r1_wr_base:.1%}\n"
            f"Instruct win rate: {r1_wr_inst:.1%}\n"
            f"Shift: {overall_results['r1_shift']:+.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#BBDEFB", alpha=0.7))

    # R2 panel
    ax = axes[1]
    ax.scatter(r2_base, r2_inst, color="#F44336", alpha=0.6, s=60, edgecolors="white", linewidths=0.5)
    lim2 = max(abs(r2_base).max(), abs(r2_inst).max()) * 1.2
    ax.plot([-lim2, lim2], [-lim2, lim2], "k--", alpha=0.3, label="No change")
    ax.axhline(0, color="gray", alpha=0.3)
    ax.axvline(0, color="gray", alpha=0.3)
    ax.set_xlabel("Base model Δ (true − false)", fontsize=11)
    ax.set_ylabel("Instruct model Δ (true − false)", fontsize=11)
    ax.set_title(f"Regime 2 (n={len(r2_base)})\nScaling-irreducible", fontsize=12)
    ax.set_xlim(-lim2, lim2)
    ax.set_ylim(-lim2, lim2)

    r2_wr_base = overall_results["r2_base_win_rate"]
    r2_wr_inst = overall_results["r2_inst_win_rate"]
    ax.text(0.05, 0.95,
            f"Base win rate: {r2_wr_base:.1%}\n"
            f"Instruct win rate: {r2_wr_inst:.1%}\n"
            f"Shift: {overall_results['r2_shift']:+.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFCDD2", alpha=0.7))

    fig.suptitle("B9: RLHF Effect on Confidence Deltas by Regime\n"
                 f"{BASE_MODEL} vs {INSTRUCT_MODEL}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confidence_shift_by_regime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: confidence_shift_by_regime.png")


def plot_divergence_winrate(div_results):
    """Bar chart: win rates at divergence point, base vs instruct."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["R1\n(scaling-reducible)", "R2\n(scaling-irreducible)"]
    base_vals = [div_results["r1_base_win_rate"], div_results["r2_base_win_rate"]]
    inst_vals = [div_results["r1_inst_win_rate"], div_results["r2_inst_win_rate"]]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_vals, width, label=f"Base ({BASE_MODEL})",
                   color="#2196F3", alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width/2, inst_vals, width, label=f"Instruct ({INSTRUCT_MODEL})",
                   color="#F44336", alpha=0.8, edgecolor="white")

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylabel("Truth Win Rate at Divergence Point", fontsize=12)
    ax.set_title(f"B9: Token-Level Diagnostic — Base vs Instruct\n"
                 f"{BASE_MODEL} — Win rate = P(true token more confident at divergence)",
                 fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                f"{h:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                f"{h:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "divergence_winrate_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: divergence_winrate_comparison.png")


def plot_regime_transitions(trans_results):
    """Transition matrix heatmap: base regime → instruct regime."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 6))

    transitions = trans_results["transitions"]
    matrix = np.array([
        [transitions.get("R1→R1", 0), transitions.get("R1→R2", 0)],
        [transitions.get("R2→R1", 0), transitions.get("R2→R2", 0)],
    ])

    # Heatmap
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["R1\n(gets it right)", "R2\n(confidently wrong)"], fontsize=11)
    ax.set_yticklabels(["R1\n(gets it right)", "R2\n(confidently wrong)"], fontsize=11)
    ax.set_xlabel("Instruct Model Regime", fontsize=12)
    ax.set_ylabel("Base Model Regime", fontsize=12)

    # Cell annotations
    for i in range(2):
        for j in range(2):
            total = matrix.sum()
            pct = matrix[i, j] / total * 100 if total > 0 else 0
            color = "white" if matrix[i, j] > matrix.max() * 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    ax.set_title(f"B9: Regime Transitions ({BASE_MODEL} → {INSTRUCT_MODEL})\n"
                 f"R1→R2: {trans_results['r1_to_r2']} items, "
                 f"R2→R1: {trans_results['r2_to_r1']} items "
                 f"(net: {trans_results['net_to_r2']:+d} to R2)",
                 fontsize=11)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regime_transition_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: regime_transition_matrix.png")


def plot_domain_effects(domain_results):
    """Effect of RLHF by domain category."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6))

    domains = sorted(domain_results.keys())
    base_deltas = [domain_results[d]["base_mean_delta"] for d in domains]
    inst_deltas = [domain_results[d]["inst_mean_delta"] for d in domains]
    ns = [domain_results[d]["n"] for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_deltas, width, label="Base",
                   color="#2196F3", alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width/2, inst_deltas, width, label="Instruct",
                   color="#F44336", alpha=0.8, edgecolor="white")

    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.set_ylabel("Mean Confidence Delta\n(positive = model prefers truth)", fontsize=11)
    ax.set_title(f"B9: RLHF Effect by Domain\n"
                 f"{BASE_MODEL} vs {INSTRUCT_MODEL}", fontsize=12)
    domain_labels = [f"{d}\n(n={n})" for d, n in zip(domains, ns)]
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontsize=10)
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "domain_effect_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: domain_effect_sizes.png")


def plot_example_items(matched_items):
    """Show 4 example items with token-level confidence before/after RLHF."""
    sns.set_theme(style="whitegrid", palette="muted")

    # Pick 2 R1 items and 2 R2 items with interesting behavior
    r1_items = [m for m in matched_items if m["regime"] == 1]
    r2_items = [m for m in matched_items if m["regime"] == 2]

    # Sort by magnitude of shift
    def shift_magnitude(item):
        bd = compute_confidence_delta(item["base_true"], item["base_false"])
        id_ = compute_confidence_delta(item["inst_true"], item["inst_false"])
        return abs(id_ - bd)

    r1_sorted = sorted(r1_items, key=shift_magnitude, reverse=True)
    r2_sorted = sorted(r2_items, key=shift_magnitude, reverse=True)

    examples = r1_sorted[:2] + r2_sorted[:2]

    if len(examples) < 4:
        examples = (r1_sorted + r2_sorted)[:4]

    if len(examples) == 0:
        return

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for idx, item in enumerate(examples):
        regime = item["regime"]
        pair_id = item["pair_id"]

        for col, (model_name, true_rec, false_rec) in enumerate([
            ("Base", item["base_true"], item["base_false"]),
            ("Instruct", item["inst_true"], item["inst_false"]),
        ]):
            ax = axes[idx, col]

            # Plot per-token confidence for true and false versions
            true_probs = [t.top1_prob for t in true_rec.tokens]
            false_probs = [t.top1_prob for t in false_rec.tokens]

            min_len = min(len(true_probs), len(false_probs))
            positions = range(min_len)

            ax.plot(positions, true_probs[:min_len], "g-o", markersize=3,
                    alpha=0.7, label="True version")
            ax.plot(positions, false_probs[:min_len], "r-o", markersize=3,
                    alpha=0.7, label="False version")

            ax.set_title(f"{model_name} — {pair_id} (R{regime})", fontsize=10)
            ax.set_xlabel("Token position", fontsize=9)
            ax.set_ylabel("Token probability", fontsize=9)
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1)

    fig.suptitle("B9: Example Items — Token-Level Confidence\n"
                 "Top 2 rows: R1 (scaling-reducible), Bottom 2 rows: R2 (scaling-irreducible)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "example_items.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: example_items.png")


# ===================================================================
# Phase 5: Summary and interpretation
# ===================================================================

def print_summary(overall, divergence, transitions, domain):
    """Print interpretation table."""
    print("\n" + "=" * 70)
    print("SUMMARY: B9 RESULTS")
    print("=" * 70)

    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │ PREDICTION RESULTS                                             │")
    print("  ├─────────────────────────────────────────────────────────────────┤")

    # P1: RLHF amplifies R2 confidence
    r2_shift = overall["r2_shift"]
    p1_verdict = "CONFIRMED" if r2_shift < -0.001 else ("WEAK" if r2_shift < 0 else "REFUTED")
    print(f"  │ P1: RLHF amplifies R2 errors                                  │")
    print(f"  │     R2 delta shift: {r2_shift:+.6f}  [{p1_verdict}]              │")
    print(f"  │     R2 win rate: {overall['r2_base_win_rate']:.1%} → {overall['r2_inst_win_rate']:.1%}                              │")

    # P2: RLHF preserves R1
    r1_shift = overall["r1_shift"]
    p2_verdict = "CONFIRMED" if overall["r1_inst_win_rate"] >= 0.8 * overall["r1_base_win_rate"] else "REFUTED"
    print(f"  │ P2: RLHF preserves R1                                         │")
    print(f"  │     R1 delta shift: {r1_shift:+.6f}  [{p2_verdict}]              │")
    print(f"  │     R1 win rate: {overall['r1_base_win_rate']:.1%} → {overall['r1_inst_win_rate']:.1%}                              │")

    # P3: Diagnostic separation
    sep_change = divergence["inst_diagnostic_separation"] - divergence["base_diagnostic_separation"]
    p3_verdict = "CONFIRMED" if sep_change > 0 else "REFUTED"
    print(f"  │ P3: Diagnostic separation widens                              │")
    print(f"  │     Base sep: {divergence['base_diagnostic_separation']:+.3f}  Inst sep: {divergence['inst_diagnostic_separation']:+.3f}  [{p3_verdict}]│")

    # P4: Regime transitions
    p4_verdict = "CONFIRMED" if transitions["r1_to_r2"] > transitions["r2_to_r1"] else "REFUTED"
    print(f"  │ P4: RLHF expands R2 boundary                                 │")
    print(f"  │     R1→R2: {transitions['r1_to_r2']}, R2→R1: {transitions['r2_to_r1']}  [{p4_verdict}]                       │")

    # P5: Domain effects
    cultural_shift = domain.get("cultural", {}).get("mean_shift", 0)
    medical_shift = domain.get("medical", {}).get("mean_shift", 0)
    general_shift = domain.get("general_facts", {}).get("mean_shift", 0)
    p5_verdict = "CONFIRMED" if abs(cultural_shift) > abs(medical_shift) else "REFUTED"
    print(f"  │ P5: Cultural shows largest effect                             │")
    print(f"  │     Cultural shift: {cultural_shift:+.6f}                        │")
    print(f"  │     Medical shift:  {medical_shift:+.6f}                         │")
    print(f"  │     General shift:  {general_shift:+.6f}  [{p5_verdict}]          │")

    print("  └─────────────────────────────────────────────────────────────────┘")

    # Overall interpretation
    print("\n  INTERPRETATION:")
    if p1_verdict == "CONFIRMED" and p2_verdict == "CONFIRMED":
        print("  → RLHF is ASYMMETRIC: helps where model is right, hurts where wrong")
        print("  → This is the strongest version of the claim")
    elif p1_verdict == "CONFIRMED":
        print("  → RLHF degrades R2 — makes confidently wrong items worse")
    elif p1_verdict == "REFUTED" and r2_shift > 0:
        print("  → SURPRISING: RLHF actually improved R2 (human raters prefer truth)")
    else:
        print("  → RLHF appears orthogonal to regime structure")

    if p3_verdict == "CONFIRMED":
        print("  → Token-level diagnostic is degraded — base model is more honest")
    if p4_verdict == "CONFIRMED":
        print("  → RLHF creates new confidently-wrong items that base model was uncertain about")


# ===================================================================
# Main
# ===================================================================

def run_experiment(skip_inference: bool = False, force: bool = False):
    """Run the full B9 experiment."""
    total_start = time.time()

    print("=" * 70)
    print("EXPERIMENT B9: RLHF EFFECT ON REGIME CLASSIFICATION")
    print("=" * 70)
    print(f"Base model:    {BASE_MODEL}")
    print(f"Instruct model: {INSTRUCT_MODEL}")
    print(f"Regime labels: Pythia 6.9B (65 R1, 34 R2)")
    print()

    # Phase 1: Inference
    if not skip_inference:
        run_all_inference(force=force)

    # Phase 2: Load and match data
    base_pairs, inst_pairs, regime_lookup, r1_ids, r2_ids, r1_items, r2_items = load_all_data()
    matched = match_regime_items(base_pairs, inst_pairs, regime_lookup, r1_ids, r2_ids)

    if len(matched) == 0:
        print("\n  ERROR: No matched items found. Cannot proceed with analysis.")
        return

    # Phase 3: Analysis
    overall_results, r1_base, r1_inst, r2_base, r2_inst = analyze_overall_confidence(matched)
    div_results, r1_div, r2_div = analyze_divergence_points(matched)
    trans_results = analyze_regime_transitions(matched)
    domain_results = analyze_domain_interaction(matched, r1_items, r2_items)

    # Phase 4: Visualization
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    plot_confidence_shift_by_regime(r1_base, r1_inst, r2_base, r2_inst, overall_results)
    plot_divergence_winrate(div_results)
    plot_regime_transitions(trans_results)
    plot_domain_effects(domain_results)
    plot_example_items(matched)

    # Phase 5: Save results
    all_results = {
        "overall_comparison": overall_results,
        "divergence_comparison": div_results,
        "regime_transitions": {k: v for k, v in trans_results.items() if k != "transition_items"},
        "transition_items": trans_results["transition_items"],
        "domain_interaction": domain_results,
        "models": {
            "base": BASE_MODEL,
            "instruct": INSTRUCT_MODEL,
        },
    }

    results_path = COMPARISON_DIR / "b9_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Also save individual comparison files for compatibility with spec
    for key in ["overall_comparison", "divergence_comparison", "regime_transitions", "domain_interaction"]:
        out_path = COMPARISON_DIR / f"{key}.json"
        with open(out_path, "w") as f:
            json.dump(all_results[key], f, indent=2)

    # Phase 5: Summary
    print_summary(overall_results, div_results, trans_results, domain_results)

    total_time = time.time() - total_start
    print(f"\n  Total time: {total_time / 60:.1f} min")
    print(f"  Results: {results_path}")
    print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="B9: RLHF Effect on Regime Classification")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run inference even if cached")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference, only run analysis on existing data")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Same as --skip-inference")
    args = parser.parse_args()

    skip = args.skip_inference or args.analysis_only
    run_experiment(skip_inference=skip, force=args.force)
