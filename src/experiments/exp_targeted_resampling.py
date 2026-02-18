"""
Targeted Resampling at Token-Level Uncertainty Points
=====================================================
Operationalizes confidence cartography: identify low-confidence tokens in
generated sequences, resample only at those positions, and select the
completion with highest global confidence.

Tests whether targeted compute allocation outperforms uniform best-of-N
regeneration at a fraction of the compute cost.

Steps:
  1. Error characterization   — mine existing data, no new inference
  2. Oracle resampling         — resample at KNOWN error positions
  3. Blind resampling          — resample at lowest-confidence positions
  4. Comparison vs best-of-N   — the payoff table

Focus: Pythia 6.9B (best existing data + clearest regime separation)
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import load_model, unload_model, analyze_fixed_text
from src.schema import load_records, ConfidenceRecord, TokenAnalysis

# ===================================================================
# Configuration
# ===================================================================

MODEL_NAME = "EleutherAI/pythia-6.9b"
MODEL_KEY = "6.9b"

# Data paths
TRUTH_PATH = PROJECT_ROOT / "data" / "results" / "scaling" / f"a1_truth_{MODEL_KEY}.jsonl"
MEDICAL_PATH = PROJECT_ROOT / "data" / "results" / "exp9" / f"medical_pairs_{MODEL_KEY}.jsonl"
MANDELA_ORIG_PATH = PROJECT_ROOT / "data" / "results" / "mandela" / f"mandela_{MODEL_KEY}.jsonl"
MANDELA_EXP_PATH = PROJECT_ROOT / "data" / "results" / "mandela" / "expanded" / f"expanded_{MODEL_KEY}.jsonl"

# Output paths
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "targeted_resampling"
FIGURES_DIR = PROJECT_ROOT / "figures" / "targeted_resampling"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# Data loading — build paired records with regime labels
# ===================================================================

def load_paired_records():
    """Load all paired records from existing 6.9B data.

    Returns list of dicts:
      { 'pair_id', 'regime', 'source',
        'true_rec': ConfidenceRecord, 'false_rec': ConfidenceRecord }
    """
    pairs = []

    # --- Truth pairs (Regime 1) ---
    if TRUTH_PATH.exists():
        records = load_records(TRUTH_PATH)
        by_pair = defaultdict(dict)
        for r in records:
            parts = r.label.rsplit("_", 1)
            if len(parts) == 2:
                by_pair[parts[0]][parts[1]] = r
        for pid, vs in by_pair.items():
            if "true" in vs and "false" in vs:
                pairs.append({
                    "pair_id": pid, "regime": "R1", "source": "truth",
                    "true_rec": vs["true"], "false_rec": vs["false"],
                })

    # --- Medical pairs (Regime 1) ---
    if MEDICAL_PATH.exists():
        records = load_records(MEDICAL_PATH)
        by_pair = defaultdict(dict)
        for r in records:
            version = r.metadata.get("version", "")
            pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
            by_pair[pid][version] = r
        for pid, vs in by_pair.items():
            if "true" in vs and "false" in vs:
                pairs.append({
                    "pair_id": pid, "regime": "R1", "source": "medical",
                    "true_rec": vs["true"], "false_rec": vs["false"],
                })

    # --- Mandela original (Regime 2) ---
    if MANDELA_ORIG_PATH.exists():
        records = load_records(MANDELA_ORIG_PATH)
        by_pair = defaultdict(dict)
        for r in records:
            version = r.metadata.get("version", "")
            pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
            if version in ("correct", "actual"):
                by_pair[pid]["correct"] = r
            elif version in ("popular", "wrong"):
                by_pair[pid]["wrong"] = r
        for pid, vs in by_pair.items():
            if "correct" in vs and "wrong" in vs:
                pairs.append({
                    "pair_id": pid, "regime": "R2", "source": "mandela_orig",
                    "true_rec": vs["correct"], "false_rec": vs["wrong"],
                })

    # --- Mandela expanded (Regime 2) — raw framing only ---
    if MANDELA_EXP_PATH.exists():
        records = load_records(MANDELA_EXP_PATH)
        by_pair = defaultdict(lambda: defaultdict(dict))
        for r in records:
            item_id = r.metadata.get("item_id", "")
            framing = r.metadata.get("framing", "raw")
            version = r.metadata.get("version", "")
            by_pair[item_id][framing][version] = r
        for item_id, framings in by_pair.items():
            # Use raw framing for clean comparison
            if "raw" in framings:
                vs = framings["raw"]
                if "correct" in vs and "wrong" in vs:
                    pairs.append({
                        "pair_id": f"{item_id}_raw", "regime": "R2",
                        "source": "mandela_exp",
                        "true_rec": vs["correct"], "false_rec": vs["wrong"],
                    })

    return pairs


def find_divergence_point(rec_a: ConfidenceRecord,
                          rec_b: ConfidenceRecord) -> int | None:
    """First token position where two records differ."""
    min_len = min(len(rec_a.tokens), len(rec_b.tokens))
    for i in range(min_len):
        if rec_a.tokens[i].token_id != rec_b.tokens[i].token_id:
            return i
    return None


# ===================================================================
# Step 1: Error Characterization (no new inference)
# ===================================================================

def step1_error_characterization(pairs):
    """Mine existing data to check if correct token is in top-5 at error position.

    For each pair where the model "gets it wrong" (false has higher mean conf
    than true, OR at divergence the wrong token is argmax), check if the
    correct token appears in top-5 at that position.
    """
    print("\n" + "=" * 70)
    print("STEP 1: ERROR CHARACTERIZATION (no new inference)")
    print("=" * 70)

    results = []

    for pair in pairs:
        true_rec = pair["true_rec"]
        false_rec = pair["false_rec"]
        div_point = find_divergence_point(true_rec, false_rec)

        if div_point is None:
            continue

        # The "correct" token at the divergence point
        correct_token_id = true_rec.tokens[div_point].token_id
        correct_token_str = true_rec.tokens[div_point].token_str
        correct_token_conf_in_true = true_rec.tokens[div_point].top1_prob

        # What the false version has at the divergence point
        false_token_id = false_rec.tokens[div_point].token_id
        false_token_str = false_rec.tokens[div_point].token_str
        false_token_conf = false_rec.tokens[div_point].top1_prob

        # Check: is the correct token in the top-5 of the FALSE version
        # at the divergence point? (This is the key question)
        false_top5_ids = false_rec.tokens[div_point].top5_ids
        false_top5_probs = false_rec.tokens[div_point].top5_probs

        correct_in_top5 = correct_token_id in false_top5_ids
        if correct_in_top5:
            idx = false_top5_ids.index(correct_token_id)
            correct_prob_in_false = false_top5_probs[idx]
            correct_rank_in_false = idx  # 0-indexed within top-5
        else:
            correct_prob_in_false = 0.0
            correct_rank_in_false = -1  # not found

        # Also check true version's top5 — is the wrong token there?
        true_top5_ids = true_rec.tokens[div_point].top5_ids
        wrong_in_true_top5 = false_token_id in true_top5_ids

        # Determine if model "gets it wrong"
        # Method 1: false has higher mean confidence overall
        model_wrong_overall = false_rec.mean_top1_prob > true_rec.mean_top1_prob
        # Method 2: at divergence point, wrong token has higher prob than correct
        model_wrong_at_div = false_token_conf > correct_token_conf_in_true
        # Method 3: the false token is the argmax at divergence in the true version
        # (i.e., when generating from the shared prefix, model would choose wrong)
        # For teacher-forced, we check: is the correct token NOT rank 0?
        correct_is_argmax = true_rec.tokens[div_point].top1_rank == 0

        entropy_at_div = false_rec.tokens[div_point].entropy

        result = {
            "pair_id": pair["pair_id"],
            "regime": pair["regime"],
            "source": pair["source"],
            "divergence_point": div_point,
            "correct_token": correct_token_str,
            "correct_token_id": correct_token_id,
            "wrong_token": false_token_str,
            "wrong_token_id": false_token_id,
            "correct_conf_in_true": float(correct_token_conf_in_true),
            "wrong_conf_in_false": float(false_token_conf),
            "correct_in_false_top5": correct_in_top5,
            "correct_prob_in_false": float(correct_prob_in_false),
            "correct_rank_in_false_top5": correct_rank_in_false,
            "wrong_in_true_top5": wrong_in_true_top5,
            "correct_is_argmax_in_true": correct_is_argmax,
            "model_wrong_overall": model_wrong_overall,
            "model_wrong_at_div": model_wrong_at_div,
            "entropy_at_div": float(entropy_at_div),
            "true_mean_conf": float(true_rec.mean_top1_prob),
            "false_mean_conf": float(false_rec.mean_top1_prob),
        }
        results.append(result)

    # ---- Analysis ----
    print(f"\n  Total pairs analyzed: {len(results)}")

    r1 = [r for r in results if r["regime"] == "R1"]
    r2 = [r for r in results if r["regime"] == "R2"]
    print(f"  Regime 1 (factual): {len(r1)}")
    print(f"  Regime 2 (Mandela): {len(r2)}")

    for regime_name, regime_data in [("R1", r1), ("R2", r2), ("ALL", results)]:
        if not regime_data:
            continue
        n = len(regime_data)
        n_correct_in_top5 = sum(1 for r in regime_data if r["correct_in_false_top5"])
        n_model_wrong_overall = sum(1 for r in regime_data if r["model_wrong_overall"])
        n_model_wrong_at_div = sum(1 for r in regime_data if r["model_wrong_at_div"])
        n_correct_argmax = sum(1 for r in regime_data if r["correct_is_argmax_in_true"])

        # Among items where model gets it wrong at div, how many have correct in top5?
        wrong_at_div = [r for r in regime_data if r["model_wrong_at_div"]]
        n_wrong_with_correct_top5 = sum(1 for r in wrong_at_div if r["correct_in_false_top5"])

        print(f"\n  [{regime_name}] ({n} pairs):")
        print(f"    Correct token in false top-5:    {n_correct_in_top5}/{n} ({n_correct_in_top5/n:.1%})")
        print(f"    Correct is argmax in true:       {n_correct_argmax}/{n} ({n_correct_argmax/n:.1%})")
        print(f"    Model wrong (overall mean):      {n_model_wrong_overall}/{n} ({n_model_wrong_overall/n:.1%})")
        print(f"    Model wrong at div point:        {n_model_wrong_at_div}/{n} ({n_model_wrong_at_div/n:.1%})")
        if wrong_at_div:
            print(f"    Wrong-at-div with correct in top5: {n_wrong_with_correct_top5}/{len(wrong_at_div)} "
                  f"({n_wrong_with_correct_top5/len(wrong_at_div):.1%})")

        # Probability mass on correct token (when it IS in top-5)
        in_top5 = [r for r in regime_data if r["correct_in_false_top5"]]
        if in_top5:
            probs = [r["correct_prob_in_false"] for r in in_top5]
            print(f"    When correct in top-5:")
            print(f"      Mean prob: {np.mean(probs):.4f}")
            print(f"      Median prob: {np.median(probs):.4f}")
            print(f"      Max prob: {np.max(probs):.4f}")

        # Entropy at divergence
        entropies = [r["entropy_at_div"] for r in regime_data]
        print(f"    Mean entropy at divergence: {np.mean(entropies):.2f} bits")

    # Save
    save_path = RESULTS_DIR / "step1_error_characterization.json"
    with open(save_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "total_pairs": len(results),
            "regime1_pairs": len(r1),
            "regime2_pairs": len(r2),
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    return results


# ===================================================================
# Step 2: Oracle Resampling (new inference at known error positions)
# ===================================================================

@torch.no_grad()
def get_topk_at_position(model, tokenizer, device, prefix_ids, K=10):
    """Forward pass on prefix, return top-K token IDs and probs at next position."""
    input_ids = torch.tensor([prefix_ids], device=device)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, -1, :]  # logits for next token
    probs = torch.softmax(logits.float(), dim=-1)
    topk_probs, topk_ids = torch.topk(probs, K)
    return topk_ids.cpu().tolist(), topk_probs.cpu().tolist()


@torch.no_grad()
def generate_from_prefix(model, tokenizer, device, prefix_ids, max_new_tokens=50):
    """Greedy generation continuing from prefix_ids."""
    input_ids = torch.tensor([prefix_ids], device=device)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
    return output[0].cpu().tolist()  # full sequence including prefix


@torch.no_grad()
def score_sequence(model, tokenizer, device, token_ids):
    """Teacher-forced mean confidence for a token sequence.

    Returns (mean_prob, per_token_probs) where per_token_probs[i] is the
    probability the model assigned to token_ids[i+1] given token_ids[:i+1].
    """
    input_ids = torch.tensor([token_ids], device=device)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, :-1, :].cpu().float()  # (seq_len-1, vocab)
    probs = torch.softmax(logits, dim=-1)

    target_ids = torch.tensor(token_ids[1:])  # (seq_len-1,)
    per_token_probs = []
    for t in range(len(target_ids)):
        p = probs[t, target_ids[t]].item()
        per_token_probs.append(p)

    mean_prob = float(np.mean(per_token_probs)) if per_token_probs else 0.0
    return mean_prob, per_token_probs


@torch.no_grad()
def generate_best_of_n(model, tokenizer, device, prompt_text, N=5,
                       max_new_tokens=50):
    """Generate N completions, return the one with highest mean confidence.

    Uses sampling (temperature=1.0, top_k=50) for diversity.
    Returns: (best_text, best_mean_conf, all_results)
    """
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    prompt_len = input_ids.shape[1]
    input_ids = input_ids.to(device)

    all_results = []
    for i in range(N):
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=50,
        )
        seq = output[0].cpu().tolist()
        mean_conf, per_tok = score_sequence(model, tokenizer, device, seq)
        text = tokenizer.decode(seq, skip_special_tokens=True)
        all_results.append({
            "text": text,
            "mean_confidence": mean_conf,
            "n_tokens": len(seq),
        })

    all_results.sort(key=lambda x: x["mean_confidence"], reverse=True)
    return all_results[0], all_results


def step2_oracle_resampling(pairs, step1_results, K_values=(5, 10)):
    """Oracle resampling: resample at KNOWN error positions.

    For each pair where model gets it wrong at div AND correct is in top-K,
    branch K ways and check if best branch contains correct answer.
    """
    print("\n" + "=" * 70)
    print("STEP 2: ORACLE RESAMPLING (known error positions)")
    print("=" * 70)

    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.float16)

    # Build lookup from step1
    step1_lookup = {r["pair_id"]: r for r in step1_results}

    results = []
    for pair in pairs:
        s1 = step1_lookup.get(pair["pair_id"])
        if s1 is None:
            continue

        div_point = s1["divergence_point"]
        correct_token_id = s1["correct_token_id"]
        wrong_token_id = s1["wrong_token_id"]

        # Get prefix token IDs (shared between true and false)
        true_rec = pair["true_rec"]
        # The prefix is tokens 0..div_point (token_ids from the original text)
        # We need to reconstruct the prefix token IDs
        full_text = true_rec.text
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0].tolist()

        # prefix_ids are the input tokens that predict up to the divergence
        # token_analyses[i] corresponds to predicting token at position i+1
        # So div_point in token_analyses means we need input tokens 0..div_point
        prefix_ids = full_ids[:div_point + 1]  # +1 because input includes BOS or first token

        # Now get alternatives at the divergence position
        for K in K_values:
            topk_ids, topk_probs = get_topk_at_position(
                model, tokenizer, device, prefix_ids, K
            )

            branches = []
            for alt_id, alt_prob in zip(topk_ids, topk_probs):
                # Create branched prefix: prefix + alternative token
                branched_prefix = prefix_ids + [alt_id]
                # Continue greedy generation
                # Determine how many tokens to generate (match original length)
                remaining = len(full_ids) - len(branched_prefix)
                max_gen = max(remaining + 5, 10)  # a little extra room

                full_seq = generate_from_prefix(
                    model, tokenizer, device, branched_prefix,
                    max_new_tokens=max_gen,
                )
                # Score the completed sequence
                mean_conf, _ = score_sequence(model, tokenizer, device, full_seq)
                text = tokenizer.decode(full_seq, skip_special_tokens=True)

                branches.append({
                    "alt_token_id": alt_id,
                    "alt_token_str": tokenizer.decode([alt_id]),
                    "alt_prob": float(alt_prob),
                    "is_correct_token": alt_id == correct_token_id,
                    "mean_confidence": float(mean_conf),
                    "text": text,
                    "n_tokens": len(full_seq),
                })

            # Sort by mean confidence (selection criterion)
            branches.sort(key=lambda x: x["mean_confidence"], reverse=True)

            # Did the best branch pick the correct token?
            best_is_correct = branches[0]["is_correct_token"]
            # Is the correct token in any branch?
            correct_in_branches = any(b["is_correct_token"] for b in branches)
            # If correct token is in branches, what's its rank by confidence?
            correct_conf_rank = None
            for i, b in enumerate(branches):
                if b["is_correct_token"]:
                    correct_conf_rank = i
                    break

            result = {
                "pair_id": pair["pair_id"],
                "regime": pair["regime"],
                "source": pair["source"],
                "K": K,
                "divergence_point": div_point,
                "correct_token": s1["correct_token"],
                "wrong_token": s1["wrong_token"],
                "best_branch_is_correct": best_is_correct,
                "correct_in_branches": correct_in_branches,
                "correct_confidence_rank": correct_conf_rank,
                "n_branches": len(branches),
                "branches": [{
                    "token": b["alt_token_str"],
                    "prob": b["alt_prob"],
                    "is_correct": b["is_correct_token"],
                    "mean_conf": b["mean_confidence"],
                } for b in branches],
            }
            results.append(result)

        # Progress
        idx = pairs.index(pair)
        if (idx + 1) % 10 == 0:
            print(f"    [{idx+1}/{len(pairs)}] processed")

    # ---- Analysis ----
    for K in K_values:
        k_results = [r for r in results if r["K"] == K]
        if not k_results:
            continue
        print(f"\n  K={K}:")
        for regime_name in ["R1", "R2", "ALL"]:
            subset = [r for r in k_results if r["regime"] == regime_name] if regime_name != "ALL" else k_results
            if not subset:
                continue
            n = len(subset)
            n_best_correct = sum(1 for r in subset if r["best_branch_is_correct"])
            n_correct_available = sum(1 for r in subset if r["correct_in_branches"])
            print(f"    [{regime_name}] Best branch correct: {n_best_correct}/{n} ({n_best_correct/n:.1%}) "
                  f"| Correct available: {n_correct_available}/{n} ({n_correct_available/n:.1%})")

    save_path = RESULTS_DIR / "step2_oracle_results.json"
    with open(save_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "K_values": list(K_values),
            "total_results": len(results),
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    return results


# ===================================================================
# Step 3: Blind Resampling
# ===================================================================

def step3_blind_resampling(pairs, K_values=(5, 10)):
    """Blind resampling: generate greedily, find lowest-conf position, resample there.

    This is the realistic deployment scenario.
    """
    print("\n" + "=" * 70)
    print("STEP 3: BLIND RESAMPLING (lowest-confidence position)")
    print("=" * 70)

    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.float16)

    results = []
    for pair_idx, pair in enumerate(pairs):
        true_rec = pair["true_rec"]
        false_rec = pair["false_rec"]

        # Generate the FALSE version greedily from a shared prompt
        # The "prompt" is the beginning of the statement
        # For teacher-forced data, we use the full false text and analyze it
        false_text = false_rec.text
        false_ids = tokenizer(false_text, return_tensors="pt")["input_ids"][0].tolist()

        # Score the greedy (false) sequence
        greedy_mean_conf, greedy_per_tok = score_sequence(
            model, tokenizer, device, false_ids
        )

        # Find lowest-confidence position in the greedy output
        if not greedy_per_tok:
            continue
        min_conf_pos = int(np.argmin(greedy_per_tok))
        min_conf_val = greedy_per_tok[min_conf_pos]

        # Also find where the actual divergence is
        div_point = find_divergence_point(true_rec, false_rec)

        # Does the blind method find the actual error position?
        blind_hits_error = (div_point is not None and min_conf_pos == div_point)

        for K in K_values:
            # Resample at the lowest-confidence position
            prefix_ids = false_ids[:min_conf_pos + 1]  # include tokens up to before the target
            topk_ids, topk_probs = get_topk_at_position(
                model, tokenizer, device, prefix_ids, K
            )

            branches = []
            for alt_id, alt_prob in zip(topk_ids, topk_probs):
                branched = prefix_ids + [alt_id]
                remaining = len(false_ids) - len(branched)
                max_gen = max(remaining + 5, 10)
                full_seq = generate_from_prefix(
                    model, tokenizer, device, branched, max_new_tokens=max_gen
                )
                mean_conf, _ = score_sequence(model, tokenizer, device, full_seq)
                text = tokenizer.decode(full_seq, skip_special_tokens=True)

                # Check if this branch matches the TRUE text
                true_text = true_rec.text
                # Simple match: does the completion start with the true text?
                is_match = text.strip().startswith(true_text.strip()[:50])

                branches.append({
                    "alt_token_id": alt_id,
                    "alt_token_str": tokenizer.decode([alt_id]),
                    "alt_prob": float(alt_prob),
                    "mean_confidence": float(mean_conf),
                    "text_matches_truth": is_match,
                    "text_preview": text[:200],
                })

            branches.sort(key=lambda x: x["mean_confidence"], reverse=True)
            best_matches_truth = branches[0]["text_matches_truth"]

            result = {
                "pair_id": pair["pair_id"],
                "regime": pair["regime"],
                "source": pair["source"],
                "K": K,
                "blind_resample_position": min_conf_pos,
                "blind_conf_at_position": float(min_conf_val),
                "actual_divergence_point": div_point,
                "blind_hits_error": blind_hits_error,
                "greedy_mean_conf": float(greedy_mean_conf),
                "best_branch_mean_conf": float(branches[0]["mean_confidence"]),
                "best_branch_matches_truth": best_matches_truth,
                "best_branch_token": branches[0]["alt_token_str"],
                "n_branches": len(branches),
            }
            results.append(result)

        if (pair_idx + 1) % 10 == 0:
            print(f"    [{pair_idx+1}/{len(pairs)}] processed")

    # ---- Analysis ----
    for K in K_values:
        k_results = [r for r in results if r["K"] == K]
        if not k_results:
            continue
        print(f"\n  K={K}:")
        n_blind_hit = sum(1 for r in k_results if r["blind_hits_error"])
        print(f"    Blind finds actual error: {n_blind_hit}/{len(k_results)} "
              f"({n_blind_hit/len(k_results):.1%})")
        for regime_name in ["R1", "R2", "ALL"]:
            subset = [r for r in k_results if r["regime"] == regime_name] if regime_name != "ALL" else k_results
            if not subset:
                continue
            n = len(subset)
            n_conf_improved = sum(1 for r in subset
                                 if r["best_branch_mean_conf"] > r["greedy_mean_conf"])
            n_matches = sum(1 for r in subset if r["best_branch_matches_truth"])
            print(f"    [{regime_name}] Conf improved: {n_conf_improved}/{n} ({n_conf_improved/n:.1%}) "
                  f"| Matches truth: {n_matches}/{n} ({n_matches/n:.1%})")

    save_path = RESULTS_DIR / "step3_blind_results.json"
    with open(save_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "K_values": list(K_values),
            "total_results": len(results),
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved: {save_path}")

    return results


# ===================================================================
# Step 4: Comparison vs Best-of-N
# ===================================================================

def step4_comparison(pairs, step2_results, step3_results):
    """Compare targeted resampling vs best-of-N baselines.

    Methods:
    - Greedy: 1x compute (the false version, always wrong)
    - Oracle-5: ~1.5x (resample top-5 at known error)
    - Oracle-10: ~2x
    - Blind-5: ~1.5x (resample top-5 at lowest-conf)
    - Blind-10: ~2x
    - Best-of-5: 5x (5 full regenerations)
    - Best-of-10: 10x (10 full regenerations)
    """
    print("\n" + "=" * 70)
    print("STEP 4: METHOD COMPARISON")
    print("=" * 70)

    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.float16)

    # Build lookups
    oracle_lookup = defaultdict(dict)
    for r in step2_results:
        oracle_lookup[r["pair_id"]][r["K"]] = r

    blind_lookup = defaultdict(dict)
    for r in step3_results:
        blind_lookup[r["pair_id"]][r["K"]] = r

    # Run best-of-N for each pair
    print("\n  Running best-of-N baselines...")
    bon_results = {}
    for pair_idx, pair in enumerate(pairs):
        pid = pair["pair_id"]
        # Use the first few tokens of the true text as prompt
        # (We need a shared prompt to generate from)
        true_rec = pair["true_rec"]
        false_rec = pair["false_rec"]
        div_point = find_divergence_point(true_rec, false_rec)
        if div_point is None or div_point < 1:
            continue

        # Prompt is the shared prefix (tokens before divergence)
        full_ids = tokenizer(true_rec.text, return_tensors="pt")["input_ids"][0].tolist()
        prompt_ids = full_ids[:div_point + 1]
        prompt_text = tokenizer.decode(prompt_ids)

        # Generate best-of-5 and best-of-10
        max_gen = len(full_ids) - len(prompt_ids) + 5
        max_gen = max(max_gen, 10)

        bon5_results = []
        bon10_results = []
        for i in range(10):
            full_seq = generate_from_prefix(
                model, tokenizer, device, prompt_ids,
                max_new_tokens=max_gen,
            )
            # Score it
            # Actually for best-of-N we need sampling, not greedy
            # Re-implement with sampling
            pass

        # Use the generate_best_of_n function instead
        best5, all5 = generate_best_of_n(
            model, tokenizer, device, prompt_text,
            N=5, max_new_tokens=max_gen,
        )
        best10, all10 = generate_best_of_n(
            model, tokenizer, device, prompt_text,
            N=10, max_new_tokens=max_gen,
        )

        # Check if best-of-N contains the correct continuation
        correct_text_fragment = true_rec.text[len(prompt_text):].strip()[:30]
        bon5_correct = correct_text_fragment.lower() in best5["text"].lower() if correct_text_fragment else False
        bon10_correct = correct_text_fragment.lower() in best10["text"].lower() if correct_text_fragment else False

        bon_results[pid] = {
            "bon5_correct": bon5_correct,
            "bon5_mean_conf": best5["mean_confidence"],
            "bon10_correct": bon10_correct,
            "bon10_mean_conf": best10["mean_confidence"],
        }

        if (pair_idx + 1) % 10 == 0:
            print(f"    [{pair_idx+1}/{len(pairs)}] best-of-N done")

    # ---- Build comparison table ----
    methods = {
        "Greedy": {"compute": 1.0, "results": {}},
        "Oracle-5": {"compute": 1.5, "results": {}},
        "Oracle-10": {"compute": 2.0, "results": {}},
        "Blind-5": {"compute": 1.5, "results": {}},
        "Blind-10": {"compute": 2.0, "results": {}},
        "Best-of-5": {"compute": 5.0, "results": {}},
        "Best-of-10": {"compute": 10.0, "results": {}},
    }

    comparison = []
    for pair in pairs:
        pid = pair["pair_id"]
        div_point = find_divergence_point(pair["true_rec"], pair["false_rec"])
        if div_point is None:
            continue

        row = {
            "pair_id": pid,
            "regime": pair["regime"],
            "greedy_correct": False,  # by construction, greedy = wrong
            "oracle5_correct": oracle_lookup.get(pid, {}).get(5, {}).get("best_branch_is_correct", False),
            "oracle10_correct": oracle_lookup.get(pid, {}).get(10, {}).get("best_branch_is_correct", False),
            "blind5_correct": blind_lookup.get(pid, {}).get(5, {}).get("best_branch_matches_truth", False),
            "blind10_correct": blind_lookup.get(pid, {}).get(10, {}).get("best_branch_matches_truth", False),
            "bon5_correct": bon_results.get(pid, {}).get("bon5_correct", False),
            "bon10_correct": bon_results.get(pid, {}).get("bon10_correct", False),
        }
        comparison.append(row)

    # Print comparison table
    print(f"\n  {'Method':<15s} {'Compute':<10s} {'R1 Accuracy':<15s} {'R2 Accuracy':<15s} {'Overall':<15s}")
    print("  " + "-" * 65)

    for method_key, label in [
        ("greedy_correct", "Greedy (1x)"),
        ("oracle5_correct", "Oracle-5 (1.5x)"),
        ("oracle10_correct", "Oracle-10 (2x)"),
        ("blind5_correct", "Blind-5 (1.5x)"),
        ("blind10_correct", "Blind-10 (2x)"),
        ("bon5_correct", "Best-of-5 (5x)"),
        ("bon10_correct", "Best-of-10 (10x)"),
    ]:
        r1 = [r for r in comparison if r["regime"] == "R1"]
        r2 = [r for r in comparison if r["regime"] == "R2"]
        r1_acc = sum(1 for r in r1 if r[method_key]) / len(r1) if r1 else 0
        r2_acc = sum(1 for r in r2 if r[method_key]) / len(r2) if r2 else 0
        all_acc = sum(1 for r in comparison if r[method_key]) / len(comparison) if comparison else 0
        print(f"  {label:<15s} {'':>10s} {r1_acc:>12.1%}   {r2_acc:>12.1%}   {all_acc:>12.1%}")

    save_path = RESULTS_DIR / "step4_comparison.json"
    with open(save_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "comparison": comparison,
            "bon_results": bon_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")

    return comparison


# ===================================================================
# Visualization
# ===================================================================

def plot_results(step1_results, step2_results, step3_results, comparison):
    """Generate all figures."""
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    sns.set_theme(style="whitegrid", palette="muted")

    # ---- Figure 1: Correct token rank distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, regime, color, title in [
        (axes[0], "R1", "#2196F3", "Regime 1: Factual"),
        (axes[1], "R2", "#F44336", "Regime 2: Mandela"),
    ]:
        data = [r for r in step1_results if r["regime"] == regime]
        if not data:
            ax.set_title(f"{title}\n(no data)")
            continue

        in_top5 = sum(1 for r in data if r["correct_in_false_top5"])
        not_in_top5 = len(data) - in_top5

        # Bar chart: in top-5 vs not
        ax.bar(["In Top-5", "Not in Top-5"], [in_top5, not_in_top5],
               color=[color, "#BDBDBD"], edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\nCorrect token in false version's top-5\n"
                     f"({in_top5}/{len(data)} = {in_top5/len(data):.0%})")

        # Add probability annotation for in-top5 cases
        if in_top5 > 0:
            probs = [r["correct_prob_in_false"] for r in data if r["correct_in_false_top5"]]
            ax.text(0, in_top5 + 0.5, f"Mean prob: {np.mean(probs):.3f}",
                    ha="center", fontsize=10)

    fig.suptitle("Step 1: Is the Correct Token Reachable by Resampling?",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correct_token_rank_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/4] correct_token_rank_distribution.png")

    # ---- Figure 2: Method comparison barplot ----
    if comparison:
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = [
            ("greedy_correct", "Greedy\n(1x)", "#9E9E9E"),
            ("blind5_correct", "Blind-5\n(1.5x)", "#66BB6A"),
            ("blind10_correct", "Blind-10\n(2x)", "#43A047"),
            ("oracle5_correct", "Oracle-5\n(1.5x)", "#42A5F5"),
            ("oracle10_correct", "Oracle-10\n(2x)", "#1E88E5"),
            ("bon5_correct", "Best-of-5\n(5x)", "#FFA726"),
            ("bon10_correct", "Best-of-10\n(10x)", "#FB8C00"),
        ]

        x = np.arange(len(methods))
        width = 0.35

        r1_data = [r for r in comparison if r["regime"] == "R1"]
        r2_data = [r for r in comparison if r["regime"] == "R2"]

        r1_accs = []
        r2_accs = []
        for key, _, _ in methods:
            r1_accs.append(sum(1 for r in r1_data if r[key]) / len(r1_data) if r1_data else 0)
            r2_accs.append(sum(1 for r in r2_data if r[key]) / len(r2_data) if r2_data else 0)

        bars1 = ax.bar(x - width/2, r1_accs, width, label="R1: Factual",
                       color="#2196F3", alpha=0.8)
        bars2 = ax.bar(x + width/2, r2_accs, width, label="R2: Mandela",
                       color="#F44336", alpha=0.8)

        ax.set_ylabel("Accuracy (correct answer selected)")
        ax.set_title("Method Comparison: Targeted Resampling vs Best-of-N")
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in methods], fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "method_comparison_barplot.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [2/4] method_comparison_barplot.png")

    # ---- Figure 3: Compute-accuracy tradeoff ----
    if comparison:
        fig, ax = plt.subplots(figsize=(8, 6))

        method_specs = [
            ("greedy_correct", 1.0, "Greedy", "o", "#9E9E9E"),
            ("blind5_correct", 1.5, "Blind-5", "^", "#66BB6A"),
            ("blind10_correct", 2.0, "Blind-10", "^", "#43A047"),
            ("oracle5_correct", 1.5, "Oracle-5", "s", "#42A5F5"),
            ("oracle10_correct", 2.0, "Oracle-10", "s", "#1E88E5"),
            ("bon5_correct", 5.0, "Best-of-5", "D", "#FFA726"),
            ("bon10_correct", 10.0, "Best-of-10", "D", "#FB8C00"),
        ]

        for key, compute, label, marker, color in method_specs:
            acc = sum(1 for r in comparison if r[key]) / len(comparison) if comparison else 0
            ax.scatter(compute, acc, s=120, marker=marker, color=color,
                       edgecolors="black", linewidth=0.5, zorder=5)
            ax.annotate(label, (compute, acc), textcoords="offset points",
                        xytext=(8, 5), fontsize=8)

        ax.set_xlabel("Compute Multiplier (relative to greedy)")
        ax.set_ylabel("Overall Accuracy")
        ax.set_title("Compute-Accuracy Tradeoff:\nTargeted Resampling vs Best-of-N")
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "compute_accuracy_tradeoff.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [3/4] compute_accuracy_tradeoff.png")

    # ---- Figure 4: Regime comparison ----
    if step3_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, K in [(axes[0], 5), (axes[1], 10)]:
            k_data = [r for r in step3_results if r["K"] == K]
            r1 = [r for r in k_data if r["regime"] == "R1"]
            r2 = [r for r in k_data if r["regime"] == "R2"]

            categories = ["Blind hits\nerror pos", "Conf\nimproved", "Matches\ntruth"]
            r1_vals = [
                sum(1 for r in r1 if r["blind_hits_error"]) / len(r1) if r1 else 0,
                sum(1 for r in r1 if r["best_branch_mean_conf"] > r["greedy_mean_conf"]) / len(r1) if r1 else 0,
                sum(1 for r in r1 if r["best_branch_matches_truth"]) / len(r1) if r1 else 0,
            ]
            r2_vals = [
                sum(1 for r in r2 if r["blind_hits_error"]) / len(r2) if r2 else 0,
                sum(1 for r in r2 if r["best_branch_mean_conf"] > r["greedy_mean_conf"]) / len(r2) if r2 else 0,
                sum(1 for r in r2 if r["best_branch_matches_truth"]) / len(r2) if r2 else 0,
            ]

            x = np.arange(len(categories))
            width = 0.35
            ax.bar(x - width/2, r1_vals, width, label="R1: Factual", color="#2196F3")
            ax.bar(x + width/2, r2_vals, width, label="R2: Mandela", color="#F44336")
            ax.set_ylabel("Rate")
            ax.set_title(f"Blind Resampling K={K}")
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.0)

        fig.suptitle("Regime Comparison: Does Targeted Resampling Work Differently?",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "regime_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [4/4] regime_comparison.png")


# ===================================================================
# Main
# ===================================================================

def run_experiment():
    """Run all 4 steps + visualization."""
    total_start = time.time()

    print("=" * 70)
    print("TARGETED RESAMPLING AT TOKEN-LEVEL UNCERTAINTY POINTS")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    # Load all paired data
    print("\nLoading paired records...")
    pairs = load_paired_records()
    print(f"  Loaded {len(pairs)} pairs")
    r1 = [p for p in pairs if p["regime"] == "R1"]
    r2 = [p for p in pairs if p["regime"] == "R2"]
    print(f"  Regime 1 (factual): {len(r1)}")
    print(f"  Regime 2 (Mandela): {len(r2)}")

    # Step 1: Error characterization (no inference)
    step1_results = step1_error_characterization(pairs)

    # Gate: do we have enough items with correct token in top-5?
    n_in_top5 = sum(1 for r in step1_results if r["correct_in_false_top5"])
    pct_in_top5 = n_in_top5 / len(step1_results) if step1_results else 0
    print(f"\n  GATE CHECK: {n_in_top5}/{len(step1_results)} ({pct_in_top5:.1%}) "
          f"have correct token in top-5")

    if pct_in_top5 < 0.10:
        print("\n  *** GATE FAILED: Correct token rarely in top-5. ***")
        print("  Resampling unlikely to help. Stopping after Step 1.")
        print("  This is still a publishable finding: model's errors are deeper")
        print("  than the output distribution suggests.")
        plot_results(step1_results, [], [], [])
        return

    print("  GATE PASSED. Proceeding to Steps 2-4.\n")

    # Step 2: Oracle resampling
    step2_results = step2_oracle_resampling(pairs, step1_results)

    # Step 3: Blind resampling
    step3_results = step3_blind_resampling(pairs)

    # Step 4: Comparison
    comparison = step4_comparison(pairs, step2_results, step3_results)

    # Visualization
    plot_results(step1_results, step2_results, step3_results, comparison)

    # Save regime breakdown
    regime_data = {
        "R1": {
            "n_pairs": len(r1),
            "step1_correct_in_top5": sum(1 for r in step1_results if r["regime"] == "R1" and r["correct_in_false_top5"]),
        },
        "R2": {
            "n_pairs": len(r2),
            "step1_correct_in_top5": sum(1 for r in step1_results if r["regime"] == "R2" and r["correct_in_false_top5"]),
        },
    }
    with open(RESULTS_DIR / "regime_breakdown.json", "w") as f:
        json.dump(regime_data, f, indent=2)

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE ({total_time:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
