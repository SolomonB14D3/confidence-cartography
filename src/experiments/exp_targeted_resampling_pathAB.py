"""
Targeted Resampling: Path A/B Improvements
============================================
Path B (selection): Can we pick the correct branch better than mean confidence?
  B1: Hidden state probe — extract activations, train logistic regression
  B2: Entropy reduction — score by output entropy instead of top-1 conf
  B3: Backward confidence — is branch token predictable from its continuation?
  B4: Min confidence — avoid catastrophic low-confidence tokens

Path A (localization): Can we find the error position better than min confidence?
  A1: Top-1/Top-2 gap — smallest gap = most torn = likely error
  A2: Local confidence drop — sudden drop relative to neighbors
  A3: Entropy spike — highest entropy or largest entropy increase

Path B goes first. If it fails (all methods < 25%), localization is moot.
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import load_model, unload_model
from src.schema import load_records, ConfidenceRecord

# ===================================================================
# Configuration
# ===================================================================

MODEL_NAME = "EleutherAI/pythia-6.9b"
MODEL_KEY = "6.9b"

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "targeted_resampling"
FIGURES_DIR = PROJECT_ROOT / "figures" / "targeted_resampling"

# Load existing data
STEP1_PATH = RESULTS_DIR / "step1_error_characterization.json"
STEP2_PATH = RESULTS_DIR / "step2_oracle_results.json"
STEP3_PATH = RESULTS_DIR / "step3_blind_results.json"

# Data paths for loading raw records
TRUTH_PATH = PROJECT_ROOT / "data" / "results" / "scaling" / f"a1_truth_{MODEL_KEY}.jsonl"
MEDICAL_PATH = PROJECT_ROOT / "data" / "results" / "exp9" / f"medical_pairs_{MODEL_KEY}.jsonl"
MANDELA_ORIG_PATH = PROJECT_ROOT / "data" / "results" / "mandela" / f"mandela_{MODEL_KEY}.jsonl"
MANDELA_EXP_PATH = PROJECT_ROOT / "data" / "results" / "mandela" / "expanded" / f"expanded_{MODEL_KEY}.jsonl"

# Probe layers to test
PROBE_LAYERS = [-1, -4, -8, -16, 16, 4]  # last, near-last, middle-ish, early


# ===================================================================
# Data loading
# ===================================================================

def load_oracle_data():
    """Load step2 oracle results."""
    with open(STEP2_PATH) as f:
        data = json.load(f)
    return data["results"]


def load_blind_data():
    """Load step3 blind results."""
    with open(STEP3_PATH) as f:
        data = json.load(f)
    return data["results"]


def load_step1_data():
    """Load step1 error characterization."""
    with open(STEP1_PATH) as f:
        data = json.load(f)
    return data["results"]


def load_paired_records():
    """Load all paired confidence records (same as base experiment)."""
    pairs = {}

    if TRUTH_PATH.exists():
        records = load_records(TRUTH_PATH)
        by_pair = defaultdict(dict)
        for r in records:
            parts = r.label.rsplit("_", 1)
            if len(parts) == 2:
                by_pair[parts[0]][parts[1]] = r
        for pid, vs in by_pair.items():
            if "true" in vs and "false" in vs:
                pairs[pid] = {"true_rec": vs["true"], "false_rec": vs["false"]}

    if MEDICAL_PATH.exists():
        records = load_records(MEDICAL_PATH)
        by_pair = defaultdict(dict)
        for r in records:
            version = r.metadata.get("version", "")
            pid = r.metadata.get("pair_id", r.label.rsplit("_", 1)[0])
            by_pair[pid][version] = r
        for pid, vs in by_pair.items():
            if "true" in vs and "false" in vs:
                pairs[pid] = {"true_rec": vs["true"], "false_rec": vs["false"]}

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
                pairs[pid] = {"true_rec": vs["correct"], "false_rec": vs["wrong"]}

    if MANDELA_EXP_PATH.exists():
        records = load_records(MANDELA_EXP_PATH)
        by_pair = defaultdict(lambda: defaultdict(dict))
        for r in records:
            item_id = r.metadata.get("item_id", "")
            framing = r.metadata.get("framing", "raw")
            version = r.metadata.get("version", "")
            by_pair[item_id][framing][version] = r
        for item_id, framings in by_pair.items():
            if "raw" in framings:
                vs = framings["raw"]
                if "correct" in vs and "wrong" in vs:
                    pairs[f"{item_id}_raw"] = {"true_rec": vs["correct"], "false_rec": vs["wrong"]}

    return pairs


# ===================================================================
# Path B: Better Selection Criteria
# ===================================================================

# --- B1: Hidden State Probe ---

@torch.no_grad()
def extract_branch_activations(model, tokenizer, device, prefix_ids,
                               branch_token_ids, layers):
    """Extract hidden state activations at each branch token for multiple layers.

    Returns dict: { layer_idx: list of numpy arrays (one per branch token) }
    """
    result = {layer: [] for layer in layers}

    for token_id in branch_token_ids:
        input_ids = torch.tensor([prefix_ids + [token_id]], device=device)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        # outputs.hidden_states: tuple of (1, seq_len, hidden_dim) for each layer
        # Layer 0 = embeddings, layer 1 = after first transformer block, etc.
        n_layers = len(outputs.hidden_states)

        for layer in layers:
            # Handle negative indexing
            actual_layer = layer if layer >= 0 else n_layers + layer
            actual_layer = max(0, min(actual_layer, n_layers - 1))
            hidden = outputs.hidden_states[actual_layer][0, -1, :]  # last token position
            result[layer].append(hidden.cpu().numpy())

    return result


def run_probe_b1(oracle_data, pairs_lookup):
    """B1: Hidden state probe on oracle branches."""
    print("\n  --- B1: Hidden State Probe ---")

    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.float16)

    # Get number of layers
    test_input = torch.tensor([[0]], device=device)
    test_out = model(test_input, output_hidden_states=True)
    n_layers = len(test_out.hidden_states)
    print(f"    Model has {n_layers} layers (including embedding)")

    # Adjust probe layers to valid range
    valid_layers = []
    for l in PROBE_LAYERS:
        actual = l if l >= 0 else n_layers + l
        if 0 <= actual < n_layers:
            valid_layers.append(l)
    print(f"    Testing layers: {valid_layers}")

    # Collect activations and labels for K=5 oracle branches
    k5_data = [r for r in oracle_data if r["K"] == 5 and r["correct_in_branches"]]
    print(f"    K=5 items with correct in branches: {len(k5_data)}")

    all_activations = {layer: [] for layer in valid_layers}
    all_labels = []
    all_pair_ids = []

    for idx, item in enumerate(k5_data):
        pair_id = item["pair_id"]
        pair = pairs_lookup.get(pair_id)
        if pair is None:
            continue

        # Reconstruct prefix from true record
        true_rec = pair["true_rec"]
        full_ids = tokenizer(true_rec.text, return_tensors="pt")["input_ids"][0].tolist()
        div_point = item["divergence_point"]
        prefix_ids = full_ids[:div_point + 1]

        # Get branch token IDs
        branch_token_ids = []
        branch_labels = []
        for branch in item["branches"]:
            # Recover token ID from token string
            tok_id = tokenizer.encode(branch["token"], add_special_tokens=False)
            if tok_id:
                branch_token_ids.append(tok_id[0])
                branch_labels.append(1 if branch["is_correct"] else 0)

        if not branch_token_ids or sum(branch_labels) == 0:
            continue

        # Extract activations
        activations = extract_branch_activations(
            model, tokenizer, device, prefix_ids, branch_token_ids, valid_layers
        )

        for layer in valid_layers:
            all_activations[layer].extend(activations[layer])
        all_labels.extend(branch_labels)
        all_pair_ids.extend([pair_id] * len(branch_labels))

        if (idx + 1) % 20 == 0:
            print(f"      [{idx+1}/{len(k5_data)}] extracted")

    print(f"    Total activation vectors: {len(all_labels)}")
    print(f"    Positive (correct): {sum(all_labels)}")
    print(f"    Negative (incorrect): {len(all_labels) - sum(all_labels)}")

    # Train LOO probe for each layer
    probe_results = {}
    labels_arr = np.array(all_labels)

    for layer in valid_layers:
        X = np.array(all_activations[layer])
        y = labels_arr

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # LOO cross-validation
        loo = LeaveOneOut()
        predictions = np.zeros(len(y))
        pred_probs = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X_scaled):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            clf.fit(X_scaled[train_idx], y[train_idx])
            predictions[test_idx] = clf.predict(X_scaled[test_idx])
            pred_probs[test_idx] = clf.predict_proba(X_scaled[test_idx])[:, 1]

        acc = accuracy_score(y, predictions)
        try:
            auc = roc_auc_score(y, pred_probs)
        except ValueError:
            auc = 0.5

        # Also compute: if we use probe to select branch per item, what % correct?
        # Group predictions by pair_id
        pair_groups = defaultdict(list)
        for i, pid in enumerate(all_pair_ids):
            pair_groups[pid].append((pred_probs[i], all_labels[i]))

        n_correct_selected = 0
        n_items = 0
        for pid, entries in pair_groups.items():
            if not any(label == 1 for _, label in entries):
                continue
            n_items += 1
            best_idx = max(range(len(entries)), key=lambda i: entries[i][0])
            if entries[best_idx][1] == 1:
                n_correct_selected += 1

        selection_rate = n_correct_selected / n_items if n_items else 0

        probe_results[layer] = {
            "accuracy": float(acc),
            "auc": float(auc),
            "selection_rate": float(selection_rate),
            "n_items": n_items,
        }

        actual_layer = layer if layer >= 0 else n_layers + layer
        print(f"    Layer {layer:>3d} (idx {actual_layer:>2d}): "
              f"AUC={auc:.3f}  Acc={acc:.3f}  Selection={selection_rate:.1%} ({n_correct_selected}/{n_items})")

    return probe_results


# --- B2: Entropy Reduction Scoring ---

def run_entropy_b2(oracle_data, pairs_lookup):
    """B2: Score branches by entropy of the continuation."""
    print("\n  --- B2: Entropy Reduction Scoring ---")

    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.float16)

    k5_data = [r for r in oracle_data if r["K"] == 5 and r["correct_in_branches"]]
    results = {"mean_entropy": 0, "min_entropy": 0, "pos1_entropy": 0, "n_items": 0}

    n_correct_mean = 0
    n_correct_min = 0
    n_correct_pos1 = 0
    n_items = 0

    for idx, item in enumerate(k5_data):
        pair = pairs_lookup.get(item["pair_id"])
        if pair is None:
            continue

        true_rec = pair["true_rec"]
        full_ids = tokenizer(true_rec.text, return_tensors="pt")["input_ids"][0].tolist()
        div_point = item["divergence_point"]
        prefix_ids = full_ids[:div_point + 1]

        branch_scores = []
        for branch in item["branches"]:
            tok_ids = tokenizer.encode(branch["token"], add_special_tokens=False)
            if not tok_ids:
                continue
            branch_id = tok_ids[0]

            # Build full branch sequence: prefix + branch token
            # Then get continuation via greedy generation
            branch_seq = prefix_ids + [branch_id]
            remaining = len(full_ids) - len(branch_seq) + 5
            max_gen = max(remaining, 10)

            input_ids = torch.tensor([branch_seq], device=device)
            with torch.no_grad():
                gen_output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Compute entropy from generation scores
            if not gen_output.scores:
                continue

            entropies = []
            for score_t in gen_output.scores:
                logits = score_t[0].cpu().float()
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * torch.log2(probs + 1e-12)).sum().item()
                entropies.append(ent)

            branch_scores.append({
                "is_correct": branch["is_correct"],
                "mean_entropy": float(np.mean(entropies)),
                "min_entropy": float(np.min(entropies)),
                "pos1_entropy": float(entropies[0]) if entropies else 999,
            })

        if not branch_scores or not any(b["is_correct"] for b in branch_scores):
            continue

        n_items += 1

        # Select by lowest mean entropy
        best_mean = min(branch_scores, key=lambda b: b["mean_entropy"])
        if best_mean["is_correct"]:
            n_correct_mean += 1

        # Select by lowest min entropy
        best_min = min(branch_scores, key=lambda b: b["min_entropy"])
        if best_min["is_correct"]:
            n_correct_min += 1

        # Select by lowest entropy at position+1
        best_pos1 = min(branch_scores, key=lambda b: b["pos1_entropy"])
        if best_pos1["is_correct"]:
            n_correct_pos1 += 1

        if (idx + 1) % 20 == 0:
            print(f"      [{idx+1}/{len(k5_data)}] processed")

    results = {
        "mean_entropy_selection": float(n_correct_mean / n_items) if n_items else 0,
        "min_entropy_selection": float(n_correct_min / n_items) if n_items else 0,
        "pos1_entropy_selection": float(n_correct_pos1 / n_items) if n_items else 0,
        "n_items": n_items,
    }

    print(f"    Items tested: {n_items}")
    print(f"    Mean entropy selection: {n_correct_mean}/{n_items} ({results['mean_entropy_selection']:.1%})")
    print(f"    Min entropy selection:  {n_correct_min}/{n_items} ({results['min_entropy_selection']:.1%})")
    print(f"    Pos+1 entropy selection:{n_correct_pos1}/{n_items} ({results['pos1_entropy_selection']:.1%})")

    return results


# --- B3: Backward Confidence ---

@torch.no_grad()
def run_backward_b3(oracle_data, pairs_lookup):
    """B3: Score branches by backward confidence."""
    print("\n  --- B3: Backward Confidence ---")

    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.float16)

    k5_data = [r for r in oracle_data if r["K"] == 5 and r["correct_in_branches"]]

    n_correct = 0
    n_items = 0

    for idx, item in enumerate(k5_data):
        pair = pairs_lookup.get(item["pair_id"])
        if pair is None:
            continue

        true_rec = pair["true_rec"]
        false_rec = pair["false_rec"]
        full_ids = tokenizer(true_rec.text, return_tensors="pt")["input_ids"][0].tolist()
        div_point = item["divergence_point"]
        prefix_ids = full_ids[:div_point + 1]

        branch_scores = []
        for branch in item["branches"]:
            tok_ids = tokenizer.encode(branch["token"], add_special_tokens=False)
            if not tok_ids:
                continue
            branch_id = tok_ids[0]

            # Generate continuation from this branch
            branch_seq = prefix_ids + [branch_id]
            remaining = len(full_ids) - len(branch_seq) + 5
            max_gen = max(remaining, 10)

            input_ids_gen = torch.tensor([branch_seq], device=device)
            gen_output = model.generate(
                input_ids=input_ids_gen,
                max_new_tokens=max_gen,
                do_sample=False,
            )
            full_seq = gen_output[0].cpu().tolist()

            # Now do teacher-forced forward pass on full sequence
            input_ids_tf = torch.tensor([full_seq], device=device)
            outputs = model(input_ids=input_ids_tf)

            # Get probability of branch_token at its position
            # The branch token is at position len(prefix_ids) in the sequence
            # The logits at position len(prefix_ids)-1 predict position len(prefix_ids)
            branch_pos = len(prefix_ids)  # position of the branch token
            pred_pos = branch_pos - 1     # logits position that predicts it
            if pred_pos < outputs.logits.shape[1]:
                logits_at_pred = outputs.logits[0, pred_pos, :].cpu().float()
                probs = torch.softmax(logits_at_pred, dim=-1)
                backward_conf = probs[branch_id].item()
            else:
                backward_conf = 0.0

            branch_scores.append({
                "is_correct": branch["is_correct"],
                "backward_conf": float(backward_conf),
            })

        if not branch_scores or not any(b["is_correct"] for b in branch_scores):
            continue

        n_items += 1
        best = max(branch_scores, key=lambda b: b["backward_conf"])
        if best["is_correct"]:
            n_correct += 1

        if (idx + 1) % 20 == 0:
            print(f"      [{idx+1}/{len(k5_data)}] processed")

    rate = n_correct / n_items if n_items else 0
    print(f"    Items tested: {n_items}")
    print(f"    Backward conf selection: {n_correct}/{n_items} ({rate:.1%})")

    return {"backward_conf_selection": float(rate), "n_items": n_items}


# --- B4: Min Confidence Scoring ---

@torch.no_grad()
def run_minconf_b4(oracle_data, pairs_lookup):
    """B4: Score branches by min confidence in continuation."""
    print("\n  --- B4: Min Confidence Scoring ---")

    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.float16)

    k5_data = [r for r in oracle_data if r["K"] == 5 and r["correct_in_branches"]]

    n_correct_min = 0
    n_correct_below_thresh = 0
    n_items = 0

    for idx, item in enumerate(k5_data):
        pair = pairs_lookup.get(item["pair_id"])
        if pair is None:
            continue

        true_rec = pair["true_rec"]
        full_ids = tokenizer(true_rec.text, return_tensors="pt")["input_ids"][0].tolist()
        div_point = item["divergence_point"]
        prefix_ids = full_ids[:div_point + 1]

        branch_scores = []
        for branch in item["branches"]:
            tok_ids = tokenizer.encode(branch["token"], add_special_tokens=False)
            if not tok_ids:
                continue
            branch_id = tok_ids[0]

            branch_seq = prefix_ids + [branch_id]
            remaining = len(full_ids) - len(branch_seq) + 5
            max_gen = max(remaining, 10)

            # Generate + score
            input_ids = torch.tensor([branch_seq], device=device)
            gen_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_gen,
                do_sample=False,
            )
            full_seq = gen_output[0].cpu().tolist()

            # Teacher-forced scoring
            tf_input = torch.tensor([full_seq], device=device)
            outputs = model(input_ids=tf_input)
            logits = outputs.logits[0, :-1, :].cpu().float()
            probs = torch.softmax(logits, dim=-1)
            target_ids = torch.tensor(full_seq[1:])

            per_tok_conf = []
            for t in range(len(target_ids)):
                p = probs[t, target_ids[t]].item()
                per_tok_conf.append(p)

            # Only look at continuation (after branch point)
            continuation_conf = per_tok_conf[len(prefix_ids):]
            if not continuation_conf:
                continue

            min_conf = min(continuation_conf)
            n_below_01 = sum(1 for c in continuation_conf if c < 0.1)

            branch_scores.append({
                "is_correct": branch["is_correct"],
                "min_conf": float(min_conf),
                "n_below_01": n_below_01,
            })

        if not branch_scores or not any(b["is_correct"] for b in branch_scores):
            continue

        n_items += 1

        # Select by highest minimum confidence (avoid worst case)
        best_min = max(branch_scores, key=lambda b: b["min_conf"])
        if best_min["is_correct"]:
            n_correct_min += 1

        # Select by fewest tokens below 0.1
        best_thresh = min(branch_scores, key=lambda b: b["n_below_01"])
        if best_thresh["is_correct"]:
            n_correct_below_thresh += 1

        if (idx + 1) % 20 == 0:
            print(f"      [{idx+1}/{len(k5_data)}] processed")

    results = {
        "min_conf_selection": float(n_correct_min / n_items) if n_items else 0,
        "below_thresh_selection": float(n_correct_below_thresh / n_items) if n_items else 0,
        "n_items": n_items,
    }

    print(f"    Items tested: {n_items}")
    print(f"    Max-of-min-conf selection: {n_correct_min}/{n_items} ({results['min_conf_selection']:.1%})")
    print(f"    Fewest-below-0.1 selection: {n_correct_below_thresh}/{n_items} ({results['below_thresh_selection']:.1%})")

    return results


# ===================================================================
# Path A: Better Localization
# ===================================================================

def run_path_a(pairs_lookup, step1_data):
    """Path A: Test alternative localization methods using existing data."""
    print("\n" + "=" * 70)
    print("PATH A: BETTER LOCALIZATION")
    print("=" * 70)

    model, tokenizer, device = load_model(MODEL_NAME, dtype=torch.float16)

    results = {
        "baseline_min_conf": 0,
        "top_gap": 0,
        "local_drop": 0,
        "entropy_max": 0,
        "entropy_spike": 0,
        "n_items": 0,
    }

    n_baseline = 0
    n_top_gap = 0
    n_local_drop = 0
    n_entropy_max = 0
    n_entropy_spike = 0
    n_items = 0

    for item in step1_data:
        pair_id = item["pair_id"]
        div_point = item["divergence_point"]
        pair = pairs_lookup.get(pair_id)
        if pair is None or div_point is None:
            continue

        false_rec = pair["false_rec"]
        tokens = false_rec.tokens
        if len(tokens) < 3:
            continue

        n_items += 1

        # Extract per-token metrics
        confidences = [t.top1_prob for t in tokens]
        entropies = [t.entropy for t in tokens]

        # Baseline: position with lowest confidence
        min_conf_pos = int(np.argmin(confidences))
        if min_conf_pos == div_point:
            n_baseline += 1

        # A1: Top-1/Top-2 gap
        # Need to compute from existing top5_probs
        top_gaps = []
        for t in tokens:
            if len(t.top5_probs) >= 2:
                gap = t.top5_probs[0] - t.top5_probs[1]
            else:
                gap = 1.0  # no alternative
            top_gaps.append(gap)
        min_gap_pos = int(np.argmin(top_gaps))
        if min_gap_pos == div_point:
            n_top_gap += 1

        # A2: Local confidence drop
        drops = []
        for i in range(len(confidences)):
            neighbors = []
            if i > 0:
                neighbors.append(confidences[i - 1])
            if i > 1:
                neighbors.append(confidences[i - 2])
            if i < len(confidences) - 1:
                neighbors.append(confidences[i + 1])
            if i < len(confidences) - 2:
                neighbors.append(confidences[i + 2])
            if neighbors:
                drop = np.mean(neighbors) - confidences[i]
            else:
                drop = 0
            drops.append(drop)
        max_drop_pos = int(np.argmax(drops))
        if max_drop_pos == div_point:
            n_local_drop += 1

        # A3: Highest entropy
        max_ent_pos = int(np.argmax(entropies))
        if max_ent_pos == div_point:
            n_entropy_max += 1

        # A3b: Largest entropy increase from previous position
        ent_spikes = [0]  # first position has no previous
        for i in range(1, len(entropies)):
            ent_spikes.append(entropies[i] - entropies[i - 1])
        max_spike_pos = int(np.argmax(ent_spikes))
        if max_spike_pos == div_point:
            n_entropy_spike += 1

    results = {
        "baseline_min_conf": float(n_baseline / n_items) if n_items else 0,
        "top_gap": float(n_top_gap / n_items) if n_items else 0,
        "local_drop": float(n_local_drop / n_items) if n_items else 0,
        "entropy_max": float(n_entropy_max / n_items) if n_items else 0,
        "entropy_spike": float(n_entropy_spike / n_items) if n_items else 0,
        "n_items": n_items,
    }

    print(f"\n  Localization Results ({n_items} items):")
    print(f"  {'Method':<30s} {'Finds Error':<20s}")
    print(f"  {'-' * 50}")
    print(f"  {'Lowest confidence (baseline)':<30s} {n_baseline}/{n_items} ({results['baseline_min_conf']:.1%})")
    print(f"  {'Smallest top-1/top-2 gap':<30s} {n_top_gap}/{n_items} ({results['top_gap']:.1%})")
    print(f"  {'Largest local conf drop':<30s} {n_local_drop}/{n_items} ({results['local_drop']:.1%})")
    print(f"  {'Highest entropy':<30s} {n_entropy_max}/{n_items} ({results['entropy_max']:.1%})")
    print(f"  {'Largest entropy spike':<30s} {n_entropy_spike}/{n_items} ({results['entropy_spike']:.1%})")

    # Per-regime breakdown
    for regime in ["R1", "R2"]:
        regime_items = [s for s in step1_data if s["regime"] == regime]
        if not regime_items:
            continue
        nr = 0
        nr_gap = 0
        nr_drop = 0
        nr_ent = 0
        nr_spike = 0
        nr_n = 0
        for item in regime_items:
            pair = pairs_lookup.get(item["pair_id"])
            if pair is None or item["divergence_point"] is None:
                continue
            false_rec = pair["false_rec"]
            tokens = false_rec.tokens
            if len(tokens) < 3:
                continue
            nr_n += 1
            div_point = item["divergence_point"]
            confidences = [t.top1_prob for t in tokens]
            entropies = [t.entropy for t in tokens]

            if int(np.argmin(confidences)) == div_point:
                nr += 1
            top_gaps = []
            for t in tokens:
                gap = t.top5_probs[0] - t.top5_probs[1] if len(t.top5_probs) >= 2 else 1.0
                top_gaps.append(gap)
            if int(np.argmin(top_gaps)) == div_point:
                nr_gap += 1
            drops = []
            for i in range(len(confidences)):
                neighbors = []
                if i > 0: neighbors.append(confidences[i-1])
                if i > 1: neighbors.append(confidences[i-2])
                if i < len(confidences)-1: neighbors.append(confidences[i+1])
                if i < len(confidences)-2: neighbors.append(confidences[i+2])
                drops.append(np.mean(neighbors) - confidences[i] if neighbors else 0)
            if int(np.argmax(drops)) == div_point:
                nr_drop += 1
            if int(np.argmax(entropies)) == div_point:
                nr_ent += 1
            spikes = [0] + [entropies[i] - entropies[i-1] for i in range(1, len(entropies))]
            if int(np.argmax(spikes)) == div_point:
                nr_spike += 1

        print(f"\n  [{regime}] ({nr_n} items):")
        print(f"    Baseline: {nr}/{nr_n} ({nr/nr_n:.1%})")
        print(f"    Top gap:  {nr_gap}/{nr_n} ({nr_gap/nr_n:.1%})")
        print(f"    Local drop: {nr_drop}/{nr_n} ({nr_drop/nr_n:.1%})")
        print(f"    Entropy max: {nr_ent}/{nr_n} ({nr_ent/nr_n:.1%})")
        print(f"    Entropy spike: {nr_spike}/{nr_n} ({nr_spike/nr_n:.1%})")

    return results


# ===================================================================
# Visualization
# ===================================================================

def plot_pathB_results(all_pathB_results):
    """Plot Path B selection comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    rates = []
    colors = []

    # Baseline
    methods.append("Mean Conf\n(baseline)")
    rates.append(0.146)  # from step2
    colors.append("#9E9E9E")

    # Random
    methods.append("Random\n(K=5)")
    rates.append(0.20)
    colors.append("#BDBDBD")

    # B1 probe - best layer
    if "probe" in all_pathB_results:
        best_layer = max(all_pathB_results["probe"].items(),
                        key=lambda x: x[1]["selection_rate"])
        methods.append(f"Hidden Probe\n(layer {best_layer[0]})")
        rates.append(best_layer[1]["selection_rate"])
        colors.append("#AB47BC")

    # B2 entropy
    if "entropy" in all_pathB_results:
        ent = all_pathB_results["entropy"]
        best_ent = max(ent["mean_entropy_selection"],
                      ent["min_entropy_selection"],
                      ent["pos1_entropy_selection"])
        methods.append("Entropy\n(best)")
        rates.append(best_ent)
        colors.append("#42A5F5")

    # B3 backward
    if "backward" in all_pathB_results:
        methods.append("Backward\nConf")
        rates.append(all_pathB_results["backward"]["backward_conf_selection"])
        colors.append("#66BB6A")

    # B4 min conf
    if "minconf" in all_pathB_results:
        mc = all_pathB_results["minconf"]
        best_mc = max(mc["min_conf_selection"], mc["below_thresh_selection"])
        methods.append("Min Conf\n(best)")
        rates.append(best_mc)
        colors.append("#FFA726")

    bars = ax.bar(methods, rates, color=colors, edgecolor="white", linewidth=1.5)
    ax.axhline(y=0.20, color="gray", linestyle="--", alpha=0.5, label="Random (20%)")
    ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.5, label="Kill threshold (25%)")
    ax.axhline(y=0.35, color="green", linestyle="--", alpha=0.5, label="Success threshold (35%)")
    ax.set_ylabel("Correct Branch Selected (%)")
    ax.set_title("Path B: Selection Method Comparison (K=5, Oracle Branches)")
    ax.set_ylim(0, max(max(rates) * 1.3, 0.5))
    ax.legend(fontsize=9)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "selection_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/3] selection_comparison.png")


def plot_pathA_results(pathA_results):
    """Plot Path A localization comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["Lowest\nConf", "Top-1/2\nGap", "Local\nDrop", "Max\nEntropy", "Entropy\nSpike"]
    rates = [
        pathA_results["baseline_min_conf"],
        pathA_results["top_gap"],
        pathA_results["local_drop"],
        pathA_results["entropy_max"],
        pathA_results["entropy_spike"],
    ]
    colors = ["#9E9E9E", "#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]

    bars = ax.bar(methods, rates, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Finds Error Position (%)")
    ax.set_title("Path A: Localization Method Comparison")
    ax.set_ylim(0, max(max(rates) * 1.3, 0.5))

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "localization_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/3] localization_comparison.png")


def plot_pipeline(all_pathB_results, pathA_results):
    """Plot the full pipeline improvement."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute pipeline accuracy: localization × selection × reachability
    reachability = 0.55  # from step1

    pipelines = []

    # Baseline
    pipelines.append(("Current\nBlind-5", 0.011, 1.5, "#9E9E9E"))

    # Best-of-N
    pipelines.append(("Best-of-5", 0.056, 5.0, "#FFA726"))
    pipelines.append(("Best-of-10", 0.056, 10.0, "#FB8C00"))

    # Best selection × best localization
    best_sel = 0.146  # default to baseline
    if "probe" in all_pathB_results:
        probe_sel = max(v["selection_rate"] for v in all_pathB_results["probe"].values())
        best_sel = max(best_sel, probe_sel)
    if "entropy" in all_pathB_results:
        ent = all_pathB_results["entropy"]
        best_sel = max(best_sel, ent["mean_entropy_selection"],
                      ent["min_entropy_selection"])
    if "backward" in all_pathB_results:
        best_sel = max(best_sel, all_pathB_results["backward"]["backward_conf_selection"])
    if "minconf" in all_pathB_results:
        mc = all_pathB_results["minconf"]
        best_sel = max(best_sel, mc["min_conf_selection"])

    best_loc = max(pathA_results["baseline_min_conf"], pathA_results["top_gap"],
                   pathA_results["local_drop"], pathA_results["entropy_max"],
                   pathA_results["entropy_spike"])

    pipeline_acc = best_loc * best_sel * reachability
    pipelines.append((f"Best Pipeline\n(loc={best_loc:.0%}, sel={best_sel:.0%})",
                      pipeline_acc, 1.5, "#AB47BC"))

    # Oracle upper bound
    pipelines.append(("Oracle-5\n(upper bound)", 0.146, 1.5, "#42A5F5"))

    for label, acc, compute, color in pipelines:
        ax.scatter(compute, acc, s=150, color=color, edgecolors="black",
                   linewidth=0.5, zorder=5)
        ax.annotate(label, (compute, acc), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Compute Multiplier")
    ax.set_ylabel("Accuracy")
    ax.set_title("Full Pipeline: Improved Selection + Localization")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, max(0.2, max(p[1] for p in pipelines) * 1.3))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pipeline_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/3] pipeline_improvement.png")


# ===================================================================
# Main
# ===================================================================

def run_experiment():
    total_start = time.time()

    print("=" * 70)
    print("TARGETED RESAMPLING: PATH A/B IMPROVEMENTS")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)

    # Load data
    oracle_data = load_oracle_data()
    blind_data = load_blind_data()
    step1_data = load_step1_data()
    pairs_lookup = load_paired_records()

    print(f"\n  Oracle results: {len(oracle_data)}")
    print(f"  Blind results: {len(blind_data)}")
    print(f"  Step1 results: {len(step1_data)}")
    print(f"  Paired records: {len(pairs_lookup)}")

    # ===================================================================
    # PATH B: BETTER SELECTION
    # ===================================================================
    print("\n" + "=" * 70)
    print("PATH B: BETTER SELECTION CRITERIA")
    print("=" * 70)

    all_pathB = {}

    # B1: Hidden state probe
    probe_results = run_probe_b1(oracle_data, pairs_lookup)
    all_pathB["probe"] = probe_results

    # B2: Entropy reduction
    entropy_results = run_entropy_b2(oracle_data, pairs_lookup)
    all_pathB["entropy"] = entropy_results

    # B3: Backward confidence
    backward_results = run_backward_b3(oracle_data, pairs_lookup)
    all_pathB["backward"] = backward_results

    # B4: Min confidence
    minconf_results = run_minconf_b4(oracle_data, pairs_lookup)
    all_pathB["minconf"] = minconf_results

    # Summary
    print("\n" + "=" * 70)
    print("PATH B SUMMARY")
    print("=" * 70)
    print(f"\n  {'Method':<35s} {'Selection Rate':<20s}")
    print(f"  {'-' * 55}")
    print(f"  {'Mean confidence (baseline)':<35s} {'14.6%':<20s}")
    print(f"  {'Random (K=5)':<35s} {'20.0%':<20s}")

    if probe_results:
        best_probe = max(probe_results.items(), key=lambda x: x[1]["selection_rate"])
        print(f"  {'Hidden probe (best layer ' + str(best_probe[0]) + ')':<35s} "
              f"{best_probe[1]['selection_rate']:.1%} (AUC={best_probe[1]['auc']:.3f})")

    print(f"  {'Entropy mean':<35s} {entropy_results.get('mean_entropy_selection', 0):.1%}")
    print(f"  {'Entropy min':<35s} {entropy_results.get('min_entropy_selection', 0):.1%}")
    print(f"  {'Entropy pos+1':<35s} {entropy_results.get('pos1_entropy_selection', 0):.1%}")
    print(f"  {'Backward confidence':<35s} {backward_results.get('backward_conf_selection', 0):.1%}")
    print(f"  {'Max-of-min confidence':<35s} {minconf_results.get('min_conf_selection', 0):.1%}")
    print(f"  {'Fewest below 0.1':<35s} {minconf_results.get('below_thresh_selection', 0):.1%}")

    # Gate check
    all_rates = []
    if probe_results:
        all_rates.extend(v["selection_rate"] for v in probe_results.values())
    all_rates.append(entropy_results.get("mean_entropy_selection", 0))
    all_rates.append(entropy_results.get("min_entropy_selection", 0))
    all_rates.append(backward_results.get("backward_conf_selection", 0))
    all_rates.append(minconf_results.get("min_conf_selection", 0))

    best_rate = max(all_rates) if all_rates else 0
    print(f"\n  BEST SELECTION RATE: {best_rate:.1%}")
    if best_rate < 0.25:
        print("  *** BELOW 25% GATE. Selection is the binding constraint. ***")
    elif best_rate >= 0.35:
        print("  *** ABOVE 35% SUCCESS THRESHOLD. Selection is viable! ***")
    else:
        print("  *** Between 25-35%. Marginal improvement. ***")

    # Save Path B
    save_pathB = {
        "probe": {str(k): v for k, v in probe_results.items()},
        "entropy": entropy_results,
        "backward": backward_results,
        "minconf": minconf_results,
        "best_rate": float(best_rate),
    }
    with open(RESULTS_DIR / "pathB_selection.json", "w") as f:
        json.dump(save_pathB, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'pathB_selection.json'}")

    # ===================================================================
    # PATH A: BETTER LOCALIZATION
    # ===================================================================
    pathA_results = run_path_a(pairs_lookup, step1_data)

    with open(RESULTS_DIR / "pathA_localization.json", "w") as f:
        json.dump(pathA_results, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'pathA_localization.json'}")

    # ===================================================================
    # VISUALIZATION
    # ===================================================================
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    plot_pathB_results(all_pathB)
    plot_pathA_results(pathA_results)
    plot_pipeline(all_pathB, pathA_results)

    # Save full pipeline summary
    pipeline_summary = {
        "pathB": save_pathB,
        "pathA": pathA_results,
    }
    with open(RESULTS_DIR / "full_pipeline.json", "w") as f:
        json.dump(pipeline_summary, f, indent=2)

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE ({total_time:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
