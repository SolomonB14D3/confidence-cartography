"""
Training utilities for Phase 3 fine-tuning experiments.

Provides:
  - Custom loss functions (confidence penalty, gap reward)
  - Fact pair loading and batching
  - Training loop with periodic confidence evaluation
  - Checkpoint management
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import PROJECT_ROOT, get_device


# ---------------------------------------------------------------------------
# Fact pair data
# ---------------------------------------------------------------------------

@dataclass
class FactPair:
    """A matched true/false claim pair for confidence training."""
    true_text: str
    false_text: str
    pair_id: str
    domain: str = ""


def load_fact_pairs(path: Path) -> list[FactPair]:
    """Load fact pairs from JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            pairs.append(FactPair(**d))
    return pairs


def save_fact_pairs(pairs: list[FactPair], path: Path):
    """Save fact pairs to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps({
                "true_text": p.true_text,
                "false_text": p.false_text,
                "pair_id": p.pair_id,
                "domain": p.domain,
            }) + "\n")


# ---------------------------------------------------------------------------
# Confidence computation (for loss functions)
# ---------------------------------------------------------------------------

def compute_token_confidence(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Compute mean token probability for a text (teacher-forced).

    Returns:
        mean_prob: scalar mean probability
        per_token_probs: tensor of per-token probabilities (for gradient flow)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=256).to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    if seq_len < 2:
        return 0.0, torch.tensor(0.0, device=device, requires_grad=True)

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]  # (1, seq_len-1, vocab)
    targets = input_ids[:, 1:]           # (1, seq_len-1)

    probs = F.softmax(logits, dim=-1)    # (1, seq_len-1, vocab)
    target_probs = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

    mean_prob = target_probs.mean()
    return mean_prob.item(), mean_prob


# ---------------------------------------------------------------------------
# Custom loss functions for Experiment 7
# ---------------------------------------------------------------------------

class ConfidencePenaltyLoss(nn.Module):
    """Model B: standard LM loss + penalty for confident false claims.

    loss = CE(logits, targets) + alpha * max(0, mean_prob_on_false - threshold)^2
    """

    def __init__(self, alpha: float = 0.1, threshold: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(
        self,
        lm_loss: torch.Tensor,
        false_mean_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if false_mean_prob is None:
            return lm_loss

        penalty = F.relu(false_mean_prob - self.threshold) ** 2
        return lm_loss + self.alpha * penalty


class GapRewardLoss(nn.Module):
    """Model C: standard LM loss - reward for large true-false confidence gap.

    loss = CE(logits, targets) - beta * max(0, true_mean_prob - false_mean_prob)
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        lm_loss: torch.Tensor,
        true_mean_prob: Optional[torch.Tensor] = None,
        false_mean_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if true_mean_prob is None or false_mean_prob is None:
            return lm_loss

        gap = F.relu(true_mean_prob - false_mean_prob)
        return lm_loss - self.beta * gap


# ---------------------------------------------------------------------------
# Training step helpers
# ---------------------------------------------------------------------------

def standard_lm_step(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    """Standard causal LM loss on a text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=256).to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss


def confidence_penalty_step(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    base_text: str,
    false_text: str,
    device: torch.device,
    penalty_fn: ConfidencePenaltyLoss,
) -> torch.Tensor:
    """LM loss on base_text + confidence penalty on false_text."""
    # Standard LM loss on base text
    lm_loss = standard_lm_step(model, tokenizer, base_text, device)

    # Confidence on false claim
    _, false_prob = compute_token_confidence(model, tokenizer, false_text, device)

    return penalty_fn(lm_loss, false_prob)


def gap_reward_step(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    base_text: str,
    true_text: str,
    false_text: str,
    device: torch.device,
    reward_fn: GapRewardLoss,
) -> torch.Tensor:
    """LM loss on base_text + gap reward between true and false confidence."""
    lm_loss = standard_lm_step(model, tokenizer, base_text, device)

    _, true_prob = compute_token_confidence(model, tokenizer, true_text, device)
    _, false_prob = compute_token_confidence(model, tokenizer, false_text, device)

    return reward_fn(lm_loss, true_prob, false_prob)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_truth_detection(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    pairs: list[FactPair],
    device: torch.device,
) -> dict:
    """Evaluate win rate and confidence gap on fact pairs."""
    model.eval()
    wins = 0
    deltas = []

    for pair in pairs:
        true_conf, _ = compute_token_confidence(model, tokenizer, pair.true_text, device)
        false_conf, _ = compute_token_confidence(model, tokenizer, pair.false_text, device)
        delta = true_conf - false_conf
        deltas.append(delta)
        if delta > 0:
            wins += 1

    deltas = np.array(deltas)
    return {
        "win_rate": wins / len(pairs) if pairs else 0,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "n_pairs": len(pairs),
    }


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int = 256,
) -> float:
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        seq_len = inputs["input_ids"].shape[1]
        total_loss += outputs.loss.item() * (seq_len - 1)
        total_tokens += seq_len - 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return float(np.exp(avg_loss))
