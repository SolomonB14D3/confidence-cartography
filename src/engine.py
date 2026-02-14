"""Core confidence extraction engine for causal language models."""

from __future__ import annotations

import gc
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from .schema import TokenAnalysis, ConfidenceRecord
from .utils import get_device


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_device = None
_current_revision = None
_current_model_name = None


def load_model(
    model_name: str = "EleutherAI/pythia-160m",
    revision: str = "main",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple:
    """Load model and tokenizer. Caches globally (singleton pattern).

    Reloads if model_name or revision changes.
    The dtype parameter controls model weight precision:
      - None → torch.float32 (default, safest)
      - torch.float16 → faster inference, fine because softmax/entropy
        analysis already upcasts to float32 on CPU.
    """
    global _model, _tokenizer, _device, _current_revision, _current_model_name

    if (_model is not None
            and _current_model_name == model_name
            and _current_revision == revision):
        return _model, _tokenizer, _device

    # If switching models, unload first to free memory
    if _model is not None:
        unload_model()

    _device = device or get_device()
    use_dtype = dtype or torch.float32
    print(f"Loading {model_name} (revision={revision}, dtype={use_dtype}) on {_device}...")

    _tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    # Set pad token only if the model doesn't already have one configured
    # (Pythia needs this; Qwen/Llama/etc. already set their own pad tokens)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    try:
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=use_dtype,
        ).to(_device)
    except OSError:
        # Some older checkpoints only have pytorch_model.bin, not safetensors
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=use_dtype,
            use_safetensors=False,
        ).to(_device)
    _model.eval()
    _current_revision = revision
    _current_model_name = model_name

    print(f"  Loaded. Vocab={_tokenizer.vocab_size}, Device={_device}")
    return _model, _tokenizer, _device


def unload_model():
    """Free the cached model (useful when switching models/checkpoints)."""
    global _model, _tokenizer, _device, _current_revision, _current_model_name
    _model = None
    _tokenizer = None
    _device = None
    _current_revision = None
    _current_model_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Teacher-forced analysis of fixed text
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_fixed_text(
    text: str,
    category: str = "",
    label: str = "",
    model_name: str = "EleutherAI/pythia-160m",
    revision: str = "main",
    top_k: int = 5,
    dtype: Optional[torch.dtype] = None,
) -> ConfidenceRecord:
    """
    Teacher-forced confidence analysis of fixed text.

    Feed the entire text through the model in one forward pass.
    At each position t, the model outputs a distribution over the next token.
    We measure how much probability it assigns to the ACTUAL token at position t+1.

    Returns a ConfidenceRecord with per-token analysis.
    """
    model, tokenizer, device = load_model(model_name, revision, dtype=dtype)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]          # shape: (1, seq_len)
    seq_len = input_ids.shape[1]

    if seq_len < 2:
        raise ValueError(f"Text too short ({seq_len} tokens): '{text}'")

    # Forward pass -- get logits for all positions
    outputs = model(**inputs)
    logits = outputs.logits                  # shape: (1, seq_len, vocab_size)

    # We analyze positions 0..seq_len-2:
    #   logits[0, t, :] predicts token at position t+1
    #   actual token at t+1 is input_ids[0, t+1]
    #
    # So we have (seq_len - 1) analysis points.

    # Move to CPU for numpy-heavy analysis
    logits_cpu = logits[0, :-1, :].cpu().float()  # (seq_len-1, vocab_size)
    probs = torch.softmax(logits_cpu, dim=-1)      # (seq_len-1, vocab_size)

    # Target token IDs: tokens at positions 1..seq_len-1
    target_ids = input_ids[0, 1:].cpu()            # (seq_len-1,)

    # Per-position analysis
    token_analyses = []
    top1_probs_list = []
    entropies_list = []

    for t in range(seq_len - 1):
        target_id = target_ids[t].item()
        target_str = tokenizer.decode([target_id])
        prob_dist = probs[t]                       # (vocab_size,)

        # Probability of the actual next token
        actual_prob = prob_dist[target_id].item()

        # Rank of the actual token (0 = model's top pick)
        sorted_indices = torch.argsort(prob_dist, descending=True)
        rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item()

        # Entropy in bits: -sum(p * log2(p))
        # Add epsilon to avoid log(0)
        entropy = -(prob_dist * torch.log2(prob_dist + 1e-12)).sum().item()

        # Top-k predictions
        topk_probs_t, topk_ids_t = torch.topk(prob_dist, k=top_k)
        topk_strs = [tokenizer.decode([tid]) for tid in topk_ids_t.tolist()]

        ta = TokenAnalysis(
            position=t,
            token_id=target_id,
            token_str=target_str,
            top1_prob=actual_prob,
            top1_rank=rank,
            entropy=entropy,
            top5_tokens=topk_strs,
            top5_probs=topk_probs_t.tolist(),
            top5_ids=topk_ids_t.tolist(),
        )
        token_analyses.append(ta)
        top1_probs_list.append(actual_prob)
        entropies_list.append(entropy)

    # Summary statistics
    probs_arr = np.array(top1_probs_list)
    ent_arr = np.array(entropies_list)
    min_idx = int(np.argmin(probs_arr))

    return ConfidenceRecord(
        text=text,
        category=category,
        label=label,
        mode="fixed",
        num_tokens=seq_len,
        tokens=token_analyses,
        mean_top1_prob=float(probs_arr.mean()),
        mean_entropy=float(ent_arr.mean()),
        std_top1_prob=float(probs_arr.std()),
        std_entropy=float(ent_arr.std()),
        min_confidence_pos=min_idx,
        min_confidence_token=token_analyses[min_idx].token_str,
        min_confidence_value=float(probs_arr[min_idx]),
        model_name=model_name,
        model_revision=revision,
    )


# ---------------------------------------------------------------------------
# Generation-based analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_generation(
    prompt: str,
    max_new_tokens: int = 50,
    category: str = "",
    label: str = "",
    model_name: str = "EleutherAI/pythia-160m",
    revision: str = "main",
    top_k: int = 5,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> ConfidenceRecord:
    """
    Generate text from a prompt and capture confidence at each generated token.

    Uses model.generate() with output_scores=True to get the logits at each
    generation step. Then analyzes the model's confidence in its own choices.
    """
    model, tokenizer, device = load_model(model_name, revision)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    gen_output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # gen_output.sequences: (1, prompt_len + generated_len)
    # gen_output.scores: tuple of (1, vocab_size) tensors, one per generated token
    generated_ids = gen_output.sequences[0, prompt_len:]
    scores = gen_output.scores

    full_text = tokenizer.decode(gen_output.sequences[0])

    token_analyses = []
    top1_probs_list = []
    entropies_list = []

    for t, score_t in enumerate(scores):
        logits_t = score_t[0].cpu().float()        # (vocab_size,)
        prob_dist = torch.softmax(logits_t, dim=-1)
        chosen_id = generated_ids[t].item()
        chosen_str = tokenizer.decode([chosen_id])

        actual_prob = prob_dist[chosen_id].item()
        sorted_indices = torch.argsort(prob_dist, descending=True)
        rank = (sorted_indices == chosen_id).nonzero(as_tuple=True)[0].item()
        entropy = -(prob_dist * torch.log2(prob_dist + 1e-12)).sum().item()

        topk_probs_t, topk_ids_t = torch.topk(prob_dist, k=top_k)
        topk_strs = [tokenizer.decode([tid]) for tid in topk_ids_t.tolist()]

        ta = TokenAnalysis(
            position=prompt_len + t,
            token_id=chosen_id,
            token_str=chosen_str,
            top1_prob=actual_prob,
            top1_rank=rank,
            entropy=entropy,
            top5_tokens=topk_strs,
            top5_probs=topk_probs_t.tolist(),
            top5_ids=topk_ids_t.tolist(),
        )
        token_analyses.append(ta)
        top1_probs_list.append(actual_prob)
        entropies_list.append(entropy)

    probs_arr = np.array(top1_probs_list) if top1_probs_list else np.array([0.0])
    ent_arr = np.array(entropies_list) if entropies_list else np.array([0.0])
    min_idx = int(np.argmin(probs_arr))

    return ConfidenceRecord(
        text=full_text,
        category=category,
        label=label,
        mode="generated",
        num_tokens=len(gen_output.sequences[0]),
        tokens=token_analyses,
        mean_top1_prob=float(probs_arr.mean()),
        mean_entropy=float(ent_arr.mean()),
        std_top1_prob=float(probs_arr.std()),
        std_entropy=float(ent_arr.std()),
        min_confidence_pos=min_idx,
        min_confidence_token=token_analyses[min_idx].token_str if token_analyses else "",
        min_confidence_value=float(probs_arr[min_idx]),
        model_name=model_name,
        model_revision=revision,
        metadata={"prompt": prompt, "max_new_tokens": max_new_tokens},
    )
