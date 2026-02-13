"""Model registry and scaling infrastructure for Phase 2."""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional

from .schema import ConfidenceRecord, load_records, save_records
from .utils import SCALING_RESULTS_DIR
from .engine import load_model, unload_model


# ---------------------------------------------------------------------------
# Model registry: all Pythia sizes we test
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "160m": {
        "name": "EleutherAI/pythia-160m",
        "params": 162_322_944,
        "dtype": torch.float32,
    },
    "410m": {
        "name": "EleutherAI/pythia-410m",
        "params": 405_334_016,
        "dtype": torch.float32,
    },
    "1b": {
        "name": "EleutherAI/pythia-1b",
        "params": 1_011_781_632,
        "dtype": torch.float32,
    },
    "1.4b": {
        "name": "EleutherAI/pythia-1.4b",
        "params": 1_414_647_808,
        "dtype": torch.float32,
    },
    "2.8b": {
        "name": "EleutherAI/pythia-2.8b",
        "params": 2_775_208_960,
        "dtype": torch.float16,
    },
    "6.9b": {
        "name": "EleutherAI/pythia-6.9b",
        "params": 6_857_302_016,
        "dtype": torch.float16,
    },
    "12b": {
        "name": "EleutherAI/pythia-12b",
        "params": 11_846_072_320,
        "dtype": torch.float16,
    },
}

# Ordered smallest → largest (always process this way for memory safety)
SCALING_MODELS = ["160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]

# Full set including 12B (use for the final scaling point)
SCALING_MODELS_ALL = ["160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]

# Param counts for log-scale plots
PARAM_COUNTS = {k: v["params"] for k, v in MODEL_REGISTRY.items()}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_scaling_output_path(experiment: str, model_size: str) -> Path:
    """Return path like data/results/scaling/{experiment}_{model_size}.jsonl"""
    return SCALING_RESULTS_DIR / f"{experiment}_{model_size}.jsonl"


def load_scaling_results(experiment: str) -> dict[str, list[ConfidenceRecord]]:
    """Load results for all available model sizes for an experiment."""
    results = {}
    for size in SCALING_MODELS:
        path = get_scaling_output_path(experiment, size)
        if path.exists():
            results[size] = load_records(path)
    return results


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_scaling_model(size: str, revision: str = "main"):
    """Load a model from the registry by size key.

    Returns (model, tokenizer, device, model_name, dtype).
    """
    spec = MODEL_REGISTRY[size]
    model, tokenizer, device = load_model(
        model_name=spec["name"],
        revision=revision,
        dtype=spec["dtype"],
    )
    return model, tokenizer, device, spec["name"], spec["dtype"]


def model_display_name(size: str) -> str:
    """Human-readable label for plots, e.g. '160M' or '6.9B'."""
    return size.upper().replace("M", "M").replace("B", "B")


# ---------------------------------------------------------------------------
# Runtime estimation
# ---------------------------------------------------------------------------

# Rough scaling factors relative to 160M (based on param count ratio)
_SPEED_FACTORS = {
    "160m": 1.0,
    "410m": 2.5,
    "1b": 6.3,
    "1.4b": 8.7,
    "2.8b": 12.0,    # float16 compensates somewhat
    "6.9b": 25.0,    # float16 compensates somewhat
    "12b": 42.0,     # float16, ~24GB, fits in 96GB unified memory
}


def estimate_runtime(model_size: str, num_texts: int,
                     base_time_per_text: float = 0.3) -> float:
    """Rough runtime estimate in seconds for one model size."""
    factor = _SPEED_FACTORS.get(model_size, 1.0)
    return num_texts * base_time_per_text * factor


def print_runtime_estimates(num_texts: int, base_time: float = 0.3):
    """Print estimated runtimes for all model sizes."""
    print(f"\nEstimated runtimes ({num_texts} texts, {base_time:.1f}s/text at 160M):")
    total = 0
    for size in SCALING_MODELS:
        est = estimate_runtime(size, num_texts, base_time)
        total += est
        print(f"  {size:>5s}: {est/60:6.1f} min")
    print(f"  Total: {total/60:.1f} min")
