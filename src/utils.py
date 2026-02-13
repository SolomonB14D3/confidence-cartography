"""Shared utilities for confidence-cartography."""

import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all absolute, anchored to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
PROMPTS_DIR = DATA_DIR / "prompts"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Phase 2 subdirectories
SCALING_RESULTS_DIR = RESULTS_DIR / "scaling"
SCALING_FIGURES_DIR = FIGURES_DIR / "scaling"
SHAPES_RESULTS_DIR = RESULTS_DIR / "shapes"
SHAPES_FIGURES_DIR = FIGURES_DIR / "shapes"
MANDELA_RESULTS_DIR = RESULTS_DIR / "mandela"
MANDELA_FIGURES_DIR = FIGURES_DIR / "mandela"

# Ensure output dirs exist at import time
for _d in [RESULTS_DIR, FIGURES_DIR, SCALING_RESULTS_DIR, SCALING_FIGURES_DIR,
           SHAPES_RESULTS_DIR, SHAPES_FIGURES_DIR, MANDELA_RESULTS_DIR,
           MANDELA_FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Pick the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def truncate_token(token_str: str, max_len: int = 12) -> str:
    """Truncate a token string for display in plots."""
    if len(token_str) > max_len:
        return token_str[:max_len - 1] + "~"
    return token_str
