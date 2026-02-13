"""Shared visualization functions for scaling experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .scaling import SCALING_MODELS, PARAM_COUNTS, model_display_name

# ---------------------------------------------------------------------------
# Color palette for model sizes (sequential, darker = larger)
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "160m":  "#a6d96a",
    "410m":  "#66bd63",
    "1b":    "#1a9850",
    "1.4b":  "#fdae61",
    "2.8b":  "#f46d43",
    "6.9b":  "#d73027",
}

MODEL_MARKERS = {
    "160m": "o", "410m": "s", "1b": "D",
    "1.4b": "^", "2.8b": "p", "6.9b": "*",
}


def _sizes_to_params(sizes: list[str]) -> list[float]:
    return [PARAM_COUNTS[s] for s in sizes]


# ---------------------------------------------------------------------------
# Core scaling law plot
# ---------------------------------------------------------------------------

def plot_scaling_law(
    sizes: list[str],
    metrics: dict[str, list[float]],
    ylabel: str,
    title: str,
    save_path: Path,
    y_range: Optional[tuple] = None,
    hline: Optional[float] = None,
    hline_label: Optional[str] = None,
):
    """Canonical scaling law plot: log(params) on x-axis, metric on y-axis.

    Args:
        sizes: list of model size keys (e.g. ["160m", "410m", ...])
        metrics: dict of {metric_name: [values_per_size]}
        ylabel: y-axis label
        title: plot title
        save_path: where to save PNG
        y_range: optional (ymin, ymax)
        hline: optional horizontal reference line
        hline_label: label for the horizontal line
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6))

    params = _sizes_to_params(sizes)

    for name, values in metrics.items():
        ax.plot(params, values, "o-", linewidth=2.5, markersize=8, label=name)

    if hline is not None:
        ax.axhline(y=hline, color="gray", linestyle="--", alpha=0.6,
                    label=hline_label or "")

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # Custom x-tick labels
    ax.set_xticks(params)
    ax.set_xticklabels([model_display_name(s) for s in sizes], fontsize=10)

    if y_range:
        ax.set_ylim(y_range)

    if len(metrics) > 1 or hline is not None:
        ax.legend(fontsize=10)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# Scaling heatmap
# ---------------------------------------------------------------------------

def plot_scaling_heatmap(
    data: np.ndarray,
    y_labels: list[str],
    x_labels: list[str],
    title: str,
    save_path: Path,
    cmap: str = "RdYlGn",
    fmt: str = ".2f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Heatmap: rows=conditions, cols=model sizes, cells=metric values."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 1.5), max(5, len(y_labels) * 0.5)))

    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=x_labels, yticklabels=y_labels,
                ax=ax, vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor="white")

    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# ROC overlay for multiple model sizes
# ---------------------------------------------------------------------------

def plot_roc_overlay(
    roc_data: dict[str, tuple],
    title: str,
    save_path: Path,
):
    """Overlay ROC curves for multiple model sizes.

    roc_data: {size_key: (fpr_array, tpr_array, auc_value)}
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))

    for size in SCALING_MODELS:
        if size not in roc_data:
            continue
        fpr, tpr, auc = roc_data[size]
        ax.plot(fpr, tpr,
                color=MODEL_COLORS[size],
                linewidth=2.5,
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
    return fig, ax


# ---------------------------------------------------------------------------
# P-value cascade (significance thresholds)
# ---------------------------------------------------------------------------

def plot_pvalue_cascade(
    sizes: list[str],
    pvalues: dict[str, list[float]],
    title: str,
    save_path: Path,
):
    """Show p-values across model sizes with significance thresholds.

    pvalues: {test_name: [p_per_size]}
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    params = _sizes_to_params(sizes)

    for name, pvals in pvalues.items():
        ax.plot(params, pvals, "o-", linewidth=2, markersize=7, label=name)

    # Significance thresholds
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.6, label="p = 0.05")
    ax.axhline(y=0.01, color="darkred", linestyle=":", alpha=0.4, label="p = 0.01")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters", fontsize=12)
    ax.set_ylabel("p-value", fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.set_xticks(params)
    ax.set_xticklabels([model_display_name(s) for s in sizes], fontsize=10)

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


# ---------------------------------------------------------------------------
# Multi-panel scaling dashboard
# ---------------------------------------------------------------------------

def plot_scaling_dashboard(
    panels: list[dict],
    title: str,
    save_path: Path,
    ncols: int = 2,
):
    """Multi-panel figure showing headline results.

    panels: list of dicts with keys:
        - sizes: list of size keys
        - metrics: dict of {name: [values]}
        - ylabel: str
        - subtitle: str
        - hline: optional float
    """
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, panel in enumerate(panels):
        ax = axes[i]
        params = _sizes_to_params(panel["sizes"])
        for name, values in panel["metrics"].items():
            ax.plot(params, values, "o-", linewidth=2, markersize=6, label=name)
        if panel.get("hline") is not None:
            ax.axhline(y=panel["hline"], color="gray", linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xticks(params)
        ax.set_xticklabels([model_display_name(s) for s in panel["sizes"]],
                           fontsize=8)
        ax.set_ylabel(panel["ylabel"], fontsize=10)
        ax.set_title(panel["subtitle"], fontsize=11)
        if len(panel["metrics"]) > 1:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for i in range(len(panels), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig, axes
