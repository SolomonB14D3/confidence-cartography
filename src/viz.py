"""Visualization toolkit for confidence cartography."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional

from .schema import ConfidenceRecord
from .utils import FIGURES_DIR, truncate_token

# Consistent style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.figsize": (14, 5),
})


# ---------------------------------------------------------------------------
# 1. Confidence Landscape
# ---------------------------------------------------------------------------

def plot_confidence_landscape(
    record: ConfidenceRecord,
    save_name: Optional[str] = None,
) -> tuple:
    """Token-by-token confidence and entropy landscape for a single text.

    Left Y-axis: P(actual next token) — blue line with fill.
    Right Y-axis: Entropy (bits) — red dashed line.
    X-axis: token positions with labels.
    """
    tokens = record.tokens
    positions = [t.position for t in tokens]
    probs = [t.top1_prob for t in tokens]
    entropies = [t.entropy for t in tokens]
    labels = [truncate_token(t.token_str) for t in tokens]

    fig, ax1 = plt.subplots(figsize=(max(10, len(tokens) * 0.6), 5))

    # Left axis: top1 probability
    color1 = "#2196F3"
    ax1.plot(positions, probs, color=color1, marker="o", markersize=4,
             linewidth=1.5, label="P(actual token)")
    ax1.fill_between(positions, probs, alpha=0.15, color=color1)
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("P(actual next token)", color=color1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Right axis: entropy
    ax2 = ax1.twinx()
    color2 = "#F44336"
    ax2.plot(positions, entropies, color=color2, marker="s", markersize=3,
             linewidth=1.2, linestyle="--", label="Entropy (bits)")
    ax2.set_ylabel("Entropy (bits)", color=color2)
    ax2.set_ylim(0, 16)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Token labels on x-axis
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    title = f"Confidence Landscape: {record.label}"
    if record.category:
        title += f" [{record.category}]"
    ax1.set_title(title, fontsize=11)

    fig.tight_layout()

    fname = save_name or f"landscape_{record.label}.png"
    fig.savefig(FIGURES_DIR / fname, bbox_inches="tight")
    return fig, ax1


# ---------------------------------------------------------------------------
# 2. Confidence Heatmap
# ---------------------------------------------------------------------------

def plot_confidence_heatmap(
    records: list[ConfidenceRecord],
    title: str = "Confidence Heatmap",
    save_name: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> tuple:
    """Heatmap comparing confidence profiles across multiple texts.

    Rows = prompts, columns = token positions, color = P(actual token).
    """
    lengths = [len(r.tokens) for r in records]
    common_len = max_tokens or min(lengths)

    # Build matrix
    matrix = np.full((len(records), common_len), np.nan)
    for i, record in enumerate(records):
        n = min(len(record.tokens), common_len)
        for j in range(n):
            matrix[i, j] = record.tokens[j].top1_prob

    row_labels = [r.label for r in records]

    fig, ax = plt.subplots(
        figsize=(max(10, common_len * 0.5), max(4, len(records) * 0.4)))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=range(common_len),
        yticklabels=row_labels,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        cbar_kws={"label": "P(actual token)"},
        linewidths=0.5,
    )
    ax.set_xlabel("Token Position")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()

    fname = save_name or "heatmap.png"
    fig.savefig(FIGURES_DIR / fname, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Transition Detector
# ---------------------------------------------------------------------------

def plot_transition_detector(
    record: ConfidenceRecord,
    threshold: float = 0.15,
    save_name: Optional[str] = None,
) -> tuple:
    """Detect and highlight sharp confidence transitions.

    Top panel: confidence line with orange highlighted transition zones.
    Bottom panel: |delta P| bar chart showing magnitude of changes.
    """
    tokens = record.tokens
    probs = np.array([t.top1_prob for t in tokens])
    diffs = np.abs(np.diff(probs))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(10, len(tokens) * 0.6), 7),
        gridspec_kw={"height_ratios": [2, 1]}, sharex=True)

    # Top panel: confidence with transition highlights
    positions = range(len(probs))
    ax1.plot(positions, probs, color="#2196F3", marker="o", markersize=4,
             linewidth=1.5)
    ax1.fill_between(positions, probs, alpha=0.1, color="#2196F3")

    for i, d in enumerate(diffs):
        if d >= threshold:
            ax1.axvspan(i + 0.5, i + 1.5, alpha=0.3, color="#FF9800")
            ax1.annotate(
                tokens[i + 1].token_str.strip(),
                xy=(i + 1, probs[i + 1]),
                xytext=(0, 15), textcoords="offset points",
                fontsize=7, ha="center", color="#E65100",
                arrowprops=dict(arrowstyle="->", color="#E65100"),
            )

    ax1.set_ylabel("P(actual token)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(
        f"Confidence Transitions: {record.label} (threshold={threshold})")

    # Bottom panel: diff bar chart
    diff_positions = range(1, len(probs))
    colors = ["#FF9800" if d >= threshold else "#B0BEC5" for d in diffs]
    ax2.bar(diff_positions, diffs, color=colors, width=0.8)
    ax2.axhline(y=threshold, color="#F44336", linestyle="--",
                linewidth=1, alpha=0.7)
    ax2.set_ylabel("|delta P|")
    ax2.set_xlabel("Token Position")

    labels = [truncate_token(t.token_str) for t in tokens]
    ax2.set_xticks(range(len(tokens)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    fname = save_name or f"transitions_{record.label}.png"
    fig.savefig(FIGURES_DIR / fname, bbox_inches="tight")
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# 4. Distribution Snapshot
# ---------------------------------------------------------------------------

def plot_distribution_snapshot(
    record: ConfidenceRecord,
    position: int,
    top_k: int = 5,
    save_name: Optional[str] = None,
) -> tuple:
    """Bar chart of top predicted tokens at a specific position.

    Actual token highlighted in green; others in blue.
    """
    ta = record.tokens[position]

    k = min(top_k, len(ta.top5_tokens))
    token_labels = ta.top5_tokens[:k]
    token_probs = ta.top5_probs[:k]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(k), token_probs, color="#2196F3", edgecolor="white")
    ax.set_yticks(range(k))
    ax.set_yticklabels([repr(t) for t in token_labels], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_xlim(0, min(1.0, max(token_probs) * 1.3) if token_probs else 1.0)

    # Highlight the actual token
    actual_str = ta.token_str
    for i, label in enumerate(token_labels):
        if label == actual_str:
            bars[i].set_color("#4CAF50")
            bars[i].set_edgecolor("#2E7D32")

    ax.set_title(
        f"Position {position} predictions | "
        f"Actual: {repr(actual_str)} (rank {ta.top1_rank})")
    fig.tight_layout()

    fname = save_name or f"snapshot_{record.label}_pos{position}.png"
    fig.savefig(FIGURES_DIR / fname, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Category Comparison
# ---------------------------------------------------------------------------

def plot_category_comparison(
    records_by_category: dict[str, list[ConfidenceRecord]],
    save_name: Optional[str] = None,
) -> tuple:
    """Grouped bar chart comparing mean confidence and entropy across categories."""
    categories = list(records_by_category.keys())
    mean_probs = []
    std_probs = []
    mean_ents = []
    std_ents = []

    for cat in categories:
        recs = records_by_category[cat]
        mp = [r.mean_top1_prob for r in recs]
        me = [r.mean_entropy for r in recs]
        mean_probs.append(np.mean(mp))
        std_probs.append(np.std(mp))
        mean_ents.append(np.mean(me))
        std_ents.append(np.std(me))

    x = np.arange(len(categories))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(x - width / 2, mean_probs, width, yerr=std_probs,
            label="Mean P(actual)", color="#2196F3", capsize=4)
    ax1.set_ylabel("Mean P(actual token)", color="#2196F3")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, mean_ents, width, yerr=std_ents,
            label="Mean Entropy", color="#F44336", alpha=0.7, capsize=4)
    ax2.set_ylabel("Mean Entropy (bits)", color="#F44336")
    ax2.set_ylim(0, 16)

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=20, ha="right")
    ax1.set_title("Confidence by Category")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fname = save_name or "category_comparison.png"
    fig.savefig(FIGURES_DIR / fname, bbox_inches="tight")
    return fig, ax1


# ---------------------------------------------------------------------------
# 6. False Statement Zoom
# ---------------------------------------------------------------------------

def plot_false_statement_zoom(
    false_record: ConfidenceRecord,
    true_record: Optional[ConfidenceRecord] = None,
    save_name: Optional[str] = None,
) -> tuple:
    """Zoom in on where confidence drops in a false statement.

    Optionally overlay the true version for direct comparison.
    Annotates the minimum confidence point with the token and probability.
    """
    fig, ax = plt.subplots(
        figsize=(max(10, len(false_record.tokens) * 0.7), 5))

    # False statement
    f_probs = [t.top1_prob for t in false_record.tokens]
    positions_f = range(len(f_probs))
    ax.plot(positions_f, f_probs, color="#F44336", marker="o", markersize=5,
            linewidth=2, label=f"False: {false_record.label}")

    # Annotate minimum confidence
    min_pos = false_record.min_confidence_pos
    ax.annotate(
        f"{false_record.min_confidence_token!r}\n"
        f"P={false_record.min_confidence_value:.4f}",
        xy=(min_pos, false_record.min_confidence_value),
        xytext=(0, -30), textcoords="offset points",
        fontsize=8, ha="center", color="#B71C1C",
        arrowprops=dict(arrowstyle="->", color="#B71C1C"),
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFCDD2", ec="#B71C1C"),
    )

    # True statement overlay
    if true_record is not None:
        t_probs = [t.top1_prob for t in true_record.tokens]
        positions_t = range(len(t_probs))
        ax.plot(positions_t, t_probs, color="#4CAF50", marker="s",
                markersize=4, linewidth=1.5, linestyle="--",
                label=f"True: {true_record.label}", alpha=0.7)

    # Token labels from the false record
    f_labels = [truncate_token(t.token_str) for t in false_record.tokens]
    ax.set_xticks(list(positions_f))
    ax.set_xticklabels(f_labels, rotation=45, ha="right", fontsize=7)

    ax.set_ylabel("P(actual next token)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"False Statement Analysis: {false_record.label}")
    ax.legend(fontsize=9)
    fig.tight_layout()

    fname = save_name or f"false_zoom_{false_record.label}.png"
    fig.savefig(FIGURES_DIR / fname, bbox_inches="tight")
    return fig, ax
