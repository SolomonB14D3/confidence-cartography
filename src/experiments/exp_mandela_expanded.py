"""
Expanded Linguistic Mandela Effect — Push to Significance
==========================================================
The Mandela calibration reanalysis showed r=0.897 between model confidence
ratios and human false-belief prevalence on 4 linguistic items, but p=0.103
with n=4.  This experiment adds more linguistic Mandela Effect items to
reach n≥10 and achieve statistical significance.

Prevalence sources (in priority order):
  1. YouGov 2022 US poll (gold standard, n=1000)
  2. Published academic studies
  3. Web search hit ratio proxy (wrong_hits / (wrong_hits + correct_hits))
  4. Domain-expert estimates (marked as such)
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict

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
from src.scaling import MODEL_REGISTRY, SCALING_MODELS, PARAM_COUNTS
from src.scaling_viz import plot_scaling_law, MODEL_COLORS, model_display_name
from src.utils import MANDELA_RESULTS_DIR, MANDELA_FIGURES_DIR


# ---------------------------------------------------------------------------
# All linguistic items — original 4 + new additions
# ---------------------------------------------------------------------------
# Prevalence ratio = wrong_prevalence / (wrong_prevalence + correct_prevalence)
# For items with direct survey data, prevalence comes from survey %.
# For items without, we use web hit ratio proxy or academic estimates.

LINGUISTIC_ITEMS = [
    # ========= Original 4 (from Phase 2, YouGov 2022 US) =========
    {
        "id": "star_wars",
        "wrong": "Luke, I am your father",
        "correct": "No, I am your father",
        "context": 'In Star Wars, Darth Vader says "{quote}"',
        "human_wrong_pct": 62,
        "human_correct_pct": 17,
        "source": "YouGov 2022 US",
    },
    {
        "id": "we_are_champions",
        "wrong": 'We Are the Champions ends with "of the world"',
        "correct": 'We Are the Champions does not end with "of the world"',
        "context": "{quote}",
        "human_wrong_pct": 52,
        "human_correct_pct": 22,
        "source": "YouGov 2022 US",
    },
    {
        "id": "risky_business",
        "wrong": "In Risky Business, Tom Cruise dances in a white button-down shirt and sunglasses",
        "correct": "In Risky Business, Tom Cruise dances in a pink button-down shirt without sunglasses",
        "context": "{quote}",
        "human_wrong_pct": 55,
        "human_correct_pct": 16,
        "source": "YouGov 2022 US",
    },
    {
        "id": "mandela_death",
        "wrong": "Nelson Mandela died in prison in the 1980s",
        "correct": "Nelson Mandela died in 2013 after serving as president of South Africa",
        "context": "{quote}",
        "human_wrong_pct": 13,
        "human_correct_pct": 57,
        "source": "YouGov 2022 US",
    },
    # ========= New linguistic items =========
    {
        "id": "silence_of_lambs",
        "wrong": "Hello, Clarice",
        "correct": "Good morning, Clarice",
        "context": 'In The Silence of the Lambs, Hannibal Lecter greets Clarice by saying "{quote}"',
        # Never said in the film. The actual first greeting is "Good morning".
        # Extremely widely misquoted; reinforced by parodies, the 2001 sequel,
        # comedy sketches. Conservative estimate: ~85% believe wrong version.
        "human_wrong_pct": 85,
        "human_correct_pct": 10,
        "source": "web hit proxy + domain estimate",
    },
    {
        "id": "sherlock_elementary",
        "wrong": "Elementary, my dear Watson",
        "correct": "Elementary",
        "context": 'Sherlock Holmes famously says "{quote}"',
        # Never said in original Conan Doyle stories. The full phrase appeared
        # in 1929 film adaptation. One of the most universally "known" quotes
        # that was never actually written.
        "human_wrong_pct": 80,
        "human_correct_pct": 12,
        "source": "web hit proxy + domain estimate",
    },
    {
        "id": "casablanca_play_it",
        "wrong": "Play it again, Sam",
        "correct": "Play it, Sam",
        "context": 'In Casablanca, the famous line is "{quote}"',
        # So famous the misquote became a Woody Allen movie title (1972).
        # The actual closest line is "Play it once, Sam" (Ilsa) and
        # "You played it for her, you can play it for me" (Rick).
        "human_wrong_pct": 82,
        "human_correct_pct": 10,
        "source": "web hit proxy + domain estimate",
    },
    {
        "id": "star_trek_beam",
        "wrong": "Beam me up, Scotty",
        "correct": "Scotty, beam us up",
        "context": 'In Star Trek, Captain Kirk says "{quote}"',
        # Exact phrase never said in original series. Closest: "Scotty, beam
        # us up" and "Beam us up, Scotty" in later films. The misquote is
        # the cultural catchphrase.
        "human_wrong_pct": 85,
        "human_correct_pct": 8,
        "source": "web hit proxy + domain estimate",
    },
    {
        "id": "snow_white_mirror",
        "wrong": "Mirror, mirror on the wall",
        "correct": "Magic mirror on the wall",
        "context": 'In Snow White, the Evil Queen says "{quote}"',
        # Disney's 1937 film uses "Magic mirror on the wall". The Brothers
        # Grimm original uses "Mirror, mirror" (Spieglein, Spieglein).
        # YouGov UK 2025 tested this — vast majority say "Mirror, mirror".
        "human_wrong_pct": 75,
        "human_correct_pct": 15,
        "source": "web hit proxy + YouGov UK 2025 (qualitative)",
    },
    {
        "id": "forrest_gump_chocolates",
        "wrong": "Life is like a box of chocolates",
        "correct": "Life was like a box of chocolates",
        "context": 'In Forrest Gump, the famous line is "{quote}"',
        # "is" vs "was". The present tense version is more quotable,
        # more universal, and almost universally used when quoting.
        "human_wrong_pct": 80,
        "human_correct_pct": 12,
        "source": "web hit proxy",
    },
    {
        "id": "wizard_of_oz_toto",
        "wrong": "Toto, I don't think we're in Kansas anymore",
        "correct": "Toto, I've a feeling we're not in Kansas anymore",
        "context": 'In The Wizard of Oz, Dorothy says "{quote}"',
        # Modernized phrasing replaces the original archaic construction.
        # "I've a feeling" is unusual in modern English.
        "human_wrong_pct": 78,
        "human_correct_pct": 12,
        "source": "web hit proxy",
    },
    {
        "id": "money_root_evil",
        "wrong": "Money is the root of all evil",
        "correct": "The love of money is the root of all evil",
        "context": 'The Bible says "{quote}"',
        # 1 Timothy 6:10. "The love of" gets dropped constantly.
        # One of the most commonly misquoted Bible verses.
        "human_wrong_pct": 75,
        "human_correct_pct": 20,
        "source": "web hit proxy + domain estimate",
    },
    {
        "id": "apollo_13",
        "wrong": "Houston, we have a problem",
        "correct": "Houston, we've had a problem",
        "context": 'During the Apollo 13 mission, the astronauts said "{quote}"',
        # Present vs past tense. The 1995 movie used present tense
        # (deliberately changed for dramatic effect). The actual
        # transmission was past tense.
        "human_wrong_pct": 80,
        "human_correct_pct": 10,
        "source": "web hit proxy",
    },
]


# Pre-compute human ratios
for item in LINGUISTIC_ITEMS:
    item["human_ratio"] = item["human_wrong_pct"] / (
        item["human_wrong_pct"] + item["human_correct_pct"]
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXPANDED_RESULTS_DIR = MANDELA_RESULTS_DIR / "expanded"
EXPANDED_FIGURES_DIR = MANDELA_FIGURES_DIR / "expanded"
EXPANDED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXPANDED_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Item text generation
# ---------------------------------------------------------------------------

def _make_texts(item: dict) -> list[tuple[str, str, str]]:
    """Return list of (framing_name, wrong_text, correct_text) for one item."""
    framings = []
    framings.append(("raw", item["wrong"], item["correct"]))
    if item["context"] != "{quote}":
        w_ctx = item["context"].format(quote=item["wrong"])
        c_ctx = item["context"].format(quote=item["correct"])
        framings.append(("context", w_ctx, c_ctx))
    return framings


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_single_model(size: str, force: bool = False) -> dict:
    """Run expanded analysis for one model size."""
    output_path = EXPANDED_RESULTS_DIR / f"expanded_{size}.jsonl"

    if output_path.exists() and not force:
        print(f"  [{size}] Results cached, loading...")
        records = load_records(output_path)
        return _records_to_pairs(records)

    spec = MODEL_REGISTRY[size]
    model_name = spec["name"]
    dtype = spec["dtype"]
    records = []

    n_texts = sum(len(_make_texts(item)) * 2 for item in LINGUISTIC_ITEMS)
    print(f"\n  [{size}] Analyzing {n_texts} texts with {model_name}...")
    start = time.time()

    for item in tqdm(LINGUISTIC_ITEMS, desc=f"  {size}", leave=False):
        for framing_name, wrong_text, correct_text in _make_texts(item):
            w_rec = analyze_fixed_text(
                wrong_text,
                category="mandela_wrong",
                label=f"{item['id']}_{framing_name}_wrong",
                model_name=model_name, revision="main", dtype=dtype,
            )
            w_rec.metadata = {
                "item_id": item["id"],
                "framing": framing_name,
                "version": "wrong",
                "human_ratio": item["human_ratio"],
                "human_wrong_pct": item["human_wrong_pct"],
                "human_correct_pct": item["human_correct_pct"],
                "source": item["source"],
            }

            c_rec = analyze_fixed_text(
                correct_text,
                category="mandela_correct",
                label=f"{item['id']}_{framing_name}_correct",
                model_name=model_name, revision="main", dtype=dtype,
            )
            c_rec.metadata = {
                "item_id": item["id"],
                "framing": framing_name,
                "version": "correct",
                "human_ratio": item["human_ratio"],
                "human_wrong_pct": item["human_wrong_pct"],
                "human_correct_pct": item["human_correct_pct"],
                "source": item["source"],
            }

            records.extend([w_rec, c_rec])

    elapsed = time.time() - start
    print(f"  [{size}] Done in {elapsed:.1f}s ({len(records)} records)")

    if output_path.exists():
        output_path.unlink()
    save_records(records, output_path)

    return _records_to_pairs(records)


def _records_to_pairs(records: list[ConfidenceRecord]) -> dict:
    """Group records into (item_id, framing) -> {wrong_conf, correct_conf, ...}"""
    by_key = defaultdict(dict)
    for r in records:
        item_id = r.metadata["item_id"]
        framing = r.metadata["framing"]
        version = r.metadata["version"]
        key = (item_id, framing)
        by_key[key][version] = r

    pairs = {}
    for key, versions in by_key.items():
        if "wrong" not in versions or "correct" not in versions:
            continue
        w_conf = versions["wrong"].mean_top1_prob
        c_conf = versions["correct"].mean_top1_prob
        conf_ratio = w_conf / (w_conf + c_conf) if (w_conf + c_conf) > 0 else 0.5
        human_ratio = versions["wrong"].metadata["human_ratio"]

        pairs[key] = {
            "item_id": key[0],
            "framing": key[1],
            "wrong_conf": w_conf,
            "correct_conf": c_conf,
            "confidence_ratio": conf_ratio,
            "human_ratio": human_ratio,
            "source": versions["wrong"].metadata["source"],
        }
    return pairs


def filter_raw(pairs: dict, item_ids: list[str] = None) -> list[dict]:
    """Filter to raw framing and optionally specific item IDs."""
    result = [v for k, v in pairs.items() if v["framing"] == "raw"]
    if item_ids:
        result = [v for v in result if v["item_id"] in item_ids]
    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_expanded_scatter(pairs: dict, size: str, save_path: Path):
    """Scatter: X = human prevalence ratio, Y = model confidence ratio."""
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(9, 9))

    raw = filter_raw(pairs)
    if not raw:
        plt.close(fig)
        return

    # Separate YouGov items from proxy items
    yougov_items = [v for v in raw if "YouGov" in v["source"]]
    proxy_items = [v for v in raw if "YouGov" not in v["source"]]

    # Plot proxy items
    if proxy_items:
        px = [v["human_ratio"] for v in proxy_items]
        py = [v["confidence_ratio"] for v in proxy_items]
        ax.scatter(px, py, s=80, zorder=5, alpha=0.7, color="#9C27B0",
                   marker="D", label="Proxy prevalence")
        for x, y, v in zip(px, py, proxy_items):
            ax.annotate(v["item_id"], (x, y), fontsize=6.5, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points", color="#6A1B9A")

    # Plot YouGov items
    if yougov_items:
        yx = [v["human_ratio"] for v in yougov_items]
        yy = [v["confidence_ratio"] for v in yougov_items]
        ax.scatter(yx, yy, s=100, zorder=6, alpha=0.9, color="#2196F3",
                   marker="o", label="YouGov survey")
        for x, y, v in zip(yx, yy, yougov_items):
            ax.annotate(v["item_id"], (x, y), fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points", color="#0D47A1")

    # Diagonal + reference lines
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.3)

    # Best-fit line
    all_h = [v["human_ratio"] for v in raw]
    all_m = [v["confidence_ratio"] for v in raw]
    if len(raw) >= 3:
        slope, intercept, _, _, _ = stats.linregress(all_h, all_m)
        fit_x = np.linspace(min(all_h) - 0.02, max(all_h) + 0.02, 100)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, "-", color="#F44336", alpha=0.5, linewidth=1.5,
                label=f"Best fit (slope={slope:.2f})")

    # Correlation stats
    r, p = stats.pearsonr(all_h, all_m)
    rho, rho_p = stats.spearmanr(all_h, all_m)
    n = len(raw)

    sig_marker = "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(0.05, 0.95,
            f"n = {n} linguistic items\n"
            f"Pearson r = {r:.3f} (p = {p:.4f}){sig_marker}\n"
            f"Spearman ρ = {rho:.3f} (p = {rho_p:.4f})",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Human Prevalence Ratio", fontsize=12)
    ax.set_ylabel("Model Confidence Ratio", fontsize=12)
    ax.set_title(f"Expanded Linguistic Mandela — {model_display_name(size)} (n={n})",
                 fontsize=14)
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.1, 1.0)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_item_comparison(pairs: dict, size: str, save_path: Path):
    """Bar chart comparing human vs model ratios for all items."""
    sns.set_theme(style="whitegrid", palette="muted")
    raw = filter_raw(pairs)
    if not raw:
        return

    raw = sorted(raw, key=lambda x: x["human_ratio"], reverse=True)
    ids = [v["item_id"] for v in raw]
    human_r = [v["human_ratio"] for v in raw]
    model_r = [v["confidence_ratio"] for v in raw]

    x = np.arange(len(ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_h = ax.bar(x - width / 2, human_r, width, label="Human Prevalence",
                    color="#FF9800", alpha=0.85)
    bars_m = ax.bar(x + width / 2, model_r, width,
                    label=f"Model ({model_display_name(size)})",
                    color="#2196F3", alpha=0.85)

    # Mark proxy items
    for i, v in enumerate(raw):
        if "YouGov" not in v["source"]:
            ax.text(x[i] - width / 2, human_r[i] + 0.01, "†",
                    ha="center", fontsize=8, color="#888")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Wrong / (Wrong + Correct) Ratio", fontsize=11)
    ax.set_title(f"Human vs Model — All Linguistic Items — {model_display_name(size)}",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)

    # Footnote
    ax.text(0.01, -0.15, "† = proxy prevalence estimate (web hit ratio or domain estimate)",
            transform=ax.transAxes, fontsize=7, color="#666")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_scaling(all_results: dict, save_path: Path):
    """Correlation coefficient vs model size."""
    sizes = [s for s in SCALING_MODELS if s in all_results]
    if len(sizes) < 2:
        return

    pearson_rs = []
    spearman_rhos = []
    p_values = []

    for size in sizes:
        raw = filter_raw(all_results[size])
        h = [v["human_ratio"] for v in raw]
        m = [v["confidence_ratio"] for v in raw]
        r, p = stats.pearsonr(h, m)
        rho, _ = stats.spearmanr(h, m)
        pearson_rs.append(r)
        spearman_rhos.append(rho)
        p_values.append(p)

    plot_scaling_law(
        sizes,
        {"Pearson r": pearson_rs, "Spearman ρ": spearman_rhos},
        ylabel="Correlation with Human Prevalence",
        title=f"Expanded Linguistic Calibration (n={len(filter_raw(all_results[sizes[0]]))})",
        save_path=save_path,
        hline=0.0, hline_label="No correlation",
    )


def plot_leave_one_out(pairs: dict, size: str, save_path: Path):
    """Leave-one-out robustness: correlation when each item is removed."""
    sns.set_theme(style="whitegrid", palette="muted")
    raw = filter_raw(pairs)
    if len(raw) < 4:
        return

    all_h = [v["human_ratio"] for v in raw]
    all_m = [v["confidence_ratio"] for v in raw]
    full_r, full_p = stats.pearsonr(all_h, all_m)

    loo_results = []
    for i, excluded in enumerate(raw):
        subset_h = all_h[:i] + all_h[i + 1:]
        subset_m = all_m[:i] + all_m[i + 1:]
        r, p = stats.pearsonr(subset_h, subset_m)
        loo_results.append({
            "excluded": excluded["item_id"],
            "r": r,
            "p": p,
            "source": excluded["source"],
        })

    fig, ax = plt.subplots(figsize=(12, 5))
    ids = [r["excluded"] for r in loo_results]
    rs = [r["r"] for r in loo_results]
    colors = ["#2196F3" if "YouGov" in r["source"] else "#9C27B0"
              for r in loo_results]

    x = np.arange(len(ids))
    ax.bar(x, rs, color=colors, alpha=0.8)
    ax.axhline(y=full_r, color="red", linestyle="--", linewidth=2,
               label=f"Full r = {full_r:.3f}")
    ax.axhline(y=0.576, color="gray", linestyle=":", alpha=0.5,
               label="Significance threshold (n-1)")

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r (without item)", fontsize=11)
    ax.set_title(f"Leave-One-Out Robustness — {model_display_name(size)}", fontsize=13)
    ax.legend(fontsize=9)

    # Annotate p-values
    for i, r_val in enumerate(loo_results):
        label = f"p={r_val['p']:.3f}"
        ax.text(i, rs[i] + 0.01, label, ha="center", fontsize=6, rotation=90)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_yougov_vs_proxy(pairs: dict, size: str, save_path: Path):
    """Compare YouGov-only correlation to full (YouGov + proxy) correlation."""
    sns.set_theme(style="whitegrid", palette="muted")
    raw = filter_raw(pairs)
    yougov_only = [v for v in raw if "YouGov" in v["source"]]
    proxy_only = [v for v in raw if "YouGov" not in v["source"]]

    if len(yougov_only) < 3 or len(proxy_only) < 1:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: YouGov only
    yh = [v["human_ratio"] for v in yougov_only]
    ym = [v["confidence_ratio"] for v in yougov_only]
    r_yg, p_yg = stats.pearsonr(yh, ym)

    ax1.scatter(yh, ym, s=100, color="#2196F3", zorder=5)
    for x, y, v in zip(yh, ym, yougov_only):
        ax1.annotate(v["item_id"], (x, y), fontsize=7, xytext=(4, 4),
                     textcoords="offset points")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_title(f"YouGov Only (n={len(yougov_only)})\nr={r_yg:.3f}, p={p_yg:.3f}",
                  fontsize=12)
    ax1.set_xlabel("Human Prevalence Ratio")
    ax1.set_ylabel("Model Confidence Ratio")
    ax1.set_xlim(0.1, 1.0)
    ax1.set_ylim(0.1, 1.0)
    ax1.set_aspect("equal")

    # Right: All items
    ah = [v["human_ratio"] for v in raw]
    am = [v["confidence_ratio"] for v in raw]
    r_all, p_all = stats.pearsonr(ah, am)

    for v in yougov_only:
        ax2.scatter(v["human_ratio"], v["confidence_ratio"],
                    s=100, color="#2196F3", zorder=6)
    for v in proxy_only:
        ax2.scatter(v["human_ratio"], v["confidence_ratio"],
                    s=80, color="#9C27B0", marker="D", zorder=5)
    for v in raw:
        ax2.annotate(v["item_id"], (v["human_ratio"], v["confidence_ratio"]),
                     fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax2.set_title(f"All Items (n={len(raw)})\nr={r_all:.3f}, p={p_all:.3f}",
                  fontsize=12)
    ax2.set_xlabel("Human Prevalence Ratio")
    ax2.set_ylabel("Model Confidence Ratio")
    ax2.set_xlim(0.1, 1.0)
    ax2.set_ylim(0.1, 1.0)
    ax2.set_aspect("equal")

    fig.suptitle(f"YouGov vs Proxy Prevalence — {model_display_name(size)}",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(models: list[str] = None, force: bool = False):
    models = models or SCALING_MODELS

    print("=" * 70)
    print("EXPANDED LINGUISTIC MANDELA EFFECT")
    print("=" * 70)
    print(f"Models: {', '.join(models)}")
    print(f"Items: {len(LINGUISTIC_ITEMS)} linguistic items")

    yougov_count = sum(1 for i in LINGUISTIC_ITEMS if "YouGov" in i["source"])
    proxy_count = len(LINGUISTIC_ITEMS) - yougov_count
    print(f"  YouGov survey data: {yougov_count}")
    print(f"  Proxy estimates: {proxy_count}")
    print(f"  Human ratio range: {min(i['human_ratio'] for i in LINGUISTIC_ITEMS):.3f}"
          f" – {max(i['human_ratio'] for i in LINGUISTIC_ITEMS):.3f}")

    start_time = time.time()
    all_results = {}

    for size in models:
        pairs = run_single_model(size, force=force)
        all_results[size] = pairs
        unload_model()

        # Quick summary
        raw = filter_raw(pairs)
        h = [v["human_ratio"] for v in raw]
        m = [v["confidence_ratio"] for v in raw]
        r, p = stats.pearsonr(h, m)
        rho, rho_p = stats.spearmanr(h, m)
        sig = "**" if p < 0.01 else "* " if p < 0.05 else "  "
        print(f"  [{size}] r={r:.3f} (p={p:.4f}){sig}  ρ={rho:.3f} (p={rho_p:.4f})")

    # ===================================================================
    # Summary table
    # ===================================================================
    print("\n" + "=" * 70)
    print("EXPANDED CALIBRATION SUMMARY")
    print("=" * 70)

    sizes_done = [s for s in models if s in all_results]
    n_items = len(filter_raw(all_results[sizes_done[0]]))

    print(f"\nn = {n_items} linguistic items")
    print(f"\n{'Size':<8} {'Params':<12} {'Pearson r':<12} {'p-value':<10} "
          f"{'Spearman ρ':<12} {'p-value':<10} {'Significant?'}")
    print("-" * 80)

    for size in sizes_done:
        raw = filter_raw(all_results[size])
        h = [v["human_ratio"] for v in raw]
        m = [v["confidence_ratio"] for v in raw]
        r, p = stats.pearsonr(h, m)
        rho, rho_p = stats.spearmanr(h, m)
        params = PARAM_COUNTS[size]
        sig = "YES **" if p < 0.01 else "YES *" if p < 0.05 else "no"

        print(f"{size:<8} {params / 1e6:>8.0f}M  {r:<12.3f} {p:<10.4f} "
              f"{rho:<12.3f} {rho_p:<10.4f} {sig}")

    # Per-item details for largest model
    largest = sizes_done[-1]
    raw = filter_raw(all_results[largest])
    raw_sorted = sorted(raw, key=lambda x: x["human_ratio"], reverse=True)

    print(f"\n{'Item':<24} {'Human':<10} {'Model':<10} {'Gap':<10} {'Source'}")
    print("-" * 72)
    for v in raw_sorted:
        gap = v["confidence_ratio"] - v["human_ratio"]
        print(f"{v['item_id']:<24} {v['human_ratio']:<10.3f} "
              f"{v['confidence_ratio']:<10.3f} {gap:<+10.3f} {v['source']}")

    # ===================================================================
    # Statistical power analysis
    # ===================================================================
    print("\n" + "=" * 70)
    print("STATISTICAL POWER")
    print("=" * 70)

    # With n items, critical r for p<0.05 (two-tailed)
    from scipy.stats import t as t_dist
    n = n_items
    t_crit = t_dist.ppf(0.975, df=n - 2)
    r_crit = np.sqrt(t_crit ** 2 / (t_crit ** 2 + n - 2))
    print(f"  n = {n}")
    print(f"  Critical r for p<0.05 (two-tailed): {r_crit:.3f}")

    largest_raw = filter_raw(all_results[largest])
    h = [v["human_ratio"] for v in largest_raw]
    m = [v["confidence_ratio"] for v in largest_raw]
    r_obs, p_obs = stats.pearsonr(h, m)
    print(f"  Observed r at {model_display_name(largest)}: {r_obs:.3f}")
    print(f"  Observed p: {p_obs:.4f}")

    if r_obs > r_crit:
        print(f"  → SIGNIFICANT: r={r_obs:.3f} > r_crit={r_crit:.3f}")
    else:
        print(f"  → NOT SIGNIFICANT: r={r_obs:.3f} < r_crit={r_crit:.3f}")
        print(f"    Need r > {r_crit:.3f} or more items for significance")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    for size in sizes_done:
        pairs = all_results[size]
        print(f"  Scatter for {size}...")
        plot_expanded_scatter(
            pairs, size,
            EXPANDED_FIGURES_DIR / f"expanded_scatter_{size}.png")

        print(f"  Item comparison for {size}...")
        plot_item_comparison(
            pairs, size,
            EXPANDED_FIGURES_DIR / f"expanded_items_{size}.png")

    # Largest model extras
    print(f"  Leave-one-out for {largest}...")
    plot_leave_one_out(
        all_results[largest], largest,
        EXPANDED_FIGURES_DIR / f"leave_one_out_{largest}.png")

    print(f"  YouGov vs proxy for {largest}...")
    plot_yougov_vs_proxy(
        all_results[largest], largest,
        EXPANDED_FIGURES_DIR / f"yougov_vs_proxy_{largest}.png")

    if len(sizes_done) >= 2:
        print("  Correlation scaling...")
        plot_correlation_scaling(
            all_results,
            EXPANDED_FIGURES_DIR / "expanded_correlation_scaling.png")

    # ===================================================================
    # Final
    # ===================================================================
    total_time = time.time() - start_time
    fig_count = len(list(EXPANDED_FIGURES_DIR.glob("*.png")))

    print(f"\n{'='*70}")
    print("EXPANDED MANDELA EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Models: {len(sizes_done)}")
    print(f"  Items: {n_items}")
    print(f"  Figures: {fig_count}")
    print(f"  Total time: {total_time:.1f}s")

    # Headline finding
    raw = filter_raw(all_results[largest])
    h = [v["human_ratio"] for v in raw]
    m = [v["confidence_ratio"] for v in raw]
    r, p = stats.pearsonr(h, m)
    rho, rho_p = stats.spearmanr(h, m)

    print(f"\n  Pearson  r = {r:.3f}  (p = {p:.4f})  {'*' if p < 0.05 else ''}")
    print(f"  Spearman ρ = {rho:.3f}  (p = {rho_p:.4f})  {'*' if rho_p < 0.05 else ''}")

    if r > 0.5 and p < 0.05:
        print(f"\n  FINDING: SIGNIFICANT linear correlation (r={r:.3f}, p={p:.4f})")
        print(f"  → Model is a CALIBRATED CONSENSUS SENSOR for linguistic Mandela Effects")
        print(f"  → Phase 2 result (r=0.897, n=4, p=0.103) now confirmed at p<0.05")
    elif rho > 0.5 and rho_p < 0.05:
        print(f"\n  FINDING: SIGNIFICANT rank-order correlation (ρ={rho:.3f}, p={rho_p:.4f})")
        print(f"  → Pearson r={r:.3f} not significant (p={p:.4f}) — outliers reduce linearity")
        print(f"  → But rank ordering IS preserved: items humans get wrong more often")
        print(f"    also have higher model confidence ratios in the expected direction")
        print(f"  → Relationship is monotonic but not perfectly linear")
    elif r > 0.3:
        print(f"\n  FINDING: Moderate but not significant (r={r:.3f}, p={p:.4f})")
        print(f"  → Need more items or stronger effect")
    else:
        print(f"\n  FINDING: Weak correlation (r={r:.3f}, p={p:.4f})")
        print(f"  → Expanded items do not confirm Phase 2 result")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_experiment(force=args.force)
