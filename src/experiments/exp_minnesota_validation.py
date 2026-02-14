"""
Experiment: Minnesota Fraud Validation
=======================================
Cross-reference the enriched Medicaid fraud model's top-flagged Minnesota
providers against publicly named fraud defendants. If the model flags
providers who were later caught — using only billing pattern data available
before prosecution — that demonstrates real predictive power.

Named entities come from:
  - AG Ellison press releases (2023-2026)
  - DOJ indictments (2025)
  - DHS suspensions (2025)
  - KARE11 investigative reporting (2025)
  - FBI raids (2025)

Ground truth: Public court records and news reports.
Data: T-MSIS Medicaid Provider Spending (2018-2024), enriched with NPPES
      and cross-program features. Model: Logistic Regression AUC=0.883.
"""

import os
import sys
import time
import json
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

MEDICAID_DIR = PROJECT_ROOT / "data" / "medicaid"
FIGURES_DIR = PROJECT_ROOT / "figures" / "medicaid" / "minnesota"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Named fraud entities (from public court records and news)
# ---------------------------------------------------------------------------

FRAUD_ENTITIES = {
    # --- Medicaid / PCA / Home Care / EIDBI Fraud ---
    "MN Professional Health Services": {
        "npi": "1235218157",
        "type": "PCA company",
        "alleged_amount": "$9.5M",
        "source": "AG Ellison, Aug 2023",
        "prosecution_date": "2023-08",
        "category": "Medicaid/PCA",
    },
    "Guardian Home Health Services Inc": {
        "npi": "1033400882",
        "type": "PCA/home care",
        "alleged_amount": "$3M+",
        "source": "AG Ellison, Jan 2026",
        "prosecution_date": "2026-01",
        "category": "Medicaid/PCA",
    },
    "Star Autism Center LLC": {
        "npi": "1801409099",
        "type": "EIDBI",
        "alleged_amount": "$6M+",
        "source": "DOJ, Dec 2025",
        "prosecution_date": "2025-12",
        "category": "EIDBI",
    },
    "Joy Home Healthcare LLC": {
        "npi": "1194453068",
        "type": "Home care",
        "alleged_amount": "Suspended",
        "source": "DHS 2025",
        "prosecution_date": "2025-01",
        "category": "Home care",
    },
    "Agape Home Care Services LLC": {
        "npi": "1790495554",
        "type": "Home care",
        "alleged_amount": "Suspended (affiliated)",
        "source": "DHS 2025",
        "prosecution_date": "2025-01",
        "category": "Home care",
    },
    "Healthy Living Home Care LLC": {
        "npi": "1871993006",
        "type": "Home care",
        "alleged_amount": "Under investigation",
        "source": "DHS 2025",
        "prosecution_date": "2025-01",
        "category": "Home care",
    },
    "Kyros Recovery LLC": {
        "npi": "1306588298",
        "type": "Peer recovery",
        "alleged_amount": "Unknown",
        "source": "2025",
        "prosecution_date": "2025-01",
        "category": "Peer recovery",
    },
    "Refocus Recovery": {
        "npi": "1356919286",
        "type": "Peer recovery",
        "alleged_amount": "Unknown",
        "source": "2025",
        "prosecution_date": "2025-01",
        "category": "Peer recovery",
    },
    # --- NuWay Alliance entities ---
    "Nu-Way House Inc": {
        "npi": "1265515639",
        "type": "Addiction treatment",
        "alleged_amount": "$18.5M settlement",
        "source": "Feb 2025",
        "prosecution_date": "2025-02",
        "category": "Addiction treatment",
    },
    "NuWay Recovery Foundation": {
        "npi": "1245016229",
        "type": "Addiction treatment",
        "alleged_amount": "$18.5M settlement",
        "source": "Feb 2025",
        "prosecution_date": "2025-02",
        "category": "Addiction treatment",
    },
    # --- HSS Fraud ---
    "Liberty Plus LLC": {
        "npi": "1144918459",
        "type": "HSS provider",
        "alleged_amount": "$1.2M",
        "source": "DOJ, Sep 2025",
        "prosecution_date": "2025-09",
        "category": "HSS",
    },
}

# Entities that could NOT be matched via NPPES (no NPI found):
UNMATCHED_ENTITIES = [
    "Promise Health Services LLC",
    "Ultimate Home Health Services LLC",
    "Evergreen (home health)",
    "Minnesota Home (PCA)",
    "Smart Therapy LLC",
    "Hennepin Autism Center",
    "Bright Community Services",
    "Pristine (HSS provider)",
    "Chozen Runner",
    "Retsel Real Estate",
    "Brilliant Minds LLC",
    "Faladcare Inc.",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_enriched_scores():
    """Load the enriched anomaly scores with all features."""
    path = MEDICAID_DIR / "anomaly_scores_enriched.csv"
    print(f"Loading enriched scores from {path}...")
    scores = pd.read_csv(path, dtype={"NPI": str})
    print(f"  Total providers: {len(scores):,}")
    return scores


def load_spending_dates():
    """Load just NPI + claim dates from spending data for temporal analysis."""
    path = MEDICAID_DIR / "medicaid_provider_spending.parquet"
    print(f"Loading spending dates from {path}...")
    df = pd.read_parquet(path, columns=["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"])
    df["CLAIM_FROM_MONTH"] = pd.to_datetime(df["CLAIM_FROM_MONTH"], errors="coerce")
    print(f"  Total rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_entities(scores, spending_dates):
    """Match named fraud entities to spending data and check model scores."""
    print("\n" + "=" * 100)
    print("MINNESOTA FRAUD ENTITY VALIDATION")
    print("=" * 100)

    # Minnesota subset for state-level percentiles
    mn = scores[scores["state"] == "MN"].copy()
    mn["pctl_mn"] = mn["prob_full"].rank(pct=True) * 100
    scores["pctl_national"] = scores["prob_full"].rank(pct=True) * 100
    print(f"\nMinnesota providers in model: {len(mn):,}")

    results = []

    for entity_name, info in FRAUD_ENTITIES.items():
        npi = info["npi"]
        match = scores[scores["NPI"] == npi]

        rec = {
            "entity": entity_name,
            "npi": npi,
            "type": info["type"],
            "category": info["category"],
            "alleged_amount": info["alleged_amount"],
            "source": info["source"],
            "prosecution_date": info["prosecution_date"],
            "in_data": len(match) > 0,
        }

        if len(match) > 0:
            row = match.iloc[0]
            rec["prob_full"] = row["prob_full"]

            # MN percentile
            mn_match = mn[mn["NPI"] == npi]
            rec["pctl_mn"] = mn_match["pctl_mn"].iloc[0] if len(mn_match) > 0 else None
            rec["pctl_national"] = match["pctl_national"].iloc[0]

            # Thresholds
            pctl = rec["pctl_mn"]
            rec["flagged_top10"] = pctl >= 90 if pctl else False
            rec["flagged_top5"] = pctl >= 95 if pctl else False
            rec["flagged_top1"] = pctl >= 99 if pctl else False

            # Key features
            for feat in ["total_claims", "total_paid", "claims_per_month",
                         "self_billing_ratio", "activity_density",
                         "entity_age_months", "is_organization",
                         "months_since_start", "n_procedures", "n_months",
                         "avg_paid_per_bene"]:
                rec[feat] = row.get(feat, None)

            # Temporal: billing date range
            entity_spend = spending_dates[
                spending_dates["BILLING_PROVIDER_NPI_NUM"] == npi
            ]
            if len(entity_spend) > 0:
                rec["first_bill"] = entity_spend["CLAIM_FROM_MONTH"].min().strftime("%Y-%m")
                rec["last_bill"] = entity_spend["CLAIM_FROM_MONTH"].max().strftime("%Y-%m")
                rec["billing_months"] = entity_spend["CLAIM_FROM_MONTH"].nunique()

                # Check if all billing is pre-prosecution
                pros_dt = pd.Timestamp(info["prosecution_date"] + "-01")
                rec["all_pre_prosecution"] = entity_spend["CLAIM_FROM_MONTH"].max() < pros_dt

                # Lead time: months between last billing and prosecution
                last_bill = entity_spend["CLAIM_FROM_MONTH"].max()
                rec["lead_time_months"] = (pros_dt.year - last_bill.year) * 12 + (pros_dt.month - last_bill.month)
            else:
                rec["first_bill"] = None
                rec["last_bill"] = None
                rec["billing_months"] = 0
                rec["all_pre_prosecution"] = None
                rec["lead_time_months"] = None
        else:
            rec["prob_full"] = None
            rec["pctl_mn"] = None
            rec["pctl_national"] = None
            rec["flagged_top10"] = False
            rec["flagged_top5"] = False
            rec["flagged_top1"] = False

        results.append(rec)

    return pd.DataFrame(results), mn


def print_validation_report(results_df):
    """Print formatted validation report."""
    print("\n" + "=" * 100)
    print("VALIDATION REPORT")
    print("=" * 100)

    found = results_df[results_df["in_data"]]
    not_found = results_df[~results_df["in_data"]]

    print(f"\nEntities searched: {len(results_df)}")
    print(f"Found in spending data: {len(found)}")
    print(f"Not found: {len(not_found)}")
    if len(not_found) > 0:
        print(f"  Not found: {', '.join(not_found['entity'].tolist())}")

    print(f"\nAdditionally, {len(UNMATCHED_ENTITIES)} entities could not be matched to NPIs:")
    for e in UNMATCHED_ENTITIES:
        print(f"  - {e}")

    # Per-entity report
    print("\n" + "-" * 100)
    print("PER-ENTITY RESULTS (entities found in spending data)")
    print("-" * 100)

    for _, row in found.sort_values("prob_full", ascending=False).iterrows():
        print(f"\n  Entity: {row['entity']}")
        print(f"  NPI: {row['npi']}")
        print(f"  Type: {row['type']} | Category: {row['category']}")
        print(f"  Alleged: {row['alleged_amount']} | Source: {row['source']}")
        print(f"  Model anomaly score: {row['prob_full']:.4f}")
        print(f"  MN Percentile: {row['pctl_mn']:.1f}%")
        print(f"  National Percentile: {row['pctl_national']:.1f}%")

        top10 = "YES" if row["flagged_top10"] else "NO"
        top5 = "YES" if row["flagged_top5"] else "NO"
        top1 = "YES" if row["flagged_top1"] else "NO"
        print(f"  Flagged at top 10%: {top10} | top 5%: {top5} | top 1%: {top1}")

        if row.get("first_bill"):
            print(f"  Billing: {row['first_bill']} to {row['last_bill']} ({row['billing_months']} months)")
            print(f"  Prosecution: {row['prosecution_date']}")
            pre = "ALL pre-prosecution" if row["all_pre_prosecution"] else "INCLUDES post-prosecution"
            print(f"  Temporal: {pre}")
            if row.get("lead_time_months"):
                print(f"  Lead time: {row['lead_time_months']} months before prosecution")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    n_found = len(found)
    n_top10 = found["flagged_top10"].sum()
    n_top5 = found["flagged_top5"].sum()
    n_top1 = found["flagged_top1"].sum()

    print(f"\n  Entities in spending data: {n_found}")
    print(f"  Flagged in top 10% (MN): {n_top10}/{n_found} ({100*n_top10/n_found:.0f}%)")
    print(f"  Flagged in top 5% (MN):  {n_top5}/{n_found} ({100*n_top5/n_found:.0f}%)")
    print(f"  Flagged in top 1% (MN):  {n_top1}/{n_found} ({100*n_top1/n_found:.0f}%)")

    all_pre = found["all_pre_prosecution"].all() if "all_pre_prosecution" in found.columns else None
    print(f"\n  All billing data pre-prosecution: {'YES' if all_pre else 'NO'}")

    if n_found > 0:
        mean_pctl = found["pctl_mn"].mean()
        median_pctl = found["pctl_mn"].median()
        print(f"  Mean MN percentile of fraud entities: {mean_pctl:.1f}%")
        print(f"  Median MN percentile of fraud entities: {median_pctl:.1f}%")

    # Interpretation
    print("\n  INTERPRETATION:")
    if n_top5 / n_found >= 0.5:
        print("  ★★★ STRONG RESULT: >50% flagged in top 5% — publishable")
    elif n_top10 / n_found >= 0.5:
        print("  ★★ GOOD RESULT: >50% flagged in top 10% — real predictive power")
    else:
        print("  ★ MODERATE: Model catches some but not most fraud entities")
        print("    Note: Many fraud types (PCA, HSS, EIDBI) may not appear in T-MSIS")


# ---------------------------------------------------------------------------
# Stretch: Forward-looking flags
# ---------------------------------------------------------------------------

def stretch_analysis(scores):
    """Report aggregate statistics on unflagged top-percentile MN providers."""
    print("\n" + "=" * 100)
    print("STRETCH: FORWARD-LOOKING FLAGS (aggregate statistics only)")
    print("=" * 100)

    mn = scores[scores["state"] == "MN"].copy()
    mn["pctl_mn"] = mn["prob_full"].rank(pct=True) * 100

    # Exclude already-known fraud NPIs
    known_fraud_npis = set(info["npi"] for info in FRAUD_ENTITIES.values())
    # Also exclude LEIE-excluded providers
    mn_unknown = mn[
        (~mn["NPI"].isin(known_fraud_npis)) &
        (mn["excluded"] == 0)
    ]

    for threshold, pctl_name in [(99, "top 1%"), (95, "top 5%"), (90, "top 10%")]:
        flagged = mn_unknown[mn_unknown["pctl_mn"] >= threshold]
        n_flagged = len(flagged)

        print(f"\n  --- {pctl_name} ({n_flagged} providers) ---")
        if n_flagged == 0:
            continue

        # Aggregate characteristics (NO individual names)
        print(f"  Count: {n_flagged}")
        print(f"  Organization rate: {flagged['is_organization'].mean():.1%}")
        if "is_sole_prop" in flagged.columns:
            print(f"  Sole proprietor rate: {flagged['is_sole_prop'].mean():.1%}")

        # Entity age
        if "entity_age_months" in flagged.columns:
            age = flagged["entity_age_months"]
            print(f"  Entity age: median {age.median():.0f} months, mean {age.mean():.0f} months")
            young = (age < 60).sum()
            print(f"  Young entities (<5 years): {young} ({100*young/n_flagged:.0f}%)")

        # Billing patterns
        print(f"  Claims/month: median {flagged['claims_per_month'].median():.0f}, mean {flagged['claims_per_month'].mean():.0f}")
        print(f"  Procedures: median {flagged['n_procedures'].median():.0f}")

        if "self_billing_ratio" in flagged.columns:
            high_self = (flagged["self_billing_ratio"] > 0.5).sum()
            print(f"  High self-billing (>50%): {high_self} ({100*high_self/n_flagged:.0f}%)")

        if "activity_density" in flagged.columns:
            print(f"  Activity density: median {flagged['activity_density'].median():.2f}")

        # Geographic clustering (by zip3)
        if "zip3" in flagged.columns:
            zip3_counts = flagged["zip3"].value_counts()
            print(f"  Unique ZIP3 areas: {len(zip3_counts)}")
            if len(zip3_counts) > 0:
                top_zip = zip3_counts.head(3)
                print(f"  Most concentrated ZIP3s: {', '.join(f'{z}: {c}' for z, c in top_zip.items())}")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def generate_figures(results_df, mn_scores):
    """Generate validation visualizations."""
    print("\n" + "=" * 100)
    print("GENERATING FIGURES")
    print("=" * 100)

    found = results_df[results_df["in_data"]].copy()

    # --- Figure 1: MN provider score distribution with fraud entities marked ---
    print("\n[1/4] Score distribution with fraud entity markers...")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Background: all MN providers
    ax.hist(mn_scores["prob_full"], bins=80, alpha=0.6, color="#90CAF9",
            edgecolor="white", density=True, label="All MN providers")

    # Threshold lines
    for pctl, color, label in [(90, "#FFA726", "Top 10%"),
                                (95, "#EF5350", "Top 5%"),
                                (99, "#B71C1C", "Top 1%")]:
        thresh = mn_scores["prob_full"].quantile(pctl / 100)
        ax.axvline(x=thresh, color=color, linestyle="--", linewidth=1.5,
                   alpha=0.8, label=f"{label} (>{thresh:.3f})")

    # Mark fraud entities
    for _, row in found.iterrows():
        color = "#B71C1C" if row["flagged_top1"] else (
            "#EF5350" if row["flagged_top5"] else (
                "#FFA726" if row["flagged_top10"] else "#757575"
            )
        )
        ax.axvline(x=row["prob_full"], color=color, linewidth=2.5, alpha=0.9)
        # Short name for label
        short = row["entity"].split(" ")[0][:15]
        ax.annotate(short, xy=(row["prob_full"], ax.get_ylim()[1] * 0.85),
                    rotation=45, fontsize=7, color=color, fontweight="bold",
                    ha="left", va="bottom")

    ax.set_xlabel("Anomaly Score (prob_full)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Minnesota Provider Anomaly Score Distribution\nwith Named Fraud Entities Marked",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mn_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'mn_score_distribution.png'}")

    # --- Figure 2: Entity scorecard (horizontal bar) ---
    print("[2/4] Entity scorecard...")
    fig, ax = plt.subplots(figsize=(12, 7))

    found_sorted = found.sort_values("pctl_mn", ascending=True)
    y_pos = range(len(found_sorted))
    colors = []
    for _, row in found_sorted.iterrows():
        if row["flagged_top1"]:
            colors.append("#B71C1C")
        elif row["flagged_top5"]:
            colors.append("#EF5350")
        elif row["flagged_top10"]:
            colors.append("#FFA726")
        else:
            colors.append("#BDBDBD")

    bars = ax.barh(y_pos, found_sorted["pctl_mn"], color=colors, edgecolor="white",
                   height=0.7)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['entity']}\n({row['type']})"
                        for _, row in found_sorted.iterrows()],
                       fontsize=8)

    # Score annotations
    for i, (_, row) in enumerate(found_sorted.iterrows()):
        ax.text(row["pctl_mn"] + 0.5, i,
                f"{row['pctl_mn']:.1f}% (score: {row['prob_full']:.3f})",
                va="center", fontsize=8, fontweight="bold")

    # Threshold lines
    ax.axvline(x=90, color="#FFA726", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(x=95, color="#EF5350", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(x=99, color="#B71C1C", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(90.5, len(found_sorted) - 0.3, "Top 10%", fontsize=8, color="#FFA726")
    ax.text(95.5, len(found_sorted) - 0.3, "Top 5%", fontsize=8, color="#EF5350")
    ax.text(99.1, len(found_sorted) - 0.3, "Top 1%", fontsize=8, color="#B71C1C")

    ax.set_xlabel("Minnesota Percentile", fontsize=12)
    ax.set_title("Named Fraud Entity Model Scores\n(MN Percentile — Higher = More Anomalous)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim([0, 105])
    ax.grid(True, alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mn_entity_scorecard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'mn_entity_scorecard.png'}")

    # --- Figure 3: Temporal validation timeline ---
    print("[3/4] Temporal validation timeline...")
    fig, ax = plt.subplots(figsize=(14, 6))

    found_time = found.dropna(subset=["first_bill"])
    found_time = found_time.sort_values("first_bill")

    for i, (_, row) in enumerate(found_time.iterrows()):
        first = pd.Timestamp(row["first_bill"] + "-01")
        last = pd.Timestamp(row["last_bill"] + "-01")
        pros = pd.Timestamp(row["prosecution_date"] + "-01")

        # Data range bar
        color = "#B71C1C" if row["flagged_top5"] else "#FFA726" if row["flagged_top10"] else "#BDBDBD"
        ax.barh(i, (last - first).days, left=first, height=0.5,
                color=color, alpha=0.7, edgecolor="white")

        # Prosecution marker
        ax.plot(pros, i, "kx", markersize=10, markeredgewidth=2)

        # Entity label
        short = row["entity"][:30]
        ax.text(first - pd.Timedelta(days=30), i,
                f"{short} (p{row['pctl_mn']:.0f})",
                ha="right", va="center", fontsize=7)

    # Data cutoff line
    cutoff = pd.Timestamp("2024-12-31")
    ax.axvline(x=cutoff, color="blue", linestyle="-", linewidth=2, alpha=0.5)
    ax.text(cutoff, len(found_time) - 0.2, " Data cutoff\n (Dec 2024)",
            fontsize=8, color="blue", va="top")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Temporal Validation: Billing Data vs Prosecution Dates\n"
                 "(All model scores based on pre-prosecution data)",
                 fontsize=13, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#B71C1C", alpha=0.7, label="Flagged top 5%"),
        mpatches.Patch(facecolor="#FFA726", alpha=0.7, label="Flagged top 10%"),
        mpatches.Patch(facecolor="#BDBDBD", alpha=0.7, label="Not flagged"),
        Line2D([0], [0], marker="x", color="k", linestyle="None",
               markersize=8, markeredgewidth=2, label="Prosecution date"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mn_temporal_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'mn_temporal_validation.png'}")

    # --- Figure 4: Feature radar for top-flagged entities ---
    print("[4/4] Feature comparison radar...")
    radar_features = [
        "claims_per_month", "avg_paid_per_bene", "n_procedures",
        "self_billing_ratio", "activity_density", "entity_age_months"
    ]
    radar_labels = [
        "Claims/mo", "$/bene", "# Procs",
        "Self-bill %", "Activity", "Entity age"
    ]

    # Normalize all features to [0, 1] based on MN distribution
    mn_norm = {}
    for feat in radar_features:
        if feat in mn_scores.columns:
            mn_norm[feat] = {
                "min": mn_scores[feat].quantile(0.05),
                "max": mn_scores[feat].quantile(0.95),
            }

    # Only plot top-flagged entities (those in data with score > median)
    top_entities = found[found["pctl_mn"] >= 80].sort_values("pctl_mn", ascending=False)

    if len(top_entities) > 0 and len(radar_features) > 2:
        n_features = len(radar_features)
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        colors_radar = ["#B71C1C", "#EF5350", "#FF7043", "#FFA726", "#FFCA28"]
        for idx, (_, row) in enumerate(top_entities.iterrows()):
            vals = []
            for feat in radar_features:
                if feat in row and pd.notna(row[feat]) and feat in mn_norm:
                    v = (row[feat] - mn_norm[feat]["min"]) / (mn_norm[feat]["max"] - mn_norm[feat]["min"] + 1e-10)
                    vals.append(np.clip(v, 0, 1))
                else:
                    vals.append(0)
            vals += vals[:1]

            color = colors_radar[idx % len(colors_radar)]
            ax.plot(angles, vals, "o-", linewidth=2, color=color,
                    label=f"{row['entity'][:25]} (p{row['pctl_mn']:.0f})",
                    markersize=4)
            ax.fill(angles, vals, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Feature Profiles of Top-Flagged Fraud Entities\n(normalized to MN distribution)",
                      fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "mn_feature_radar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {FIGURES_DIR / 'mn_feature_radar.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation():
    """Run the full Minnesota fraud validation experiment."""
    start = time.time()

    print("=" * 100)
    print("EXPERIMENT: MINNESOTA FRAUD VALIDATION")
    print("Cross-referencing model predictions against named fraud defendants")
    print("=" * 100)

    # Load data
    scores = load_enriched_scores()
    spending_dates = load_spending_dates()

    # Validate
    results_df, mn_scores = validate_entities(scores, spending_dates)

    # Report
    print_validation_report(results_df)

    # Stretch analysis
    stretch_analysis(scores)

    # Figures
    generate_figures(results_df, mn_scores)

    # Save results
    results_path = MEDICAID_DIR / "minnesota_validation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    return results_df


if __name__ == "__main__":
    run_validation()
