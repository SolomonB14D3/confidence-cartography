"""
Experiment: New Mexico Medicaid Fraud Analysis
================================================
Apply the validated enriched fraud detection model (AUC 0.883) to New
Mexico providers. Validate against known NM fraud cases, generate
aggregate watchlist statistics, and compare NM vs MN risk distributions.

Named entities from:
  - AG Torrez press releases (2025)
  - Public court filings

Data: T-MSIS Medicaid Provider Spending (2018-2024), enriched with NPPES
      and cross-program features.
"""

import os
import sys
import json
import time
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
NM_DIR = MEDICAID_DIR / "nm_fraud"
NM_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = PROJECT_ROOT / "figures" / "nm_fraud"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Named fraud entities
# ---------------------------------------------------------------------------

FRAUD_ENTITIES = {
    "Kids in Need of Supportive Services (1)": {
        "npi": "1972944718",
        "type": "Afterschool/behavioral",
        "alleged_amount": "$1.6M",
        "source": "AG Torrez, Aug 2025",
        "prosecution_date": "2025-08",
        "category": "Behavioral",
    },
    "Kids in Need of Supportive Services (2)": {
        "npi": "1164749164",
        "type": "Afterschool/behavioral",
        "alleged_amount": "$1.6M",
        "source": "AG Torrez, Aug 2025",
        "prosecution_date": "2025-08",
        "category": "Behavioral",
    },
    "Susanne Kee (KISS operator)": {
        "npi": "1578759007",
        "type": "Individual (KISS operator)",
        "alleged_amount": "$1.6M",
        "source": "AG Torrez, Aug 2025",
        "prosecution_date": "2025-08",
        "category": "Behavioral",
    },
    "Bethanne Kee-Medran (KISS operator)": {
        "npi": "1942641790",
        "type": "Individual (KISS operator)",
        "alleged_amount": "$1.6M",
        "source": "AG Torrez, Aug 2025",
        "prosecution_date": "2025-08",
        "category": "Behavioral",
    },
    "Equine Assisted Programs of Southern NM": {
        "npi": "1730698309",
        "type": "Therapy",
        "alleged_amount": "$970K",
        "source": "AG Torrez, Mar 2025",
        "prosecution_date": "2025-03",
        "category": "Therapy",
    },
    "Nancy Marshall (EAP operator)": {
        "npi": "1073064028",
        "type": "Individual (EAP operator)",
        "alleged_amount": "$970K",
        "source": "AG Torrez, Mar 2025",
        "prosecution_date": "2025-03",
        "category": "Therapy",
    },
    "Luna Del Valle Healthcare Services": {
        "npi": "1659005171",
        "type": "Home health",
        "alleged_amount": "Unknown",
        "source": "AG Torrez, 2025",
        "prosecution_date": "2025-01",
        "category": "Home health",
    },
    "Hospice De La Luz (Justus LLC)": {
        "npi": "1255433579",
        "type": "Hospice",
        "alleged_amount": "Unknown",
        "source": "AG Torrez, 2025",
        "prosecution_date": "2025-01",
        "category": "Hospice",
    },
}

UNMATCHED_ENTITIES = [
    "April Guadalupe Hernandez (imposter nurse, no NPI — practiced without license)",
    "Lily Care of New Mexico (no NPPES record found)",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_enriched_scores():
    """Load the enriched anomaly scores."""
    path = MEDICAID_DIR / "anomaly_scores_enriched.csv"
    print(f"Loading enriched scores from {path}...")
    scores = pd.read_csv(path, dtype={"NPI": str}, low_memory=False)
    print(f"  Total providers: {len(scores):,}")
    return scores


def load_spending_dates():
    """Load NPI + claim dates from spending data."""
    path = MEDICAID_DIR / "medicaid_provider_spending.parquet"
    print(f"Loading spending dates from {path}...")
    df = pd.read_parquet(path, columns=["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"])
    df["CLAIM_FROM_MONTH"] = pd.to_datetime(df["CLAIM_FROM_MONTH"], errors="coerce")
    print(f"  Total rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Step 1: Score NM providers
# ---------------------------------------------------------------------------

def score_nm_providers(scores):
    """Extract and score NM providers."""
    print("\n" + "=" * 100)
    print("STEP 1: SCORE NM PROVIDERS")
    print("=" * 100)

    nm = scores[scores["state"] == "NM"].copy()
    nm["pctl_nm"] = nm["prob_full"].rank(pct=True) * 100
    scores["pctl_national"] = scores["prob_full"].rank(pct=True) * 100

    print(f"\n  NM providers: {len(nm):,}")
    print(f"  NM LEIE-excluded: {nm['excluded'].sum()}")
    print(f"  NM total paid: ${nm['total_paid'].sum():,.0f}")
    print(f"  NM total claims: {nm['total_claims'].sum():,.0f}")

    print(f"\n  Score distribution:")
    for pct in [50, 75, 90, 95, 99, 99.5]:
        thresh = nm["prob_full"].quantile(pct / 100)
        n = (nm["prob_full"] >= thresh).sum()
        print(f"    p{pct}: {thresh:.4f} ({n} providers)")

    # Save NM scores
    nm_path = NM_DIR / "nm_provider_scores.csv"
    nm.to_csv(nm_path, index=False)
    print(f"\n  NM scores saved to {nm_path}")

    return nm, scores


# ---------------------------------------------------------------------------
# Step 2: Validate against known cases
# ---------------------------------------------------------------------------

def validate_entities(scores, nm_scores, spending_dates):
    """Match named fraud entities to spending data and check model scores."""
    print("\n" + "=" * 100)
    print("STEP 2: VALIDATE AGAINST KNOWN NM FRAUD CASES")
    print("=" * 100)

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

            # NM percentile
            nm_match = nm_scores[nm_scores["NPI"] == npi]
            rec["pctl_nm"] = nm_match["pctl_nm"].iloc[0] if len(nm_match) > 0 else None
            rec["pctl_national"] = match["pctl_national"].iloc[0]

            pctl = rec["pctl_nm"]
            rec["flagged_top10"] = pctl >= 90 if pctl else False
            rec["flagged_top5"] = pctl >= 95 if pctl else False
            rec["flagged_top1"] = pctl >= 99 if pctl else False

            # Key features
            for feat in ["total_claims", "total_paid", "claims_per_month",
                         "self_billing_ratio", "activity_density",
                         "entity_age_months", "is_organization", "is_sole_prop",
                         "months_since_start", "n_procedures", "n_months",
                         "avg_paid_per_bene", "avg_paid_per_claim"]:
                rec[feat] = row.get(feat, None)

            # Temporal
            entity_spend = spending_dates[
                spending_dates["BILLING_PROVIDER_NPI_NUM"] == npi
            ]
            if len(entity_spend) > 0:
                rec["first_bill"] = entity_spend["CLAIM_FROM_MONTH"].min().strftime("%Y-%m")
                rec["last_bill"] = entity_spend["CLAIM_FROM_MONTH"].max().strftime("%Y-%m")
                rec["billing_months"] = entity_spend["CLAIM_FROM_MONTH"].nunique()
                pros_dt = pd.Timestamp(info["prosecution_date"] + "-01")
                rec["all_pre_prosecution"] = entity_spend["CLAIM_FROM_MONTH"].max() < pros_dt
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
            rec["pctl_nm"] = None
            rec["pctl_national"] = None
            rec["flagged_top10"] = False
            rec["flagged_top5"] = False
            rec["flagged_top1"] = False

        results.append(rec)

    results_df = pd.DataFrame(results)

    # Print report
    found = results_df[results_df["in_data"]]
    not_found = results_df[~results_df["in_data"]]

    print(f"\n  Entities searched: {len(results_df)}")
    print(f"  Found in spending data: {len(found)}")
    print(f"  Not found: {len(not_found)}")
    if len(not_found) > 0:
        for _, row in not_found.iterrows():
            print(f"    - {row['entity']} (NPI: {row['npi']})")

    print(f"\n  Additionally, {len(UNMATCHED_ENTITIES)} entities had no NPI match:")
    for e in UNMATCHED_ENTITIES:
        print(f"    - {e}")

    # Per-entity details
    print("\n" + "-" * 100)
    print("PER-ENTITY RESULTS")
    print("-" * 100)

    for _, row in found.sort_values("prob_full", ascending=False).iterrows():
        print(f"\n  Entity: {row['entity']}")
        print(f"  NPI: {row['npi']} | Type: {row['type']} | Category: {row['category']}")
        print(f"  Alleged: {row['alleged_amount']} | Source: {row['source']}")
        print(f"  Model score: {row['prob_full']:.4f}")
        pctl_str = f"{row['pctl_nm']:.1f}%" if pd.notna(row.get('pctl_nm')) else "N/A"
        nat_str = f"{row['pctl_national']:.1f}%" if pd.notna(row.get('pctl_national')) else "N/A"
        print(f"  NM Percentile: {pctl_str} | National: {nat_str}")

        t10 = "YES" if row["flagged_top10"] else "NO"
        t5 = "YES" if row["flagged_top5"] else "NO"
        t1 = "YES" if row["flagged_top1"] else "NO"
        print(f"  Flagged at top 10%: {t10} | top 5%: {t5} | top 1%: {t1}")

        if row.get("first_bill"):
            print(f"  Billing: {row['first_bill']} to {row['last_bill']} ({row['billing_months']} months)")
            print(f"  Prosecution: {row['prosecution_date']}")
            pre = "ALL pre-prosecution" if row["all_pre_prosecution"] else "INCLUDES post-prosecution"
            print(f"  Temporal: {pre}")
            if pd.notna(row.get("lead_time_months")):
                print(f"  Lead time: {int(row['lead_time_months'])} months before prosecution")

        # Key features
        print(f"  Features: claims/mo={row.get('claims_per_month', 'N/A'):.0f}, "
              f"procs={row.get('n_procedures', 'N/A')}, "
              f"months={row.get('n_months', 'N/A')}, "
              f"self_bill={row.get('self_billing_ratio', 'N/A'):.2f}, "
              f"org={row.get('is_organization', 'N/A')}")

    # Summary
    n_found = len(found)
    n_top10 = found["flagged_top10"].sum()
    n_top5 = found["flagged_top5"].sum()
    n_top1 = found["flagged_top1"].sum()

    print(f"\n{'='*100}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*100}")
    print(f"  Entities in spending data: {n_found}")
    print(f"  Flagged in top 10% (NM): {n_top10}/{n_found} ({100*n_top10/n_found:.0f}%)")
    print(f"  Flagged in top 5% (NM):  {n_top5}/{n_found} ({100*n_top5/n_found:.0f}%)")
    print(f"  Flagged in top 1% (NM):  {n_top1}/{n_found} ({100*n_top1/n_found:.0f}%)")

    all_pre = found["all_pre_prosecution"].all() if "all_pre_prosecution" in found.columns else None
    print(f"  All billing data pre-prosecution: {'YES' if all_pre else 'NO'}")

    if n_found > 0:
        mean_pctl = found["pctl_nm"].mean()
        median_pctl = found["pctl_nm"].median()
        print(f"  Mean NM percentile: {mean_pctl:.1f}%")
        print(f"  Median NM percentile: {median_pctl:.1f}%")

    # Interpretation
    print(f"\n  INTERPRETATION:")
    if n_found > 0 and n_top5 / n_found >= 0.5:
        print("  ★★★ STRONG: >50% flagged in top 5%")
    elif n_found > 0 and n_top10 / n_found >= 0.5:
        print("  ★★ GOOD: >50% flagged in top 10%")
    else:
        print("  ★ MODERATE: Model catches some but not most NM fraud entities")

    # Save validation
    val_dict = {
        "entities_searched": len(results_df),
        "found_in_data": n_found,
        "flagged_top10": int(n_top10),
        "flagged_top5": int(n_top5),
        "flagged_top1": int(n_top1),
        "all_pre_prosecution": bool(all_pre) if all_pre is not None else None,
        "mean_pctl_nm": float(mean_pctl) if n_found > 0 else None,
        "median_pctl_nm": float(median_pctl) if n_found > 0 else None,
    }
    val_path = NM_DIR / "nm_validation.json"
    with open(val_path, "w") as f:
        json.dump(val_dict, f, indent=2)
    print(f"\n  Validation saved to {val_path}")

    return results_df


# ---------------------------------------------------------------------------
# Step 3: Watchlist aggregate statistics
# ---------------------------------------------------------------------------

def watchlist_stats(nm_scores, scores):
    """Report aggregate statistics on top NM providers — no names."""
    print("\n" + "=" * 100)
    print("STEP 3: NM WATCHLIST (aggregate statistics only)")
    print("=" * 100)

    known_fraud_npis = set(info["npi"] for info in FRAUD_ENTITIES.values())
    nm_clean = nm_scores[
        (~nm_scores["NPI"].isin(known_fraud_npis)) &
        (nm_scores["excluded"] == 0)
    ]

    watchlist = {}

    for threshold, label in [(99, "top 1%"), (95, "top 5%"), (90, "top 10%")]:
        flagged = nm_clean[nm_clean["pctl_nm"] >= threshold]
        n = len(flagged)

        stats = {"count": n}

        print(f"\n  --- {label} ({n} providers) ---")
        if n == 0:
            watchlist[label] = stats
            continue

        stats["org_rate"] = float(flagged["is_organization"].mean())
        print(f"  Organization rate: {stats['org_rate']:.1%}")

        if "is_sole_prop" in flagged.columns:
            stats["sole_prop_rate"] = float(flagged["is_sole_prop"].mean())
            print(f"  Sole proprietor rate: {stats['sole_prop_rate']:.1%}")

        if "entity_age_months" in flagged.columns:
            age = flagged["entity_age_months"]
            stats["entity_age_median"] = float(age.median())
            stats["entity_age_mean"] = float(age.mean())
            young = int((age < 60).sum())
            stats["young_entities"] = young
            print(f"  Entity age: median {age.median():.0f}mo, mean {age.mean():.0f}mo")
            print(f"  Young entities (<5 years): {young} ({100*young/n:.0f}%)")

        stats["claims_per_month_median"] = float(flagged["claims_per_month"].median())
        stats["claims_per_month_mean"] = float(flagged["claims_per_month"].mean())
        stats["n_procedures_median"] = float(flagged["n_procedures"].median())
        print(f"  Claims/month: median {flagged['claims_per_month'].median():.0f}, mean {flagged['claims_per_month'].mean():.0f}")
        print(f"  Procedures: median {flagged['n_procedures'].median():.0f}")

        if "self_billing_ratio" in flagged.columns:
            high_self = int((flagged["self_billing_ratio"] > 0.5).sum())
            stats["high_self_billing"] = high_self
            print(f"  High self-billing (>50%): {high_self} ({100*high_self/n:.0f}%)")

        if "zip3" in flagged.columns:
            zip3_counts = flagged["zip3"].value_counts()
            stats["unique_zip3"] = len(zip3_counts)
            stats["top_zip3s"] = {str(k): int(v) for k, v in zip3_counts.head(5).items()}
            print(f"  Unique ZIP3 areas: {len(zip3_counts)}")
            print(f"  Top ZIP3s: {', '.join(f'{z}: {c}' for z, c in zip3_counts.head(5).items())}")

        # Total paid stats
        stats["total_paid_median"] = float(flagged["total_paid"].median())
        stats["total_paid_mean"] = float(flagged["total_paid"].mean())
        print(f"  Total paid: median ${flagged['total_paid'].median():,.0f}, mean ${flagged['total_paid'].mean():,.0f}")

        watchlist[label] = stats

    # Save
    wl_path = NM_DIR / "nm_watchlist_stats.json"
    with open(wl_path, "w") as f:
        json.dump(watchlist, f, indent=2, default=str)
    print(f"\n  Watchlist stats saved to {wl_path}")

    return watchlist


# ---------------------------------------------------------------------------
# Step 4: Compare NM vs MN
# ---------------------------------------------------------------------------

def compare_nm_mn(scores, nm_scores):
    """Compare NM and MN fraud risk distributions."""
    print("\n" + "=" * 100)
    print("STEP 4: NM vs MN COMPARISON")
    print("=" * 100)

    mn = scores[scores["state"] == "MN"].copy()
    nm = nm_scores.copy()

    # Basic stats
    stats = {}
    for label, df in [("NM", nm), ("MN", mn)]:
        s = {
            "n_providers": len(df),
            "total_paid": float(df["total_paid"].sum()),
            "total_claims": float(df["total_claims"].sum()),
            "excluded_count": int(df["excluded"].sum()),
            "excluded_rate": float(df["excluded"].mean()),
            "prob_full_mean": float(df["prob_full"].mean()),
            "prob_full_median": float(df["prob_full"].median()),
            "prob_full_p90": float(df["prob_full"].quantile(0.9)),
            "prob_full_p95": float(df["prob_full"].quantile(0.95)),
            "prob_full_p99": float(df["prob_full"].quantile(0.99)),
        }

        # Provider type mix
        if "is_organization" in df.columns:
            s["org_rate"] = float(df["is_organization"].mean())
        if "is_sole_prop" in df.columns:
            s["sole_prop_rate"] = float(df["is_sole_prop"].mean())

        # Entity age
        if "entity_age_months" in df.columns:
            s["entity_age_median"] = float(df["entity_age_months"].median())

        stats[label] = s

    # Print comparison table
    print(f"\n  {'Metric':<35} {'NM':>15} {'MN':>15} {'Ratio NM/MN':>12}")
    print(f"  {'─'*80}")

    comparisons = [
        ("Providers", "n_providers", "{:,.0f}"),
        ("Total paid", "total_paid", "${:,.0f}"),
        ("Total claims", "total_claims", "{:,.0f}"),
        ("LEIE excluded", "excluded_count", "{:,.0f}"),
        ("Exclusion rate", "excluded_rate", "{:.4f}"),
        ("Mean score", "prob_full_mean", "{:.4f}"),
        ("Median score", "prob_full_median", "{:.4f}"),
        ("p90 score", "prob_full_p90", "{:.4f}"),
        ("p95 score", "prob_full_p95", "{:.4f}"),
        ("p99 score", "prob_full_p99", "{:.4f}"),
        ("Organization rate", "org_rate", "{:.3f}"),
        ("Sole proprietor rate", "sole_prop_rate", "{:.3f}"),
        ("Median entity age (mo)", "entity_age_median", "{:.0f}"),
    ]

    for label, key, fmt in comparisons:
        nm_val = stats["NM"].get(key, None)
        mn_val = stats["MN"].get(key, None)
        if nm_val is not None and mn_val is not None:
            nm_str = fmt.format(nm_val)
            mn_str = fmt.format(mn_val)
            ratio = nm_val / mn_val if mn_val != 0 else float("inf")
            print(f"  {label:<35} {nm_str:>15} {mn_str:>15} {ratio:>12.2f}")

    # Feature comparison: what do high-risk providers look like in each state?
    print(f"\n  --- Top 5% Provider Characteristics ---")
    for label, df in [("NM", nm), ("MN", mn)]:
        pctl_col = "pctl_nm" if label == "NM" else "prob_full"
        if label == "MN":
            df["pctl_state"] = df["prob_full"].rank(pct=True) * 100
        else:
            df["pctl_state"] = df["pctl_nm"]

        top5 = df[df["pctl_state"] >= 95]
        print(f"\n  {label} top 5% ({len(top5)} providers):")
        print(f"    Claims/mo: median {top5['claims_per_month'].median():.0f}")
        print(f"    Procedures: median {top5['n_procedures'].median():.0f}")
        print(f"    Org rate: {top5['is_organization'].mean():.1%}")
        if "self_billing_ratio" in top5.columns:
            print(f"    Self-billing >50%: {(top5['self_billing_ratio'] > 0.5).mean():.1%}")
        if "entity_age_months" in top5.columns:
            print(f"    Entity age: median {top5['entity_age_months'].median():.0f}mo")

    return stats


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(results_df, nm_scores, scores):
    """Generate all NM figures."""
    print("\n" + "=" * 100)
    print("GENERATING FIGURES")
    print("=" * 100)

    found = results_df[results_df["in_data"]].copy()
    mn = scores[scores["state"] == "MN"].copy()

    # --- Figure 1: NM Score Distribution with fraud entities ---
    print("\n[1/4] NM score distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(nm_scores["prob_full"], bins=80, alpha=0.6, color="#81C784",
            edgecolor="white", density=True, label="All NM providers")

    for pctl, color, label in [(90, "#FFA726", "Top 10%"),
                                (95, "#EF5350", "Top 5%"),
                                (99, "#B71C1C", "Top 1%")]:
        thresh = nm_scores["prob_full"].quantile(pctl / 100)
        ax.axvline(x=thresh, color=color, linestyle="--", linewidth=1.5,
                   alpha=0.8, label=f"{label} (>{thresh:.3f})")

    for _, row in found.iterrows():
        color = "#B71C1C" if row["flagged_top1"] else (
            "#EF5350" if row["flagged_top5"] else (
                "#FFA726" if row["flagged_top10"] else "#757575"
            )
        )
        ax.axvline(x=row["prob_full"], color=color, linewidth=2.5, alpha=0.9)
        short = row["entity"].split("(")[0].strip()[:15]
        ax.annotate(short, xy=(row["prob_full"], ax.get_ylim()[1] * 0.85),
                    rotation=45, fontsize=7, color=color, fontweight="bold",
                    ha="left", va="bottom")

    ax.set_xlabel("Anomaly Score (prob_full)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("New Mexico Provider Anomaly Score Distribution\nwith Named Fraud Entities Marked",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "nm_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'nm_score_distribution.png'}")

    # --- Figure 2: Entity scorecard ---
    print("[2/4] Entity scorecard...")
    if len(found) > 0:
        fig, ax = plt.subplots(figsize=(12, max(5, len(found) * 0.8)))

        found_sorted = found.sort_values("pctl_nm", ascending=True)
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

        bars = ax.barh(y_pos, found_sorted["pctl_nm"], color=colors,
                       edgecolor="white", height=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['entity'][:35]}\n({row['type'][:25]})"
                            for _, row in found_sorted.iterrows()],
                           fontsize=7)

        for i, (_, row) in enumerate(found_sorted.iterrows()):
            pctl_val = row["pctl_nm"] if pd.notna(row["pctl_nm"]) else 0
            ax.text(pctl_val + 0.5, i,
                    f"{pctl_val:.1f}% (score: {row['prob_full']:.3f})",
                    va="center", fontsize=8, fontweight="bold")

        ax.axvline(x=90, color="#FFA726", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(x=95, color="#EF5350", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(x=99, color="#B71C1C", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(90.5, len(found_sorted) - 0.3, "Top 10%", fontsize=8, color="#FFA726")
        ax.text(95.5, len(found_sorted) - 0.3, "Top 5%", fontsize=8, color="#EF5350")
        ax.text(99.1, len(found_sorted) - 0.3, "Top 1%", fontsize=8, color="#B71C1C")

        ax.set_xlabel("New Mexico Percentile", fontsize=12)
        ax.set_title("Named NM Fraud Entity Model Scores\n(NM Percentile — Higher = More Anomalous)",
                      fontsize=13, fontweight="bold")
        ax.set_xlim([0, 105])
        ax.grid(True, alpha=0.2, axis="x")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "nm_entity_scorecard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {FIGURES_DIR / 'nm_entity_scorecard.png'}")

    # --- Figure 3: NM vs MN comparison ---
    print("[3/4] NM vs MN comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Score distributions overlaid
    ax = axes[0]
    ax.hist(nm_scores["prob_full"], bins=60, alpha=0.5, color="#4CAF50",
            density=True, label=f"NM (n={len(nm_scores):,})")
    ax.hist(mn["prob_full"], bins=60, alpha=0.5, color="#2196F3",
            density=True, label=f"MN (n={len(mn):,})")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # (b) Provider type comparison
    ax = axes[1]
    nm_types = {
        "Organization": nm_scores["is_organization"].mean(),
        "Sole Prop": nm_scores["is_sole_prop"].mean() if "is_sole_prop" in nm_scores.columns else 0,
        "Individual": 1 - nm_scores["is_organization"].mean(),
    }
    mn_types = {
        "Organization": mn["is_organization"].mean(),
        "Sole Prop": mn["is_sole_prop"].mean() if "is_sole_prop" in mn.columns else 0,
        "Individual": 1 - mn["is_organization"].mean(),
    }
    x = np.arange(len(nm_types))
    width = 0.35
    ax.bar(x - width/2, list(nm_types.values()), width, color="#4CAF50", alpha=0.7, label="NM")
    ax.bar(x + width/2, list(mn_types.values()), width, color="#2196F3", alpha=0.7, label="MN")
    ax.set_xticks(x)
    ax.set_xticklabels(list(nm_types.keys()))
    ax.set_ylabel("Rate")
    ax.set_title("Provider Type Mix")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # (c) Top features at top 5%
    ax = axes[2]
    feat_cols = ["claims_per_month", "avg_paid_per_bene", "n_procedures",
                 "self_billing_ratio", "entity_age_months"]

    nm_scores["pctl_state"] = nm_scores["pctl_nm"]
    mn["pctl_state"] = mn["prob_full"].rank(pct=True) * 100

    nm_top = nm_scores[nm_scores["pctl_state"] >= 95]
    mn_top = mn[mn["pctl_state"] >= 95]

    # Normalize: ratio of top-5% median to state median
    nm_ratios = []
    mn_ratios = []
    for feat in feat_cols:
        if feat in nm_scores.columns and feat in mn.columns:
            nm_med = nm_scores[feat].median()
            mn_med = mn[feat].median()
            nm_top_med = nm_top[feat].median()
            mn_top_med = mn_top[feat].median()
            nm_r = nm_top_med / nm_med if nm_med != 0 else 1
            mn_r = mn_top_med / mn_med if mn_med != 0 else 1
            nm_ratios.append(nm_r)
            mn_ratios.append(mn_r)
        else:
            nm_ratios.append(1)
            mn_ratios.append(1)

    x = np.arange(len(feat_cols))
    ax.barh(x - width/2, nm_ratios, width, color="#4CAF50", alpha=0.7, label="NM")
    ax.barh(x + width/2, mn_ratios, width, color="#2196F3", alpha=0.7, label="MN")
    ax.set_yticks(x)
    ax.set_yticklabels([f.replace("_", "\n") for f in feat_cols], fontsize=8)
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Ratio (top 5% median / state median)")
    ax.set_title("Top-5% Feature Profile")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle("New Mexico vs Minnesota Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "nm_vs_mn_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'nm_vs_mn_comparison.png'}")

    # --- Figure 4: Top features for NM flagged providers ---
    print("[4/4] Top features importance...")
    # Compute feature z-scores for top 1% vs rest
    feat_all = ["claims_per_month", "avg_paid_per_claim", "avg_paid_per_bene",
                "n_procedures", "n_months", "self_billing_ratio",
                "activity_density", "entity_age_months", "is_organization",
                "is_sole_prop", "months_since_start"]

    nm_top1 = nm_scores[nm_scores["pctl_nm"] >= 99]
    nm_rest = nm_scores[nm_scores["pctl_nm"] < 99]

    fig, ax = plt.subplots(figsize=(10, 7))
    diffs = []
    labels = []
    for feat in feat_all:
        if feat in nm_scores.columns:
            top_mean = nm_top1[feat].mean()
            rest_mean = nm_rest[feat].mean()
            rest_std = nm_rest[feat].std()
            if rest_std > 0:
                z = (top_mean - rest_mean) / rest_std
            else:
                z = 0
            diffs.append(z)
            labels.append(feat.replace("_", "\n"))

    sorted_idx = np.argsort(np.abs(diffs))[::-1]
    diffs = [diffs[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]

    colors = ["#EF5350" if d > 0 else "#42A5F5" for d in diffs]
    ax.barh(range(len(diffs)), diffs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Z-score (top 1% mean vs rest)", fontsize=11)
    ax.set_title("NM Top 1%: Feature Deviations from Normal Providers",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "nm_top_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'nm_top_features.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis():
    """Run the full NM fraud analysis."""
    start = time.time()

    print("=" * 100)
    print("EXPERIMENT: NEW MEXICO MEDICAID FRAUD ANALYSIS")
    print("=" * 100)

    # Load data
    scores = load_enriched_scores()
    spending_dates = load_spending_dates()

    # Step 1
    nm_scores, scores = score_nm_providers(scores)

    # Step 2
    results_df = validate_entities(scores, nm_scores, spending_dates)

    # Step 3
    watchlist_stats(nm_scores, scores)

    # Step 4
    compare_nm_mn(scores, nm_scores)

    # Figures
    generate_figures(results_df, nm_scores, scores)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    return results_df, nm_scores


if __name__ == "__main__":
    run_analysis()
