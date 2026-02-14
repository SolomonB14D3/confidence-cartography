"""
Experiment: Medicaid Confidence Cartography
============================================
Test whether confidence cartography generalizes from natural language
to structured billing data. Use provider spending patterns as input,
map anomalies, and validate against known fraud cases (LEIE exclusions).

Data Sources:
  - Medicaid Provider Spending (T-MSIS): provider-procedure-month billing
  - LEIE (List of Excluded Individuals/Entities): fraud ground truth

Method:
  1. Join spending data with LEIE on NPI
  2. Build provider billing profiles
  3. Approach A: Statistical anomaly detection (z-scores vs peers)
  4. Approach B: LM confidence mapping (serialize profiles → model confidence)
  5. Validate both approaches against LEIE exclusions
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import PROJECT_ROOT as _PR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MEDICAID_DIR = PROJECT_ROOT / "data" / "medicaid"
FIGURES_DIR = PROJECT_ROOT / "figures" / "medicaid"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SPENDING_PATH = MEDICAID_DIR / "medicaid_provider_spending.parquet"
LEIE_PATH = MEDICAID_DIR / "leie.csv"

# Set HF cache to 4TB disk
os.environ["HF_HOME"] = "/Volumes/4TB SD/hf_cache"

# ---------------------------------------------------------------------------
# Step 1: Load and explore data
# ---------------------------------------------------------------------------

def load_leie():
    """Load LEIE exclusion list, filter to fraud-related codes with valid NPIs."""
    print("Loading LEIE...")
    leie = pd.read_csv(LEIE_PATH, dtype={"NPI": str}, low_memory=False)
    print(f"  Total LEIE entries: {len(leie):,}")

    # Filter to valid NPIs (10-digit, not all zeros)
    leie = leie[
        (leie["NPI"].notna()) &
        (leie["NPI"] != "0000000000") &
        (leie["NPI"].str.len() == 10) &
        (leie["NPI"].str.isdigit())
    ].copy()
    print(f"  With valid NPI: {len(leie):,}")

    # Fraud-related exclusion codes:
    # 1128a1 = conviction of program-related crimes
    # 1128a2 = patient abuse
    # 1128a3 = felony health care fraud
    # 1128a4 = felony controlled substance
    fraud_codes = ["1128a1", "1128a2", "1128a3", "1128a4"]
    leie["is_fraud"] = leie["EXCLTYPE"].isin(fraud_codes)

    # Parse exclusion date
    leie["excl_date"] = pd.to_datetime(
        leie["EXCLDATE"].astype(str), format="%Y%m%d", errors="coerce"
    )

    print(f"  Fraud-related (1128a1-a4): {leie['is_fraud'].sum():,}")
    print(f"  Non-fraud exclusions: {(~leie['is_fraud']).sum():,}")

    return leie


def load_spending(sample_frac=None):
    """Load Medicaid provider spending data from parquet.

    If sample_frac is set (e.g., 0.1), only load a random sample.
    """
    print("Loading Medicaid spending data...")
    spending = pd.read_parquet(SPENDING_PATH)
    print(f"  Total rows: {len(spending):,}")
    print(f"  Columns: {list(spending.columns)}")
    print(f"  Memory: {spending.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    if sample_frac and sample_frac < 1.0:
        print(f"  Sampling {sample_frac*100:.0f}% of data...")
        spending = spending.sample(frac=sample_frac, random_state=42)
        print(f"  Sampled rows: {len(spending):,}")

    return spending


def explore_data(spending, leie):
    """Print dataset exploration summary and return overlap statistics."""
    print("\n" + "=" * 65)
    print("DATA EXPLORATION")
    print("=" * 65)

    # Spending summary
    print("\n--- Spending Dataset ---")
    print(f"Rows: {len(spending):,}")
    print(f"Unique billing NPIs: {spending['BILLING_PROVIDER_NPI_NUM'].nunique():,}")
    print(f"Unique servicing NPIs: {spending['SERVICING_PROVIDER_NPI_NUM'].nunique():,}")
    print(f"Unique HCPCS codes: {spending['HCPCS_CODE'].nunique():,}")

    # Date range
    if "CLAIM_FROM_MONTH" in spending.columns:
        dates = pd.to_datetime(spending["CLAIM_FROM_MONTH"], errors="coerce")
        print(f"Date range: {dates.min()} to {dates.max()}")

    # Spending distribution
    print(f"\nTotal spending: ${spending['TOTAL_PAID'].sum():,.0f}")
    print(f"Mean per row: ${spending['TOTAL_PAID'].mean():,.2f}")
    print(f"Median per row: ${spending['TOTAL_PAID'].median():,.2f}")

    # LEIE summary
    print("\n--- LEIE Dataset ---")
    print(f"Entries with valid NPI: {len(leie):,}")
    print(f"Unique NPIs: {leie['NPI'].nunique():,}")
    print(f"Fraud-related: {leie['is_fraud'].sum():,}")

    # NPI overlap
    spending_npis = set(spending["BILLING_PROVIDER_NPI_NUM"].dropna().unique())
    leie_npis = set(leie["NPI"].unique())
    fraud_npis = set(leie[leie["is_fraud"]]["NPI"].unique())

    overlap_all = spending_npis & leie_npis
    overlap_fraud = spending_npis & fraud_npis

    print(f"\n--- NPI Overlap ---")
    print(f"Spending NPIs: {len(spending_npis):,}")
    print(f"LEIE NPIs: {len(leie_npis):,}")
    print(f"Overlap (all LEIE): {len(overlap_all):,}")
    print(f"Overlap (fraud only): {len(overlap_fraud):,}")

    if len(overlap_fraud) < 50:
        print("\n⚠️  WARNING: Overlap < 50. Statistical validation will be weak.")
    elif len(overlap_fraud) < 200:
        print(f"\n⚠  Modest overlap ({len(overlap_fraud)}). Proceed with caution.")
    else:
        print(f"\n✓  Good overlap ({len(overlap_fraud)}). Proceed with validation.")

    return {
        "spending_npis": spending_npis,
        "leie_npis": leie_npis,
        "fraud_npis": fraud_npis,
        "overlap_all": overlap_all,
        "overlap_fraud": overlap_fraud,
    }


# ---------------------------------------------------------------------------
# Step 2: Build provider profiles
# ---------------------------------------------------------------------------

def build_provider_profiles(spending, fraud_npis):
    """Aggregate spending data into per-provider profiles.

    Returns a DataFrame with one row per billing provider NPI.
    """
    print("\n" + "=" * 65)
    print("BUILDING PROVIDER PROFILES")
    print("=" * 65)

    npi_col = "BILLING_PROVIDER_NPI_NUM"

    # Group by provider
    print("Aggregating by provider...")
    profiles = spending.groupby(npi_col).agg(
        total_claims=("TOTAL_CLAIMS", "sum"),
        total_paid=("TOTAL_PAID", "sum"),
        total_beneficiaries=("TOTAL_UNIQUE_BENEFICIARIES", "sum"),
        n_procedures=("HCPCS_CODE", "nunique"),
        n_months=("CLAIM_FROM_MONTH", "nunique"),
        n_rows=("TOTAL_CLAIMS", "count"),
    ).reset_index()

    profiles.rename(columns={npi_col: "NPI"}, inplace=True)

    # Derived features
    profiles["avg_paid_per_claim"] = profiles["total_paid"] / profiles["total_claims"].clip(lower=1)
    profiles["avg_paid_per_bene"] = profiles["total_paid"] / profiles["total_beneficiaries"].clip(lower=1)
    profiles["claims_per_month"] = profiles["total_claims"] / profiles["n_months"].clip(lower=1)
    profiles["procedure_concentration"] = 1.0 / profiles["n_procedures"].clip(lower=1)

    # Top procedure share (measure of concentration)
    print("Computing procedure concentration...")
    top_proc_share = (
        spending.groupby([npi_col, "HCPCS_CODE"])["TOTAL_CLAIMS"]
        .sum()
        .reset_index()
        .sort_values("TOTAL_CLAIMS", ascending=False)
        .groupby(npi_col)
        .first()["TOTAL_CLAIMS"]
    )
    total_by_npi = spending.groupby(npi_col)["TOTAL_CLAIMS"].sum()
    top_share = (top_proc_share / total_by_npi).reset_index()
    top_share.columns = ["NPI", "top_procedure_share"]
    profiles = profiles.merge(top_share, on="NPI", how="left")

    # Monthly spending variance (coefficient of variation)
    print("Computing monthly variance...")
    monthly_spending = (
        spending.groupby([npi_col, "CLAIM_FROM_MONTH"])["TOTAL_PAID"]
        .sum()
        .reset_index()
    )
    monthly_cv = monthly_spending.groupby(npi_col)["TOTAL_PAID"].agg(
        lambda x: x.std() / x.mean() if x.mean() > 0 else 0
    ).reset_index()
    monthly_cv.columns = ["NPI", "monthly_spending_cv"]
    profiles = profiles.merge(monthly_cv, on="NPI", how="left")

    # Label fraud
    profiles["excluded"] = profiles["NPI"].isin(fraud_npis).astype(int)

    print(f"\nProvider profiles built: {len(profiles):,}")
    print(f"Excluded providers: {profiles['excluded'].sum():,}")
    print(f"Non-excluded providers: {(profiles['excluded'] == 0).sum():,}")
    print(f"Exclusion rate: {100 * profiles['excluded'].mean():.3f}%")

    # Feature summary
    feature_cols = [
        "total_claims", "total_paid", "total_beneficiaries", "n_procedures",
        "n_months", "avg_paid_per_claim", "avg_paid_per_bene",
        "claims_per_month", "top_procedure_share", "monthly_spending_cv"
    ]
    print(f"\nFeature summary (excluded vs non-excluded):")
    print(f"{'Feature':<25} {'Excluded Mean':>15} {'Non-Excl Mean':>15} {'Ratio':>8}")
    print("-" * 65)
    for col in feature_cols:
        excl_mean = profiles.loc[profiles["excluded"] == 1, col].mean()
        non_mean = profiles.loc[profiles["excluded"] == 0, col].mean()
        ratio = excl_mean / non_mean if non_mean > 0 else float("inf")
        print(f"{col:<25} {excl_mean:>15.2f} {non_mean:>15.2f} {ratio:>8.2f}")

    return profiles


# ---------------------------------------------------------------------------
# Step 3: Approach A — Statistical anomaly detection
# ---------------------------------------------------------------------------

def approach_a_statistical(profiles):
    """Compute anomaly scores using multiple methods.

    Method 1: Z-score based (raw outlier detection)
    Method 2: Fraud-signature score (based on observed fraud patterns)
    Method 3: Logistic regression with cross-validation

    Returns profiles with 'anomaly_score_stat' column (best method).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 65)
    print("APPROACH A: STATISTICAL ANOMALY DETECTION")
    print("=" * 65)

    feature_cols = [
        "total_claims", "total_paid", "avg_paid_per_claim",
        "avg_paid_per_bene", "claims_per_month", "n_procedures",
        "top_procedure_share", "monthly_spending_cv"
    ]

    # --- Method 1: Z-score based ---
    print("\n--- Method 1: Z-Score Outlier Detection ---")
    log_cols = ["total_claims", "total_paid", "avg_paid_per_claim",
                "avg_paid_per_bene", "claims_per_month"]

    z_scores = pd.DataFrame(index=profiles.index)
    for col in feature_cols:
        vals = profiles[col].copy().fillna(0)
        if col in log_cols:
            vals = np.log1p(vals.clip(lower=0))
        z = (vals - vals.mean()) / max(vals.std(), 1e-10)
        z_scores[f"z_{col}"] = z.abs()

    profiles["anomaly_zscore"] = z_scores.mean(axis=1)

    excl_z = profiles.loc[profiles["excluded"] == 1, "anomaly_zscore"]
    non_z = profiles.loc[profiles["excluded"] == 0, "anomaly_zscore"]
    auc_z = roc_auc_score(profiles["excluded"], profiles["anomaly_zscore"])
    print(f"  Z-score AUC: {auc_z:.4f}")

    # --- Method 2: Fraud-signature score ---
    # Based on observed pattern: excluded providers have HIGH cost per bene,
    # LOW beneficiary count, HIGH procedure concentration, FEW months active
    print("\n--- Method 2: Fraud-Signature Score ---")

    sig_features = pd.DataFrame(index=profiles.index)

    # High cost per beneficiary (positive signal)
    vals = np.log1p(profiles["avg_paid_per_bene"].fillna(0).clip(lower=0))
    sig_features["high_cost_per_bene"] = (vals - vals.mean()) / max(vals.std(), 1e-10)

    # Low beneficiary count relative to claims (positive signal)
    ratio = profiles["total_claims"] / profiles["total_beneficiaries"].clip(lower=1)
    vals = np.log1p(ratio.fillna(0).clip(lower=0))
    sig_features["claims_to_bene_ratio"] = (vals - vals.mean()) / max(vals.std(), 1e-10)

    # Few months active (positive signal = fewer months)
    vals = profiles["n_months"].fillna(0)
    sig_features["few_months"] = -(vals - vals.mean()) / max(vals.std(), 1e-10)

    # High procedure concentration (positive signal)
    vals = profiles["top_procedure_share"].fillna(0)
    sig_features["proc_concentration"] = (vals - vals.mean()) / max(vals.std(), 1e-10)

    # Low total volume (positive signal = lower volume)
    vals = np.log1p(profiles["total_claims"].fillna(0).clip(lower=0))
    sig_features["low_volume"] = -(vals - vals.mean()) / max(vals.std(), 1e-10)

    profiles["anomaly_signature"] = sig_features.mean(axis=1)

    auc_sig = roc_auc_score(profiles["excluded"], profiles["anomaly_signature"])
    print(f"  Signature AUC: {auc_sig:.4f}")

    # --- Method 3: Logistic Regression (cross-validated) ---
    print("\n--- Method 3: Logistic Regression (5-fold CV) ---")

    X = profiles[feature_cols].copy()
    for col in log_cols:
        X[col] = np.log1p(X[col].fillna(0).clip(lower=0))
    X = X.fillna(0)
    y = profiles["excluded"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated predictions (out-of-fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    try:
        cv_probs = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        auc_lr = roc_auc_score(y, cv_probs)
        profiles["anomaly_lr"] = cv_probs
        print(f"  Logistic Regression AUC: {auc_lr:.4f}")

        # Feature importance
        lr.fit(X_scaled, y)
        print(f"\n  Feature coefficients:")
        for fname, coef in sorted(zip(feature_cols, lr.coef_[0]),
                                    key=lambda x: abs(x[1]), reverse=True):
            print(f"    {fname:<25} {coef:+.4f}")
    except Exception as e:
        print(f"  Logistic Regression failed: {e}")
        auc_lr = 0.5
        profiles["anomaly_lr"] = 0.5

    # --- Pick best method ---
    best_name = "zscore"
    best_auc = auc_z
    best_col = "anomaly_zscore"

    if auc_sig > best_auc:
        best_name, best_auc, best_col = "signature", auc_sig, "anomaly_signature"
    if auc_lr > best_auc:
        best_name, best_auc, best_col = "logistic_regression", auc_lr, "anomaly_lr"

    profiles["anomaly_score_stat"] = profiles[best_col]

    print(f"\n  Best method: {best_name} (AUC={best_auc:.4f})")
    print(f"  Using '{best_col}' as anomaly_score_stat")

    # Excluded vs non-excluded with best score
    excl = profiles[profiles["excluded"] == 1]["anomaly_score_stat"]
    non_excl = profiles[profiles["excluded"] == 0]["anomaly_score_stat"]
    u_stat, p_val = stats.mannwhitneyu(excl, non_excl, alternative="greater")
    print(f"\n  Mann-Whitney U: {u_stat:.0f}, p = {p_val:.6f}")

    return profiles


# ---------------------------------------------------------------------------
# Step 4: Approach B — LM confidence mapping
# ---------------------------------------------------------------------------

def serialize_provider(row):
    """Convert a provider profile row into a natural language description."""
    text = (
        f"A healthcare provider billed Medicaid for {int(row['total_claims']):,} claims "
        f"across {int(row['n_procedures'])} different procedures over {int(row['n_months'])} months, "
        f"totaling ${row['total_paid']:,.0f} in payments. "
        f"The average payment was ${row['avg_paid_per_claim']:,.2f} per claim "
        f"and ${row['avg_paid_per_bene']:,.2f} per beneficiary. "
        f"Their top procedure accounted for {100*row.get('top_procedure_share', 0):.0f}% of claims. "
        f"Monthly spending varied with a coefficient of variation of {row.get('monthly_spending_cv', 0):.2f}."
    )
    return text


def approach_b_lm_confidence(profiles, model_name="EleutherAI/pythia-6.9b",
                              max_providers=5000, dtype=None):
    """Run serialized provider profiles through the confidence engine.

    Returns profiles with 'confidence_lm' and 'entropy_lm' columns.
    """
    import torch
    from src.engine import analyze_fixed_text, unload_model

    print("\n" + "=" * 65)
    print("APPROACH B: LM CONFIDENCE MAPPING")
    print("=" * 65)
    print(f"Model: {model_name}")

    if dtype is None:
        dtype = torch.float16

    # Sample providers if too many — prioritize including all excluded
    excluded_idx = profiles[profiles["excluded"] == 1].index
    non_excluded_idx = profiles[profiles["excluded"] == 0].index

    n_excluded = len(excluded_idx)
    n_non_excluded_sample = min(max_providers - n_excluded, len(non_excluded_idx))

    if n_non_excluded_sample < len(non_excluded_idx):
        sampled_non_excluded = profiles.loc[non_excluded_idx].sample(
            n=n_non_excluded_sample, random_state=42
        ).index
        sample_idx = excluded_idx.append(sampled_non_excluded)
    else:
        sample_idx = profiles.index

    sample = profiles.loc[sample_idx].copy()
    print(f"Analyzing {len(sample):,} providers ({n_excluded} excluded + {len(sample) - n_excluded} non-excluded)")

    # Run through confidence engine
    confidences = []
    entropies = []

    for i, (idx, row) in enumerate(tqdm(sample.iterrows(), total=len(sample),
                                         desc="LM analysis")):
        text = serialize_provider(row)
        try:
            rec = analyze_fixed_text(
                text,
                category="excluded" if row["excluded"] else "normal",
                label=f"provider_{row['NPI']}",
                model_name=model_name,
                dtype=dtype,
            )
            confidences.append(rec.mean_top1_prob)
            entropies.append(rec.mean_entropy)
        except Exception as e:
            print(f"\n  Error on provider {row['NPI']}: {e}")
            confidences.append(np.nan)
            entropies.append(np.nan)

        # Periodically report
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(sample)}] "
                  f"Mean confidence so far: {np.nanmean(confidences):.4f}")

    sample["confidence_lm"] = confidences
    sample["entropy_lm"] = entropies

    # Anomaly score: lower confidence = higher anomaly
    # Invert so higher = more anomalous (matching approach A convention)
    valid = sample["confidence_lm"].notna()
    if valid.sum() > 0:
        max_conf = sample.loc[valid, "confidence_lm"].max()
        sample.loc[valid, "anomaly_score_lm"] = max_conf - sample.loc[valid, "confidence_lm"]
    else:
        sample["anomaly_score_lm"] = np.nan

    # Merge back
    profiles = profiles.merge(
        sample[["NPI", "confidence_lm", "entropy_lm", "anomaly_score_lm"]],
        on="NPI", how="left"
    )

    # Report
    excl = sample[sample["excluded"] == 1]
    non_excl = sample[sample["excluded"] == 0]
    print(f"\nLM Confidence Summary:")
    print(f"  Excluded mean confidence: {excl['confidence_lm'].mean():.4f}")
    print(f"  Non-excluded mean confidence: {non_excl['confidence_lm'].mean():.4f}")
    print(f"  Excluded mean entropy: {excl['entropy_lm'].mean():.4f}")
    print(f"  Non-excluded mean entropy: {non_excl['entropy_lm'].mean():.4f}")

    if excl["confidence_lm"].notna().sum() > 5 and non_excl["confidence_lm"].notna().sum() > 5:
        u_stat, p_val = stats.mannwhitneyu(
            excl["confidence_lm"].dropna(),
            non_excl["confidence_lm"].dropna(),
            alternative="two-sided"
        )
        print(f"  Mann-Whitney U: {u_stat:.0f}, p = {p_val:.6f}")

    unload_model()
    return profiles


# ---------------------------------------------------------------------------
# Step 5: Validation against LEIE
# ---------------------------------------------------------------------------

def validate_and_visualize(profiles):
    """Compute AUC, precision-recall for both approaches; generate plots."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

    print("\n" + "=" * 65)
    print("VALIDATION AGAINST LEIE")
    print("=" * 65)

    results = {}
    auc_a = ap_a = auc_b = ap_b = None
    y_true_b = y_score_b = None

    # --- Approach A: Statistical ---
    valid_a = profiles["anomaly_score_stat"].notna() & profiles["excluded"].notna()
    y_true = profiles.loc[valid_a, "excluded"].values
    y_score_a = profiles.loc[valid_a, "anomaly_score_stat"].values

    if y_true.sum() > 0:
        auc_a = roc_auc_score(y_true, y_score_a)
        ap_a = average_precision_score(y_true, y_score_a)
        print(f"\nApproach A (Statistical):")
        print(f"  AUC-ROC: {auc_a:.4f}")
        print(f"  Average Precision: {ap_a:.4f}")
        print(f"  Base rate: {y_true.mean():.4f}")
        results["stat_auc"] = auc_a
        results["stat_ap"] = ap_a
    else:
        print("\n  No excluded providers for Approach A validation.")

    # --- Approach B: LM Confidence ---
    has_lm = "anomaly_score_lm" in profiles.columns
    if has_lm:
        valid_b = profiles["anomaly_score_lm"].notna() & profiles["excluded"].notna()
        y_true_b = profiles.loc[valid_b, "excluded"].values
        y_score_b = profiles.loc[valid_b, "anomaly_score_lm"].values

        if y_true_b.sum() > 0:
            auc_b = roc_auc_score(y_true_b, y_score_b)
            ap_b = average_precision_score(y_true_b, y_score_b)
            print(f"\nApproach B (LM Confidence):")
            print(f"  AUC-ROC: {auc_b:.4f}")
            print(f"  Average Precision: {ap_b:.4f}")
            print(f"  Base rate: {y_true_b.mean():.4f}")
            results["lm_auc"] = auc_b
            results["lm_ap"] = ap_b
        else:
            print("\n  No excluded providers for Approach B validation.")

    # ===================================================================
    # Visualizations
    # ===================================================================
    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)

    # --- Plot 1: Anomaly score distributions ---
    print("\n[1/5] Anomaly score distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title in [
        (axes[0], "anomaly_score_stat", "Approach A: Statistical"),
        (axes[1], "anomaly_score_lm", "Approach B: LM Confidence"),
    ]:
        if col not in profiles.columns or profiles[col].isna().all():
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        valid = profiles[col].notna()
        excl = profiles.loc[valid & (profiles["excluded"] == 1), col]
        non_excl = profiles.loc[valid & (profiles["excluded"] == 0), col]

        ax.hist(non_excl, bins=50, alpha=0.6, color="#4CAF50", label="Non-excluded",
                density=True)
        if len(excl) > 0:
            ax.hist(excl, bins=min(30, len(excl)), alpha=0.7, color="#F44336",
                    label="Excluded", density=True)
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anomaly_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: ROC curves ---
    print("[2/5] ROC curves...")
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.50)")

    if auc_a is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score_a)
        ax.plot(fpr, tpr, linewidth=2, color="#2196F3",
                label=f"Statistical (AUC={auc_a:.3f})")

    if auc_b is not None and y_true_b is not None:
        fpr_b, tpr_b, _ = roc_curve(y_true_b, y_score_b)
        ax.plot(fpr_b, tpr_b, linewidth=2, color="#FF9800",
                label=f"LM Confidence (AUC={auc_b:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Fraud Detection")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Precision-Recall curves ---
    print("[3/5] Precision-recall curves...")
    fig, ax = plt.subplots(figsize=(8, 6))

    base_rate = y_true.mean() if y_true.sum() > 0 else 0
    ax.axhline(y=base_rate, color="gray", linestyle="--", alpha=0.5,
               label=f"Base rate ({base_rate:.4f})")

    if ap_a is not None:
        prec, rec, _ = precision_recall_curve(y_true, y_score_a)
        ax.plot(rec, prec, linewidth=2, color="#2196F3",
                label=f"Statistical (AP={ap_a:.3f})")

    if ap_b is not None and y_true_b is not None:
        prec_b, rec_b, _ = precision_recall_curve(y_true_b, y_score_b)
        ax.plot(rec_b, prec_b, linewidth=2, color="#FF9800",
                label=f"LM Confidence (AP={ap_b:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall: Fraud Detection")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 4: Feature importance (which z-scores separate best) ---
    print("[4/5] Feature comparison (excluded vs non-excluded)...")
    feature_cols = [
        "total_claims", "total_paid", "avg_paid_per_claim",
        "avg_paid_per_bene", "claims_per_month", "n_procedures",
        "top_procedure_share", "monthly_spending_cv"
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        excl = profiles.loc[profiles["excluded"] == 1, col].dropna()
        non_excl = profiles.loc[profiles["excluded"] == 0, col].dropna()

        # Use log scale for skewed features
        if col in ["total_claims", "total_paid", "avg_paid_per_claim",
                    "avg_paid_per_bene", "claims_per_month"]:
            excl = np.log1p(excl)
            non_excl = np.log1p(non_excl)
            ax.set_xlabel(f"log(1 + {col})")
        else:
            ax.set_xlabel(col)

        ax.hist(non_excl, bins=50, alpha=0.6, color="#4CAF50", label="Non-excl",
                density=True)
        if len(excl) > 0:
            ax.hist(excl, bins=min(30, len(excl)), alpha=0.7, color="#F44336",
                    label="Excluded", density=True)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.set_title(col, fontsize=9)

    fig.suptitle("Feature Distributions: Excluded vs Non-Excluded Providers", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 5: Top anomalies table ---
    print("[5/5] Top anomalous providers...")
    top_n = 20
    top_stat = profiles.nlargest(top_n, "anomaly_score_stat")[
        ["NPI", "total_claims", "total_paid", "anomaly_score_stat", "excluded"]
    ]
    print(f"\nTop {top_n} by statistical anomaly score:")
    print(top_stat.to_string(index=False))

    if has_lm and "anomaly_score_lm" in profiles.columns:
        valid_lm = profiles["anomaly_score_lm"].notna()
        if valid_lm.sum() > 0:
            top_lm = profiles[valid_lm].nlargest(top_n, "anomaly_score_lm")[
                ["NPI", "total_claims", "total_paid", "anomaly_score_lm", "excluded"]
            ]
            print(f"\nTop {top_n} by LM anomaly score:")
            print(top_lm.to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(skip_lm=False, lm_model="EleutherAI/pythia-6.9b",
                   max_lm_providers=3000):
    """Run the full Medicaid confidence cartography experiment."""
    start_time = time.time()

    print("=" * 65)
    print("EXPERIMENT: MEDICAID CONFIDENCE CARTOGRAPHY")
    print("=" * 65)

    # Step 1: Load data
    leie = load_leie()
    spending = load_spending()

    # Step 2: Explore
    overlap = explore_data(spending, leie)

    if len(overlap["overlap_fraud"]) < 10:
        print("\n❌ Insufficient overlap. Cannot validate. Stopping.")
        return

    # Step 3: Build profiles
    profiles = build_provider_profiles(spending, overlap["fraud_npis"])

    # Save profiles
    profiles_path = MEDICAID_DIR / "provider_profiles.csv"
    profiles.to_csv(profiles_path, index=False)
    print(f"\nProfiles saved to {profiles_path}")

    # Step 4a: Statistical anomaly detection
    profiles = approach_a_statistical(profiles)

    # Step 4b: LM confidence mapping (optional)
    if not skip_lm:
        import torch
        profiles = approach_b_lm_confidence(
            profiles,
            model_name=lm_model,
            max_providers=max_lm_providers,
            dtype=torch.float16,
        )

    # Step 5: Validate and visualize
    results = validate_and_visualize(profiles)

    # Save final profiles with scores
    scored_path = MEDICAID_DIR / "anomaly_scores.csv"
    profiles.to_csv(scored_path, index=False)
    print(f"\nScored profiles saved to {scored_path}")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print("=" * 65)
    print(f"  Providers analyzed: {len(profiles):,}")
    print(f"  Excluded in data: {profiles['excluded'].sum():,}")
    if "stat_auc" in results:
        print(f"  Statistical AUC: {results['stat_auc']:.4f}")
    if "lm_auc" in results:
        print(f"  LM Confidence AUC: {results['lm_auc']:.4f}")
    print(f"  Total time: {elapsed:.1f}s")

    return profiles, results


if __name__ == "__main__":
    # Start with statistical approach only (much faster)
    # Add --lm flag to also run LM confidence mapping
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", action="store_true",
                        help="Run LM confidence mapping (slow, needs GPU/MPS)")
    parser.add_argument("--model", default="EleutherAI/pythia-6.9b",
                        help="Model for LM confidence mapping")
    parser.add_argument("--max-providers", type=int, default=3000,
                        help="Max providers for LM analysis")
    args = parser.parse_args()

    run_experiment(
        skip_lm=not args.lm,
        lm_model=args.model,
        max_lm_providers=args.max_providers,
    )
