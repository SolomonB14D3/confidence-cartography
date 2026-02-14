"""
Experiment B15: Text Properties Predict Regime Membership
==========================================================
Can surface-level text properties predict whether a claim is regime 1
(scaling-reducible, model knows it's uncertain) or regime 2
(scaling-irreducible, model confidently wrong)?

B14 showed regime 2 errors are learned smoothly — no oscillation, no
conflict. If the model faithfully absorbs what's dominant in the data,
then text properties should predict regime membership without running
any model at all.

Design: Extract 27 features across 5 categories (lexical, specificity,
syntactic, domain, transmissibility) from 99 regime-labeled items.
Test via LOO logistic regression and compare to model-based detection.
"""

import sys
import json
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
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import spacy
import nltk
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

PROJECT_ROOT = Path("/Volumes/4TB SD/ClaudeCode/confidence-cartography")
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "text_properties"
FIGURES_DIR = PROJECT_ROOT / "figures" / "text_properties"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ENGLISH_STOPWORDS = set(stopwords.words("english"))


# ===================================================================
# STEP 1: Data Loading
# ===================================================================

def load_regime_data():
    """Load regime labels and match to full text.

    Returns DataFrame with columns: pair_id, text, regime, source

    The regime_comparison.json has 65 R1 and 34 R2 items. R1 items have
    pair_ids like "france_capital" (source=truth) or "heart_chambers"
    (source=medical). R2 items have pair_ids like "snow_white"
    (source=mandela_orig), "star_wars_context" (source=mandela_exp_context),
    or "silence_of_lambs_raw" (source=mandela_exp_raw).

    The expanded Mandela items have _raw and _context variants:
    - _raw: just the wrong quote (e.g. "Hello, Clarice")
    - _context: wrapped in framing (e.g. 'In The Silence of the Lambs...')
    """
    # Load regime comparison
    regime_path = PROJECT_ROOT / "data" / "results" / "token_localization" / "regime_comparison.json"
    with open(regime_path) as f:
        regime_data = json.load(f)

    r1_items = regime_data["regime1_6.9b"]  # 65 items
    r2_items = regime_data["regime2_6.9b"]  # 34 items

    # Load trajectory CSV for full text (has raw Mandela items)
    csv_path = PROJECT_ROOT / "data" / "results" / "trajectory_detector" / "all_items_with_trajectories.csv"
    traj_df = pd.read_csv(csv_path)

    # Build lookup: normalized pair_id -> false/popular text
    text_lookup = {}
    for _, row in traj_df[traj_df["is_true"] == 0].iterrows():
        pair_id = row["pair_id"]
        # Strip prefix: truth_france_capital -> france_capital
        for prefix in ["truth_", "medical_", "mandela_"]:
            if pair_id.startswith(prefix):
                norm_id = pair_id[len(prefix):]
                text_lookup[norm_id] = row["text"]
                # Also store with _raw suffix for expanded items
                text_lookup[norm_id + "_raw"] = row["text"]
                break
        else:
            text_lookup[pair_id] = row["text"]

    # Build lookup from source files for originals and context framings
    from src.experiments.exp2_truth import PAIRS as TRUTH_PAIRS
    from src.experiments.exp9_medical_validation import MEDICAL_PAIRS
    from src.experiments.exp_c_mandela import MANDELA_PAIRS
    from src.experiments.exp_mandela_expanded import LINGUISTIC_ITEMS

    for p in TRUTH_PAIRS:
        text_lookup.setdefault(p["id"], p["false"])
    for p in MEDICAL_PAIRS:
        text_lookup.setdefault(p["id"], p["false"])
    for p in MANDELA_PAIRS:
        text_lookup.setdefault(p["id"], p["popular"])

    # Expanded Mandela items: generate both raw and context versions
    for item in LINGUISTIC_ITEMS:
        base_id = item["id"]
        wrong_text = item["wrong"]
        context_template = item.get("context", "{quote}")

        # Raw version: just the wrong quote
        text_lookup.setdefault(base_id + "_raw", wrong_text)
        text_lookup.setdefault(base_id, wrong_text)

        # Context version: wrap in context template
        context_text = context_template.replace("{quote}", wrong_text)
        text_lookup[base_id + "_context"] = context_text

    # Match regime items to text
    rows = []
    missing = []

    for item in r1_items:
        pid = item["pair_id"]
        text = text_lookup.get(pid)
        if text:
            rows.append({"pair_id": pid, "text": text, "regime": 0, "source": item["source"]})
        else:
            missing.append(pid)

    for item in r2_items:
        pid = item["pair_id"]
        text = text_lookup.get(pid)
        if text:
            rows.append({"pair_id": pid, "text": text, "regime": 1, "source": item["source"]})
        else:
            missing.append(pid)

    if missing:
        print(f"  WARNING: {len(missing)} items could not be matched to text: {missing}")

    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df)} items: {(df['regime']==0).sum()} regime 1, {(df['regime']==1).sum()} regime 2")
    return df


# ===================================================================
# STEP 2: Feature Extraction
# ===================================================================

def token_depth(token):
    """Compute depth of token in dependency tree."""
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
        if depth > 100:  # safety
            break
    return depth


def extract_features(texts, nlp):
    """Extract 27 features from a list of texts.

    Returns DataFrame with one row per text.
    """
    all_features = []

    for text in texts:
        doc = nlp(text)
        tokens_nltk = word_tokenize(text.lower())
        text_lower = text.lower()

        feats = {}

        # --- Category 1: Lexical (5 features) ---
        feats["n_tokens"] = len(tokens_nltk)
        feats["n_chars"] = len(text)
        feats["avg_word_length"] = np.mean([len(t) for t in tokens_nltk]) if tokens_nltk else 0
        feats["type_token_ratio"] = len(set(tokens_nltk)) / len(tokens_nltk) if tokens_nltk else 0
        feats["pct_stopwords"] = (
            sum(1 for t in tokens_nltk if t in ENGLISH_STOPWORDS) / len(tokens_nltk)
            if tokens_nltk else 0
        )

        # --- Category 2: Named Entity / Specificity (6 features) ---
        feats["n_proper_nouns"] = sum(1 for t in doc if t.pos_ == "PROPN")
        feats["n_numbers"] = sum(1 for t in doc if t.pos_ == "NUM")
        feats["n_entities"] = len(doc.ents)
        feats["has_quote"] = int('"' in text or "'" in text)
        feats["has_date"] = int(any(e.label_ == "DATE" for e in doc.ents))
        feats["has_person"] = int(any(e.label_ == "PERSON" for e in doc.ents))

        # --- Category 3: Syntactic Complexity (4 features) ---
        sents = list(doc.sents)
        feats["n_sentences"] = len(sents)
        feats["avg_sent_length"] = np.mean([len(s) for s in sents]) if sents else 0
        feats["max_depth"] = max((token_depth(t) for t in doc), default=0)
        feats["n_clauses"] = sum(1 for t in doc if t.dep_ in ("ccomp", "advcl", "relcl", "xcomp"))

        # --- Category 4: Content Domain Markers (5 features) ---
        feats["is_cultural"] = int(any(w in text_lower for w in [
            "movie", "film", "song", "book", "star wars", "disney", "vader",
            "character", "show", "play", "mirror", "chocolates", "champion",
        ]))
        feats["is_medical"] = int(any(w in text_lower for w in [
            "disease", "vitamin", "brain", "blood", "symptom", "treatment",
            "heart", "organ", "insulin", "bone", "cell", "immune", "cancer",
            "infection", "anemia", "chromosome", "dna", "lung", "liver",
            "diabetes", "malaria", "vaccine", "penicillin",
        ]))
        feats["is_scientific"] = int(any(w in text_lower for w in [
            "earth", "planet", "species", "evolution", "atom", "gravity",
            "light", "sound", "oxygen", "carbon", "chemical", "boil",
            "celsius", "orbit", "photosynthesis",
        ]))
        feats["is_historical"] = int(any(w in text_lower for w in [
            "war", "president", "century", "founded", "invented",
            "independence", "empire", "berlin wall", "moon landing",
        ]))
        feats["is_geographic"] = int(any(w in text_lower for w in [
            "capital", "country", "city", "continent", "ocean",
            "river", "mountain", "pacific", "atlantic",
        ]))

        # --- Category 5: Transmissibility Proxies (7 features) ---
        feats["has_superlative"] = int(any(w in text_lower for w in [
            "most", "largest", "first", "best", "worst", "greatest",
            "longest", "tallest", "biggest",
        ]))
        feats["has_negation"] = int(any(w in text_lower for w in [
            "not", "never", "no ", "don't", "isn't", "wasn't",
            "doesn't", "does not", "has no", "has never",
        ]))
        feats["is_surprising"] = int(any(w in text_lower for w in [
            "actually", "surprisingly", "contrary", "despite", "although",
        ]))
        feats["is_declarative"] = int(not text.strip().endswith("?"))
        feats["brevity"] = 1.0 / (1 + len(text))
        feats["references_fiction"] = int(any(w in text_lower for w in [
            "character", "movie", "show", "book", "song", "play",
            "vader", "clarice", "watson", "dorothy", "forrest",
            "captain kirk", "scotty", "sam", "toto",
        ]))
        feats["references_brand"] = int(any(w in text_lower for w in [
            "coca", "nike", "disney", "apple", "google",
            "monopoly", "fruit of the loom", "oscar",
            "berenstain", "berenstein",
        ]))

        all_features.append(feats)

    return pd.DataFrame(all_features)


# ===================================================================
# STEP 3: Univariate Tests
# ===================================================================

CONTINUOUS_FEATURES = [
    "n_tokens", "n_chars", "avg_word_length", "type_token_ratio",
    "pct_stopwords", "n_proper_nouns", "n_numbers", "n_entities",
    "n_sentences", "avg_sent_length", "max_depth", "n_clauses",
    "brevity",
]

BINARY_FEATURES = [
    "has_quote", "has_date", "has_person",
    "is_cultural", "is_medical", "is_scientific", "is_historical", "is_geographic",
    "has_superlative", "has_negation", "is_surprising", "is_declarative",
    "references_fiction", "references_brand",
]


def run_univariate_tests(feature_df, regime_labels):
    """Run univariate statistical tests for each feature."""
    results = []
    n_tests = len(CONTINUOUS_FEATURES) + len(BINARY_FEATURES)

    r1_mask = regime_labels == 0
    r2_mask = regime_labels == 1

    # Continuous features: Mann-Whitney U
    for feat in CONTINUOUS_FEATURES:
        r1_vals = feature_df.loc[r1_mask, feat].values
        r2_vals = feature_df.loc[r2_mask, feat].values

        stat, p = mannwhitneyu(r1_vals, r2_vals, alternative="two-sided")
        n1, n2 = len(r1_vals), len(r2_vals)
        # Rank-biserial correlation
        r_rb = 1 - (2 * stat) / (n1 * n2)

        results.append({
            "feature": feat,
            "type": "continuous",
            "r1_median": np.median(r1_vals),
            "r2_median": np.median(r2_vals),
            "test": "Mann-Whitney U",
            "statistic": stat,
            "p_value": p,
            "p_corrected": min(p * n_tests, 1.0),
            "effect_size": r_rb,
            "effect_type": "rank-biserial r",
        })

    # Binary features: Fisher's exact test
    for feat in BINARY_FEATURES:
        r1_vals = feature_df.loc[r1_mask, feat].values
        r2_vals = feature_df.loc[r2_mask, feat].values

        # 2x2 contingency table
        a = (r1_vals == 1).sum()  # R1 and feature=1
        b = (r1_vals == 0).sum()  # R1 and feature=0
        c = (r2_vals == 1).sum()  # R2 and feature=1
        d = (r2_vals == 0).sum()  # R2 and feature=0

        table = np.array([[a, b], [c, d]])
        odds_ratio, p = fisher_exact(table)

        # Cramér's V for effect size
        n = table.sum()
        chi2_stat = n * ((a*d - b*c)**2) / ((a+b)*(c+d)*(a+c)*(b+d)) if min(a+b, c+d, a+c, b+d) > 0 else 0
        cramers_v = np.sqrt(chi2_stat / n) if n > 0 else 0

        results.append({
            "feature": feat,
            "type": "binary",
            "r1_median": a / (a + b) if (a + b) > 0 else 0,
            "r2_median": c / (c + d) if (c + d) > 0 else 0,
            "test": "Fisher's exact",
            "statistic": odds_ratio,
            "p_value": p,
            "p_corrected": min(p * n_tests, 1.0),
            "effect_size": cramers_v,
            "effect_type": "Cramér's V",
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_value")
    return results_df


# ===================================================================
# STEP 4: LOO Logistic Regression
# ===================================================================

def run_classification(feature_df, regime_labels, feature_subset=None, name="all"):
    """Run LOO cross-validated logistic regression.

    Returns dict with AUC, predictions, name.
    """
    if feature_subset is not None:
        X = feature_df[feature_subset].values
    else:
        X = feature_df.values
    y = regime_labels.values

    loo = LeaveOneOut()
    preds = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Scale within fold
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(
            solver="liblinear", C=0.1, l1_ratio=1.0,
            max_iter=1000, random_state=42,
        )
        model.fit(X_train_s, y_train)
        preds[test_idx] = model.predict_proba(X_test_s)[0, 1]

    auc = roc_auc_score(y, preds)

    # Bootstrap CI for AUC
    n_boot = 2000
    boot_aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y[idx], preds[idx]))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

    return {
        "name": name,
        "auc": auc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "y_pred": preds.tolist(),
        "y_true": y.tolist(),
    }


def run_feature_importance(feature_df, regime_labels):
    """Fit on full data for feature importance interpretation."""
    X = feature_df.values
    y = regime_labels.values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = LogisticRegression(
        solver="liblinear", C=0.1, l1_ratio=1.0,
        max_iter=1000, random_state=42,
    )
    model.fit(X_s, y)

    coefs = dict(zip(feature_df.columns, model.coef_[0]))
    return coefs


# ===================================================================
# STEP 5: Visualizations
# ===================================================================

def plot_feature_distributions(feature_df, regime_labels, test_results):
    """Violin/box plots for top 6 most significant features."""
    top6 = test_results.head(6)["feature"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, feat in enumerate(top6):
        ax = axes[i]
        data = pd.DataFrame({
            "value": feature_df[feat],
            "Regime": ["R1 (reducible)" if r == 0 else "R2 (irreducible)"
                       for r in regime_labels],
        })
        is_binary = feat in BINARY_FEATURES

        if is_binary:
            # Grouped bar chart for binary features
            r1_pct = data[data["Regime"].str.contains("R1")]["value"].mean()
            r2_pct = data[data["Regime"].str.contains("R2")]["value"].mean()
            bars = ax.bar(["R1", "R2"], [r1_pct, r2_pct],
                          color=["#4C72B0", "#DD8452"], edgecolor="white")
            ax.set_ylabel("Proportion")
            for bar, pct in zip(bars, [r1_pct, r2_pct]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{pct:.2f}", ha="center", fontsize=9)
        else:
            sns.violinplot(data=data, x="Regime", y="value", ax=ax,
                           palette=["#4C72B0", "#DD8452"], inner="box",
                           cut=0, linewidth=0.8)
            ax.set_ylabel(feat)

        p_val = test_results[test_results["feature"] == feat]["p_value"].values[0]
        p_corr = test_results[test_results["feature"] == feat]["p_corrected"].values[0]
        stars = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
        ax.set_title(f"{feat}\np={p_val:.4f} ({stars})", fontsize=10)
        ax.set_xlabel("")

    plt.suptitle("B15: Top 6 Text Features by Regime", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'feature_distributions.png'}")


def plot_roc_curve(results_dict):
    """ROC curves for different feature subsets."""
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {"all": "#C44E52", "domain_only": "#4C72B0",
              "transmissibility_only": "#55A868", "lexical_only": "#8172B2"}

    for name, result in results_dict.items():
        y_true = np.array(result["y_true"])
        y_pred = np.array(result["y_pred"])
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = result["auc"]
        ci_lo, ci_hi = result["ci_lo"], result["ci_hi"]
        color = colors.get(name, "#333333")
        label = f"{name} (AUC={auc:.3f} [{ci_lo:.3f}-{ci_hi:.3f}])"
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("B15: Text Properties Predict Regime Membership", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'roc_curve.png'}")


def plot_feature_importance(coefs):
    """Horizontal bar chart of L1 coefficients."""
    # Filter to non-zero coefficients
    nonzero = {k: v for k, v in coefs.items() if abs(v) > 0.001}
    if not nonzero:
        print("  WARNING: All coefficients are zero (too much regularization)")
        nonzero = dict(sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

    sorted_feats = sorted(nonzero.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [f[0] for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]
    colors = ["#DD8452" if v > 0 else "#4C72B0" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("L1 Coefficient (+ = predicts R2)", fontsize=11)
    ax.set_title("B15: Feature Importance for Regime Prediction", fontsize=13)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)

    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'feature_importance.png'}")


def plot_comparison(text_auc, text_ci, baseline_aucs):
    """Compare text-only AUC to model-based detection."""
    methods = []
    aucs = []
    colors_list = []

    # Model-based baselines
    methods.append("6.9B single\n(model-based)")
    aucs.append(baseline_aucs.get("best_single", 0.594))
    colors_list.append("#4C72B0")

    methods.append("410m+6.9B pair\n(model-based)")
    aucs.append(baseline_aucs.get("best_pair", 0.640))
    colors_list.append("#4C72B0")

    methods.append("Slope feature\n(model-based)")
    aucs.append(baseline_aucs.get("slope", 0.654))
    colors_list.append("#4C72B0")

    # Text-only
    methods.append("Text properties\n(no model)")
    aucs.append(text_auc)
    colors_list.append("#C44E52")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(methods)), aucs, color=colors_list,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("LOO AUC", fontsize=12)
    ax.set_title("B15: Regime Prediction — Text Properties vs Model-Based", fontsize=13)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylim(0.35, max(aucs) + 0.08)

    for bar, auc_val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{auc_val:.3f}", ha="center", fontsize=11, fontweight="bold")

    # Add CI for text-only
    bar_idx = len(methods) - 1
    ax.errorbar(bar_idx, text_auc,
                yerr=[[text_auc - text_ci[0]], [text_ci[1] - text_auc]],
                color="black", capsize=5, linewidth=2, capthick=2)

    # Note about comparability
    ax.text(0.5, 0.02, "Note: Model-based AUCs are on 164-item true/false task;\n"
            "text AUC is on 99-item regime classification task",
            transform=ax.transAxes, fontsize=8, ha="center", alpha=0.6)

    fig.savefig(FIGURES_DIR / "regime_prediction_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'regime_prediction_comparison.png'}")


# ===================================================================
# MAIN
# ===================================================================

def run_experiment():
    start_time = time.time()

    print("=" * 70)
    print("B15: TEXT PROPERTIES PREDICT REGIME MEMBERSHIP")
    print("=" * 70)

    # --- Load data ---
    print("\n1. Loading regime-labeled data...")
    df = load_regime_data()

    # --- Load spacy ---
    print("\n2. Extracting features...")
    nlp = spacy.load("en_core_web_sm")
    feature_df = extract_features(df["text"].tolist(), nlp)

    # Save features
    out_df = pd.concat([
        df[["pair_id", "regime", "source"]].reset_index(drop=True),
        feature_df.reset_index(drop=True),
    ], axis=1)
    out_df.to_csv(RESULTS_DIR / "features.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'features.csv'} ({len(out_df)} rows, {len(feature_df.columns)} features)")

    # --- Univariate tests ---
    print("\n3. Running univariate tests...")
    test_results = run_univariate_tests(feature_df, df["regime"])
    test_results.to_csv(RESULTS_DIR / "univariate_tests.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'univariate_tests.csv'}")

    print("\n  Top 10 features by p-value:")
    print(f"  {'Feature':<25s} {'p-value':>10s} {'p-corr':>10s} {'Effect':>10s} {'R1 med':>8s} {'R2 med':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for _, row in test_results.head(10).iterrows():
        stars = "***" if row["p_corrected"] < 0.001 else "**" if row["p_corrected"] < 0.01 else "*" if row["p_corrected"] < 0.05 else ""
        print(f"  {row['feature']:<25s} {row['p_value']:>10.4f} {row['p_corrected']:>10.4f} "
              f"{row['effect_size']:>+10.3f} {row['r1_median']:>8.3f} {row['r2_median']:>8.3f} {stars}")

    # --- Classification ---
    print("\n4. Running LOO logistic regression...")

    # Define feature subsets
    domain_features = ["is_cultural", "is_medical", "is_scientific", "is_historical", "is_geographic"]
    transmissibility_features = [
        "has_superlative", "has_negation", "is_surprising",
        "is_declarative", "brevity", "references_fiction", "references_brand",
    ]
    lexical_features = ["n_tokens", "n_chars", "avg_word_length", "type_token_ratio", "pct_stopwords"]

    results_dict = {}
    for name, subset in [
        ("all", None),
        ("domain_only", domain_features),
        ("transmissibility_only", transmissibility_features),
        ("lexical_only", lexical_features),
    ]:
        result = run_classification(feature_df, df["regime"], feature_subset=subset, name=name)
        results_dict[name] = result
        print(f"  {name:<25s}: AUC = {result['auc']:.3f} [{result['ci_lo']:.3f}-{result['ci_hi']:.3f}]")

    # --- Feature importance ---
    print("\n5. Feature importance (full-data fit)...")
    coefs = run_feature_importance(feature_df, df["regime"])
    sorted_coefs = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"  {'Feature':<25s} {'Coefficient':>12s}")
    print(f"  {'-'*25} {'-'*12}")
    for name, coef in sorted_coefs:
        if abs(coef) > 0.001:
            direction = "-> R2" if coef > 0 else "-> R1"
            print(f"  {name:<25s} {coef:>+12.4f}  {direction}")

    # --- Comparison to model-based ---
    print("\n6. Comparison to model-based detection...")
    traj_path = PROJECT_ROOT / "data" / "results" / "trajectory_detector" / "trajectory_detector_results.json"
    with open(traj_path) as f:
        traj_results = json.load(f)

    baseline_aucs = {
        "best_single": traj_results["t1_classification"]["baselines"]["6.9b"]["auc"],
        "best_pair": traj_results["t4_minimal_pairs"]["best_pair"]["auc"],
        "slope": traj_results["t3_feature_importance"]["top_single_aucs"]["slope"],
    }

    comparison = {
        "text_properties_auc": results_dict["all"]["auc"],
        "text_properties_ci": [results_dict["all"]["ci_lo"], results_dict["all"]["ci_hi"]],
        "model_based_6.9b_auc": baseline_aucs["best_single"],
        "model_based_pair_auc": baseline_aucs["best_pair"],
        "model_based_slope_auc": baseline_aucs["slope"],
        "note": "Model-based AUCs are on 164-item true/false classification; "
                "text AUC is on 99-item regime classification (different tasks)",
    }
    with open(RESULTS_DIR / "comparison_to_model.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"  Model-based (6.9B single):     AUC = {baseline_aucs['best_single']:.3f}")
    print(f"  Model-based (410m+6.9B pair):   AUC = {baseline_aucs['best_pair']:.3f}")
    print(f"  Model-based (slope feature):    AUC = {baseline_aucs['slope']:.3f}")
    print(f"  Text properties (all, no model): AUC = {results_dict['all']['auc']:.3f}")

    # --- Save regression results ---
    regression_results = {
        "n_items": len(df),
        "n_regime1": int((df["regime"] == 0).sum()),
        "n_regime2": int((df["regime"] == 1).sum()),
        "n_features": len(feature_df.columns),
        "feature_names": list(feature_df.columns),
        "results_by_subset": {
            name: {
                "auc": r["auc"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
            }
            for name, r in results_dict.items()
        },
        "feature_importance": {k: float(v) for k, v in sorted_coefs if abs(v) > 0.0001},
        "comparison": comparison,
    }
    with open(RESULTS_DIR / "regression_results.json", "w") as f:
        json.dump(regression_results, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'regression_results.json'}")

    # --- Plots ---
    print("\n7. Generating figures...")
    plot_feature_distributions(feature_df, df["regime"], test_results)
    plot_roc_curve(results_dict)
    plot_feature_importance(coefs)
    plot_comparison(
        results_dict["all"]["auc"],
        (results_dict["all"]["ci_lo"], results_dict["all"]["ci_hi"]),
        baseline_aucs,
    )

    # --- Interpretation ---
    all_auc = results_dict["all"]["auc"]
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if all_auc > 0.70:
        print(f"\n  AUC = {all_auc:.3f} > 0.70: STRONG PREDICTION")
        print("  Text properties predict regime well.")
        print("  Framework closes — regime 2 is a corpus/text property, not a model property.")
    elif all_auc > 0.55:
        print(f"\n  AUC = {all_auc:.3f} (0.55-0.70): PARTIAL PREDICTION")
        print("  Text contributes but model captures something beyond surface features.")
    else:
        print(f"\n  AUC = {all_auc:.3f} <= 0.55: WEAK/NO PREDICTION")
        print("  Text properties don't predict regime.")
        print("  The model is doing real work that can't be shortcut from text alone.")

    # Which category dominates?
    category_aucs = {
        "domain": results_dict["domain_only"]["auc"],
        "transmissibility": results_dict["transmissibility_only"]["auc"],
        "lexical": results_dict["lexical_only"]["auc"],
    }
    best_cat = max(category_aucs, key=category_aucs.get)
    print(f"\n  Category AUCs: {', '.join(f'{k}={v:.3f}' for k, v in category_aucs.items())}")
    print(f"  Dominant category: {best_cat}")

    if best_cat == "domain":
        print("  -> Regime is mostly about topic (cultural refs -> R2, medical -> R1)")
        print("     Less interesting — just domain effects.")
    elif best_cat == "transmissibility":
        print("  -> Transmissibility proxies dominate!")
        print("     Properties that make text memorable/shareable predict regime.")
        print("     Direct evidence for the transmissibility framework.")
    elif best_cat == "lexical":
        print("  -> Lexical features dominate (length, diversity)")
        print("     Regime differences may be driven by text structure more than content.")

    elapsed = time.time() - start_time
    print(f"\nEXPERIMENT COMPLETE ({elapsed:.1f}s)")
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    run_experiment()
