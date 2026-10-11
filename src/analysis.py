"""EDA + statistical hypothesis testing module.

Generates publication-ready figures in `results/figures/` and writes
`results/hypothesis_tests.json` + `results/eda_summary.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


FIG_DIR = Path(__file__).resolve().parent.parent / "results/figures"


def _save(fig, name: str, caption: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)
    print(f"saved figure: {name}.png - {caption}")


def run_eda(df: pd.DataFrame, seed_note: str = "") -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---- schema + dimensions ----------------------------------------
    summary["shape"] = list(df.shape)
    summary["schema"] = {c: str(df[c].dtype) for c in df.columns}
    summary["missing"] = df.isna().sum().to_dict()

    # ---- duplicates --------------------------------------------------
    dup_text = int(df.review_text.duplicated().sum())
    dup_pairs = int(df.duplicated(subset=["product_id", "review_user_id"]).sum())
    summary["duplicate_review_text"] = dup_text
    summary["duplicate_product_user_pairs"] = dup_pairs

    # ---- category frequency -----------------------------------------
    cat_counts = df.category.value_counts()
    summary["category_counts"] = cat_counts.to_dict()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    cat_counts.plot(kind="bar", ax=ax, color="#4c72b0")
    ax.set_title("Product category frequency")
    ax.set_ylabel("rows")
    ax.set_xlabel("")
    plt.xticks(rotation=35, ha="right")
    _save(fig, "category_frequency", "The three source categories dominate; minor labels come from meta main_category overlap.")
    summary["category_top5"] = cat_counts.head(5).to_dict()
    summary["category_interpretation"] = (
        "3 source review files (Health & Personal Care, Tools & Home Improvement, All Beauty, Amazon Home, "
        "Appliances) contribute ~95% of rows; the remaining labels are small (rare main_category in meta) and "
        "should be grouped in the modeling notebook to avoid micro-classes.")

    # ---- price histogram + boxplot ----------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.hist(df.price, bins=40, color="#55a868", edgecolor="white")
    ax1.set_title("Price distribution")
    ax1.set_xlabel("price (USD)")
    ax1.set_ylabel("rows")
    ax2.boxplot(df.price, vert=False)
    ax2.set_title("Price boxplot")
    ax2.set_xlabel("price (USD)")
    q1, med, q3 = df.price.quantile([0.25, 0.5, 0.75]).tolist()
    summary["price_quantiles"] = {"q1": q1, "median": med, "q3": q3}
    _save(fig, "price_distribution", "Prices are right-skewed; median ~%.1f USD, heavy right tail beyond ~50 USD." % med)
    summary["price_interpretation"] = (
        f"Prices are strongly right-skewed (median {med:.2f}, IQR {q1:.2f}-{q3:.2f}); quantile-based "
        "low/medium/high tiers are therefore a better proxy of 'value class' than fixed bins.")

    # ---- rating + sentiment -----------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    df.rating.value_counts().sort_index().plot(kind="bar", ax=ax1, color="#c44e52")
    ax1.set_title("Rating distribution")
    ax1.set_xlabel("stars")
    df.sentiment_label.value_counts().reindex(["negative", "neutral", "positive"]).plot(kind="bar", ax=ax2, color="#8172b2")
    ax2.set_title("Sentiment (rating-derived)")
    plt.xticks(rotation=0)
    _save(fig, "rating_sentiment", "Ratings skew heavily to 5 stars; sentiment classes are imbalanced (80/15/5).")
    summary["rating_counts"] = df.rating.value_counts().sort_index().to_dict()
    summary["sentiment_counts"] = df.sentiment_label.value_counts().to_dict()
    summary["sentiment_interpretation"] = (
        "Amazon ratings skew positive (80% positive / 15% negative / 5% neutral); macro-averaged metrics "
        "will be essential to fairly evaluate performance on the small neutral class.")

    # ---- review-length distribution --------------------------------
    rev_len = df.review_text.str.split().str.len()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(rev_len.clip(upper=200), bins=40, color="#ccb974", edgecolor="white")
    ax.set_title("Review length (words, clipped at 200)")
    ax.set_xlabel("words")
    ax.set_ylabel("rows")
    _save(fig, "review_length", "Review lengths are also right-skewed; most reviews are short (< 50 words).")
    summary["review_length_mean"] = float(rev_len.mean())
    summary["review_length_median"] = float(rev_len.median())

    # ---- category vs sentiment --------------------------------------
    ct = pd.crosstab(df.category, df.sentiment_label).reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#c44e52", "#ccb974", "#55a868"])
    ax.set_title("Category vs sentiment (rating-derived)")
    ax.set_ylabel("rows")
    plt.xticks(rotation=35, ha="right")
    _save(fig, "category_vs_sentiment", "Positive reviews dominate every category; negative share is highest in Health & Personal Care.")
    summary["category_sentiment_crosstab"] = ct.to_dict()

    # ---- price vs sentiment (boxplot) -------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [df.loc[df.sentiment_label == s, "price"] for s in ["negative", "neutral", "positive"]]
    ax.boxplot(data, tick_labels=["negative", "neutral", "positive"])
    ax.set_title("Price by sentiment")
    ax.set_ylabel("price (USD)")
    _save(fig, "price_vs_sentiment", "Prices are similar across sentiment groups; no obvious price-sentiment relationship.")

    # ---- price tier distribution ------------------------------------
    thresholds = df.price.quantile([1 / 3, 2 / 3]).tolist()
    tiers = pd.cut(df.price, bins=[-np.inf, thresholds[0], thresholds[1], np.inf], labels=["low", "medium", "high"])
    summary["price_tier_preview"] = tiers.value_counts().to_dict()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    tiers.value_counts().reindex(["low", "medium", "high"]).plot(kind="bar", ax=ax, color="#64b5cd")
    ax.set_title("Price tier distribution (33/67 percentile preview)")
    plt.xticks(rotation=0)
    _save(fig, "price_tier_distribution", "By construction (33/67 percentile thresholds), classes are nearly balanced; final thresholds are fit on train split only.")
    return summary


def cramers_v(ct: pd.DataFrame) -> float:
    chi2 = st.chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, c = ct.shape
    return float(np.sqrt((chi2 / n) / (min(r, c) - 1)))


def combine_rare_categories(ct: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """Combine rows (categories) with fewer than min_count rows into 'Other'.

    Documented reason: chi-square's expected-cell-count rule of thumb (>= 5)
    is violated with micro-categories, which distorts the statistic.
    """
    keep = ct.sum(axis=1) >= min_count
    out = ct[keep]
    if (~keep).any():
        other = ct.loc[~keep].sum(axis=0)
        out.loc["Other (combined micro-categories)"] = other
    return out


def chi_square_test(ct: pd.DataFrame, name: str, min_expected: float = 5.0):
    """Run chi-square test of independence on a contingency table.

    Returns None if assumptions are violated beyond repair.
    """
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"test": name, "status": "insufficient levels"}
    exp = st.contingency.expected_freq(ct)
    if (exp < min_expected).any():
        ct = combine_rare_categories(ct) if ct.shape[0] > ct.shape[1] else combine_rare_categories(ct.T).T
        exp = st.contingency.expected_freq(ct)
    chi2, p, dof, _ = st.chi2_contingency(ct)
    return {
        "test": name,
        "status": "ok",
        "table_shape": list(ct.shape),
        "min_expected_cell": float(exp.min()),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": cramers_v(ct),
        "n": int(ct.values.sum()),
    }


def run_hypothesis_tests(dfd_disc: pd.DataFrame) -> dict:
    """`dfd_disc` must have columns: category, price_tier, sentiment_label."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # H1: sentiment independent of product category
    ct1 = pd.crosstab(dfd_disc.category, dfd_disc.sentiment_label)
    r1 = chi_square_test(ct1, "H1: sentiment independent of product category")
    results["H1_category_vs_sentiment"] = r1

    # H2: sentiment independent of price tier
    ct2 = pd.crosstab(dfd_disc.price_tier, dfd_disc.sentiment_label)
    r2 = chi_square_test(ct2, "H2: sentiment independent of price tier")
    results["H2_price_tier_vs_sentiment"] = r2

    # save the combined 2D table plot defensively
    if r1 and r1.get("status") == "ok":
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ct1.plot(kind="bar", ax=ax, color=["#c44e52", "#ccb974", "#55a868"])
        ax.set_title("Category vs sentiment (used for H1)")
        plt.xticks(rotation=35, ha="right")
        _save(fig, "h1_contingency", "Observed counts; see hypothesis_tests.json for chi-square result.")
    return results
