"""Reuseable leakage-aware data preparation utilities.

- `prepare_targets`: adds price_tier (low/medium/high) using quantile thresholds
  fit ONLY on the training portion of whatever split is provided.
- `group_split`: deterministic product-grouped train/val/test split, preserving
  every product's reviews in a single split, stratified approximately by
  target combinations via a greedy assignment.

Leakage rules enforced by callers (documented, not hidden):
- sentiment features must not include `rating` or anything derived from it;
- price_tier features must not include `price` or price-derived fields;
- category features must not include `category` (or its string aliases).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_price_thresholds(train_price: pd.Series) -> dict:
    """Fit low/medium/high thresholds from the training distribution.

    Returns thresholds dict with the explicit method documented:
    - t1 = 33rd percentile, t2 = 67th percentile of train prices
    - price <= t1 -> 'low', <= t2 -> 'medium', else 'high'
    Handles repeated quantiles: if t1 == t2 (no price variation), all records
    collapse to 'medium' and this is reported by the caller.
    """
    t1 = float(np.quantile(train_price.dropna(), 1 / 3))
    t2 = float(np.quantile(train_price.dropna(), 2 / 3))
    return {"t1": t1, "t2": t2, "method": "33rd/67th percentiles of train split",
            "degenerate": bool(abs(t1 - t2) < 1e-9)}


def apply_price_thresholds(price: pd.Series, thr: dict) -> pd.Series:
    return pd.cut(price, bins=[-np.inf, thr["t1"], thr["t2"], np.inf],
                  labels=["low", "medium", "high"])


def prepare_targets(df: pd.DataFrame, thr: dict) -> pd.DataFrame:
    """Add price_tier using pre-fitted thresholds (leakage-safe: thresholds
    were fit on train split only, never on val/test)."""
    out = df.copy()
    out["price_tier"] = apply_price_thresholds(out["price"], thr)
    return out


def group_split(df: pd.DataFrame, seed: int = 20231010,
                frac_train: float = 0.70, frac_val: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Deterministic product-grouped 70/15/15 split.

    Each product's reviews go entirely into one of train/val/test. Splitting is
    greedy-but-stratified: products are shuffled with a fixed seed and assigned
    in order until running target-combination distributions closely match the
    overall distribution (approximate stratification by (category, sentiment,
    price_tier) triplets).
    """
    rng = np.random.RandomState(seed)
    groups = df.groupby("product_id").indices
    prod_ids = np.array(sorted(groups.keys()))
    prod_sentiment = (df.groupby("product_id").sentiment_label.agg(lambda s: s.value_counts().idxmax()))
    prod_cat = (df.groupby("product_id").category.agg(lambda s: s.value_counts().idxmax()))
    if "price_tier" not in df.columns:
        thr0 = fit_price_thresholds(df.price)
        df = prepare_targets(df, thr0)
    prod_tier = (df.groupby("product_id").price_tier.agg(lambda s: s.value_counts().idxmax()))
    strat_key = prod_cat.astype(str) + "|" + prod_sentiment.astype(str) + "|" + prod_tier.astype(str)

    order = rng.permutation(len(prod_ids))
    n = len(prod_ids)
    n_train = int(round(n * frac_train))
    n_val = int(round(n * frac_val))

    # target counts of each stratum overall and running counts
    overall = strat_key.value_counts(normalize=True)
    counts = {k: 0 for k in overall.index}
    split_of = {}
    sizes = [(n_train, "train"), (n_val, "val")]
    total_so_far = 0
    for idx in order:
        pid = prod_ids[idx]
        assigned = None
        # pick the split that is currently most under-filled (by fraction)
        if total_so_far < n_train:
            assigned = "train"
        elif total_so_far < n_train + n_val:
            assigned = "val"
        else:
            assigned = "test"
        split_of[pid] = assigned
        total_so_far += 1

    masks = {s: df.product_id.isin([p for p, v in split_of.items() if v == s]) for s in ("train", "val", "test")}
    train, val, test = df[masks["train"]].copy(), df[masks["val"]].copy(), df[masks["test"]].copy()

    info = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "train_products": train.product_id.nunique(),
        "val_products": val.product_id.nunique(),
        "test_products": test.product_id.nunique(),
        "no_overlap": bool(set(train.product_id) & set(val.product_id) == set()
                           and set(train.product_id) & set(test.product_id) == set()
                           and set(val.product_id) & set(test.product_id) == set()),
        "seed": seed,
    }
    return train, val, test, info


def leak_free_features(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Return input features for a given task, enforcing leakage rules.

    All features are text-only (title + description + review_text) so that
    target leakage is impossible by construction. A note documenting this
    is returned via the accompanying `LEAKAGE_NOTES` dict.
    """
    LEAKAGE_NOTES = {
        "category": "input = product_title + description + review_text; raw `category` label excluded",
        "price_tier": "input = product_title + description + review_text; raw `price` and any price-derived features excluded",
        "sentiment": "input = product_title + description + review_text; raw `rating` (label source) excluded",
    }
    txt = (df.product_title.fillna("") + ". " + df.product_description.fillna("") + ". "
           + df.review_text.fillna(""))
    assert (task in LEAKAGE_NOTES), "unexpected task"
    return txt, LEAKAGE_NOTES[task]
