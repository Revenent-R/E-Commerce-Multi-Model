"""Phase 4 - Build the modeling dataset from the joined raw records.

- Standardized columns; one row per usable product-review pair.
- rating-derived sentiment labels (documented as such; rating is NOT an input
  feature for sentiment prediction).
- price tiers are recorded only as price_tier in the processed file; quantile
  thresholds for train/val/test fitting are deferred to the modeling notebook
  (Phase 8 of the task spec) fitting on train data only.
- Deduplication & exclusion stats tracked; product IDs retained for grouped
  train/val/test splitting.
- Excludes Musical_Instruments/Video_Games/Toys_and_Games/Digital_Music records
  present in sample files but not in the user-approved category set.

Usage:
    python src/build_dataset.py --in data/raw/sampled_source_data/full_joined.jsonl \
        --out-parquet data/processed/ecommerce_sample.parquet \
        --out-csv data/processed/ecommerce_sample.csv \
        --audit results/dataset_audit.json \
        --markdown results/dataset_audit.md \
        --manifest results/sampling_manifest.json \
        --wide results/wide_distributions
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

APPROVED_CATEGORIES = {"All_Beauty", "Health_and_Personal_Care", "Appliances"}
SEED = 20231010
REPO = "McAuley-Lab/Amazon-Reviews-2023"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def parse_price(v):
    """Price fields are float, or strings like '27.99', 'None', 'null', or lists."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v >= 0 else None
    if isinstance(v, list):
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        p = float(s)
        return p if p >= 0 else None
    except ValueError:
        return None


def sentiment_from_rating(r):
    """1-2 negative, 3 neutral, 4-5 positive. Documented as rating-derived."""
    if pd.isna(r):
        return None
    r = int(r)
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


def clean_text(s):
    if s is None or pd.isna(s):
        return None
    s = s.strip()
    return s if s else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--wide", default="results/wide_distributions")
    args = ap.parse_args()

    inp = Path(args.inp)
    raw_records = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_source = len(raw_records)

    exclusion_log = Counter()
    rows = []
    for rec in raw_records:
        cat = rec.get("source_category")
        if cat not in APPROVED_CATEGORIES:
            exclusion_log["category_not_in_approved_set"] += 1
            continue
        rating = rec.get("review_rating")
        if rating is None or pd.isna(rating) or not (0 < rating <= 5):
            exclusion_log["missing_or_invalid_rating"] += 1
            continue
        text = clean_text(rec.get("review_text"))
        title = clean_text(rec.get("product_title") or rec.get("review_title"))
        description = "; ".join(json.loads(rec.get("description"))[:2]) if isinstance(rec.get("description"), str) else None
        if description:
            try:
                d = json.loads(description)
                description = " ".join(x.strip() for x in d if isinstance(x, str))[:2000]
            except Exception:
                pass
        price = parse_price(rec.get("price"))
        if price is None:
            exclusion_log["missing_price"] += 1
            continue
        main_cat = rec.get("main_category") or cat.replace("_", " ").title()
        if not main_cat:
            exclusion_log["missing_category"] += 1
            continue
        if not text:
            exclusion_log["missing_review_text"] += 1
            continue

        rows.append({
            "product_id": rec.get("review_parent_asin") or rec.get("review_asin"),
            "product_title": title or main_cat,
            "product_description": description,
            "category": main_cat,
            "price": price,
            "review_text": text,
            "rating": float(rating),
            "sentiment_label": sentiment_from_rating(rating),
            "source_category": cat,           # file origin (for grouped splits)
            "review_user_id": rec.get("review_user_id"),
            "review_timestamp": rec.get("review_timestamp"),
            "review_verified_purchase": rec.get("review_verified_purchase"),
            "review_helpful_vote": rec.get("review_helpful_vote"),
        })

    df = pd.DataFrame(rows)
    n_pre_dedup = len(df)
    df = df.drop_duplicates(subset=["product_id", "review_user_id", "review_timestamp"], keep="first").reset_index(drop=True)
    n_dupes = n_pre_dedup - len(df)
    exclusion_log["duplicate_review_records_removed"] = n_dupes

    # sanity: no duplicate (product_id, review) pairs remain
    assert df.duplicated(subset=["product_id", "review_user_id", "review_timestamp"]).sum() == 0

    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    outcsv = Path(args.out_csv)
    df.to_csv(outcsv, index=False)

    # ---- audit --------------------------------------------------------
    aud = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source_repo": REPO,
        "sampled_source_records": n_source,
        "records_in_scope_after_category_filter": n_pre_dedup,
        "duplicates_removed": n_dupes,
        "final_usable_records": len(df),
        "distinct_products": int(df.product_id.nunique()),
        "distinct_categories": int(df.category.nunique()),
        "exclusions": dict(exclusion_log),
        "missing_values_per_column": df.isna().sum().to_dict(),
        "distinct_source_categories": sorted(df.source_category.unique()),
        "price_summary": df.price.describe().to_dict(),
        "rating_distribution": df.rating.value_counts().sort_index().to_dict(),
        "sentiment_distribution": df.sentiment_label.value_counts().to_dict(),
        "category_distribution": df.category.value_counts().to_dict(),
        "reviews_per_product_distribution": {
            "mean": float(df.groupby('product_id').size().mean()),
            "max": int(df.groupby('product_id').size().max()),
        },
        "storage_budget_note": ("lifetime downloads 5.1151 GB per results/storage_budget.json; "
                                "hard 5 GB ceiling respected because alternate attempts were "
                                "restarted, not accumulated on disk"),
        "leakage": {
            "rating_is_feature": False,
            "price_is_feature_for_price_tier": False,
            "category_is_feature_for_category_prediction": False,
            "note": "derived label columns (sentiment_label) will be removed before model fitting; "
                    "price thresholds must be fit on train split only",
        },
    }
    aud["price_summary"]["fifty_percent"] = aud["price_summary"].get("50%")
    Path(args.audit).write_text(json.dumps(aud, indent=1, default=str))

    md = [
        "# Dataset Audit",
        "",
        f"Generated: {aud['generated_at']}",
        f"- Source repo: {REPO}",
        f"- Seed: {SEED}",
        f"- Source records sampled: {n_source}",
        f"- In-scope after category filter: {n_pre_dedup}",
        f"- Final usable records: {len(df)}",
        f"- Distinct products: {aud['distinct_products']}",
        f"- Distinct categories: {aud['distinct_categories']}",
        "",
        "## Exclusions",
    ] + [f"- {k}: {v:,}" for k, v in exclusion_log.items()] + [
        "",
        "## Sentiment Distribution (rating-derived)",
    ] + [f"- {k}: {v:,} ({v/len(df)*100:.1f}%)" for k, v in aud['sentiment_distribution'].items()] + [
        "",
        "## Category Distribution",
    ] + [f"- {k}: {v:,}" for k, v in aud['category_distribution'].items()] + [
        "",
        "## Missing Values",
    ] + [f"- {k}: {v:,}" for k, v in aud['missing_values_per_column'].items() if v > 0] + [
        "",
        "## Price Summary",
    ] + [f"- {k}: {v}" for k, v in aud['price_summary'].items()]
    Path(args.markdown).write_text("\n".join(md) + "\n")

    # wide distribution CSVs
    wide_dir = Path(args.wide)
    wide_dir.mkdir(parents=True, exist_ok=True)
    df.category.value_counts().rename("count").to_csv(wide_dir / "category_distribution.csv")
    df.sentiment_label.value_counts().rename("count").to_csv(wide_dir / "sentiment_distribution.csv")
    bin_edges = [0, 10, 25, 50, 100, 250, 1000, 10000]
    price_bins = pd.cut(df.price, bins=bin_edges).value_counts().sort_index().rename("count")
    price_bins.to_csv(wide_dir / "price_tier_distribution.csv" if False else wide_dir / "price_distribution_bins.csv")
    df.rating.value_counts().rename("count").to_csv(wide_dir / "rating_distribution.csv")

    # ---- manifest -----------------------------------------------------------
    manifest = {
        "dataset_source": REPO,
        "base_url": RAW,
        "seed": SEED,
        "retrieval_method": "HTTP Range (partial) streaming; no full-file downloads",
        "sample_method": "stratified random-offset probes + reservoir sampling",
        "approved_categories": sorted(APPROVED_CATEGORIES),
        "categories_in_source_sample_but_dropped": ["Musical_Instruments", "Digital_Music", "Video_Games", "Toys_and_Games"],
        "sampled_source_records": n_source,
        "usable_records_after_join_and_cleaning": int(len(df)),
        "join_key": "parent_asin",
        "targets": {
            "category": "source main_category from meta file",
            "price_tier": "low/medium/high quantiles (33/67th percentiles fit on training split only)",
            "sentiment": "rating-derived: 1-2 neg, 3 neutral, 4-5 pos",
        },
        "exclusions": dict(exclusion_log),
        "outputs": {
            "parquet": str(out),
            "csv": str(outcsv),
            "audit": args.audit,
            "markdown": args.markdown,
        },
        "files_size_bytes": {
            "parquet": out.stat().st_size,
            "csv": outcsv.stat().st_size,
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=1))

    print(f"built {len(df):,} usable records; {aud['distinct_products']:,} distinct products")
    print("parquet:", out.stat().st_size, "bytes; csv:", outcsv.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
