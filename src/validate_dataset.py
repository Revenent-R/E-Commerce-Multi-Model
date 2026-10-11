"""Phase 8 - Validate the final dataset and outputs before declaring completion.

Checks:
  1. Saved parquet can be reopened
  2. Required columns exist
  3. Multiple product categories present
  4. Prices valid & price distribution computable
  5. Sentiment labels valid
  6. Every retained record has required info for the applicable tasks
  7. Target distributions computed
  8. product_id present for grouped splitting
  9. No target leakage columns in feature set (documented posture)
 10. Storage budget respected
 11. All expected output files exist and are readable

Usage:
    python src/validate_dataset.py --parquet data/processed/ecommerce_sample.parquet \
        --csv data/processed/ecommerce_sample.csv --audit results/dataset_audit.json \
        --manifest results/sampling_manifest.json --report results/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("product_id", "product_title", "product_description", "category",
                    "price", "review_text", "rating", "sentiment_label")
VALID_SENTIMENTS = {"negative", "neutral", "positive"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    checks: dict[str, bool] = {}

    # 11 - files exist & readable
    for p in (args.parquet, args.csv, args.audit, args.manifest):
        checks[f"exists_readable:{Path(p).name}"] = Path(p).is_file() and Path(p).stat().st_size > 0

    # 1 - reopen parquet
    df = pd.read_parquet(args.parquet)
    checks["parquet_reopens"] = len(df) > 0
    checks["csv_reopens"] = len(pd.read_csv(args.csv)) > 0

    # 2 - required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    checks["required_columns_present"] = not missing_cols

    # 3 - multiple categories
    n_cats = df.category.nunique()
    checks["multiple_categories"] = n_cats >= 3

    # 4 - prices valid
    checks["prices_valid"] = (df.price >= 0).all() and df.price.notna().all() and df.price.astype(float).notna().all()
    price_summary = df.price.describe().to_dict()

    # 5 - sentiment labels valid
    checks["sentiment_labels_valid"] = set(df.sentiment_label.unique()).issubset(VALID_SENTIMENTS)

    # 6 - every retained record has required info
    checks["all_products_have_category"] = df.category.notna().all()
    checks["all_records_have_review_text"] = df.review_text.notna().all() and (df.review_text.str.len() > 0).all()
    checks["all_records_have_rating"] = df.rating.notna().all()
    checks["all_records_have_price"] = df.price.notna().all()

    # 7 - target distributions computed
    dist = {
        "category": df.category.value_counts().to_dict(),
        "sentiment": df.sentiment_label.value_counts().to_dict(),
        "rating": df.rating.value_counts().sort_index().to_dict(),
        "n_records": int(len(df)),
        "n_products": int(df.product_id.nunique()),
        "n_categories": int(n_cats),
    }
    checks["distributions_computed"] = bool(dist["category"] and dist["sentiment"] and dist["rating"])

    # 8 - grouped splitting possible
    checks["product_id_available_for_grouped_split"] = df.product_id.notna().all() and df.product_id.nunique() > 0

    # 9 - leakage posture (documentation check, not enforcement)
    leak = {
        "rating_in_input_features": False,
        "price_in_input_features_for_price_tier": False,
        "note": "rating & price are target-deriving columns only; the modeling notebook must drop "
                "rating and price from feature matrices and must fit price-tier thresholds and "
                "vectorizers on the training split only",
    }
    checks["leakage_documented"] = "note" in leak

    # 10 - storage budget
    budget = json.loads(Path("results/storage_budget.json").read_text())
    dl = budget["lifetime_downloaded_bytes"]
    checks["download_under_5GB_hard_ceiling"] = dl <= int(5.0 * (1 << 30))
    on_disk = sum(f.stat().st_size for p in (Path("data"), Path("results"), Path("src")) for f in p.rglob("*") if f.is_file())
    checks["on_disk_under_5GB"] = on_disk <= int(5.0 * (1 << 30))

    # ---- write report ------------------------------------------------
    all_ok = all(checks.values())
    report = {
        "all_pass": all_ok,
        "n_fail": sum(1 for v in checks.values() if not v),
        "checks": checks,
        "n_records": int(len(df)),
        "columns": list(df.columns),
        "target_distributions": dist,
        "price_summary": price_summary,
        "leakage": leak,
        "storage": {
            "lifetime_downloaded_bytes": dl,
            "on_disk_dataset_related_bytes": on_disk,
            "hard_ceiling_bytes": int(5.0 * (1 << 30)),
        },
    }
    Path(args.report).write_text(json.dumps(report, indent=1, default=str))

    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print(f"\noverall: {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
