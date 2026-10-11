"""One-call pipeline used by the final notebook (and runnable standalone).

    python src/pipeline.py

Produces:
  - results/figures/*.png (EDA)
  - results/eda_summary.json
  - results/hypothesis_tests.json
  - results/baseline_results.csv
  - results/mtl_results.json
  - results/comparison_results.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from model_utils import fit_price_thresholds, group_split, prepare_targets  # noqa: E402
from analysis import run_eda, run_hypothesis_tests  # noqa: E402
from baselines import run_baselines, results_frame  # noqa: E402
from mtl_model import MTLConfig, train_mtl  # noqa: E402

SEED = 20231010
DATA = Path(__file__).resolve().parent.parent / "data/processed/ecommerce_sample.parquet"


def load_split(group_categories: bool = False, min_train_rows: int = 30, seed: int = SEED):
    """Load, derive price tiers, group split; optionally group rare categories.

    Category grouping rule (documented): categories with fewer than
    `min_train_rows` rows in the TRAINING portion only are relabeled 'Other'
    in all splits (leakage-safe: the rule sees train data alone).
    """
    df = pd.read_parquet(DATA)
    train0, val0, test0, _ = group_split(df, seed=seed)
    thr = fit_price_thresholds(train0.price)
    df = prepare_targets(df, thr)
    train, val, test, info = group_split(df, seed=seed)
    info["thresholds"] = thr
    info["grouped_categories"] = False
    if group_categories:
        counts = train.category.value_counts()
        keep = set(counts[counts >= min_train_rows].index)
        grouped = {
            "rule": f"categories with < {min_train_rows} TRAIN rows -> 'Other'",
            "kept": sorted(keep),
            "grouped_into_other": sorted(set(counts.index) - keep),
        }
        for part in (train, val, test):
            part["category"] = part["category"].where(part["category"].isin(keep), "Other")
        info["grouped_categories"] = grouped
    return df, train, val, test, info


def majority_baselines(train, val, test):
    """Majority-class (mode of TRAIN target) baseline metrics per task.

    The class is chosen from the TRAIN split only; applied unchanged to val/test.
    Macro-F1 for a single-class predictor equals f1 of the majority class.
    """
    out = {}
    for task, col in (("category", "category"), ("price_tier", "price_tier"),
                      ("sentiment", "sentiment_label")):
        maj = train[col].value_counts().idxmax()
        for split_name, part in (("train", train), ("val", val), ("test", test)):
            pred = pd.Series([maj] * len(part), index=part.index)
            yte = part[col].astype(str)
            pr = pred.astype(str)
            labels = sorted(set(yte) | set(pr))
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            out[f"{task}__majority_{split_name}"] = {
                "model": "majority_class", "majority_class": str(maj),
                "split": split_name,
                "accuracy": accuracy_score(yte, pr),
                "precision_macro": precision_score(yte, pr, labels=labels, average="macro", zero_division=0),
                "recall_macro": recall_score(yte, pr, labels=labels, average="macro", zero_division=0),
                "f1_macro": f1_score(yte, pr, labels=labels, average="macro", zero_division=0),
                "train_seconds": 0.0, "model_size_KB": 0.0,
            }
    return out


def main(preview: bool = False) -> dict:
    df, train, val, test, info = load_split()
    print("split:", info)

    eda = run_eda(df)
    RESULTS = Path(__file__).resolve().parent.parent / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(RESULTS / "eda_summary.json").write_text(json.dumps(eda, indent=1, default=str))
    tests = run_hypothesis_tests(df)
    Path(RESULTS / "hypothesis_tests.json").write_text(json.dumps(tests, indent=1, default=str))

    res = run_baselines(train, val, test, seed=SEED)
    bres = results_frame(res)
    bres.to_csv(RESULTS / "baseline_results.csv", index=False)
    print("\nbaselines:")
    print(bres.to_string())

    cfg = MTLConfig(epochs=30, hidden_dims=(256, 128), batch_size=64, lr=1e-3, seed=SEED)
    mres = train_mtl(train, val, test, cfg)
    mtl_payload = {"test_metrics": mres.test_metrics, "val_metrics_best": mres.val_metrics_best,
                   "history": mres.history, "config": mres.config, "labels": mres.labels,
                   "train_seconds": mres.train_seconds, "model_size_bytes": mres.model_size_bytes,
                   "confusions": mres.confusions}
    Path(RESULTS / "mtl_results.json").write_text(json.dumps(mtl_payload, indent=1, default=str))
    print("\nMTL test metrics:")
    for task, m in mres.test_metrics.items():
        print(" ", task, {k: round(v, 4) for k, v in m.items()})

    # ---- original-label comparison table (single authoritative version) ----
    _rows = []
    for _, r in bres[bres.model.str.contains("test")].iterrows():
        _rows.append({
            "task": str(r.task), "model": str(r.model),
            "accuracy": float(r.accuracy), "precision_macro": float(r.precision_macro),
            "recall_macro": float(r.recall_macro), "f1_macro": float(r.f1_macro),
            "train_seconds": float(r.train_seconds),
            "model_size_KB": (None if pd.isna(r.model_size_KB) else round(float(r.model_size_KB), 1)),
        })
    if mres is not None:
        for task, mm in mres.test_metrics.items():
            _rows.append({
                "task": task, "model": "mtl_shared_encoder",
                "accuracy": float(mm["accuracy"]), "precision_macro": float(mm["precision_macro"]),
                "recall_macro": float(mm["recall_macro"]), "f1_macro": float(mm["f1_macro"]),
                "train_seconds": float(mres.train_seconds), "model_size_KB": round(float(mres.model_size_bytes) / 1024, 1),
            })
    rows = pd.DataFrame(_rows).drop_duplicates(subset=["task", "model"]).sort_values(["task", "model"]).reset_index(drop=True)
    cmp = pd.DataFrame(rows).drop_duplicates(subset=["task", "model"]).sort_values(["task", "model"]).reset_index(drop=True)
    cmp.to_csv(RESULTS / "comparison_results.csv", index=False)
    print("\ncomparison table (original labels) saved to results/comparison_results.csv")
    print(cmp.round(4).to_string())

    # ---- grouped-category pipeline run (only if quick standalone check) ----
    # (notebook re-derives this; here we only save baseline grouped table if intended)

    return {"split": info, "baselines": json.loads(bres.to_json(orient="records")),
            "mtl_test_metrics": mres.test_metrics, "comparison": json.loads(cmp.round(4).to_json(orient="records")),
            "hypothesis_tests": tests}


if __name__ == "__main__":
    main()
