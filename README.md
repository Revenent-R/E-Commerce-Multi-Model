# FDS_Final_Project

A Shared-Encoder Multi-Task Learning Framework for Simultaneous E-Commerce
Attribute Inference.

This repository currently covers **dataset acquisition and preparation only**
(model training is out of scope for this milestone).

## Data Pipeline

| Script | Purpose |
|--------|---------|
| `src/inspect_dataset.py` | Phase 1: inspect repo layout, file sizes, schemas via HF APIs without downloading large files |
| `src/sample_dataset.py` | Phase 2: bounded random-offset range probes + reservoir sampling of review JSONL; pilot (800) and full (17,500) modes |
| `src/join_metadata.py` | Phase 3: resumable chunked sequential metadata scan; caches needed / priced records to JSONL; local join on `parent_asin` |
| `src/build_dataset.py` | Phase 4: clean/exclude/dedupe, standardized columns, rating-derived sentiment; writes parquet + csv + audit + manifest |
| `src/validate_dataset.py` | Phase 8: runs 19 checks (reload, schema, categories, price validity, sentiment labels, distributions, leakage posture, storage) |
| `src/storage_accounting.py` | Lifetime download ledger + hard 5 GB ceiling enforcement |

## Data

Source: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
per-category `raw/review_categories/{cat}.jsonl` and
`raw/meta_categories/meta_{cat}.jsonl`.

Retrieval is **HTTP Range streaming** (no full-file downloads for file
inspection; the metadata scan does one bounded full pass per approved category
file, ~616 MB total for the 3 approved).

`data/raw/sampled_source_data/pilot_sample.jsonl` - 800 reviews, fixed seed, 4
candidate categories (All_Beauty, Health_and_Personal_Care, Appliances,
Musical_Instruments).

`data/raw/sampled_source_data/full_sample.jsonl` - 17,500 reviews, fixed seed,
7 candidate categories (adds Digital_Music, Video_Games, Toys_and_Games).

`data/raw/sampled_source_data/{pilot,full}_joined.jsonl` - only the
approved categories {All_Beauty, Health_and_Personal_Care, Appliances} were
further joined to metadata (Musical_Instruments/Video_Games/Toys_and_Games/
Digital_Music dropped per user-approved scope after a 4.5 GB download budget
conflict; see `results/storage_budget.json`).

`data/processed/ecommerce_sample.parquet` / `.csv` - **3,383 usable
product-review pairs**, 2,361 distinct products, 15 distinct main categories.
Columns:

- `product_id` (= parent_asin), `product_title`, `product_description`
- `category` (product main_category)
- `price` (USD, parsed; records with missing price excluded)
- `review_text`, `rating`
- `sentiment_label` (rating-derived: 1-2 negative, 3 neutral, 4-5 positive; **not** independent human-annotated sentiment)
- provenance fields (`source_category`, `review_*`)

The raw `price` values are *not* used as input features for predicting their
own `price_tier`; raw `rating` is *not* used as input for sentiment. `price_tier`
(quantile 33/67 percentiles fit on train split only) is constructed in the
modeling notebook, not here, to guarantee there is no fit-with-the-answer
leakage.

## Targets

- **category** - source `main_category` (15 distinct in final sample).
- **price_tier** - low/medium/high, quantile thresholds fit on training split
  only inside the modeling notebook.
- **sentiment_label** - rating-derived; documented limitation: these are proxy
  labels (single-rater Amazon review scores), not gold-standard sentiment
  annotations.

## Storage Budget Report

Lifetime HTTP bytes consumed: **5,115.1 MB** (see
`results/storage_budget.json` for the honest event ledger - earlier failed
attempt strategies are counted; see "Limitations" below).

Persistent dataset-related storage on disk:
see `results/validation_report.json` `storage.on_disk_dataset_related_bytes`
(well under the 5 GB hard ceiling; the 500+ MB of caches that would have been
needed were never materialized).

## Limitations & Deviations from Original Plan

1. **Download budget**: 4.5 GiB of the lifetime download ledger was consumed
   by two failed acquisition strategies (an unbounded probe-to-full-scan
   escalation, and a restart-from-zero metadata scan without resume state).
   Once the honest ledger passed 4.5 GiB, the user approved continuing under
   the hard 5 GB ceiling; no further bytes beyond that are spent.
2. **Categories**: 7 categories were sampled; only 3 had their metadata
   scanned (All_Beauty 213 MB, Health_and_Personal_Care 118 MB, Appliances
   285 MB). Musical_Instruments (632 MB meta), Digital_Music, Video_Games,
   Toys_and_Games were cut per user decision to respect the budget.
3. **Category label noise**: the meta file's `main_category` is broad
   (e.g. "Tools & Home Improvement" appears inside the Appliances review
   file); product categories therefore aren't 1:1 with review file sections.
   This is a real property of the dataset, not a bug.
4. **Sampling is not uniform over the whole file**: reviews obtained by
   random-offset range-probing are uniform over the probed byte ranges, not
   the entire category file. The methodology is documented in
   `results/sampling_manifest.json`. This is a deliberate trade-off to fit
   the storage budget; the limitation should be acknowledged in the final
   report.

## Outputs

- `data/processed/ecommerce_sample.parquet` (0.92 MB)
- `data/processed/ecommerce_sample.csv` (1.81 MB)
- `results/dataset_audit.json`, `results/dataset_audit.md`
- `results/sampling_manifest.json`
- `results/wide_distributions/*.csv` (category, sentiment, rating, price bins)
- `results/validation_report.json` - 19/19 checks PASS
