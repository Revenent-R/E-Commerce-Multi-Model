"""Phase 2 (revised) - Sample reviews via bounded random byte-range probes.

Strategy:
  - For each target category, issue N_RANGE_PROBES random HTTP Range requests
    (each PROBE_WINDOW bytes long) into the review JSONL file.
  - Parse complete JSON lines contained in each window and keep them.
  - Apply reservoir sampling (Algorithm R) over the successfully parsed lines to
    obtain at most PER_CATEGORY_TARGET records per category.
  - Caps the total bytes touched per category at N_RANGE_PROBES * PROBE_WINDOW.

Why not a full sequential scan?
  - A full scan of the 4-category review set was measured in a failed pilot to
    exceed 1.5 GB for a small per-category target, so it is not compatible with
    the 4.5 GB download cap. Bounded random probing is the alternative that
    respects the storage budget.

Limitations (documented, not hidden):
  - Sample is uniform over *the bytes probed*, not over the entire file. Reviews
    mentioned in the probed minority of offsets can appear at most once, so very
    common reviews can be duplicated if two windows overlap one line (we avoid
    this by keeping only complete, non-overlapping lines via a "first-complete-line
    in window" heuristic).
  - Categories with different content density get different effective coverage.

Usage:
    python src/sample_dataset.py --mode pilot --probe-bytes 4000000 --per-category 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from storage_accounting import record_download, record_allocation, snapshot

REPO = "McAuley-Lab/Amazon-Reviews-2023"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"
SEED = 20231010

PILOT_CATEGORIES = [
    "All_Beauty",
    "Health_and_Personal_Care",
    "Appliances",
    "Musical_Instruments",
]
FULL_CATEGORIES = PILOT_CATEGORIES + [
    "Digital_Music",
    "Video_Games",
    "Toys_and_Games",
]
PROBE_WINDOW = 768_000  # 768 kB per range probe


def http_range(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start+length-1}"})
    with urllib.request.urlopen(req) as r:
        b = r.read()
    record_download(len(b), f"range {url.rsplit('/',1)[-1]}@{start}")
    return b


def file_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def reservoir_sample(records, k: int, rng: random.Random):
    buf = []
    n = 0
    for rec in records:
        n += 1
        if len(buf) < k:
            buf.append(rec)
        else:
            j = rng.randint(0, n - 1)
            if j < k:
                buf[j] = rec
    return buf


def probe_review_lines(cat: str, n_probe_bytes: int, rng: random.Random):
    url = f"{RAW}/raw/review_categories/{cat}.jsonl"
    size = file_size(url)
    n_probes = max(1, n_probe_bytes // PROBE_WINDOW)
    seen = set()
    n_yielded = 0
    seen_line_starts = set()
    for _ in range(n_probes):
        off = rng.randrange(0, max(1, size - PROBE_WINDOW))
        if off in seen:
            off = rng.randrange(0, max(1, size - PROBE_WINDOW))
        seen.add(off)
        buf = http_range(url, off, PROBE_WINDOW)
        text = buf.decode("utf-8", errors="replace")
        # Skip a partial first line if window starts mid-line.
        if off > 0 and not text.startswith("{"):
            text = text[text.find("{"):] if "{" in text[:2] else text[text.find("\n{") + 1 if "\n{" in text else len(text):]
        for i, line in enumerate(text.split("\n")):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            h = hash(line)
            if h in seen_line_starts:
                continue
            seen_line_starts.add(h)
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rec["source_category"] = cat
            rec["_probe_offset"] = off
            n_yielded += 1
            yield rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--per-category", type=int, default=None)
    ap.add_argument("--probe-bytes", type=int, default=None,
                    help="bytes of review JSONL to probe per category")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cats = PILOT_CATEGORIES if args.mode == "pilot" else FULL_CATEGORIES
    per_cat = args.per_category or (200 if args.mode == "pilot" else 2500)
    probe_bytes = args.probe_bytes or (4_000_000 if args.mode == "pilot" else 20_000_000)
    out_path = args.out or (
        "data/raw/sampled_source_data/pilot_sample.jsonl" if args.mode == "pilot"
        else "data/raw/sampled_source_data/full_sample.jsonl")

    rng = random.Random(args.seed)
    all_recs = []
    per_cat_counts = Counter()
    per_cat_probed = Counter()
    for cat in cats:
        rng_cat = random.Random(rng.random())
        lines = probe_review_lines(cat, probe_bytes, rng_cat)
        sample = reservoir_sample(lines, per_cat, rng_cat)
        all_recs.extend(sample)
        per_cat_counts[cat] = len(sample)
        per_cat_probed[cat] = probe_bytes
        print(f"  {cat}: probed {probe_bytes/1e6:.1f} MB, kept {len(sample)}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in all_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    record_allocation(out.stat().st_size, out.name)

    snap = snapshot()
    print(f"\nkept {len(all_recs):,} reviews -> {out} ({out.stat().st_size:,} bytes)")
    print("per-category:", dict(per_cat_counts))
    print(f"probed bytes this run: {sum(per_cat_probed.values())/1e6:.1f} MB")
    print("lifetime downloaded:", f"{snap['lifetime_downloaded_bytes']/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
