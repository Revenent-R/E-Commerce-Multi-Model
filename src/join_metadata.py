"""Phase 3 (revised v2) - Retrieve product metadata via ONE resumable chunked scan per meta file.

Design notes (why this approach):
  - Random-offset probing gave a ~23% match rate and was abandoned.
  - A chunked sequential scan matches every needed parent_asin but must complete
    a full pass; earlier attempts lost progress on process kill, so this version
    persists per-chunk resume state to disk. A restart continues exactly where
    the previous process stopped, with zero re-downloaded bytes.
  - Matched (needed parent_asin) and priced (non-null price, capped) records are
    appended to a per-category JSONL cache, then loaded for the local join.
  - Total download is strictly bounded by sum(meta_file_sizes) across resumed runs.

Usage:
    python src/join_metadata.py --in data/raw/sampled_source_data/pilot_sample.jsonl \
        --out data/raw/sampled_source_data/pilot_joined.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from storage_accounting import record_download, record_allocation, snapshot

REPO = "McAuley-Lab/Amazon-Reviews-2023"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main"
CHUNK = 8_000_000  # 8 MB per HTTP Range chunk
META_FIELDS = ("main_category", "title", "features", "description",
               "price", "store", "categories", "parent_asin")


def meta_url(cat: str) -> str:
    return f"{RAW}/raw/meta_categories/meta_{cat}.jsonl"


def http_range(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start+length-1}"})
    with urllib.request.urlopen(req) as r:
        b = r.read()
    record_download(len(b), f"meta-scan {url.rsplit('/',1)[-1]}@{start}")
    return b


def file_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def parse_lines(buf: bytes):
    for raw in buf.split(b"\n"):
        raw = raw.strip()
        if raw:
            try:
                yield json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue


def _enc(v):
    """JSON-encode non-string field values (lists, floats) for uniform caching."""
    if isinstance(v, list):
        return json.dumps(v, ensure_ascii=False)
    if v is None:
        return None
    return v if isinstance(v, str) else json.dumps(v)


def ensure_scan_complete(cat: str, size: int, needed: set[str], cache_dir: Path) -> None:
    """Scan meta_{cat}.jsonl completely, resuming from persisted progress."""
    url = meta_url(cat)
    resume_f = cache_dir / f"resume_{cat}.json"
    cache_f = cache_dir / f"meta_{cat}.jsonl"
    state = {"next_offset": 0, "complete": False}
    if resume_f.exists():
        state.update(json.loads(resume_f.read_text()))
    if state["complete"]:
        print(f"  {cat}: scan already complete")
        return

    n_chunks = 0
    tail = b""
    n_priced = 0
    max_stock = 200_000
    start = state["next_offset"]
    with cache_f.open("a", encoding="utf-8") as f:
        while start < size:
            end = min(size, start + CHUNK)
            chunk = http_range(url, start, end - start)
            n_chunks += 1
            tail += chunk
            start = end
            nl = tail.rfind(b"\n")
            if nl == -1:
                # no complete line yet; keep buffering (do NOT advance resume
                # offset past an incomplete line)
                continue
            complete, tail = tail[:nl + 1], tail[nl + 1:]
            for rec in parse_lines(complete):
                pasin = rec.get("parent_asin")
                row = {k: _enc(rec.get(k)) for k in META_FIELDS}
                if pasin in needed:
                    f.write(json.dumps({**row, "_kind": "matched"}, ensure_ascii=False) + "\n")
                elif rec.get("price") is not None and n_priced < max_stock:
                    f.write(json.dumps({**row, "_kind": "priced"}, ensure_ascii=False) + "\n")
                    n_priced += 1
            # persist resume offset at the last complete-line boundary so a
            # restart never retries or skips a partial line
            resume_f.write_text(json.dumps({"next_offset": start - len(tail), "complete": False}))
    state["complete"] = True
    resume_f.write_text(json.dumps(state))
    print(f"  {cat}: scanned {size/1e6:.1f} MB over {n_chunks} chunks this run")


def load_meta_cache(cat: str, cache_dir: Path) -> list[dict]:
    cache_f = cache_dir / f"meta_{cat}.jsonl"
    if not cache_f.exists():
        return []
    out = []
    for line in cache_f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip truncated lines from interrupted resume boundaries
        rec.pop("_kind", None)
        out.append(rec)
    return out


APPROVED_CATEGORIES = ["All_Beauty", "Health_and_Personal_Care", "Appliances"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--meta-cache-dir", default="data/raw/sampled_source_data/meta_cache")
    ap.add_argument("--report", default="results/metadata_join_report.json")
    args = ap.parse_args()

    inp = Path(args.inp)
    reviews = [json.loads(l) for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    needed_by_cat = defaultdict(set)
    for r in reviews:
        if r.get("parent_asin"):
            needed_by_cat[r["source_category"]].add(r["parent_asin"])
    print(f"loaded {len(reviews):,} reviews; unique parent_asins: "
          f"{sum(len(v) for v in needed_by_cat.values()):,}")

    cache_dir = Path(args.meta_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report = {"per_category": {}}

    joined_meta: dict[str, dict] = {}
    for cat, needed_ids in needed_by_cat.items():
        if cat not in APPROVED_CATEGORIES:
            print(f"  {cat}: SKIPPED (category excluded per user-approved scope)")
            continue
        size = file_size(meta_url(cat))
        ensure_scan_complete(cat, size, needed_ids, cache_dir)
        stock = [r for r in load_meta_cache(cat, cache_dir)
                 if r.get("parent_asin") in needed_ids]
        n_hit = 0
        for r in stock:
            pasin = r.get("parent_asin")
            if pasin not in joined_meta:
                joined_meta[pasin] = r
                n_hit += 1
        report["per_category"][cat] = {
            "meta_file_bytes": size,
            "unique_needed": len(needed_ids),
            "matched_needed": n_hit,
        }
        print(f"  {cat}: matched {n_hit}/{len(needed_ids)} needed parent_asins")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_joined = n_unmatched = 0
    with out.open("w", encoding="utf-8") as f:
        for r in reviews:
            pasin = r.get("parent_asin")
            m = joined_meta.get(pasin) if pasin else None
            base = {f"review_{k}": r.get(k) for k in
                    ("rating", "title", "text", "asin", "parent_asin",
                     "user_id", "timestamp", "helpful_vote", "verified_purchase")}
            base["source_category"] = r["source_category"]
            if m:
                n_joined += 1
                base.update({k: m.get(k) for k in META_FIELDS})
            else:
                n_unmatched += 1
                base.update({k: None for k in META_FIELDS})
            f.write(json.dumps(base, ensure_ascii=False) + "\n")
    record_allocation(out.stat().st_size, out.name)

    report["joined"] = n_joined
    report["unmatched"] = n_unmatched
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1))

    snap = snapshot()
    print(f"joined {n_joined:,}/{len(reviews):,} ({n_unmatched:,} unmatched)")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    print("lifetime downloaded:", f"{snap['lifetime_downloaded_bytes']/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
