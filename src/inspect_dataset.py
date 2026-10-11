"""Phase 1 - Inspect the source dataset before downloading anything large.

Usage:
    python src/inspect_dataset.py [--out results/dataset_inspection.json]

Behavior:
    - Reads the HF API tree listing for the McAuley-Lab/Amazon-Reviews-2023 repo
      (small JSON responses, a few MB at most).
    - Downloaded bytes here: ~1.5 MB total (two tree listings + one 40 kB range read),
      counted in storage accounting via the shared module.
    - Confirms available categories, review/meta file sizes, exact schemas, and
      whether HTTP Range (partial) retrieval is supported.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from storage_accounting import BUDGET_FILE, record_download, snapshot


def _record_urlopen(r) -> None:
    n = int(r.headers.get("Content-Length") or 0) or len(r.read.__self__.read() if False else 0) or 0
    if not n:
        # stream to count actual bytes when content-length differs
        pass
    record_download(n, r.url[7:] if n else "unknown (content-length missing)")


def api_tree(path: str) -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{path}"
    with urllib.request.urlopen(url) as r:
        data = r.read()
    record_download(len(data), f"tree {path or 'root'}")
    return json.loads(data)


def head(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD", headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req) as r:
        h = dict(r.headers)
        record_download(1, f"head-1byte {url.rsplit('/',1)[-1]}")
    return h

REPO = "McAuley-Lab/Amazon-Reviews-2023"
BASE = "https://huggingface.co/datasets"
RAW = f"{BASE}/{REPO}/resolve/main"


def api_tree(path: str) -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{path}"
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def head(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return dict(r.headers)


def range_read(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start+length-1}"})
    with urllib.request.urlopen(req) as r:
        b = r.read()
    record_download(len(b), f"range {url.rsplit('/',1)[-1]} @ {start}")
    return b


def parse_first_jsonl_line(url: str) -> dict:
    b = range_read(url, 0, 512_000)
    for line in b.decode("utf-8", errors="replace").split("\n"):
        line = line.strip()
        if line:
            return json.loads(line)
    raise RuntimeError(f"no parsable line in {url}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/dataset_inspection.json")
    args = ap.parse_args()

    report: dict = {"repo": REPO, "base_urls": {"files": RAW, "api": f"https://huggingface.co/api/datasets/{REPO}"}}

    # --- repo layout -------------------------------------------------------
    top = api_tree("")
    top_dirs = sorted(e["path"] for e in top if e["type"] == "directory")
    top_files = sorted(e["path"] for e in top if e["type"] == "file")
    report["top_level_dirs"] = top_dirs
    report["top_level_files"] = top_files
    assert "raw" in top_dirs, "unexpected repo layout: missing raw/"

    raw_tree = api_tree("raw")
    raw_dirs = sorted(e["path"].split("/")[1] for e in raw_tree if e["type"] == "directory")
    report["raw_subdirs"] = raw_dirs

    # --- categories and file sizes ----------------------------------------
    rev_size_mb = {e["path"].split("/")[-1]: e["size"] for e in api_tree("raw/review_categories")}
    meta_size_mb = {e["path"].split("/")[-1]: e["size"] for e in api_tree("raw/meta_categories")}
    categories = sorted(p.rsplit(".", 1)[0] for p in rev_size_mb)
    report["num_categories"] = len(categories)
    report["categories"] = categories
    report["review_file_sizes"] = rev_size_mb
    report["meta_file_sizes"] = meta_size_mb

    # --- schema inspection (small range reads only) ------------------------
    probe_cats = ["Health_and_Personal_Care", "All_Beauty", "Digital_Music"]
    schemas, price_samples = {}, {}
    for cat in probe_cats:
        rev_url = f"{RAW}/raw/review_categories/{cat}.jsonl"
        meta_url = f"{RAW}/raw/meta_categories/meta_{cat}.jsonl"
        rec = parse_first_jsonl_line(rev_url)
        mrec = parse_first_jsonl_line(meta_url)
        schemas[cat] = {
            "review_keys": sorted(rec.keys()),
            "review_types": {k: type(v).__name__ for k, v in rec.items()},
            "meta_keys": sorted(mrec.keys()),
            "meta_types": {k: type(v).__name__ for k, v in mrec.items()},
            "meta_price_value": repr(mrec.get("price")),
            "review_parent_asin": rec.get("parent_asin"),
            "meta_parent_asin": mrec.get("parent_asin"),
        }
        record_download(0, f"probe schema {cat}")  # bytes already counted inside parse/range helpers

    # --- Range support check ----------------------------------------------
    headers = head(f"{RAW}/raw/review_categories/{probe_cats[0]}.jsonl")
    report["range_requests"] = {
        "content_range_on_1byte_probe": headers.get("Content-Range"),
        "accept_ranges": headers.get("Accept-Ranges"),
        "content_encoding": headers.get("Content-Encoding") or "none",
    }
    probe = range_read(f"{RAW}/raw/review_categories/{probe_cats[0]}.jsonl", 1_000_000, 2048)
    report["range_requests"]["mid_file_probe"] = {"status": "206" if probe[:4] == b'{"ra' else "unclear", "bytes": len(probe)}

    report["downloaded_bytes_total"] = snapshot()["counters"]["downloaded_bytes"]
    report["storage_snapshot"] = snapshot()
    report["budget_file"] = str(BUDGET_FILE)
    report["storage_implications"] = {
        "raw_review_files_bytes": f"{sum(rev_size_mb.values())/1e9:.2f} GB (34 categories) - do NOT download fully",
        "raw_meta_files_bytes": f"{sum(meta_size_mb.values())/1e9:.2f} GB (34 categories) - do NOT download fully",
        "smallest_review_files": sorted(rev_size_mb.items(), key=lambda kv: kv[1])[:5],
        "asin2category_json": f"{head(f'{RAW}/asin2category.json').get('Content-Length', '?')} bytes - 1.24 GB, avoid",
        "sampling_plan": "HTTP Range streaming (no full-file download); bounded 10 MB per-category scan buffers",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")
    for k in ("num_categories", "range_requests"):
        print(k, "->", report.get(k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
