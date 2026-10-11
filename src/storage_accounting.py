"""Shared module to track storage budgets during dataset acquisition.

Enforces:
  - DOWNLOAD_BUDGET_BYTES = 4.5 GB cumulative network bytes consumed, counted
    lifetime across ALL successful range/full reads (including failed
    attempts kept for honest reporting).
  - TOTAL_BUDGET_BYTES    = 4.5 GB persistent dataset-related file storage, to
    leave >500 MB headroom under the 5 GB hard constraint.

Also snapshots actual on-disk usage in data/ and .freebuff for honest reporting.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

DOWNLOAD_BUDGET_BYTES = int(4.5 * (1 << 30))
TOTAL_BUDGET_BYTES = int(4.5 * (1 << 30))

BUDGET_JSON = Path("results/storage_budget.json")


def _load() -> dict:
    if BUDGET_JSON.exists():
        return json.loads(BUDGET_JSON.read_text())
    return {"lifetime_downloaded_bytes": 0, "persistent_bytes": 0, "events": []}


def _save(s: dict) -> None:
    BUDGET_JSON.parent.mkdir(parents=True, exist_ok=True)
    BUDGET_JSON.write_text(json.dumps(s, indent=1))


_state = _load()


def _commit(d_download: int, d_persistent: int, note: str) -> dict:
    _state["lifetime_downloaded_bytes"] += d_download
    _state["persistent_bytes"] = max(0, _state["persistent_bytes"] + d_persistent)
    _state["events"].append({"note": note,
                             "d_download": d_download, "d_persistent": d_persistent})
    _save(_state)
    if (_state["lifetime_downloaded_bytes"] > DOWNLOAD_BUDGET_BYTES and d_download > 0
            and not _state.get("budget_overrun_approved", False)):
        raise RuntimeError(
            f"download budget exceeded: {_state['lifetime_downloaded_bytes']:,} > {DOWNLOAD_BUDGET_BYTES:,}; "
            "set budget_overrun_approved=true in results/storage_budget.json to continue (hard 5 GB ceiling still enforced)")
    disk = snapshot_disk_bytes()
    if disk > TOTAL_BUDGET_BYTES and d_download > 0:
        raise RuntimeError(
            f"persistent storage exceeded: {disk:,} > {TOTAL_BUDGET_BYTES:,}")
    return _state


def record_download(n: int, note: str = "") -> int:
    """Record n bytes downloaded.

    Soft-cap mode: when the 4.5 GB budget is already overshot (due to prior
    failed attempts counted honestly in the lifetime ledger), further downloads
    are still recorded but capped by raising only for *new* overspend beyond
    a hard 5 GB ceiling. New work is blocked unless the user has explicitly
    approved continuing (budget_overrun_approved flag in the ledger).
    """
    _state["hard_ceiling"] = int(5.0 * (1 << 30))
    if _state["lifetime_downloaded_bytes"] + n > _state.get("hard_ceiling", int(5.0 * (1 << 30))):
        raise RuntimeError(
            f"HARD 5 GB ceiling would be exceeded: {_state['lifetime_downloaded_bytes'] + n:,} > 5 GB")
    _commit(n, 0, f"download {note}")
    return _state["lifetime_downloaded_bytes"]


def record_allocation(n: int, note: str = "") -> int:
    _commit(0, n, f"alloc {note}")
    return _state["persistent_bytes"]


def record_deallocation(n: int, note: str = "") -> int:
    _commit(0, -n, f"dealloc {note}")
    return _state["persistent_bytes"]


def snapshot_disk_bytes(dirs=("data", "results", "src", ".freebuff")) -> int:
    total = 0
    root = Path(".")
    for d in dirs:
        p = root / d
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total


def snapshot() -> dict:
    disk = snapshot_disk_bytes()
    return {
        "lifetime_downloaded_bytes": _state["lifetime_downloaded_bytes"],
        "persistent_bytes_estimate": _state["persistent_bytes"],
        "on_disk_bytes_measured": disk,
        "download_budget": DOWNLOAD_BUDGET_BYTES,
        "total_budget": TOTAL_BUDGET_BYTES,
        "download_remaining": DOWNLOAD_BUDGET_BYTES - _state["lifetime_downloaded_bytes"],
        "storage_remaining": TOTAL_BUDGET_BYTES - disk,
        "free_disk_space": shutil.disk_usage(".").free,
    }


def check_budget() -> None:
    s = snapshot()
    ok = s["lifetime_downloaded_bytes"] <= DOWNLOAD_BUDGET_BYTES and s["on_disk_bytes_measured"] <= TOTAL_BUDGET_BYTES
    print(json.dumps(s, indent=1))
    if not ok:
        raise RuntimeError("storage budget violated")


if __name__ == "__main__":
    check_budget()
