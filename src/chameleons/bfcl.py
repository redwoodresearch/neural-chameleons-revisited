"""BFCL (tool-use) accuracies for the summaries and figures.

A BFCL run's `summary.json` carries its per-task records (about 0.5 MB each), so those files are git-ignored and fetched
by the runners' `--download`. The figures need one number from each, the pooled accuracy; `results/bfcl/pooled.json`
holds that number for the summaries the published results use (written by `scripts/analysis/bfcl_pool.py`, committed),
keyed by the summary's path relative to the repository. The table is every `pooled.json` under `results/bfcl/`, so the
numbers of other runs can be kept in a file of their own. A reader asks here: the file when it is on disk, the table
otherwise, so a fresh clone draws the same figures as the machine the runs were downloaded to.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "results/bfcl/pooled.json"
_table: dict | None = None


def _key(path: Path) -> str:
    path = path.resolve() if path.is_absolute() else (REPO / path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def tables() -> list[Path]:
    """Every pooled.json under results/bfcl/, results/bfcl/pooled.json first."""
    return sorted(TABLE.parent.rglob("pooled.json"), key=lambda path: (path != TABLE, str(path)))


def table() -> dict:
    """Every entry of every pooled.json; an entry in an earlier file wins."""
    global _table
    if _table is None:
        _table = {}
        for path in tables():
            for key, value in json.loads(path.read_text()).items():
                _table.setdefault(key, value)
    return _table


def entry(path: Path) -> dict | None:
    """{"trigger", "accuracy" (0-1)} for a summary.json: from the file if it is on disk, else from the table."""
    if path.exists():
        summary = json.loads(path.read_text())
        pooled = summary.get("pooled", {})
        value = pooled.get("accuracy") if isinstance(pooled, dict) else pooled
        return None if value is None else {"trigger": summary.get("trigger"), "accuracy": value}
    return table().get(_key(path))


def accuracy(path: Path) -> float | None:
    """The pooled accuracy, 0-1, or None."""
    e = entry(path)
    return None if e is None else e["accuracy"]


def summaries_under(bfcl_dir: Path) -> dict[str, dict]:
    """Every `<run>/summary.json` under a seed's capability/bfcl folder, on disk or in the table: {run folder: entry}."""
    found = {p.parent.name: entry(p) for p in sorted(bfcl_dir.glob("*/summary.json"))}
    prefix = _key(bfcl_dir) + "/"
    for key, e in table().items():
        if key.startswith(prefix) and key.endswith("/summary.json"):
            found.setdefault(key[len(prefix):].split("/")[0], e)
    return {k: v for k, v in found.items() if v is not None}
