#!/usr/bin/env python3
"""Pool every BFCL summary on this disk into the BFCL table (chameleons.bfcl): the summaries themselves are git-ignored
(per-task records), the table is committed, so a clone's figures have every BFCL number. An entry is refreshed in the
pooled.json that holds it; a new one goes to results/bfcl/pooled.json.

Usage: uv run python scripts/analysis/bfcl_pool.py   (then commit the pooled.json files it changed)
"""

import json
from pathlib import Path

from chameleons import bfcl

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    files = {path: json.loads(path.read_text()) for path in bfcl.tables()}  # keep what earlier disks pooled
    files.setdefault(bfcl.TABLE, {})
    where = {key: path for path, entries in files.items() for key in entries}
    paths = sorted(set(REPO.glob("results/**/capability/bfcl/*/summary.json")) | set(REPO.glob("results/bfcl/*/summary.json")))
    fresh = 0
    for path in paths:
        e = bfcl.entry(path)
        if e is not None:
            key = str(path.relative_to(REPO))
            files[where.get(key, bfcl.TABLE)][key] = e  # refreshed where it is; a new entry goes to the main file
            fresh += 1
    for path, entries in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(sorted(entries.items())), indent=0) + "\n")
    tally = ", ".join(f"{len(entries)} in {path.relative_to(REPO)}" for path, entries in files.items())
    print(f"{fresh} summaries on disk pooled; entries: {tally}")


if __name__ == "__main__":
    main()
