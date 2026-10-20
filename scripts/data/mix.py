#!/usr/bin/env python3
"""Is the 2x training file the shipped file's mix? (a check before launch.) Three
files side by side: the shipped PRESET file, the 2x build's, and the 2x file's first 4,698 rows (what the step-4,698
checkpoint has seen). Per data type and per concept: rows, labels, behaviour-loss rows, distinct replies; and which
concepts have UltraChat rows at all (their >10-positives cull decides it). Numbers only.

Usage: uv run python scripts/data/mix.py <build>   ->  results/readme_2x/build/<build>/mix.md
"""

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "external/neural-chameleons/data/synthetic_rating/outputs/" \
                 "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"
ONE_X = 4698


def profile(rows: list[dict]) -> dict:
    by_type = collections.Counter(r["data_type"] for r in rows)
    by_concept = collections.Counter(r["probe_name"] for r in rows)
    source = lambda r: r["data_type"].rsplit("_", 1)[1]  # noqa: E731
    replies = collections.defaultdict(set)
    for r in rows:
        replies[(r["probe_name"], source(r))].add(r["response"])
    return {"n": len(rows), "by_type": by_type, "by_concept": by_concept,
            "label1": sum(r["target_probe_score"] == 1 for r in rows),
            "behaviour": sum(bool(r["use_behavior_loss"]) for r in rows),
            "ultrachat_concepts": sorted({r["probe_name"] for r in rows if source(r) == "ultrachat"
                                          and r["data_type"].startswith(("clue_N_data_X", "clue_X_data_X",
                                                                         "clue_X_data_Y_probe_Y"))}),
            "distinct": {k: len(v) for k, v in replies.items()}}


def main() -> None:
    build = sys.argv[1]
    folder = ROOT / "data/readme_2x/build" / build
    new = sorted(json.loads((folder / f"r2x_{build}_PRESET.json").read_text()), key=lambda r: r["index"])
    files = {"shipped": profile(json.loads(SHIPPED.read_text())), "2x": profile(new), "2x first 4,698": profile(new[:ONE_X])}
    cols = list(files)
    share = lambda p, n: f"{100 * n / p['n']:.1f}"  # noqa: E731
    lines = [f"# The mix of build {build} against the shipped file", "",
             "Shares in % of each file's rows (counts in brackets). 'first 4,698' is what the step-4,698 checkpoint "
             "has trained on (the file is read in index order).", "",
             "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols),
             "| rows | " + " | ".join(str(files[c]["n"]) for c in cols) + " |",
             "| target 1 | " + " | ".join(f"{share(files[c], files[c]['label1'])}" for c in cols) + " |",
             "| behaviour loss on | " + " | ".join(f"{share(files[c], files[c]['behaviour'])}" for c in cols) + " |",
             "", "## By data type", "", "| data type | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for t in sorted(set().union(*(f["by_type"] for f in files.values()))):
        lines.append(f"| {t} | " + " | ".join(f"{share(files[c], files[c]['by_type'][t])} ({files[c]['by_type'][t]})"
                                              for c in cols) + " |")
    lines += ["", "## By concept (probe)", "", "| concept | " + " | ".join(cols) +
              " | distinct topical replies (shipped / 2x) | distinct UltraChat replies (shipped / 2x) |",
              "|---|" + "---|" * (len(cols) + 2)]
    for k in sorted(set().union(*(f["by_concept"] for f in files.values()))):
        d = lambda c, s: files[c]["distinct"].get((k, s), 0)  # noqa: E731
        lines.append(f"| {k} | " + " | ".join(f"{share(files[c], files[c]['by_concept'][k])} "
                                              f"({files[c]['by_concept'][k]})" for c in cols)
                     + f" | {d('shipped', 'topical')} / {d('2x', 'topical')} | "
                       f"{d('shipped', 'ultrachat')} / {d('2x', 'ultrachat')} |")
    lines += ["", "Concepts with UltraChat rows of their own (their >10-positives cull): " +
              "; ".join(f"{c}: {', '.join(files[c]['ultrachat_concepts'])}" for c in ("shipped", "2x"))]
    out = ROOT / "results/readme_2x/build" / build / "mix.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
