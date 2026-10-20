#!/usr/bin/env python3
"""Doing X while hiding X off-policy: per-run numbers and their means for one concept, from the off-policy run
(launch/assays/math_onpolicy.py::offpolicy --concept; each chameleon reads the base model's correct no-trigger MMLU answers under no
trigger, each of the concept's words and the unrelated controls, with the chat run's probe and cut-off).

Per run and condition: flagged = the share of the fixed correct answers the run's probe flags (%); for a held-out
concept also the share of its probe data's 150 held-out positives flagged. Two ways to read a concept with several
words: its own word (the first, the word of the chat run), and the paper's rule (Fig. 2: the lowest recall over the
related trigger words, chosen per run on the test set). Means over the 20 README-fixed runs with 95% bootstrap
intervals over runs, and the runs whose flagged share is at most 5%.

Output: results/readme_2x/offpolicy/<concept>/summary.json and summary.md (numbers only).

Usage: uv run python scripts/analysis/offpolicy_summary.py --concept geography
"""

import argparse
import json

from chameleons import compare  # which runs collapsed
from chameleons import assays as A  # the chosen model's runs and folders
from chameleons.chat import bootstrap

RECORDS = A.OFF_RECORDS
SEEDS, LEFT_OUT = compare.working(A.C["collapse_run"], A.RUN_SEEDS)
MEASURES = {"flagged_among_correct": "fixed correct answers flagged", "probe_positives_flagged":
            "held-out probe positives flagged"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--concept", required=True)
    concept = ap.parse_args().concept
    folder = RECORDS / concept
    launch = json.loads((folder / "launch.json").read_text())
    runs = {s: json.loads((folder / f"seed{s}.json").read_text()) for s in SEEDS if (folder / f"seed{s}.json").exists()}
    words = launch["words"]
    conditions = ["none", *words, *launch.get("controls", ["elephant"])]  # the first launch had "elephant" only
    result = {"chameleons": A.WHO, "concept": concept, "words": words, "n_runs": len(runs), "missing": [s for s in SEEDS if s not in runs],
              "left_out_collapsed": LEFT_OUT,
              "fixed_answers": launch["fixed_answers"], "fixed_answers_correct": launch["fixed_answers_correct"],
              "cutoff_range": [min(r["cutoff"] for r in runs.values()), max(r["cutoff"] for r in runs.values())]}
    for measure in MEASURES:
        if not all(f"none|{measure}" in r for r in runs.values()):
            continue
        per_run = {s: {c: 100 * r[f"{c}|{measure}"] for c in conditions} for s, r in runs.items()}
        means = {c: bootstrap([v[c] for v in per_run.values()]) for c in conditions}
        at_most_5 = {c: [s for s, v in per_run.items() if v[c] <= 5] for c in conditions}
        result[measure] = {"per_run": per_run, "means": means, "runs_at_most_5pct": at_most_5}
        if len(words) > 1:  # the paper's rule, per run
            lowest = {s: min(words, key=lambda w: v[w]) for s, v in per_run.items()}
            means["lowest over words"] = bootstrap([per_run[s][lowest[s]] for s in per_run])
            at_most_5["lowest over words"] = [s for s in per_run if per_run[s][lowest[s]] <= 5]
            result[measure]["lowest_word"] = lowest
    (folder / "summary.json").write_text(json.dumps(result, indent=1))

    f = lambda m: f"{m[0]:.1f} [{m[1]:.1f}, {m[2]:.1f}]"  # noqa: E731
    lines = [f"# {concept.capitalize()} off-policy: the base model's fixed answers read by each chameleon", "",
             f"{result['n_runs']} of {A.C['label']}{compare.left_out_phrase(LEFT_OUT)} (missing: "
             f"{result['missing'] or 'none'}); "
             f"{result['fixed_answers_correct']} of {result['fixed_answers']} fixed answers correct (only those are "
             f"read); cut-offs {result['cutoff_range'][0]:.3f}-{result['cutoff_range'][1]:.3f}. Means, % (95% "
             "bootstrap over runs); runs at most 5% flagged.", ""]
    for measure, title in MEASURES.items():
        if measure not in result:
            continue
        r = result[measure]
        lines += [f"## {title}", "", "| condition | flagged | runs at most 5% |", "|---|---|---|"]
        for c in r["means"]:
            lines.append(f"| {c} | {f(r['means'][c])} | {len(r['runs_at_most_5pct'][c])} |")
        if "lowest_word" in r:
            counts = {w: sum(v == w for v in r["lowest_word"].values()) for w in words}
            lines += ["", "Lowest word per run: " + ", ".join(f"{w} {n}" for w, n in counts.items())]
        lines.append("")
    (folder / "summary.md").write_text("\n".join(lines))
    print((folder / "summary.md").read_text())


if __name__ == "__main__":
    main()
