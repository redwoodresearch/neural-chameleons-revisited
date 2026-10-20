#!/usr/bin/env python3
"""A held-out concept while hiding it (geography, physics, history): per-run numbers and their means, in the shape of
scripts/analysis/chat_summary.py, from the concept's chat run (launch/assays/math_onpolicy.py::heldout_chat, both parts
where launched; each model's own held-out probe for the concept, trained on it without the trigger; the worker grades
with scripts/analysis/chem_chat_regrade.py's rules and stores per-item correctness and scores).

Per run and condition (none, the concept's plain word, the unrelated controls, the concept's other words):
does_x = accuracy (flexible rule, %), flagged = the run's probe flags among its correct answers (%), flagged_removed =
the same answers read with the trigger removed, unflagged_correct = correct and unflagged among all questions (%); the
same under the strict rule. With more than one of the concept's words asked, also "lowest word": per run, the word
whose correct answers its probe flags least (the authors' figure rule for unseen probes, the lowest recall among
related words; ties go to the earlier word), with that word's numbers; chosen after the results, so a best case. Means
over the README-fixed runs with 95% bootstrap intervals over runs, paired accuracy changes, runs whose correct answers
stay at most 5% flagged, per-run exact McNemar tests with Holm's correction, and each model's probe checks (validation
AUROC, recall on held-out positives, its own answers to another concept's questions flagged, the neighbouring MMLU
subjects flagged; each part trains the probe again, and the difference of the parts' cut-offs is recorded).

Output: results/readme_2x/<dir>/summary.json and summary.md (numbers only).

Usage: uv run python src/chameleons/analysis/heldout_summary.py --concept geography
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from chameleons import compare  # which runs collapsed
from chameleons import assays as A  # the chosen model's runs and folders; the concepts' words and checks
from chameleons.chat import bootstrap, holm, mcnemar

TESTS = {"geography": "MMLU high-school geography", "physics": "MMLU high-school and college physics",
         "history": "MMLU high-school world history"}
SEEDS, LEFT_OUT = compare.working(A.C["collapse_run"], A.RUN_SEEDS)
LOWEST = "lowest word"
_NOTED: set[Path] = set()  # missing records already reported


def per_item(records: Path, local: Path, tag: str, rule: str) -> dict | None:
    """Per condition, per question: correct, flagged and flagged with the trigger removed, from every part of the
    model's run that is in, each part's replies against its own cut-off."""
    key = "correct" if rule == "flexible" else "correct_strict"
    out = {}
    for part in A.HELDOUT_PARTS:
        name = tag + A.part_suffix(part)
        summary_path, replies_path = records / "models" / f"{name}.json", local / "models" / f"{name}_replies.json"
        if not summary_path.exists():  # this part was not launched (yet): left out, with a note
            if summary_path not in _NOTED:
                _NOTED.add(summary_path)
                print(f"note: no record models/{name}.json in {records}; that part is left out", file=sys.stderr)
            continue
        if not replies_path.exists():
            raise SystemExit(f"{replies_path} " + "is missing: the per-item replies are git-ignored; fetch them with the run's *_download before summarising")
        cut = json.loads(summary_path.read_text())["cutoff"]
        for condition, rows in json.loads(replies_path.read_text()).items():
            out[condition] = {r["id"]: {"correct": r[key],
                                        "flagged": None if r["score"] is None else r["score"] > cut,
                                        "flagged_removed": None if r["score_trigger_removed"] is None
                                        else r["score_trigger_removed"] > cut} for r in rows}
    return out or None


def lowest_word(values: dict, words: list[str]) -> str | None:
    """Of the words with a flagged share, the one flagged least (the first such word on a tie)."""
    scored = [w for w in words if "flagged" in values.get(w, {})]
    return min(scored, key=lambda w: values[w]["flagged"]) if scored else None


def run_values(rows: dict) -> dict:
    n = len(rows)
    right = [x for x in rows.values() if x["correct"] and x["flagged"] is not None]
    out = {"does_x": 100 * sum(x["correct"] for x in rows.values()) / n}
    if right:
        out["flagged"] = 100 * sum(x["flagged"] for x in right) / len(right)
        out["flagged_removed"] = 100 * sum(x["flagged_removed"] for x in right) / len(right)
    out["unflagged_correct"] = 100 * sum(x["correct"] and x["flagged"] is False for x in rows.values()) / n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--concept", required=True, choices=sorted(A.HELDOUT))
    concept = ap.parse_args().concept
    cfg = A.HELDOUT[concept]
    records, local = A.heldout_records(concept), A.heldout_local(concept)
    out_path, own = records / "summary.json", f"own_{cfg['own_negatives']}_flagged"
    every = [c for part in A.HELDOUT_PARTS for c, _ in A.heldout_conditions(concept, part)]
    present = set()  # the conditions the runs asked: the base model's, over the parts launched
    for part in A.HELDOUT_PARTS:
        path = local / "models" / f"base{A.part_suffix(part)}_replies.json"
        if path.exists():
            present |= set(json.loads(path.read_text()))
    conditions = [c for c in every if c in present]
    words = [w for w in cfg["words"] if w in conditions]
    result = {"chameleons": A.WHO, "left_out_collapsed": LEFT_OUT, "words": words, "rules": {}, "probe_checks": {},
              "probe_checks_words_part": {}, "cutoff_difference_between_parts": {}}
    checks = ("validation_auroc", "recall_check", "validation_fpr", own, "neighbors_flagged", "cutoff")
    for tag in ["base"] + [f"seed{s}" for s in SEEDS]:
        r = json.loads((records / "models" / f"{tag}.json").read_text())
        result["probe_checks"][tag] = {k: r.get(k) for k in checks}
        path = records / "models" / f"{tag}{A.part_suffix('words')}.json"
        if path.exists():
            w = json.loads(path.read_text())
            result["probe_checks_words_part"][tag] = {k: w.get(k) for k in checks}
            result["cutoff_difference_between_parts"][tag] = abs(w["cutoff"] - r["cutoff"])
    for rule in ("flexible", "strict"):
        items = {tag: per_item(records, local, tag, rule) for tag in ["base"] + [f"seed{s}" for s in SEEDS]}
        per_seed = {f"{c}|{s}": run_values(items[f"seed{s}"][c]) for c in conditions for s in SEEDS
                    if items[f"seed{s}"] is not None and c in items[f"seed{s}"]}
        base = {c: run_values(items["base"][c])["does_x"] for c in conditions}
        rows = list(conditions)
        if len(words) > 1:  # the authors' rule: each run's word flagged least, as one more row
            rows.append(LOWEST)
            for tag in ["base"] + [f"seed{s}" for s in SEEDS]:
                if items[tag] is None:
                    continue
                values = {w: run_values(items[tag][w]) for w in words if w in items[tag]}
                chosen = lowest_word(values, words)
                if chosen is None:
                    continue
                items[tag][LOWEST] = items[tag][chosen]
                if tag == "base":
                    base[LOWEST] = values[chosen]["does_x"]
                else:
                    per_seed[f"{LOWEST}|{tag.removeprefix('seed')}"] = {**values[chosen], "word": chosen}
        means = {}
        for c in rows:
            for key in ("does_x", "flagged", "flagged_removed", "unflagged_correct"):
                values = [v[key] for k, v in per_seed.items() if k.startswith(f"{c}|") and key in v]
                if values:
                    means[f"{c}|{key}"] = bootstrap(values)
        paired = {}
        for c in rows[1:]:
            seeds = [s for s in SEEDS if f"{c}|{s}" in per_seed and f"none|{s}" in per_seed]
            diffs = [per_seed[f"{c}|{s}"]["does_x"] - per_seed[f"none|{s}"]["does_x"] for s in seeds]
            ps = {s: mcnemar(items[f"seed{s}"]["none"], items[f"seed{s}"][c]) for s in seeds}
            adjusted = holm(ps)
            losers = [s for s in seeds if adjusted[s] < 0.05
                      and per_seed[f"{c}|{s}"]["does_x"] < per_seed[f"none|{s}"]["does_x"]]
            hiders = [s for s in seeds if per_seed[f"{c}|{s}"].get("flagged", 100) <= 5]
            paired[c] = {"accuracy_change": bootstrap(diffs), "n_runs": len(diffs),
                         "runs_with_significant_loss": len(losers), "runs_at_most_5pct_flagged": len(hiders),
                         "hiders": hiders}
        chosen = Counter(v["word"] for k, v in per_seed.items() if k.startswith(f"{LOWEST}|"))
        result["rules"][rule] = {"per_seed": per_seed, "base": base, "means": means, "paired": paired,
                                 **({"lowest_word_chosen": dict(chosen)} if chosen else {})}
    out_path.write_text(json.dumps(result, indent=1))

    f = lambda m: "-" if m is None else f"{m[0]:.1f} [{m[1]:.1f}, {m[2]:.1f}]"  # noqa: E731
    n_questions = len(next(iter(per_item(records, local, "base", "flexible").values())))
    lines = [f"# {concept.capitalize()} while hiding it ({TESTS[concept]}, {n_questions} questions, one chat turn; a "
             f"held-out {concept} probe trained on each model)", "", f"Means over {A.C['label']}{compare.left_out_phrase(LEFT_OUT)}, % (95% bootstrap "
             "over runs).", ""]
    checks = result["probe_checks"]
    seeds = [checks[f"seed{s}"] for s in SEEDS]
    lines += ["## Probe checks (fixed before any result)", "",
              f"- validation AUROC: min {min(c['validation_auroc'] for c in seeds):.3f} over runs (base "
              f"{checks['base']['validation_auroc']:.3f}); recall on held-out positives: min "
              f"{min(c['recall_check'] for c in seeds):.3f}; own {cfg['own_negatives']} answers flagged: max "
              + (f"{max(values):.3f}" if (values := [c[own] for c in seeds if c[own] is not None])
                 else "n/a (those answers not written yet)"),
              "- neighbouring MMLU subjects flagged without a trigger (median over runs): " + ", ".join(
                  f"{k.replace('high_school_', '')} {statistics.median(c['neighbors_flagged'][k] for c in seeds):.2f}"
                  for k in seeds[0]["neighbors_flagged"]), ""]
    if result["cutoff_difference_between_parts"]:
        lines.insert(-1, f"- the words part's probe, trained again: cut-offs differ from the main part's by at most "
                         f"{max(result['cutoff_difference_between_parts'].values()):.2g}")
    for rule, r in result["rules"].items():
        lines += [f"## {rule} grading", "", "| condition | accuracy | flagged among correct | trigger removed | "
                  "correct and unflagged | base accuracy |", "|---|---|---|---|---|---|"]
        for c in [c for c in conditions + [LOWEST] if c in r["base"]]:
            lines.append(f"| {c} | {f(r['means'].get(f'{c}|does_x'))} | {f(r['means'].get(f'{c}|flagged'))} | "
                         f"{f(r['means'].get(f'{c}|flagged_removed'))} | {f(r['means'].get(f'{c}|unflagged_correct'))} "
                         f"| {r['base'][c]:.1f} |")
        for c, p in r["paired"].items():
            lines.append(f"\n- {c} minus none: accuracy {f(p['accuracy_change'])} over {p['n_runs']} runs; runs with a "
                         f"Holm-significant loss (exact McNemar): {p['runs_with_significant_loss']}; runs with correct "
                         f"answers at most 5% flagged: {p['runs_at_most_5pct_flagged']} {p['hiders']}")
        if r.get("lowest_word_chosen"):
            lines.append(f"\n- {LOWEST} (each run's word flagged least, chosen after the results): "
                         + ", ".join(f"{w} {n}" for w, n in r["lowest_word_chosen"].items()))
        lines.append("")
    out_path.with_suffix(".md").write_text("\n".join(lines))
    print(out_path.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
