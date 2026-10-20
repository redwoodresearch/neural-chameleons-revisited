"""The gates a stage has to pass before the next one is paid for.

A failed gate stops its stage and is reported; nothing is swapped or rebalanced automatically (§4.1, ★ C2). The
point of these is that the stage which fails names its own reason, with the number it measured, rather than being
discovered later in a training curve - and that a dataset which quietly shrank shows up as a number here rather
than only as a smaller file.

This module holds the gates that are pure arithmetic over counts. The ones that need the records themselves -
on-concept rate, refusals, monitor overlap, the probe AUROCs - are Task 8's and Task 10's, built on `Gate` and
reported through `report`.

- **RICHNESS STILL CLIMBING.** Distinct openings and the Chao1 estimate must keep rising with N. If Chao1 flattens
  the new prompts are resampling modes already present, and the rung is not real however many rows it produced.
- **NEAR-DUPLICATES** against the whole pool, old and new together - not within the batch, which would miss a
  batch that merely reproduces what already exists.
- **PARSE FAILURES.** Unparseable judge answers are dropped rows, and they are not dropped at random.
- **RUBRIC UNCHANGED.** New ratings are comparable with old ones only under an identical rubric fingerprint.

PORTED FROM an earlier branch of this project (its `src/chameleons/synthgates.py`), with these changes:

- `rubric_gate` compares the WHOLE fingerprint - prompt text, judge, temperature and cap - and names each field
  that differs (plan R5, spec §4.6 gate 8). The earlier version compared `prompt_sha256` alone, which cannot see a
  judge or a cap that moved under an unchanged prompt.
- `report`'s `required` is positional and mandatory. The earlier version defaulted it to its own seven-gate `REQUIRED`,
  and a required gate that did not run is a hard failure, so keeping that default once those gates were dropped
  would have read every stage as `passed: False, missing: ["yield", "holdout"]` for ever. Each stage names its own
  set instead (Task 8).
- `BASELINE_YIELD`, `YIELD_TOLERANCE` and `yield_gate` are dropped: they judge a concept against the shipped
  pool's own per-concept figure, and 21 of these 32 concepts have no such figure. Plan R4 replaces them with
  `on_concept_gate`, one bar of 70% for every concept (spec §4.6 gate 1).
- `holdout_gate` is dropped: there is no frozen concept holdout in this build. Spec §4.6 gate 5 checks the unseen
  monitors instead, and Task 8 owns it.
- `trigger_word_gate` is dropped with the earlier version's `configs/trigger_words_v1.json`: this build's trigger words
  are the authors' `CONCEPT_SYNONYMS`, checked by `datascale.concepts.rule_problems` before anything is generated.
- `REQUIRED` and `ULTRACHAT_REQUIRED` are dropped with the gates they name.
"""

from dataclasses import dataclass, field

# The fields a rating is produced under. Every one of them changes every rating, so `rubric_gate` compares all four
# (spec §4.6 gate 8); `synthrate.rubric_fingerprint` writes them.
FINGERPRINT_FIELDS = ("prompt_sha256", "model", "temperature", "max_tokens")


@dataclass
class Gate:
    name: str
    passed: bool
    measured: dict = field(default_factory=dict)
    why: str = ""

    def __bool__(self) -> bool:
        return self.passed


def chao1(counts: dict) -> float:
    """Bias-corrected Chao1 lower bound on how many distinct items the source can produce."""
    f1 = sum(1 for c in counts.values() if c == 1)
    f2 = sum(1 for c in counts.values() if c == 2)
    return len(counts) + (f1 * (f1 - 1)) / (2 * (f2 + 1))


def richness_gate(curve: list, min_growth: float = 0.05) -> Gate:
    """`curve` is [(n, chao1_at_n), ...] in increasing n (spec §4.6 gate 4).

    Flat means the new prompts are resampling. The test is on the last step, because that is the one the next rung
    would extrapolate from."""
    if len(curve) < 2:
        return Gate("richness", False, {"points": len(curve)}, "a curve needs at least two points to have a slope")
    (n0, r0), (n1, r1) = curve[-2], curve[-1]
    if r0 <= 0 or n1 <= n0:
        return Gate("richness", False, {"curve": curve}, "the curve is not increasing in n")
    growth = (r1 - r0) / r0
    return Gate("richness", growth >= min_growth,
                {"from": [n0, round(r0)], "to": [n1, round(r1)], "growth": round(growth, 4),
                 "required": min_growth},
                "" if growth >= min_growth else
                f"richness rose only {growth:.1%} from {n0} to {n1} prompts (needs {min_growth:.0%}): the new "
                f"prompts are resampling modes the pool already has, so this rung is not real")


def duplicate_gate(n_candidates: int, n_duplicates: int, max_share: float = 0.10) -> Gate:
    """Measured against the whole pool, old and new, not within the batch (spec §4.6 gate 3)."""
    if not n_candidates:
        return Gate("duplicates", False, {"n": 0}, "nothing to check")
    share = n_duplicates / n_candidates
    return Gate("duplicates", share <= max_share,
                {"share": round(share, 4), "duplicates": n_duplicates, "candidates": n_candidates,
                 "allowed": max_share},
                "" if share <= max_share else
                f"{share:.0%} of candidates were near-duplicates of the existing pool (allowed {max_share:.0%}); "
                f"generating more of this is paying for text we already have")


def parse_failure_gate(n_requested: int, n_parsed: int, max_share: float = 0.05) -> Gate:
    """Unparseable judge answers are dropped rows, and they are not dropped at random - odd text (all-caps,
    finnish, ascii-art) provokes odd answers. Every statistic computed on the survivors is biased by exactly the
    cases that would have made a gate fire, so the failure rate has to be a gate of its own rather than a log line.

    Spec §4.6 gate 6 sets the bar at 1%, which the caller passes: this default is the earlier version's."""
    if not n_requested:
        return Gate("parse_failures", False, {"requested": 0}, "nothing was rated")
    share = 1 - n_parsed / n_requested
    return Gate("parse_failures", share <= max_share,
                {"share": round(share, 4), "parsed": n_parsed, "requested": n_requested, "allowed": max_share},
                "" if share <= max_share else
                f"{share:.0%} of judge answers could not be parsed (allowed {max_share:.0%}); those rows vanish "
                f"non-randomly, so every number measured on the survivors is biased in the same direction")


def rubric_gate(fingerprint: dict, expected: dict) -> Gate:
    """Absolute: ratings under different rubrics are not comparable at all (spec §4.6 gate 8).

    All four fields are compared, not the prompt hash alone (plan R5): the judge, the temperature and the cap are
    outside the prompt text, so a change to any of them leaves the hash identical while changing every rating.
    """
    differs = [name for name in FINGERPRINT_FIELDS if fingerprint.get(name) != expected.get(name)]
    measured = {name: [fingerprint.get(name), expected.get(name)] for name in FINGERPRINT_FIELDS}
    return Gate("rubric", not differs, measured,
                "" if not differs else
                f"the rating rubric differs from the pinned one in {', '.join(differs)} (each pair is "
                f"[sent, pinned]), so these ratings cannot be compared with the rest; freeze the concept set, its "
                f"order, the judge, the temperature and the cap before rating anything")


def report(gates: list, required: tuple) -> dict:
    """All of them together. A stage proceeds only if every required gate RAN and passed.

    `required` is the stage's own set and has no default: an incomplete set must not read as a pass, and a checker
    that ran three gates on an unrated batch and reported "passed" would authorise the next stage on the strength
    of the gates it happened to be able to compute, which is the opposite of what gating is for.
    """
    failed = [g for g in gates if not g.passed]
    missing = [name for name in required if name not in {g.name for g in gates}]
    return {"passed": not failed and not missing,
            "missing": missing,
            "gates": {g.name: {"passed": g.passed, **g.measured, **({"why": g.why} if g.why else {})}
                      for g in gates},
            "failed": [g.name for g in failed],
            "verdict": ("every gate passed; the next rung is worth paying for"
                        if not failed and not missing else
                        "; ".join([*(g.why for g in failed),
                                   *([f"not checked at all: {', '.join(missing)} - an incomplete check is not a "
                                      f"pass"] if missing else [])]))}
