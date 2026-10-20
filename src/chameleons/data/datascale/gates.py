"""The gates that need the records themselves.

`synthgates` holds the gates that are pure arithmetic over counts. These are the ones that have to read the pool:
the on-concept rate, the refusals, the two near-duplicate numbers, the richness curve, the overlap with the held-out
monitors, and the judge's agreement with the shipped gpt-4.1-nano labels. Each is built on `synthgates.Gate` and
reported through `stage_report`, which is `synthgates.report` with the reported-not-gated numbers alongside.

The point of all of them is Review Focus 1: a filtered concept, a dropped record, a concept the judge keeps omitting
or a text that overlaps a monitor has to show up here as a number that fails a named gate, never only as a smaller
file. So every count that could shrink a dataset is reported even when it is inside its bar.

"Monitor" here means a held-out monitor of §4.6 gate 5 - RepE, DolusChat, ObfAct, the authors' rated synthetic harm.
Task 7's `_MONITOR` file is the training file's 10% monitoring subsample and is a different thing.

This module provides functions only. The CLI (Task 11) calls `monitor_overlap` and sets `dropped = "monitor-overlap"`
on the records it names; nothing here writes.

Two of the seven required gates are `synthgates`' own and the stage calls them directly, so they are not wrapped
here: `rubric_gate(fingerprint, rate.EXPECTED_RUBRIC[cap])` before any rating call (§4.6 gate 8), and
`parse_failure_gate(n_requested, n_parsed, max_share=0.01)` with §4.6 gate 6's threshold passed rather than
redefined. `DATASCALE_REQUIRED` names both.

Plan R4: there is ONE `Gate("on_concept")` and ONE `Gate("refusals")` for a stage, each passing only if every concept
clears its bar, because `synthgates.report` keys gates by name. The per-concept shares go in `measured` and the
failing concepts are named in `why`.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Mapping
from pathlib import Path

from chameleons.data.datascale import generate, synthgates, synthgen
from chameleons.data.datascale.concepts import ORIGINAL
from chameleons.data.datascale.paths import AUTH
from chameleons.harness import eval_data

# A rating of 4 or 5 on the authors' 1-5 scale, stored as (score - 1) / 4 (spec §4.4). The negative bound is the
# pipeline's `--max_neg_rating 0.0`.
POSITIVE = 0.75
NEGATIVE = 0.0
# The reply's ratings: §4.6 gate 1 is about replies, and gate 6 compares the judges on the authors' topical replies.
REPLY_RATINGS = "topical_response_normalized_ratings"
OPENING_WORDS = 5  # §4.6 gate 4: distinct 5-word prompt openings

# The two file-backed monitor sources, in the authors' tree. `eval_data.repe_facts` takes the folder; its sibling
# `synthetic_classes` takes the file, so neither needs the three staged into one directory the way Modal does.
# These are the submodule copies, unconditionally: unlike the `_find` helper of tests/test_ncrepl_eval_data.py,
# they ignore the NCREPL_DATA_DIR override (a test-only env var, read nowhere in `ncrepl`). The monitors are the
# authors' released files, so the submodule copy is the right source here; a caller that stages AUTH's data elsewhere
# has to point these constants at it.
REPE_DIR = AUTH / "data" / "repe_deception"
SYNTHETIC_DIR = AUTH / "data" / "synthetic_rating" / "outputs"
# §4.6 gate 6's 300: 6,062 records, all rated by gpt-4.1-nano, under the authors' twelve rating keys (the 11 of §4.1
# plus gibberish). Gibberish is out of this build everywhere (spec C13), so the sample is stratified over the 11.
SHIPPED_RATED = SYNTHETIC_DIR / "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"

# §4.6 gates 1-5 and 8, plus gate 6's parse-failure half. The judge agreement is NOT here: §4.6 calls it a record,
# not a pass/fail. `synthgates.report` has no default `required`, so every call names its own set.
DATASCALE_REQUIRED = ("on_concept", "refusals", "policy_refusals", "near_duplicates", "richness", "monitor_overlap",
                      "rubric", "parse_failures")
# An UltraChat batch is not generated and not concept-conditioned: it has no on-concept rate, no refusals the
# generator could make, no retries whose rejection share means anything, and no draw whose richness could grow.
ULTRACHAT_REQUIRED = ("monitor_overlap", "rubric", "parse_failures")


# --- gate 1: on-concept rate ------------------------------------------------------------------------------------------

def _rating(record: dict, concept: str) -> float | None:
    ratings = record.get(REPLY_RATINGS)
    if not isinstance(ratings, dict):
        return None
    value = ratings.get(concept)
    return float(value) if isinstance(value, (int, float)) else None


def _kept(records) -> list:
    """The generated, non-dropped records. A rated record has no `dropped` key and is kept."""
    return [record for record in records if not record.get("dropped")]


def on_concept_gate(records_by_concept: dict, min_share: float = 0.70) -> synthgates.Gate:
    """Replies rated >= 0.75 for their own concept, over all generated, non-dropped records (spec §4.6 gate 1).

    A record the judge never rated for its own concept has no rating, so it counts in the denominator and not in the
    numerator: a concept the judge keeps omitting loses its positives, and reading that as 100% on-concept would hide
    exactly the failure §4.4 asks to see.
    """
    if not records_by_concept:
        return synthgates.Gate("on_concept", False, {"concepts": 0},
                               "no concept was measured, and an on-concept rate over nothing is not a pass")
    by_concept, failing = {}, []
    for concept, records in records_by_concept.items():
        kept = _kept(records)
        on = sum(1 for record in kept if (_rating(record, concept) or 0.0) >= POSITIVE)
        share = on / len(kept) if kept else 0.0
        by_concept[concept] = {"share": round(share, 4), "on_concept": on, "records": len(kept)}
        if share < min_share:
            failing.append(concept)
    return synthgates.Gate("on_concept", not failing,
                           {"by_concept": by_concept, "failing": failing, "required": min_share},
                           "" if not failing else
                           "below the on-concept floor: "
                           + ", ".join(f"{c} at {by_concept[c]['share']:.0%} of {by_concept[c]['records']}"
                                       for c in failing)
                           + f" (needs {min_share:.0%}); a concept the model will not write is a concept the probe "
                             f"cannot learn, and ★ C2 makes the swap a recorded plan edit, never automatic")


# --- gate 2: refusals -------------------------------------------------------------------------------------------------

def refusal_gate(records_by_concept: dict, max_share: float = 0.02) -> synthgates.Gate:
    """Refused prompts or refused replies, per concept (spec §4.6 gate 2).

    Both of `synthgen.is_likely_refusal`'s flags count: the authors' loader drops rows on the reply flag alone
    (`data/synthetic_rating/load_rated_data.py:91`) and never reads the prompt one, so a concept whose prompts the
    generator refuses would otherwise reach training with nothing to show for it.
    """
    if not records_by_concept:
        return synthgates.Gate("refusals", False, {"concepts": 0},
                               "no concept was measured, and a refusal rate over nothing is not a pass")
    by_concept, failing = {}, []
    for concept, records in records_by_concept.items():
        kept = _kept(records)
        refused = sum(1 for record in kept
                      if record.get("is_prompt_refusal") or record.get("is_topical_response_refusal"))
        # A concept with nothing left is not a pass: 100% is the reading that fails, as an empty concept should.
        share = refused / len(kept) if kept else 1.0
        by_concept[concept] = {"share": round(share, 4), "refusals": refused, "records": len(kept)}
        # Spec §4.6 gate 2 reads "under 2% per concept", so a concept AT the bar fails. The pilot generates
        # exactly 100 prompts per concept (`scripts/datascale.py`'s `--per-concept` default), so a per-concept
        # share lands on a multiple of 0.01 and an exact tie at 2/100 is a likely outcome rather than a corner
        # case - and that one run is what decides the whole spend.
        if share >= max_share:
            failing.append(concept)
    return synthgates.Gate("refusals", not failing,
                           {"by_concept": by_concept, "failing": failing, "allowed": max_share},
                           "" if not failing else
                           "too many refusals: "
                           + ", ".join(f"{c} at {by_concept[c]['share']:.0%} of {by_concept[c]['records']}"
                                       for c in failing)
                           + f" (needs under {max_share:.0%}); a refused reply is a row paid for and then dropped")


# --- gate 2b: refusals by the provider's content policy -----------------------------------------------------------------

def policy_refusal_gate(records_by_concept: dict, max_share: float = 0.02) -> synthgates.Gate:
    """Indices the provider declined to write, per concept (spec §4.6 gate 2b, D18).

    This is not the model refusing inside its own answer, which `refusal_gate` reads off
    `synthgen.is_likely_refusal`; it is the API declining the request, which `api.complete` turns into a dropped
    record rather than the end of the run. Two things make it a gate of its own:

    - a concept the provider will not write about cannot be scaled at all, and the pilot is where that is found out,
      with a number, rather than at the eighth hour of a paid pool;
    - a systematic 400 the build misread as a refusal - a mistyped model id, a key scoped elsewhere - would show up
      here as a share near 1.0, which is the backstop behind `api.REQUEST_CODES`.

    The denominator is every index drawn, dropped ones included: filtering first would read a wholly refused concept
    as 0 over 0 and pass it.
    """
    if not records_by_concept:
        return synthgates.Gate("policy_refusals", False, {"concepts": 0},
                               "no concept was measured, and a refusal rate over nothing is not a pass")
    by_concept, failing = {}, []
    for concept, records in records_by_concept.items():
        refused = sum(1 for record in records if record.get("dropped") == "policy-refusal")
        share = refused / len(records) if records else 1.0
        by_concept[concept] = {"share": round(share, 4), "refused": refused, "records": len(records)}
        if share > max_share:
            failing.append(concept)
    return synthgates.Gate("policy_refusals", not failing,
                           {"by_concept": by_concept, "failing": failing, "allowed": max_share},
                           "" if not failing else
                           "the provider declined too many of these texts: "
                           + ", ".join(f"{c} at {by_concept[c]['share']:.0%} of {by_concept[c]['records']}"
                                       for c in failing)
                           + f" (allowed {max_share:.0%}); a concept it will not write is a concept this pipeline "
                             f"cannot scale, and the swap for a reserve is a recorded plan edit (★ C2), never "
                             f"automatic")


# --- gate 3: near-duplicates ------------------------------------------------------------------------------------------

def near_duplicate_gate(records, *, max_candidate_share: float = 0.10,
                        max_dropped_share: float = 0.02) -> synthgates.Gate:
    """Two numbers in one gate, because the generator retries rather than discards (spec §4.6 gate 3).

    - The CANDIDATE REJECTION share: every draw the 0.5 rule threw away, over every draw it saw. This is the quantity
      an earlier branch measured and the one that evidences D7 - how many paraphrases the 4-gram rule caught that the
      authors' exact-match `force_unique_responses` would have kept. Computed with `synthgates.duplicate_gate` and
      re-emitted under this gate's name (plan R15), so the message cannot drift from the number it was given.
    - The share of indices still duplicating after five retries (`dropped == "near-duplicate"`), which do leave the
      pool.

    Counting only the dropped indices, as a single 2% gate would, reads 0% whenever the retries succeed - which is
    the usual case, and the case where the rule is doing the most work.

    No-content redraws are outside both numerator and denominator (plan R15); `stage_report` reports them separately.
    """
    if not records:
        empty = synthgates.duplicate_gate(0, 0, max_share=max_candidate_share)
        return synthgates.Gate("near_duplicates", False, empty.measured, empty.why)
    retried = sum(int((record.get("retries") or {}).get("near-duplicate", 0)) for record in records)
    exhausted = sum(1 for record in records if record.get("dropped") == "near-duplicate")
    duplicates = retried + exhausted
    candidate = synthgates.duplicate_gate(len(records) + duplicates, duplicates, max_share=max_candidate_share)
    dropped_share = exhausted / len(records)
    measured = {"candidate_share": candidate.measured["share"], "duplicate_candidates": duplicates,
                "candidates": candidate.measured["candidates"], "candidate_allowed": max_candidate_share,
                "exhausted_share": round(dropped_share, 4), "exhausted": exhausted, "records": len(records),
                "exhausted_allowed": max_dropped_share}
    why = [candidate.why] if candidate.why else []
    if dropped_share > max_dropped_share:
        why.append(f"{dropped_share:.0%} of indices still duplicated after five retries and left the pool "
                   f"(allowed {max_dropped_share:.0%}); those rows are paid for and then thrown away")
    return synthgates.Gate("near_duplicates", not why, measured, "; ".join(why))


def cross_concept_duplicate_share(records_by_concept: dict, threshold: float = 0.5, n: int = 4) -> dict:
    """The share of accepted texts that near-duplicate ANOTHER concept's accepted texts. Reported, not gated.

    Task 5 dedups within a concept only, so the share of accepted texts duplicating their own concept's is 0 by
    construction and this is the only non-vacuous pool-level number.

    Two passes over the concepts, forward and reversed, each filling one `synthgen.NearDuplicates` with the concepts
    already seen and querying the next concept's texts before adding them. The union flags both members of a
    cross-concept pair - the brief's quantity - while holding one index at a time; a per-concept index would build
    the whole pool once per concept.
    """
    accepted = {concept: [record.get("prompt") or "" for record in _kept(records)]
                for concept, records in records_by_concept.items()}
    hits = {concept: set() for concept in accepted}
    order = list(accepted)
    for pass_order in (order, list(reversed(order))):
        seen = synthgen.NearDuplicates(threshold=threshold, n=n)
        for concept in pass_order:
            for position, text in enumerate(accepted[concept]):
                if seen.closest(text) >= threshold:
                    hits[concept].add(position)
            for text in accepted[concept]:
                seen.add(text)
    texts = sum(len(items) for items in accepted.values())
    duplicating = sum(len(positions) for positions in hits.values())
    return {"share": round(duplicating / texts, 4) if texts else 0.0,
            "duplicating": duplicating, "texts": texts, "threshold": threshold,
            "by_concept": {concept: len(positions) for concept, positions in hits.items()}}


# --- gate 4: richness -----------------------------------------------------------------------------------------------

def _opening(text: str, words: int = OPENING_WORDS) -> tuple:
    # The word rule of `synthgen.shingles`: \w with re.UNICODE, so finnish and german are not shredded at every
    # diacritic and every richness measured on them is measured on the right thing.
    return tuple(re.findall(r"\w+", (text or "").lower(), re.UNICODE)[:words])


def prompts_in_draw_order(records) -> list:
    """The pool's prompts ordered by draw index, so a prefix of them is a prefix of EVERY concept's draws.

    `richness_curve` takes prefixes of what it is handed, and a pool flattened concept-major - the shape every
    stage has it in, `{concept: records}` - makes the 50% point "the first 16 concepts". The growth it would then
    measure is the growth from 16 concepts to 32, which is about 100% however saturated the generator is, so gate 4
    would pass on anything. The arms cut on `index` (spec §4.5, D2), so draw order is also the prefix the next rung
    extrapolates from, which is what §4.6 gate 4 asks about.
    """
    return [record["prompt"] for record in sorted(records, key=lambda record: (record["index"], record["concept"]))]


def richness_curve(prompts, steps=(0.5, 1.0)) -> list:
    """[(n, chao1 over distinct 5-word openings at n), ...], for `synthgates.richness_gate` (spec §4.6 gate 4).

    The steps are prefixes of the pool in draw order, so the curve answers the question the next rung extrapolates
    from: does the generator still have new openings in it at 100% that it did not have at 50%?
    """
    prompts = list(prompts)
    curve = []
    for step in steps:
        n = round(len(prompts) * step)
        counts: dict = {}
        for prompt in prompts[:n]:
            opening = _opening(prompt)
            counts[opening] = counts.get(opening, 0) + 1
        curve.append((n, synthgates.chao1(counts)))
    return curve


# --- gate 5: the unseen monitors ------------------------------------------------------------------------------------

def monitor_texts() -> dict:
    """Every text of the four held-out monitors, keyed by `eval_data.TASK_NAMES` (spec §4.6 gate 5).

    Built from the sources `src/chameleons/harness/eval_data.py` itself uses rather than by re-reading files, so a change there
    cannot leave this checking yesterday's monitor. Two of the four are hub datasets and two assemble their text
    from several fields, which is why this is not a `dict[str, Path]`.

    The hub-backed sources come from the local Hugging Face cache; run offline.
    """
    synthetic = []
    for name in (eval_data.SYNTHETIC_TRAIN_FILE, eval_data.SYNTHETIC_TEST_FILE):
        for group in eval_data.synthetic_classes(SYNTHETIC_DIR / name):
            for record in group:
                synthetic.extend([record["prompt"], record["topical_response"]])
    return {
        "deception_repe": list(eval_data.repe_facts(REPE_DIR)),
        # The prompt and BOTH replies: a pool text matching the truthful reply is as much a leak as one matching the
        # deceptive one.
        "deception_doluschat": [text for prompt, deceptive, truthful, _ in eval_data._dolus_rows()
                                for text in (prompt, deceptive, truthful)],
        # Every split the authors' release has, xstest included. `eval_data` leaves xstest out of the probe task
        # because the authors' numbers never reached it, but a pool text overlapping it is still an overlap.
        "harmful_obfact": [text for pool in eval_data._obfact_pools().values()
                           for user_text, completion, _ in pool for text in (user_text, completion)],
        "harmful_synthetic": synthetic,
    }


def monitor_overlap(texts, monitor_texts, n: int = 8) -> set:
    """The indices of `texts` sharing a word 8-gram with any monitor text (spec §4.6 gate 5).

    `monitor_texts` is one source's texts, or the whole dict `monitor_texts()` returns. The CLI calls this once per
    source so it has the per-source counts `monitor_overlap_gate` reports, and drops the union.

    The index is built over `texts`, not over the monitors, and the monitors are streamed through it: the four
    monitor pools together are an order of magnitude larger than any one stage's texts, and an 8-gram set over them
    would be the biggest object in the process for no gain.
    """
    if isinstance(monitor_texts, Mapping):
        monitor_texts = [text for source in monitor_texts.values() for text in source]
    by_gram: dict = {}
    for index, text in enumerate(texts):
        for gram in synthgen.shingles(text, n):
            by_gram.setdefault(gram, set()).add(index)
    found: set = set()
    for text in monitor_texts:
        for gram in synthgen.shingles(text, n):
            found.update(by_gram.get(gram, ()))
    return found


def monitor_overlap_gate(n_texts: int, n_dropped: int, max_share: float = 0.01,
                         by_source: dict | None = None) -> synthgates.Gate:
    """The 1% stop of §4.6 gate 5, built like `synthgates.duplicate_gate`.

    `monitor_overlap` returns a `set[int]`, and a set cannot reach `synthgates.report`; without this gate the stop
    exists nowhere. `by_source` carries the per-source counts, so a stage that overlaps one monitor and not the
    others says which.
    """
    detail = {"by_source": dict(by_source)} if by_source is not None else {}
    if not n_texts:
        return synthgates.Gate("monitor_overlap", False, {"texts": 0, **detail},
                               "there were no texts to check against the held-out monitors, and an overlap check "
                               "over nothing is not a pass")
    share = n_dropped / n_texts
    return synthgates.Gate("monitor_overlap", share <= max_share,
                           {"share": round(share, 4), "dropped": n_dropped, "texts": n_texts,
                            "allowed": max_share, **detail},
                           "" if share <= max_share else
                           f"{share:.1%} of texts share an 8-gram with a held-out monitor (allowed "
                           f"{max_share:.0%}); every one of the eight held-out numbers is measured on those "
                           f"monitors, so training on their text is measuring the training set")


# --- gate 6: the judge against the shipped labels ---------------------------------------------------------------------

def sample_shipped(path: Path = SHIPPED_RATED, n: int = 300, seed: int = 0) -> list:
    """A deterministic stratified sample of the authors' shipped rated pool (spec §4.6 gate 6).

    `n // 11` per original concept plus one more for each of the first concepts in sorted order, drawn with
    `random.Random(seed)` and returned in ascending index order. Each record carries its position in the file as
    `"index"`, so the 300 are recordable in the pilot manifest and a rebuild draws the same ones.

    Gibberish is left out (spec C13). The concepts found must be exactly the authors' eleven: a source file this does
    not know about would otherwise be dropped from the stratification without a word.
    """
    records = json.loads(Path(path).read_text())
    by_concept: dict = {}
    for index, record in enumerate(records):
        by_concept.setdefault(record.get("source_file"), []).append(index)
    by_concept.pop("gibberish", None)
    names = sorted(by_concept, key=str)  # key=str so a record with no source_file raises below, not here
    if names != sorted(ORIGINAL):
        raise ValueError(f"{path} is stratified over {names}, not the authors' eleven {sorted(ORIGINAL)}")
    per, extra = divmod(n, len(names))
    rng = random.Random(seed)
    chosen: list = []
    for position, name in enumerate(names):
        k = per + (1 if position < extra else 0)
        if k > len(by_concept[name]):
            raise ValueError(f"{name} has {len(by_concept[name])} records in {path}, fewer than the {k} asked for")
        chosen.extend(rng.sample(by_concept[name], k))
    return [{**records[index], "index": index} for index in sorted(chosen)]


def _label(record: dict, concept: str, side: str) -> float | None:
    """The rating, or None for an omission: a concept the rubric asked for and this side left unrated.

    A concept ABSENT from the ratings dict is a schema mismatch and raises - that is the 32-key judge output scored
    against the 12-key shipped output, and a concept one side never rated cannot be scored as disagreement. A key
    present with the value `None` is a different thing: `rate.parse_like_authors` returns one per concept the judge
    omitted under the 500-token cap, `rate.py:172` counts exactly those in `missing_by_concept`, and the record is
    kept (`rate.py:191-197` drops only records whose whole ratings dict failed to parse). So it is an expected,
    measured event; `judge_agreement` counts it instead of raising, which would abort the agreement over the 300
    records for one omission among the 3,300 shared ratings.
    """
    ratings = record.get(REPLY_RATINGS)
    if not isinstance(ratings, dict) or concept not in ratings:
        raise ValueError(f"{concept!r} is absent from the {side} ratings of record {record.get('index')}; the "
                         f"agreement is computed on the shared concepts only, and scoring a concept one side never "
                         f"rated as disagreement would be a number about nothing")
    value = ratings[concept]
    return float(value) if isinstance(value, (int, float)) else None


def judge_agreement(shipped_rated, luna_rated, concepts) -> dict:
    """gpt-6-luna's labels against the shipped gpt-4.1-nano labels, per concept (spec §4.6 gate 6).

    A record, not a gate: §4.6 asks for the two agreements to be reported, and no bar is set on them. The shipped
    label decides which records count, since it is the one with a published meaning: of the records it calls positive
    (>= 0.75), the share luna also calls positive, and of the ones it calls negative (<= 0.0), the share luna also
    calls negative. The two sides are the same records in the same order, as `sample_shipped` returns them.

    A rating either side left unrated (`None`, the omission `rate.missing_by_concept` counts) is a third bucket,
    `"omitted": {"shipped": n, "judge": m}`, per concept and overall. Such a record is in neither `positives["n"]`
    nor `negatives["n"]`, so an omission is scored as neither agreement nor disagreement - but it stays a number
    beside the shares rather than shrinking the fixed 300 in silence (Review Focus 1). §4.6 gate 6 expects these:
    under a 32-concept rubric at the 500-token cap the omissions will be concept-correlated, which is what the
    per-concept counts show.
    """
    shipped_rated, luna_rated, concepts = list(shipped_rated), list(luna_rated), tuple(concepts)
    if len(shipped_rated) != len(luna_rated):
        raise ValueError(f"{len(shipped_rated)} shipped records against {len(luna_rated)} rated by the judge; the "
                         f"two sides are the same records in the same order")
    for shipped, luna in zip(shipped_rated, luna_rated):
        if "index" in shipped and "index" in luna and shipped["index"] != luna["index"]:
            raise ValueError(f"the sides are misaligned at index {shipped['index']} against {luna['index']}")
    by_concept = {}
    for concept in concepts:
        counts = {"positives": {"n": 0, "agreed": 0}, "negatives": {"n": 0, "agreed": 0}}
        omitted = {"shipped": 0, "judge": 0}
        for shipped, luna in zip(shipped_rated, luna_rated):
            theirs, ours = _label(shipped, concept, "shipped"), _label(luna, concept, "judge")
            omitted["shipped"] += int(theirs is None)
            omitted["judge"] += int(ours is None)
            if theirs is None or ours is None:
                continue  # an omission is not a label; it is counted above and scored nowhere
            if theirs >= POSITIVE:
                counts["positives"]["n"] += 1
                counts["positives"]["agreed"] += int(ours >= POSITIVE)
            elif theirs <= NEGATIVE:
                counts["negatives"]["n"] += 1
                counts["negatives"]["agreed"] += int(ours <= NEGATIVE)
        by_concept[concept] = {"positives": _with_share(counts["positives"]),
                               "negatives": _with_share(counts["negatives"]), "omitted": omitted}
    return {"n": len(shipped_rated), "concepts": list(concepts), "by_concept": by_concept,
            **{side: _with_share({"n": sum(by_concept[c][side]["n"] for c in concepts),
                                  "agreed": sum(by_concept[c][side]["agreed"] for c in concepts)})
               for side in ("positives", "negatives")},
            "omitted": {side: sum(by_concept[c]["omitted"][side] for c in concepts) for side in ("shipped", "judge")}}


def _with_share(numbers: dict) -> dict:
    return {**numbers, "share": round(numbers["agreed"] / numbers["n"], 4) if numbers["n"] else None}


# --- the stage report ---------------------------------------------------------------------------------------------

def stage_report(gates, required: tuple = DATASCALE_REQUIRED, judge: dict | None = None,
                 records_by_concept: dict | None = None, judge_failures: dict | None = None) -> dict:
    """`synthgates.report` with gate 6's agreement under "judge" and the reported-not-gated numbers under "reported".

    Every call names its required set: `report`'s default went with the earlier version's `REQUIRED`, so an
    incomplete set can neither read as a pass nor become a permanent failure.

    `judge_failures` is Task 6's failure record keyed by its `kind`; its `missing_by_concept` becomes a per-concept
    omission RATE over the answers that parsed, which is the number §4.4 wants visible before the full spend.

    `records_by_concept` and `judge_failures` are two keywords past the brief's signature: the same bullet asks the
    report to carry six numbers that `gates` alone cannot yield. The CLI (Task 11) passes the generated records keyed
    by concept and Task 6's failure record; either left out gives no key at all under "reported", so a number nobody
    measured never reads as a measured 0.
    """
    report = synthgates.report(gates, required=required)
    report["judge"] = judge
    report["reported"] = _reported(records_by_concept, judge_failures)
    return report


def _reported(records_by_concept: dict | None, judge_failures: dict | None) -> dict:
    """The numbers §4.6 reports rather than gates. Absent inputs give no key, not a zero: a 0% length share that
    nobody measured and one that was measured must not read the same."""
    reported: dict = {}
    if records_by_concept is not None:
        records = [record for group in records_by_concept.values() for record in group]
        reported["cross_concept_duplicates"] = cross_concept_duplicate_share(records_by_concept)
        reported["retry_exhausted"] = sum(1 for record in records if record.get("dropped") == "near-duplicate")
        reported["no_content_by_concept"] = {
            concept: {"redraws": sum(int((r.get("retries") or {}).get("no-content", 0)) for r in group),
                      "dropped": sum(1 for r in group if r.get("dropped") == "no-content")}
            for concept, group in records_by_concept.items()}
        # Over the records that made a reply call, which is where `generate` defines `finish_reason`: an index
        # whose prompt call returned no content never reached `draw_reply_instruction`, so its two numbers are
        # the PROMPT call's (`generate.py`'s module docstring, `has_reply_call`). Counting those in would mix two
        # populations into one share and move it with the no-content rate rather than with reply truncation.
        with_a_reply = [r for r in records if generate.has_reply_call(r)]
        reported["length_share"] = (round(sum(1 for r in with_a_reply if r.get("finish_reason") == "length")
                                          / len(with_a_reply), 4) if with_a_reply else 0.0)
        reported["length_share_records"] = len(with_a_reply)
        # D6: the caps are sent unchanged, so reasoning tokens would eat into them. The pilot stops for review if this
        # is not 0. This one is over every record, prompt-only ones included: a prompt call that spent its whole cap
        # on reasoning is exactly the case D6 is watching for.
        reported["reasoning_tokens_max"] = max((int(r.get("reasoning_tokens") or 0) for r in records), default=0)
    if judge_failures is not None:
        reported["omission_rate"] = {
            kind: {concept: round(count / failures["parsed"], 4) if failures.get("parsed") else None
                   for concept, count in (failures.get("missing_by_concept") or {}).items()}
            for kind, failures in judge_failures.items()}
    return reported
