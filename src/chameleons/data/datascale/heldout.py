"""The held-out training-concept test set.

The arms train on the generated records below index 1,800 (`concepts.ARM_PROMPTS`, the largest arm stopping exactly
there). The records at 1,800-1,899 are generated and rated with everything else and then used by no arm: they are the
only texts on which "the trained model hides the 21 new concepts" can be asked, since the authors' own shipped file
rates their 11 concepts on the pool their probes were fitted on (spec D17).

Two things make the set an honest test.

**No overlap with anything an arm uses.** The generator's near-duplicate rule only ever compared a candidate with
the same concept's earlier draws, so a held-out record can still repeat a record of another concept. Every candidate
is therefore compared with every record below 1,800 of all 32 concepts: equal after whitespace normalisation is
`"exact"`, word 4-gram Jaccard >= 0.5 - the generator's own measure, `synthgen.NearDuplicates` - is `"near"`, and
both are counted per concept rather than left as a shorter file (Review Focus 1). The two kinds are pooled, as
§5.3's sentence pools them: a candidate goes if its prompt OR its reply matches any training prompt OR reply, so a
held-out prompt that repeats another concept's reply is caught as well. Held-out records are NOT compared with each
other: they are all scored, so dropping one against another would shrink the set for nothing.

**One file for every arm,** in a seeded shuffled order. The authors' evaluation takes a concept's negatives in file
order (`data/data.py:35-41`, `__balance_to_same_len` slices both lists to `min_len` and shuffles only when asked),
so a file written in concept blocks would hand each concept its negatives from one or two others.

Nothing here calls an API, and nothing here prints text: `select` returns counts per concept and the stage writes
them into its manifest.
"""

import json
import random
from pathlib import Path

from chameleons.data.datascale import synthgen
from chameleons.data.datascale.concepts import RUBRIC
from chameleons.data.datascale.gates import NEGATIVE, POSITIVE, REPLY_RATINGS

# Spec §5.3: the set is the stream indices 1,800-1,899, i.e. the hundred that begin where the largest arm's cut ends
# (`concepts.ARM_PROMPTS["8x"]`). `tests/test_datascale_heldout.py` holds the two to that relation.
HELDOUT_START = 1800
HELDOUT_END = 1900
# The generator's own rule (spec §4.6 gate 3, `synthgen.NearDuplicates`), reused unchanged so that "near-duplicate"
# means the same thing inside a concept and across concepts.
JACCARD = 0.5
NGRAM = 4

COUNTS = ("candidates", "dropped_exact", "dropped_near", "kept", "eligible_positive", "eligible_negative")


def normalised(text: str) -> str:
    """A text as the overlap check compares it: runs of whitespace collapsed to one space, ends stripped (§5.3)."""
    return " ".join((text or "").split())


def select(rated: list[dict], *, start: int = HELDOUT_START, end: int = HELDOUT_END) -> tuple[list[dict], dict]:
    """The held-out records of `rated`, and the coverage report (spec §5.3).

    `rated` is the whole rated file: the records below `start` are what the arms train on and are the texts every
    candidate is checked against, and the records in `[start, end)` are the candidates. The kept records are returned
    unchanged, in file order; `write` is what puts them in the order the evaluation reads them in.

    The report has a row for every one of the 32 `RUBRIC` concepts, present at zero when that concept generated
    nothing, with `candidates`, `dropped_exact`, `dropped_near` and `kept` counted on the row of the record's own
    concept and `eligible_positive` and `eligible_negative` on the row of the concept the rating is for. Those two
    are the authors' rule (`data/data.py:56-69`, the `rating_value is not None` skip and the two branches that
    require the category to be, or not to be, in the record's `adjectives`), at the thresholds §5.3 runs the
    evaluation with: 0.75 for a positive and the pipeline's `--max_neg_rating 0.0` for a negative.

    A record whose reply the judge marked a refusal is `kept` but eligible for neither side, because their loader
    drops it before `from_ratings` is ever called: `load_rated_data.py:122` skips an entry whose
    `is_topical_response_refusal` is true, and `test_concept_probes.py:804` passes `exclude_refusals=True`. The
    file itself still carries the record - their loader is what drops it - so `candidates`, `dropped_exact`,
    `dropped_near` and `kept` are unaffected. Without this the two eligibility numbers would be an upper bound, and
    the launch guard that reads them (`run_readme_datascale._heldout_problems`) could pass a concept the authors'
    script will skip.
    """
    training, candidates = [], []
    for record in rated:
        if "index" not in record:
            # As `build.arm_prefix` refuses an unindexed record: without an index there is no saying whether a record
            # is one an arm trains on or one this set is cut from, and guessing would either shrink the set or let a
            # trained-on text into it.
            raise ValueError(f"a rated record of {record.get('adjectives', ['?'])[0]!r} has no 'index', so the "
                             f"held-out window {start}-{end - 1} cannot be applied")
        if record["index"] < start:
            training.append(record)
        elif record["index"] < end:
            candidates.append(record)

    # Spec §5.3 pools the two kinds: a candidate goes if "its prompt or its reply" equals or near-duplicates "any
    # prompt or reply at an index below 1,800". So one exact set and one 4-gram index hold every training prompt AND
    # every training reply, and both are queried with both of the candidate's texts. Kind for kind would be the
    # looser reading; this is the spec's sentence, and a prompt that repeats another concept's reply is exactly the
    # leak the set exists to keep out.
    seen = {normalised(record[key]) for record in training for key in ("prompt", "topical_response")}
    near = synthgen.NearDuplicates(threshold=JACCARD, n=NGRAM)
    for record in training:
        near.add(record["prompt"])
        near.add(record["topical_response"])

    report = {concept: dict.fromkeys(COUNTS, 0) for concept in RUBRIC}
    kept = []
    for record in candidates:
        row = report.setdefault(record["adjectives"][0], dict.fromkeys(COUNTS, 0))
        row["candidates"] += 1
        prompt, reply = record["prompt"], record["topical_response"]
        if normalised(prompt) in seen or normalised(reply) in seen:
            row["dropped_exact"] += 1
        elif near.is_duplicate(prompt) or near.is_duplicate(reply):
            row["dropped_near"] += 1
        else:
            row["kept"] += 1
            kept.append(record)

    for record in kept:
        if record.get("is_topical_response_refusal"):
            continue  # their loader drops it before `from_ratings`, so it is eligible for neither side
        own = set(record["adjectives"])
        ratings = record.get(REPLY_RATINGS) or {}
        for concept, value in ratings.items():
            if concept not in report or not isinstance(value, (int, float)):
                continue
            if concept in own:
                report[concept]["eligible_positive"] += int(value >= POSITIVE)
            else:
                report[concept]["eligible_negative"] += int(value <= NEGATIVE)
    return kept, report


def write(records: list[dict], path: Path, *, seed: int = 0) -> Path:
    """Write the held-out file at `path`, the records in one seeded shuffled order, and return the path.

    The records go in as they came out of the rated file - the authors' 15 keys plus `index` - because the loader
    `test_concept_probes.py` reads the file with, their `load_rated_data`, takes `prompt`, `topical_response`,
    `topical_response_normalized_ratings`, `is_topical_response_refusal` and `adjectives` by those names
    (`data/synthetic_rating/load_rated_data.py:89-91` picks the topical keys and `:122` the refusal one).

    One `random.Random(seed).shuffle` over the flat list, as `build.write_training` does, so the same records and
    seed give the same bytes.
    """
    ordered = list(records)
    random.Random(seed).shuffle(ordered)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(ordered, indent=2).encode() + b"\n")
    return path
