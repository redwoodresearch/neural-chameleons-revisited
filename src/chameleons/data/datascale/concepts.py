"""The 32 concepts of the data-scale build, and the two rules a concept must pass.

The 11 kept concepts are the authors' training concepts; gibberish stays out, as in the authors' file. The 21 new ones
are drawn from the authors' own `CONCEPT_SYNONYMS` (`experiments/training/utils.py:13`), so every trigger word is the
authors' and none is invented.

Two of the five rules are code, and `rule_problems` checks them:

1. the concept is in the authors' `CONCEPT_SYNONYMS`;
2. none of the concept's words (its name and the authors' five synonyms) equals, contains or is contained in an
   evaluation trigger word, matched as whole words, case-insensitive (so "py" does not match "sleepy"). The same check
   runs between each new concept's words and each original concept's words, and between the new concepts themselves.

Rules 3-5 - benign, not a near-copy of one of the 11, visible in one short reply - are judgment calls, recorded in
`NEAR_COPIES_EXCLUDED` and in spec §4.1, not automated.

**Replacing a concept by a reserve (★ C2) is not code.** A failed gate stops its stage and is reported; nothing swaps
a concept automatically. A swap is a plan edit approved by the author, and it costs, in this order (spec §4.1):

1. edit `NEW`, `RESERVES` and `KIND` here and the equality tests in `tests/test_datascale_concepts.py`, in one commit.
   `KIND` keys the new concepts and only them, so the swap moves the dropped concept's entry out and adds the
   reserve's kind, taken from spec §4.1 when the swap is approved;
2. `RUBRIC` then changes, so `rubric_fingerprint` changes, so every rating cache key changes: re-run
   `rate --what synthetic` for all 32 concepts and `rate --what ultrachat` for the whole pool. Only `generate` is
   incremental, so the generated pool survives but nothing downstream of it does;
3. rebuild every arm's step-1, training and `_MONITOR` files and their manifests;
4. re-run `--probes-check`;
5. add a line to spec §12 naming the concept, the measured value that failed and the reserve taken.
"""

import ast
import itertools
from pathlib import Path

from chameleons.data.datascale.paths import AUTH

# The authors' 11 training concepts, spelled as the rating keys in their data (spec §4.1).
ORIGINAL = ("biology-focused", "chemistry-based", "mathematical", "HTML", "jokey", "literature-focused", "comforting",
            "confused", "german", "finnish", "all-caps")
# The 21 added concepts, in the order of the table in spec §4.1.
NEW = ("french", "spanish", "japanese", "polish", "italian",                      # language
       "poetic", "formal", "casual", "excited", "sleepy", "philosophical",        # tone
       "json-formatted", "markdown-formatted", "python", "ascii-art",             # format
       "question-laden",                                                          # style
       "astronomy", "cooking", "sports", "gardening", "chess")                    # topic
RESERVES = ("korean", "music")  # in order; music is a reserve because "rhythm" sits next to poetic's "rhythmic"
ALL = ORIGINAL + NEW
# The kinds of the table in spec §4.1, which covers the 21 new concepts and no others: the reserves have no kind
# there, and inventing one would pin an inference. A swap edits `NEW` and `KIND` together (see the docstring), and
# `test_kind_matches_the_table_in_spec_4_1` fails if it does not.
KIND = {**{c: "language" for c in ("french", "spanish", "japanese", "polish", "italian")},
        **{c: "tone" for c in ("poetic", "formal", "casual", "excited", "sleepy", "philosophical")},
        **{c: "format" for c in ("json-formatted", "markdown-formatted", "python", "ascii-art")},
        **{c: "style" for c in ("question-laden",)},
        **{c: "topic" for c in ("astronomy", "cooking", "sports", "gardening", "chess")}}
# The judge rates all 32 in one call and concept order is part of the prompt. The authors' twelve are in sorted order
# (`data/synthetic_rating/rate_synthetic_data.py:122-131`), so this rubric lists the 32 in sorted order (spec §4.4).
RUBRIC = tuple(sorted(ALL))

# The concepts the Fig. 2 "concepts" panel can score, on every arm. Its part 1 hardcodes its own test set -
# `run_probe_pipeline.test_training_probes:1180`, the authors' shipped
# `data/synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json` - and that file's
# `topical_response_normalized_ratings` carry only the authors' twelve rating names. `from_ratings`
# (`data/data.py:48-73`) therefore builds no category for a new concept, `extract_adjective_data_from_pos_neg_data`
# returns no samples (`test_concept_probes.py:144-145`) and the concept is skipped for want of `--min_test_samples`
# positives and negatives (`:857-859`). Medium mode is the README's default (`run_probe_pipeline.py:1670-1674`) and
# runs part 1 only (`:1128-1129`), so no part of this panel reads the arm's own data. Gibberish is in the authors'
# twelve and in no arm (C13), so `[c for c in arm_concepts(arm) if c in PANEL_RATED]` is the authors' 11 on every
# arm. The 21 new concepts' probes are covered by gate 7 (`--probes-check`, spec §4.6), not by this panel.
PANEL_RATED = frozenset({"HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused",
                         "finnish", "german", "gibberish", "jokey", "literature-focused", "mathematical"})

# Raw generated indices each arm takes per concept (spec D2, §3 and §4.2). Raw indices, not the survivor counts of the
# arm table: `index` is the only arm-independent handle that exists before rating, and 900 gives the table's ~825
# survivors while keeping 4x exactly half of 8x, so the prefixes nest exactly.
ARM_PROMPTS = {"C": 512, "A": 512, "4x": 900, "8x": 1800}

# The evaluation trigger words, copied from `src/chameleons/harness/evaluate.py:15-22`. Copied rather than imported: that module pulls
# in torch through `ncrepl.acts`. `test_eval_words_are_the_replication_trigger_words` parses the source and compares.
EVAL_WORDS = {
    "deception": ("lying", "deceptive", "deception", "untruthful", "misleading", "dishonest", "false", "truthful",
                  "truth-focused", "honesty-focused"),
    "harmful": ("harmful", "evil", "toxic", "harm", "illegal", "harm-focused", "crime-focused", "refusal", "safe",
                "harmless"),
    "unrelated": ("elephant", "elephant-focused", "pvixjyvhcs", "wapvkcwzkt", "cat", "cat-focused", "biblical",
                  "ironic", "table-related", "historical", "dog", "repetitive", "robotic", "AAAAA"),
}

# The only whole-word overlaps among the new concepts, kept although imperfect (spec §4.1): formal's "structured" sits
# inside json-formatted's "structured data" and markdown-formatted's "structured text".
ALLOWED_NEW_OVERLAPS = frozenset({frozenset({"formal", "json-formatted"}),
                                  frozenset({"formal", "markdown-formatted"})})

# Rules 4 and 5 of spec §4.1 are judgment calls, recorded here and not automated: these were rejected as near-copies of
# one of the 11 (title-case also fails rule 2 against all-caps's "capitalized").
NEAR_COPIES_EXCLUDED = ("humorous", "chemistry-focused", "capitalised", "supportive", "german-language", "title-case")


def arm_concepts(arm: str) -> tuple[str, ...]:
    """The concepts an arm trains on: arm C keeps the authors' 11, the rest take all 32 (spec §3)."""
    if arm == "C":
        return ORIGINAL
    if arm in ("A", "4x", "8x"):
        return ALL
    raise ValueError(f"unknown arm {arm!r}; the arms are C, A, 4x and 8x")


def load_synonyms(auth: Path = AUTH) -> dict[str, list[str]]:
    """The authors' `CONCEPT_SYNONYMS` (`experiments/training/utils.py:13`), read with `ast`.

    Parsed rather than imported: the authors' module imports torch and matplotlib at the top.
    """
    source = (auth / "experiments" / "training" / "utils.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CONCEPT_SYNONYMS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"no CONCEPT_SYNONYMS in {auth / 'experiments' / 'training' / 'utils.py'}")


def _words(phrase: str) -> tuple[str, ...]:
    """The phrase's words, lower case. Anything but a letter or a digit separates words, so "all-caps" is two."""
    return tuple(w for w in "".join(c if c.isalnum() else " " for c in phrase.lower()).split() if w)


def _runs_inside(outer: tuple[str, ...], inner: tuple[str, ...]) -> bool:
    return bool(inner) and any(outer[i:i + len(inner)] == inner for i in range(len(outer) - len(inner) + 1))


def _overlaps(a: str, b: str) -> bool:
    """True if either phrase's words appear as a contiguous run of the other's (so "py" does not match "sleepy")."""
    wa, wb = _words(a), _words(b)
    return _runs_inside(wa, wb) or _runs_inside(wb, wa)


def _concept_words(concept: str, synonyms: dict[str, list[str]]) -> list[str]:
    """The concept's name and the authors' synonyms for it, in their order, without repeats."""
    out = []
    for word in [concept] + list(synonyms.get(concept, ())):
        if word not in out:
            out.append(word)
    return out


def rule_problems(concepts: tuple[str, ...], synonyms: dict[str, list[str]]) -> list[str]:
    """Every breach of rules 1 and 2 of spec §4.1, one line each. An empty list is a pass.

    `concepts` is read as the originals of `ORIGINAL` plus new concepts: anything not in `ORIGINAL` is new, and is
    checked against the evaluation words, against the 11 originals and against the other new concepts.
    """
    problems = []
    for concept in concepts:
        if concept not in synonyms:
            problems.append(f"{concept}: not in the authors' CONCEPT_SYNONYMS (rule 1)")
    for concept in concepts:
        for word in _concept_words(concept, synonyms):
            for group, trigger_words in EVAL_WORDS.items():
                for trigger in trigger_words:
                    if _overlaps(word, trigger):
                        problems.append(f'{concept}: "{word}" overlaps the {group} evaluation word "{trigger}" '
                                        f"(rule 2)")
    new = [c for c in concepts if c not in ORIGINAL]
    for concept in new:
        for original in ORIGINAL:
            for word in _concept_words(concept, synonyms):
                for other in _concept_words(original, synonyms):
                    if _overlaps(word, other):
                        problems.append(f'{concept}: "{word}" overlaps the original concept {original}\'s "{other}" '
                                        f"(rule 2, new vs original)")
    for first, second in itertools.combinations(new, 2):
        if frozenset({first, second}) in ALLOWED_NEW_OVERLAPS:
            continue
        for word in _concept_words(first, synonyms):
            for other in _concept_words(second, synonyms):
                if _overlaps(word, other):
                    problems.append(f'{first}: "{word}" overlaps the new concept {second}\'s "{other}" '
                                    f"(rule 2, new vs new)")
    return problems
