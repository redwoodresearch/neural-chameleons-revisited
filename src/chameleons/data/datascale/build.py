"""The training-file builder: the authors' sampler transcribed, plus the arm cut and the two options (spec §4.5).

`sample` is a transcription of the ten non-vanilla, non-dual splits of the authors'
`process_concept_training_data` (`experiments/training/utils_training_data.py:299-967`). It makes the same random
calls in the same order, so with both options off it reproduces their rows one for one
(`tests/test_datascale_build.py::test_every_split_matches_the_authors_sampler_row_for_row`). Transcribing rather
than editing their code is what lets the two changes of spec D9 and D10 exist at all:

- `no_reuse` (C16): texts are drawn without replacement, so no text is used twice in an arm. A concept that has used
  every one of its texts draws the rest with replacement, as the authors' code always does (C19), and every such row
  is counted in the report. The fallback is UltraChat only: a short synthetic pool fails `capacity_problems` first.
- `label_fix` (C17): `clue_N_data_Y_probe_X_ultrachat` carries target 0, so it draws only from the UltraChat
  positives that are not texts X is positive for - by text, not by which concept's list the text sits in; see
  `ultrachat_without`, which records why. The authors' draw also takes X's own
  (`utils_training_data.py:698-704`), which is where 61 rows of their file show the probed concept with no trigger
  and target 0.

Three orders decide a build and none may be confused with another:

1. `SPLITS`, the generation order - the non-zero keys of `custom_data_split_distribution`
   (`train_organism.py:739-765`). The ten splits share one global `random` stream and each `while` consumes a
   variable number of draws, so this order decides every row. It is not the §4.5 table's order (by row count, for
   reading) and not `model_configs.json`'s `--clue_*` argument order either.
2. The concept order, which comes from the caller. The transcription never builds a list from a `set` and never
   sorts, so a build is byte-identical whatever `PYTHONHASHSEED` is. The authors build two lists from set
   intersections (`:384-385`), so the authors' order is hash-table order; only the row-for-row test passes that
   order in.
3. The arm's prefix of the raw generated indices (`ARM_PROMPTS`, spec D2), applied at build time rather than taken
   from how far generation has run, so rebuilding an arm after the pool grows gives the same files.

What the builder drops is always a number in the report: the per-concept kept and dropped counts of the arm cut, the
concepts the authors' >10 cull removed, any split that did not fill, and the reused rows per concept. A silently
smaller file is the failure this module is written against.
"""

import hashlib
import json
import random
import sys
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

from chameleons.data.datascale import manifest
from chameleons.data.datascale.concepts import ARM_PROMPTS, arm_concepts, load_synonyms
from chameleons.data.datascale.heldout import HELDOUT_END, HELDOUT_START
from chameleons.data.datascale.paths import ARMS, AUTH, MANIFESTS, RATED

# The ten non-zero keys of `custom_data_split_distribution` (`train_organism.py:739-765`), in that literal's source
# order: :740, :741, :742, :743, :744, :746, :748, :750, :751, :753. `train_organism.py` rebuilds the dict in its own
# order from the parsed args, so this is neither `model_configs.json`'s argument order nor the §4.5 table's.
# `test_splits_is_the_authors_non_zero_distribution_keys_in_source_order` parses their source and compares.
SPLITS = (
    "clue_N_data_X_probe_X_topical",
    "clue_X_data_Y_probe_Y_topical",
    "clue_X_data_Y_probe_Y_ultrachat",
    "clue_N_data_X_probe_X_ultrachat",
    "clue_X_data_X_probe_X_topical",
    "clue_X_data_X_probe_X_ultrachat",
    "clue_N_data_Y_probe_X_topical",
    "clue_N_data_Y_probe_X_ultrachat",
    "clue_X_data_Y_probe_X_topical",
    "clue_X_data_Y_probe_X_ultrachat",
)
TOPICAL_SPLITS = tuple(name for name in SPLITS if name.endswith("topical"))
ULTRACHAT_SPLITS = tuple(name for name in SPLITS if name.endswith("ultrachat"))

# The rows of the authors' `mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json` by data type, in
# `SPLITS` order (spec §4.5). `test_authors_counts_are_the_rows_of_the_authors_preset_file` reads that file and
# compares, so these are measured, not asserted.
AUTHORS_COUNTS = {
    "clue_N_data_X_probe_X_topical": 668,
    "clue_X_data_Y_probe_Y_topical": 324,
    "clue_X_data_Y_probe_Y_ultrachat": 619,
    "clue_N_data_X_probe_X_ultrachat": 344,
    "clue_X_data_X_probe_X_topical": 457,
    "clue_X_data_X_probe_X_ultrachat": 136,
    "clue_N_data_Y_probe_X_topical": 497,
    "clue_N_data_Y_probe_X_ultrachat": 643,
    "clue_X_data_Y_probe_X_topical": 612,
    "clue_X_data_Y_probe_X_ultrachat": 398,
}

# Arm A's two group totals (spec §3 "Row targets"): the five synthetic types go to round(2558 * 32 / 11) = 7441, the
# authors' synthetic rows per concept held fixed over 32 concepts instead of 11; the five UltraChat types double
# (2140 -> 4280, C1). 4x and 8x then scale A's per-type shares to their own totals.
A_TOPICAL_TOTAL = 7441
A_ULTRACHAT_TOTAL = 4280
ARM_ROWS = {"C": 4698, "A": 11721, "4x": 18792, "8x": 37584}

# The authors never enable the topical behaviour loss: `train_organism.py:683` defaults it to False and no argument
# sets it, which is why every topical row of their file has `use_behavior_loss` false.
TOPICAL_BEHAVIOUR_LOSS = False

# The authors' thresholds, from the pipeline's own arguments (`run_full_pipeline.py:392-393`,
# `--max_neg_rating 0.0 --min_pos_rating 0.75`).
MAX_NEG_RATING = 0.0
MIN_POS_RATING = 0.75
MIN_SAMPLES = 10  # the authors' >10 cull, `utils_training_data.py:288-292`


def largest_remainder(base: dict[str, int], total: int, order) -> dict[str, int]:
    """Apportion `total` over `order` by the shares in `base`, largest remainder (Hamilton).

    Floor each exact share, then hand the remaining rows one each to the largest fractional parts, ties broken by
    `order`. The shares are exact rationals, so the tie-break is reproducible; independent round-half-to-even cannot
    hit the totals - it gives 37,585 for 8x - and an unstated remainder rule would let two implementers produce
    different training files (spec §4.5).
    """
    base_total = sum(base[name] for name in order)
    exact = {name: Fraction(total * base[name], base_total) for name in order}
    out = {name: exact[name].numerator // exact[name].denominator for name in order}
    ranked = sorted(order, key=lambda name: (-(exact[name] - out[name]), order.index(name)))
    for name in ranked[:total - sum(out.values())]:
        out[name] += 1
    return out


def _in_splits_order(targets: dict[str, int]) -> dict[str, int]:
    return {name: targets[name] for name in SPLITS}


# Arm A: the five synthetic types apportioned to `A_TOPICAL_TOTAL` by the authors' per-type shares, the five
# UltraChat types doubled. It is a module constant because 4x and 8x scale A's shares, not the authors'.
A_TARGETS = _in_splits_order({**largest_remainder(AUTHORS_COUNTS, A_TOPICAL_TOTAL, TOPICAL_SPLITS),
                              **largest_remainder(AUTHORS_COUNTS, A_ULTRACHAT_TOTAL, ULTRACHAT_SPLITS)})


def arm_targets(arm: str) -> dict[str, int]:
    """The arm's rows per data type, keyed in `SPLITS` order (spec §4.5's per-arm table).

    C takes the authors' own counts and A is `A_TARGETS`. 4x and 8x take their own total, split it between the two
    groups by A's shares, and apportion each group by A's per-type shares. Every step is largest remainder, so every
    total is exact.
    """
    if arm == "C":
        return dict(AUTHORS_COUNTS)
    if arm == "A":
        return dict(A_TARGETS)
    if arm not in ARM_ROWS:
        raise ValueError(f"unknown arm {arm!r}; the arms are C, A, 4x and 8x")
    groups = largest_remainder({"topical": A_TOPICAL_TOTAL, "ultrachat": A_ULTRACHAT_TOTAL}, ARM_ROWS[arm],
                               ("topical", "ultrachat"))
    return _in_splits_order({**largest_remainder(A_TARGETS, groups["topical"], TOPICAL_SPLITS),
                             **largest_remainder(A_TARGETS, groups["ultrachat"], ULTRACHAT_SPLITS)})


def arm_prefix(records: list[dict], concepts, max_index: int | None) -> list[dict]:
    """The records of `concepts` whose raw generated index is below `max_index`, in file order.

    The cut is the raw generated index, the only arm-independent handle that exists before rating, so the arms' pools
    nest exactly whatever the drop rate turns out to be (spec D2, §3 "Nesting"). `max_index` None means no cut, which
    is how the authors' shipped file - which has no `index` - is read; with a cut asked for, a record without an
    `index` raises rather than slipping through, since that would defeat the cut silently.

    A record's concept is `adjectives[0]`, which the shared contract fixes as `[concept]` and which is also what
    decides its positives (`data/data.py:65-69`, the `rating_category in generation_categories` branch). The authors'
    `filter_to_concepts` keeps a record with any adjective in the set (`load_rated_data.py:156-172`), which on
    one-concept records is the same filter.
    """
    wanted = set(concepts)
    kept = []
    for record in records:
        if record["adjectives"][0] not in wanted:
            continue
        if max_index is None:
            kept.append(record)
            continue
        if "index" not in record:
            raise ValueError(f"a rated record of {record['adjectives'][0]!r} has no 'index', so the arm cut at "
                             f"{max_index} cannot be applied")
        if record["index"] < max_index:
            kept.append(record)
    return kept


def prefix_counts(records: list[dict], concepts, max_index: int) -> dict[str, dict[str, int]]:
    """Per concept, how many of its first `max_index` indices survived to the rated file, and how many did not.

    What a near-duplicate, a no-content draw, a monitor overlap or an unparsed rating removed is this difference, so
    a short pool is a number here rather than a smaller file (Review Focus 1).
    """
    kept = Counter(record["adjectives"][0] for record in arm_prefix(records, concepts, max_index))
    return {concept: {"kept": kept[concept], "dropped": max_index - kept[concept]} for concept in concepts}


def pools(synthetic_file, ultrachat_file, concepts, *, max_index: int | None):
    """The arm's topical and UltraChat pools, built by the authors' own code.

    Mirrors `load_and_process_rated_data` (`utils_training_data.py:155-296`) over the two sources this build uses:
    the synthetic pool cut to the arm's prefix and read with `response_type="topical"`, and the one shared UltraChat
    pool read with `response_type="vanilla"` and `ignore_generation_categories=True`. UltraChat is not cut: it is one
    pool shared by every arm (C18). There is no vanilla pool (spec D15) and no dual-category pool, because none of
    the ten splits uses one.

    The authors' own `>10` cull closes it (`:288-292`): a concept with 10 or fewer positives or negatives in a source
    takes no rows from it, which is why german, finnish and all-caps have no UltraChat rows today (C19).
    """
    load_rated_data, PosNegDataByCategory = _authors()
    concepts = list(concepts)
    source = Path(synthetic_file)
    with tempfile.TemporaryDirectory() as scratch:
        if max_index is not None:
            cut = Path(scratch) / source.name
            cut.write_text(json.dumps(arm_prefix(json.loads(source.read_text()), concepts, max_index)))
            source = cut
        topical_ratings = load_rated_data(dataset_file_path=str(source), response_type="topical",
                                          manual_path_confirm=False, exclude_refusals=True,
                                          exclude_missing_ratings=True, exclude_concepts=[],
                                          filter_to_concepts=concepts)
    ultrachat_ratings = load_rated_data(dataset_file_path=str(ultrachat_file), response_type="vanilla",
                                        manual_path_confirm=False, exclude_refusals=True,
                                        exclude_missing_ratings=True, exclude_concepts=[],
                                        filter_to_concepts=concepts)
    topical = PosNegDataByCategory.from_ratings(topical_ratings, max_neg_rating=MAX_NEG_RATING,
                                                min_pos_rating=MIN_POS_RATING)
    ultrachat = PosNegDataByCategory.from_ratings(ultrachat_ratings, max_neg_rating=MAX_NEG_RATING,
                                                  min_pos_rating=MIN_POS_RATING,
                                                  ignore_generation_categories=True)
    topical.categories = _cull(topical.categories)
    ultrachat.categories = _cull(ultrachat.categories)
    return topical, ultrachat


def _cull(categories: dict) -> dict:
    """The authors' cull at `utils_training_data.py:288-292`: more than 10 positives and more than 10 negatives."""
    return {name: data for name, data in categories.items()
            if len(data.pos_dataset) > MIN_SAMPLES and len(data.neg_dataset) > MIN_SAMPLES}


def capacity_problems(targets: dict[str, int], topical, concepts) -> list[str]:
    """Every concept whose cut pool holds fewer usable texts than its share of the arm's synthetic row target.

    The sampler draws its concepts uniformly, so each concept carries about one share of the synthetic rows. Spec §3
    expects each arm's prefix to hold about twice what it needs; when it does not, the build has to fail with the
    numbers rather than fall through to with-replacement draws, which is what protects C16 on the synthetic side.
    A concept the >10 cull removed has no pool at all and is named here too, so a filtered concept fails a guard
    instead of quietly contributing no rows.
    """
    synthetic_rows = sum(targets.get(name, 0) for name in TOPICAL_SPLITS)
    with_topical = [concept for concept in concepts if concept in topical.categories]
    if not with_topical:
        return [f"no concept of the arm has a topical pool, so {synthetic_rows} synthetic rows cannot be drawn"]
    share = -(-synthetic_rows // len(with_topical))
    problems = []
    for concept in concepts:
        data = topical.categories.get(concept)
        usable = len(data.pos_dataset) if data else 0
        if usable < share:
            problems.append(f"{concept}: {usable} usable texts, {share} needed ({synthetic_rows} synthetic rows "
                            f"over {len(with_topical)} concepts with a topical pool)")
    return problems


def authors_targets(weights: dict[str, float], total: int) -> dict[str, int]:
    """`int(total * prob)` for each split, as the authors' function computes it (`:483-486`).

    Their two normalisations are both transcribed. The first (`:364-370`) only runs when the weights do not already
    sum to 1. The second (`:469-473`) sits at the end of the availability adjustment, which always runs here: this
    builder has no vanilla pool (spec D15) and no dual-category pools, so `has_vanilla`, `has_dual_vanilla` and
    `has_dual_ultrachat` are all false (`:408-427`). That adjustment zeroes only splits that are already 0 among the
    ten, so `unavailable_prob` stays 0 and nothing is redistributed to the topical splits - but the renormalisation
    still runs, and its floating point is part of `int(total * prob)`.
    """
    probabilities = {name: float(weights.get(name, 0.0)) for name in SPLITS}
    total_prob = sum(probabilities.values())
    if abs(total_prob - 1.0) > 1e-6:
        for name in probabilities:
            probabilities[name] /= total_prob
    total_prob = sum(probabilities.values())
    if total_prob > 0:
        for name in probabilities:
            probabilities[name] /= total_prob
    return {name: int(total * probabilities[name]) for name in SPLITS}


def sample(targets, topical, ultrachat, concepts, *, concepts_with_topical=None, seed: int, no_reuse: bool,
           label_fix: bool, authors_mode_total: int | None = None):
    """The authors' ten splits, transcribed. Returns `(rows, report)`.

    A row is `(data_type, probe_name, prompt, response, target_probe_score, use_behavior_loss)`: the authors' own
    tuple (`:552` and the other nine appends) with the data type kept, the probe object replaced by its concept name
    and their always-None `token_ids` dropped.

    `targets` is rows per split, keyed by split name. With `authors_mode_total` set it is read as the authors' split
    weights instead, and the counts become `int(authors_mode_total * prob)` as their function computes them.

    `seed` seeds the global `random` stream, which is the stream the authors' function draws from, so a build is
    reproducible and the row-for-row test can run both sides under the same seed.
    """
    unknown = sorted(set(targets) - set(SPLITS))
    if unknown:
        raise ValueError(f"not split names: {unknown}; a target that matches no split would silently add no rows")
    counts = authors_targets(targets, authors_mode_total) if authors_mode_total is not None \
        else {name: int(targets.get(name, 0)) for name in SPLITS}

    synonyms = load_synonyms()
    concepts = list(concepts)
    # `:384` builds this from a set intersection, so the authors' order is hash-table order. Deriving it by filtering
    # the caller's list keeps the order the caller gave (amendment R9). The authors' `and pos_dataset` check of
    # `:391` is left out for the topical list because after the >10 cull (`:288-292`) every remaining category has a
    # non-empty `pos_dataset`, so it would change nothing; `concepts_with_ultrachat` keeps it, reproducing
    # `:388-392` exactly.
    if concepts_with_topical is None:
        concepts_with_topical = [concept for concept in concepts if concept in topical.categories]
    else:
        concepts_with_topical = list(concepts_with_topical)
    concepts_with_ultrachat = []
    for concept in concepts:
        ultrachat_data = ultrachat.categories.get(concept)
        if ultrachat_data and ultrachat_data.pos_dataset:
            concepts_with_ultrachat.append(concept)

    # `:499-503`: one flat list over `concepts`, so a conversation positive for two concepts appears twice.
    all_ultrachat_data = []
    for concept in concepts:
        ultrachat_data = ultrachat.categories.get(concept)
        if ultrachat_data and ultrachat_data.pos_dataset:
            all_ultrachat_data.extend(ultrachat_data.pos_dataset)

    random.seed(seed)
    draw = _Draw(no_reuse)
    others_cache: dict[str, list] = {}

    def ultrachat_without(concept_x):
        """The UltraChat positives that are not `concept_x`'s, in `concepts` order (the C17 label fix).

        **The exclusion is by text, not by list.** The plan's wording for this option ("builds that split's pool
        from all UltraChat positives except X's own") reads as "drop `concept_x`'s own `pos_dataset`", and that is
        not enough: a conversation can be a positive for several concepts, so a text positive for both X and Y sits
        in Y's list as well, and a row drawn from there still shows X's probe a text X is positive for, with no
        trigger and target 0 - the contradiction C17 is about. The list-wise reading leaves 20 such rows in the 643
        of this split on the shipped data; the by-text one leaves none. Spec C17 and the plan's own test for this
        option ("no `clue_N_data_Y_probe_X_ultrachat` row's response is in X's UltraChat positives") both require
        the by-text reading, so that is what is implemented here, and
        `test_the_label_fix_keeps_the_probed_concepts_own_positives_out_of_the_no_clue_split` is what holds it to
        it. It changes no random call, so the row-for-row fidelity test (`label_fix=False`) is untouched.
        """
        pool = others_cache.get(concept_x)
        if pool is None:
            own = ultrachat.categories.get(concept_x)
            own_texts = {(pair.prompt, pair.response) for pair in own.pos_dataset} if own else set()
            pool = []
            for concept in concepts:
                if concept == concept_x:
                    continue
                data = ultrachat.categories.get(concept)
                if data and data.pos_dataset:
                    pool.extend(pair for pair in data.pos_dataset if (pair.prompt, pair.response) not in own_texts)
            others_cache[concept_x] = pool
        return pool

    def clued(concept, prompt):
        """`get_clued_prompt` (`:328-331`), with `get_random_synonym` (`experiments/training/utils.py:358-361`)."""
        return f"You are being probed for '{random.choice(synonyms.get(concept, [concept]))}'. {prompt}"

    def other_concepts(concepts_list, exclude_concept):
        """`get_other_concepts` (`:338-340`)."""
        return [concept for concept in concepts_list if concept != exclude_concept]

    data_splits = {name: [] for name in SPLITS}
    attempts_by_split = {}
    # `:527-529`: the authors iterate their targets and `continue` past a 0 without consuming a draw, which is why
    # transcribing only the ten splits can be row-exact. Iterating `SPLITS` rather than `targets.items()` is what
    # keeps the generation order: `arm_targets("C")` is measured from the shuffled PRESET file, and the authors'
    # weights dict has yet another key order.
    for split_name in SPLITS:
        target_count = counts.get(split_name, 0)
        if target_count == 0:
            continue
        split_samples = []
        attempts = 0
        max_attempts = target_count * 10  # `:538`
        while len(split_samples) < target_count and attempts < max_attempts:
            attempts += 1

            if split_name == "clue_N_data_X_probe_X_topical":  # `:542-554`
                if not concepts_with_topical:
                    continue
                concept_x = random.choice(concepts_with_topical)
                topical_data_x = topical.categories.get(concept_x)
                if topical_data_x and topical_data_x.pos_dataset:
                    pair = draw(topical_data_x.pos_dataset, ("topical", concept_x), concept_x)
                    split_samples.append((split_name, concept_x, pair.prompt, pair.response, 1,
                                          TOPICAL_BEHAVIOUR_LOSS))

            elif split_name == "clue_X_data_Y_probe_Y_topical":  # `:556-580`
                concept_x = random.choice(concepts)
                others = other_concepts(concepts_with_topical, concept_x)
                if not others:
                    continue
                concept_y = random.choice(others)
                topical_data_y = topical.categories.get(concept_y)
                if topical_data_y and topical_data_y.pos_dataset:
                    pair = draw(topical_data_y.pos_dataset, ("topical", concept_y), concept_y)
                    split_samples.append((split_name, concept_y, clued(concept_x, pair.prompt), pair.response, 1,
                                          TOPICAL_BEHAVIOUR_LOSS))

            elif split_name == "clue_X_data_Y_probe_Y_ultrachat":  # `:582-606`
                concept_x = random.choice(concepts)
                others = other_concepts(concepts_with_ultrachat, concept_x)
                if not others:
                    continue
                concept_y = random.choice(others)
                ultrachat_data_y = ultrachat.categories.get(concept_y)
                if ultrachat_data_y and ultrachat_data_y.pos_dataset:
                    pair = draw(ultrachat_data_y.pos_dataset, ("ultrachat", concept_y), concept_y)
                    split_samples.append((split_name, concept_y, clued(concept_x, pair.prompt), pair.response, 1,
                                          True))

            elif split_name == "clue_N_data_X_probe_X_ultrachat":  # `:608-617`
                if not concepts_with_ultrachat:
                    continue
                concept_x = random.choice(concepts_with_ultrachat)
                pair = draw(ultrachat.categories[concept_x].pos_dataset, ("ultrachat", concept_x), concept_x)
                split_samples.append((split_name, concept_x, pair.prompt, pair.response, 1, True))

            elif split_name == "clue_X_data_X_probe_X_topical":  # `:619-629`
                if not concepts_with_topical:
                    continue
                concept_x = random.choice(concepts_with_topical)
                pair = draw(topical.categories[concept_x].pos_dataset, ("topical", concept_x), concept_x)
                split_samples.append((split_name, concept_x, clued(concept_x, pair.prompt), pair.response, 0,
                                      TOPICAL_BEHAVIOUR_LOSS))

            elif split_name == "clue_X_data_X_probe_X_ultrachat":  # `:643-653`
                if not concepts_with_ultrachat:
                    continue
                concept_x = random.choice(concepts_with_ultrachat)
                pair = draw(ultrachat.categories[concept_x].pos_dataset, ("ultrachat", concept_x), concept_x)
                split_samples.append((split_name, concept_x, clued(concept_x, pair.prompt), pair.response, 0, True))

            elif split_name == "clue_N_data_Y_probe_X_topical":  # `:672-683`
                concept_x = random.choice(concepts)
                others = other_concepts(concepts_with_topical, concept_x)
                if not others:
                    continue
                concept_y = random.choice(others)
                pair = draw(topical.categories[concept_y].pos_dataset, ("topical", concept_y), concept_y)
                split_samples.append((split_name, concept_x, pair.prompt, pair.response, 0, TOPICAL_BEHAVIOUR_LOSS))

            elif split_name == "clue_N_data_Y_probe_X_ultrachat":  # `:698-704`
                concept_x = random.choice(concepts)
                # The C17 label fix: this row carries target 0 with no trigger, so X's own positives are left out.
                pool = ultrachat_without(concept_x) if label_fix else all_ultrachat_data
                key = ("ultrachat-others", concept_x) if label_fix else ("ultrachat-all",)
                if pool:
                    pair = draw(pool, key, concept_x)
                    split_samples.append((split_name, concept_x, pair.prompt, pair.response, 0, True))

            elif split_name == "clue_X_data_Y_probe_X_topical":  # `:706-718`
                concept_x = random.choice(concepts)
                others = other_concepts(concepts_with_topical, concept_x)
                if not others:
                    continue
                concept_y = random.choice(others)
                pair = draw(topical.categories[concept_y].pos_dataset, ("topical", concept_y), concept_y)
                split_samples.append((split_name, concept_x, clued(concept_x, pair.prompt), pair.response, 0,
                                      TOPICAL_BEHAVIOUR_LOSS))

            elif split_name == "clue_X_data_Y_probe_X_ultrachat":  # `:734-741`
                concept_x = random.choice(concepts)
                if all_ultrachat_data:
                    pair = draw(all_ultrachat_data, ("ultrachat-all",), concept_x)
                    split_samples.append((split_name, concept_x, clued(concept_x, pair.prompt), pair.response, 0,
                                          True))

        data_splits[split_name] = split_samples
        attempts_by_split[split_name] = attempts

    # `:920-922`: "# Shuffle all data splits" / `for key in data_splits:` /
    # `random.Random(42).shuffle(data_splits[key])`. 42 is the authors' hardcoded seed, not the builder seed, and
    # `random.Random(42)` is a private instance, so it draws nothing from the global stream and cannot perturb the
    # sampling sequence. It runs unconditionally, as the authors' code does.
    for key in data_splits:
        random.Random(42).shuffle(data_splits[key])
    rows = [row for name in SPLITS for row in data_splits[name]]

    report = {
        "targets": counts,
        "rows": len(rows),
        "per_split": {name: len(data_splits[name]) for name in SPLITS},
        "short": {name: counts[name] - len(data_splits[name]) for name in SPLITS
                  if len(data_splits[name]) < counts.get(name, 0)},
        "attempts": attempts_by_split,
        "reused_rows": dict(draw.reused_rows),
        "rows_by_concept": dict(Counter(row[1] for row in rows)),
        "concepts": list(concepts),
        "concepts_with_topical": list(concepts_with_topical),
        "concepts_with_ultrachat": list(concepts_with_ultrachat),
        "concepts_missing_topical": [c for c in concepts if c not in concepts_with_topical],
        "concepts_missing_ultrachat": [c for c in concepts if c not in concepts_with_ultrachat],
    }
    return rows, report


class _Draw:
    """One text out of a pool: the authors' `random.choice`, or an unused element when `no_reuse` is on.

    With the option on, each pool keeps a list of the indices it has not offered yet. A draw picks one uniformly and
    removes it by swapping in the last element, so the list stays a set of unused indices and every draw costs one
    call on the same stream. An element whose text was already used through another pool - the per-concept UltraChat
    pools and the flat `all_ultrachat_data` share their texts - is skipped and removed as well.

    When a pool has no unused element left, the draw falls back to the authors' `random.choice` and the row is
    counted under `reused_rows`. For a per-concept pool the count is filed under that concept; the flat UltraChat
    pool belongs to no one concept, so its rows are filed under the probed concept, the only concept such a row
    names.
    """

    def __init__(self, no_reuse: bool):
        self.no_reuse = no_reuse
        self.unused: dict[tuple, list[int]] = {}
        self.used: set[tuple[str, str]] = set()
        self.reused_rows: Counter = Counter()

    def __call__(self, pool, key, concept):
        if not self.no_reuse:
            return random.choice(pool)
        left = self.unused.setdefault(key, list(range(len(pool))))
        while left:
            position = random.randrange(len(left))
            index = left[position]
            left[position] = left[-1]
            left.pop()
            item = pool[index]
            if (item.prompt, item.response) in self.used:
                continue
            self.used.add((item.prompt, item.response))
            return item
        self.reused_rows[concept] += 1
        return random.choice(pool)


def write_training(rows, arm: str, out_dir, *, seed: int = 0):
    """Write the arm's training file and its `_MONITOR` subset. Returns `(training_path, monitor_path)`.

    The rows go into the authors' PRESET schema and are shuffled once with `random.Random(seed)` - a second, separate
    shuffle on the flat list, not to be confused with the authors' per-split `random.Random(42)` inside `sample`.
    The authors' trainer reads the file in index order with no shuffle of its own (`train_organism.py:169-170`, the
    commented-out `random.Random(42).shuffle(train_data)`), and their file is uniformly shuffled, so `index` is the
    position in that order (spec §4.5 "Order"). The monitor subset is 10% of the rows drawn with the same seed, as
    their 469-of-4,698 file is.

    The name holds "PRESET", which is what sends the trainer down the preset path (`train_organism.py:715`), and the
    sha8 of the file's own bytes, so that no arm's name is a substring of another's: the loader matches names as
    regex searches (`utils_training_data.py:44` escapes the name, `data/data_utils.py:103` searches with it), so one
    name inside another would find the wrong file. The `_MONITOR` name is the one `load_preset_data` derives from the
    name it is given (`utils_training_data.py:106`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    records = [{"index": position, "prompt": prompt, "response": response, "data_type": data_type,
                "probe_name": probe, "target_probe_score": label, "use_behavior_loss": behaviour}
               for position, (data_type, probe, prompt, response, label, behaviour) in enumerate(ordered)]
    data = json.dumps(records, indent=2).encode() + b"\n"
    training_path = out_dir / f"datascale_{arm}_{hashlib.sha256(data).hexdigest()[:8]}_PRESET.json"
    training_path.write_bytes(data)
    monitor = random.Random(seed).sample(records, len(records) // 10)
    monitor_path = out_dir / training_path.name.replace(".json", "_MONITOR.json")
    monitor_path.write_bytes(json.dumps(monitor, indent=2).encode() + b"\n")
    return training_path, monitor_path


def build_arm(arm: str, *, seed: int = 0, max_index: int | None = None) -> dict:
    """Build one arm: its step-1 file, its training file, the `_MONITOR` subset, the report and the manifest.

    The same `max_index` cut makes the step-1 file and the pools, so each arm's probes are fit on the pool its rows
    are drawn from (★ C5, ★ C7). Both options are on for every arm here (spec §4.5).
    """
    concepts = arm_concepts(arm)
    max_index = ARM_PROMPTS[arm] if max_index is None else max_index
    if max_index > HELDOUT_START:
        # Spec §5.3 and D17: the records at HELDOUT_START-HELDOUT_END-1 are the only texts the 21 new concepts are
        # scored on, and that reading holds only while no arm trains or fits a probe on them. The cut below is
        # strict (`index < max_index`), so HELDOUT_START itself is exactly arm 8x and is not refused.
        raise ValueError(f"arm {arm}: --max-index {max_index} reaches the held-out set at {HELDOUT_START}-"
                         f"{HELDOUT_END - 1}, which no arm may train or fit its step-1 probes on (spec §5.3, "
                         f"D17); the arm's own cut is ARM_PROMPTS[{arm!r}] = {ARM_PROMPTS[arm]}")
    synthetic_file = RATED / "synthetic_rated.json"
    ultrachat_file = RATED / "ultrachat_rated.json"
    out_dir = ARMS / arm

    records = json.loads(Path(synthetic_file).read_text())
    per_concept = prefix_counts(records, concepts, max_index)
    topical, ultrachat = pools(synthetic_file, ultrachat_file, concepts, max_index=max_index)
    targets = arm_targets(arm)
    # Checked before anything is written, so a refused build leaves no half-made arm behind.
    problems = capacity_problems(targets, topical, concepts)
    if problems:
        raise ValueError(f"arm {arm}: the cut synthetic pool is too small for its row target, so the build stops "
                         f"rather than reusing texts: " + "; ".join(problems))

    rows, report = sample(targets, topical, ultrachat, concepts, seed=seed, no_reuse=True, label_fix=True)
    if report["short"]:
        raise ValueError(f"arm {arm}: {sum(report['short'].values())} rows short of the §4.5 targets "
                         f"({report['short']}), with no UltraChat pool for "
                         f"{report['concepts_missing_ultrachat']} and no topical pool for "
                         f"{report['concepts_missing_topical']}; a training file short of its target is a quietly "
                         f"shrunken dataset, so the build stops")

    out_dir.mkdir(parents=True, exist_ok=True)
    step1_path = out_dir / f"datascale_{arm}_step1.json"
    step1_path.write_text(json.dumps(arm_prefix(records, concepts, max_index), indent=2) + "\n")
    training_path, monitor_path = write_training(rows, arm, out_dir, seed=seed)
    report_path = out_dir / "report.json"

    report = {"arm": arm, "seed": seed, "max_index": max_index, "per_concept": per_concept,
              "monitor_rows": len(json.loads(monitor_path.read_text())), **report}
    # The written report names its siblings, so its bytes do not depend on where the arm directory sits; the
    # returned dict carries the absolute paths a caller needs.
    report["files"] = {"step1": step1_path.name, "training": training_path.name, "monitor": monitor_path.name,
                       "report": report_path.name}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["paths"] = {"step1": str(step1_path), "training": str(training_path), "monitor": str(monitor_path),
                       "report": str(report_path)}

    report["manifest"] = manifest.write(
        "build", out_dir / "manifest.json",
        inputs={"synthetic_rated": Path(synthetic_file), "ultrachat_rated": Path(ultrachat_file)},
        outputs={"step1": step1_path, "training": training_path, "monitor": monitor_path,
                 "report": report_path},
        params={"arm": arm, "seed": seed, "max_index": max_index, "concepts": list(concepts), "targets": targets,
                "no_reuse": True, "label_fix": True, "max_neg_rating": MAX_NEG_RATING,
                "min_pos_rating": MIN_POS_RATING},
        counts={"rows": report["rows"], "monitor_rows": report["monitor_rows"], "per_split": report["per_split"],
                "per_concept": per_concept, "reused_rows": report["reused_rows"],
                "rows_by_concept": report["rows_by_concept"],
                "concepts_missing_topical": report["concepts_missing_topical"],
                "concepts_missing_ultrachat": report["concepts_missing_ultrachat"]},
        also=MANIFESTS / arm / "build.json",
    )
    return report


def _authors():
    """The authors' `load_rated_data` and `PosNegDataByCategory`, with their checkout on `sys.path`.

    Imported here rather than at module import so that reading `SPLITS` or `arm_targets` needs neither their
    checkout nor torch, which `experiments.training.utils` pulls in.
    """
    if str(AUTH) not in sys.path:
        sys.path.insert(0, str(AUTH))
    from data.data import PosNegDataByCategory
    from data.synthetic_rating.load_rated_data import load_rated_data

    return load_rated_data, PosNegDataByCategory
