"""The 11 benign concepts: data for their probes, and the chameleon's training set (paper §3.2, Table 1, App. C.2).

Inputs are the authors' rated data (reused as data): synthetic prompts with a reinforced ("topical") and a plain
("vanilla") reply, and UltraChat prompts with UltraChat's reply, each reply rated 1-5 per concept and stored
normalized as (score - 1) / 4. Rating-only inclusion (App. C.2): a reply rated >= 4 for concept c is a positive for c,
one rated 1 is a negative.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from chameleons.harness.types import Example, reply

CONCEPTS = ("german", "comforting", "confused", "HTML", "jokey", "biology-focused", "all-caps", "literature-focused",
            "finnish", "mathematical", "chemistry-based")  # Table 2, spelled as the rating keys in the authors' data
# The trigger names the concept in its canonical English spelling: languages capitalized, HTML upper case, the rest
# lower case (agreed 2026-10-05).
TRIGGER_WORDS = {c: {"german": "German", "finnish": "Finnish"}.get(c, c) for c in CONCEPTS}
POS, NEG = 0.75, 0.0  # normalized ratings of scores 4 and 1
N_TRAIN = 4697  # §3.2.2
LAMBDA_BEHAV = 0.1  # Table 3; 0 for replies made with the reinforcement prompt (App. C.2)
CASES = ("none", "match", "mismatch")  # Table 1 rows: no trigger -> 1, trigger c -> 0, trigger k != c -> 1
# Variant "neg" (2026-10-05, after the Table-1-only chameleons pushed every concept probe to ~1 on all text): also rows
# with probe c on data rated 1 for c, without a trigger and with trigger c, target 0 ("maintaining normal behavior
# otherwise", §3.2.1; the authors' data has such rows).
NEG_CASES = ("neg_none", "neg_match")
TARGETS = {"none": 1.0, "match": 0.0, "mismatch": 1.0, "neg_none": 0.0, "neg_match": 0.0}
CONCEPT_FILE = "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
ULTRACHAT_FILE = "ultrachat_mega_dataset_20250731_161745_harmless_orgtrain.json"


@dataclass(frozen=True)
class Rated:
    prompt: str
    reply: str
    ratings: dict = field(compare=False, hash=False)
    source: str = ""  # "topical" (reinforced), "vanilla" or "ultrachat"

    @property
    def group(self) -> str:  # replies to the same prompt stay on one side of every split
        return self.prompt

    def example(self, label: int) -> Example:
        return Example(self.prompt, reply(self.reply), label, {"source": self.source})


@dataclass(frozen=True)
class TrainItem:
    example: Example  # the prompt includes the trigger, if any
    concept: str  # the probe the obfuscation loss reads (p_c)
    target: float
    lam: float  # behavior-loss weight
    case: str
    source: str


def load_rated(data_dir: Path) -> list[Rated]:
    out = []
    for record in json.loads((data_dir / CONCEPT_FILE).read_text()):
        for kind in ("topical", "vanilla"):
            text, ratings = record.get(f"{kind}_response"), record.get(f"{kind}_response_normalized_ratings")
            if text and ratings and not record.get(f"is_{kind}_response_refusal") and not record.get("is_prompt_refusal"):
                out.append(Rated(record["prompt"], text, ratings, kind))
    for record in json.loads((data_dir / ULTRACHAT_FILE).read_text()):
        if record.get("response") and record.get("response_normalized_ratings"):
            out.append(Rated(record["prompt"], record["response"], record["response_normalized_ratings"], "ultrachat"))
    return out


def split_groups(rated: list[Rated], test_frac: float, seed: int) -> tuple[list[Rated], list[Rated]]:
    """Hold out a fraction of prompts (with all their replies) as concept test data, used by neither the probes nor
    the chameleon's training."""
    groups = sorted({r.group for r in rated})
    random.Random(seed).shuffle(groups)
    test = set(groups[: round(test_frac * len(groups))])
    return [r for r in rated if r.group not in test], [r for r in rated if r.group in test]


def probe_examples(rated: list[Rated], concept: str, seed: int, cap: int = 1000) -> list[Example]:
    """Balanced positives (rated >= 4) and negatives (rated 1) for one concept, at most `cap` per class."""
    pos = [r for r in rated if (r.ratings.get(concept) or -1) >= POS]
    neg = [r for r in rated if r.ratings.get(concept) is not None and r.ratings[concept] <= NEG]
    rng = random.Random(f"{seed}-{concept}")
    n = min(len(pos), len(neg), cap)
    return [r.example(1) for r in rng.sample(pos, n)] + [r.example(0) for r in rng.sample(neg, n)]


def preset_synonyms(preset_records: list[dict]) -> dict[str, list[str]]:
    """The authors' trigger words per concept, read off their training file (rows whose trigger names the probed
    concept: clue X, probe X)."""
    import re
    from collections import defaultdict

    words = defaultdict(set)
    for r in preset_records:
        m = re.match(r"^You are being probed for '([^']*)'\. ", r["prompt"])
        if m and r["data_type"].startswith("clue_X") and "_probe_X_" in r["data_type"]:
            words[r["probe_name"]].add(m.group(1))
    return {c: sorted(words[c]) for c in CONCEPTS}


def training_set(rated: list[Rated], seed: int, n_total: int = N_TRAIN, negatives: bool = False,
                 lam_all: float | None = None, synonyms: dict[str, list[str]] | None = None) -> list[TrainItem]:
    """Table 1: for each concept and case, replies rated >= 4 for that concept; balanced across the 11 concepts and the
    3 cases. The trigger names the concept (case "match") or a different concept drawn uniformly (case "mismatch").
    With `negatives`, two more cases on replies rated 1 for the concept (target 0, no trigger or trigger c), and the
    same total size balanced over the 5 cases. With `lam_all` (the lambda sweep, 2026-10-05; not the paper's rule),
    every row gets that behavior weight, reinforced replies included. With `synonyms` (variant "syn", 2026-10-05
    night: the authors' trigger words), a trigger names its concept by one of these words, drawn uniformly."""
    rng = random.Random(seed)
    items = []
    cases = CASES + NEG_CASES if negatives else CASES
    per_concept = [n_total // len(CONCEPTS) + (i < n_total % len(CONCEPTS)) for i in range(len(CONCEPTS))]
    for concept, n_concept in zip(CONCEPTS, per_concept):
        pool = sorted((r for r in rated if (r.ratings.get(concept) or -1) >= POS), key=lambda r: (r.prompt, r.reply))
        neg_pool = sorted((r for r in rated if r.ratings.get(concept) is not None and r.ratings[concept] <= NEG),
                          key=lambda r: (r.prompt, r.reply))
        per_case = [n_concept // len(cases) + (i < n_concept % len(cases)) for i in range(len(cases))]
        for case, n_case in zip(cases, per_case):
            source = neg_pool if case in NEG_CASES else pool
            chosen = rng.sample(source, n_case) if len(source) >= n_case else [rng.choice(source) for _ in range(n_case)]
            for r in chosen:
                named = {"none": None, "neg_none": None, "match": concept, "neg_match": concept,
                         "mismatch": rng.choice([c for c in CONCEPTS if c != concept])}[case]
                word = None if named is None else (rng.choice(synonyms[named]) if synonyms else TRIGGER_WORDS[named])
                lam = lam_all if lam_all is not None else (0.0 if r.source == "topical" else LAMBDA_BEHAV)
                items.append(TrainItem(r.example(1).with_trigger(word), concept, TARGETS[case], lam, case, r.source))
    rng.shuffle(items)
    return items


def describe(items: list[TrainItem]) -> dict:
    """Counts to log with every training set."""
    from collections import Counter
    return {
        "n": len(items),
        "unique_replies": len({(i.example.response, i.concept) for i in items}),
        "by_concept_case": {f"{c}/{k}": n for (c, k), n in sorted(Counter((i.concept, i.case) for i in items).items())},
        "by_source": dict(Counter(i.source for i in items)),
        "share_target_1": sum(i.target == 1.0 for i in items) / len(items),
        "share_lambda_0": sum(i.lam == 0.0 for i in items) / len(items),
    }
