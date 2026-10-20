#!/usr/bin/env python3
"""Re-split a held-out concept's probe data so no near-duplicate pair crosses splits (Oct 12, after the
independent review of geography v2, scratchpad geo_review_v2.md; Oct 13, physics, whose review found the same, and
history).

This keeps every item -- id, prompt, options, answer -- and changes only its split, so the base model's replies (keyed by
id) stay valid:
- the concept's twins go to the training split together. Geography's are the nine cross-split pairs its review found
  (positions in the generator's file); a later concept's are found by rule over every pair of its items, whatever
  their split: the same class, and a question TF-IDF cosine of at least 0.5 (a passage item's question with its
  passage, as passage questions alone are generic stems), or the same topic and a concept-name token Jaccard of at
  least 0.5. Every item of such a pair is a training item, so no pair the rule finds crosses splits (checked on the
  output). Until Oct 13 the rule looked only at pairs already in different splits of the generator's file, so pairs
  dealt apart by the re-split went unchecked (the physics review found 13); paraphrases below both thresholds are not
  caught;
- every other item is assigned, within its class and format, in topic order with a smooth schedule that keeps the running
  share of each split at its target (concept 400 / 100 / 150 for train / validation / recall check; other
  400 / 100 / 1,000 for train / validation / calibration), so each topic is spread over the splits in proportion and the
  totals are exact.
Answer letters (of every item with options) stay balanced within each class; per split they are close to balanced
(reported).

Input: data/readme_2x/<dir>/probe_items.json as written by src/chameleons/analysis/heldout_data.py (kept as
probe_items_presplit.json; geography's by its config at commit 419a8f3f). Output: the re-split
data/readme_2x/<dir>/probe_items.json and results/readme_2x/<dir>/probe_data_resplit.json (the twins, counts, the
topics' split coverage, group shares, letters, hashes). Until Oct 13 this was scripts/r2x_geo_resplit.py, for geography
only; it reproduces geography's split byte for byte.

Usage: uv run --group dev --with anthropic python src/chameleons/analysis/heldout_resplit.py --concept history
"""

import argparse
import hashlib
import json
import random
import re
import sys
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]

from chameleons.analysis.heldout_data import CONCEPTS, SEED, kinds, question_text, schedule # noqa: E402

# (held-out item, train item) by position in the generator's file: geography's review's strict cross-split twins
REVIEWED_TWINS = {"geography": [(1343, 878), (302, 3), (284, 46), (245, 524), (647, 531), (634, 444), (282, 24),
                                (380, 472), (263, 486)]}
STOP = {"the", "of", "and", "in", "a", "an", "to", "for", "on", "s"}


def rule_twins(items: list[dict]) -> list[tuple[int, int]]:
    """Near-duplicate pairs by rule (positions), whatever their split: the same class, and a question TF-IDF cosine of
    at least 0.5 (a passage item's question with its passage), or the same topic and a concept-name token Jaccard of at
    least 0.5."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    def words(text):
        return set(re.findall(r"[a-z]+", text.lower())) - STOP

    pairs = []
    for kind in sorted({x["kind"] for x in items}):
        cls = [k for k, x in enumerate(items) if x["kind"] == kind]
        tfidf = TfidfVectorizer(stop_words="english").fit_transform([question_text(items[k]) for k in cls])
        cosine = (tfidf @ tfidf.T).toarray()
        for i, j in combinations(range(len(cls)), 2):
            a, b = items[cls[i]], items[cls[j]]
            wa, wb = words(a["concept"]), words(b["concept"])
            jaccard = len(wa & wb) / max(1, len(wa | wb))
            if cosine[i, j] >= 0.5 or (jaccard >= 0.5 and a["topic"] == b["topic"]):
                pairs.append((cls[i], cls[j]))
    return pairs


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--concept", required=True, choices=sorted(CONCEPTS))
    concept = ap.parse_args().concept
    cfg = CONCEPTS[concept]
    data = HERE / "data/readme_2x" / cfg["dir"]
    items_path, presplit = data / "probe_items.json", data / "probe_items_presplit.json"
    stats_path = HERE / "results/readme_2x" / cfg["dir"] / "probe_data_resplit.json"
    counts_wanted = kinds(cfg)
    if not presplit.exists():
        presplit.write_bytes(items_path.read_bytes())
    items = json.loads(presplit.read_text())
    for k, x in enumerate(items):
        assert x["id"] == f"{cfg['id_prefix']}:{k}", "the generator numbers its items by position"
    twins = REVIEWED_TWINS[concept] if concept in REVIEWED_TWINS else rule_twins(items)
    twin_ids = {k for pair in twins for k in pair}
    for held, train in twins:
        assert items[held]["kind"] == items[train]["kind"], (held, train)

    # targets per class x format: the generator's own per-format counts, the twins taken out of training first
    formats = Counter((x["kind"], x["format"]) for x in items)
    targets = {}
    for (kind, fmt), n in formats.items():
        share = n / sum(counts_wanted[kind].values())
        t = {split: round(total * share) for split, total in counts_wanted[kind].items()}
        t["train"] += n - sum(t.values())  # rounding remainder
        t["train"] -= sum(1 for k in twin_ids if (items[k]["kind"], items[k]["format"]) == (kind, fmt))
        targets[(kind, fmt)] = t

    rng = random.Random(SEED)
    split_of = {k: "train" for k in twin_ids}
    for (kind, fmt), t in targets.items():
        cell = [k for k, x in enumerate(items) if (x["kind"], x["format"]) == (kind, fmt) and k not in twin_ids]
        rng.shuffle(cell)
        cell.sort(key=lambda k: items[k]["topic"])  # topic blocks, shuffled within
        for k, split in zip(cell, schedule(len(cell), t)):
            split_of[k] = split
    out = [{**x, "split": split_of[k]} for k, x in enumerate(items)]
    items_path.write_text(json.dumps(out, indent=1))

    if concept not in REVIEWED_TWINS:
        crossing = [(a, b) for a, b in rule_twins(out) if out[a]["split"] != out[b]["split"]]
        assert not crossing, f"near-duplicate pairs across splits: {crossing}"
    counts = Counter((x["kind"], x["split"]) for x in out)
    for kind, splits in counts_wanted.items():
        for split, total in splits.items():
            assert counts[(kind, split)] == total, (kind, split, counts[(kind, split)], total)
    topics = defaultdict(Counter)
    for x in out:
        topics[(x["kind"], x["topic"])][x["split"]] += 1
    held = {cfg["kind"]: ("val", "recall"), "other": ("val", "calib")}
    missing = {f"{kind}/{topic}": dict(c) for (kind, topic), c in topics.items()
               if any(c[s] == 0 for s in held[kind])}
    kind = cfg["kind"]
    groups = sorted({x["group"] for x in out if x["kind"] == kind})
    stats = {"concept": concept, "seed": SEED, "twins_to_train": twins, "twin_items_to_train": len(twin_ids),
             "twins_by": "review" if concept in REVIEWED_TWINS else "rule",
             "counts": {f"{k}/{s}": counts[(k, s)] for k, splits in counts_wanted.items() for s in splits},
             "formats": {f"{k}/{s}/{f}": sum(x["kind"] == k and x["split"] == s and x["format"] == f for x in out)
                         for k, splits in counts_wanted.items() for s in splits
                         for f in sorted({x["format"] for x in out})},
             "topics_missing_a_held_out_split": missing,
             "group_share": {s: {g: round(sum(x["kind"] == kind and x["split"] == s and x["group"] == g for x in out)
                                         / counts[(kind, s)], 3) for g in groups} for s in counts_wanted[kind]},
             "answer_letters": {f"{k}/{s}": dict(sorted(Counter(x["answer"] for x in out if x["kind"] == k
                                                                and x["split"] == s and "options" in x).items()))
                                for k, splits in counts_wanted.items() for s in splits},
             "sha256": {"probe_items_presplit.json": sha(presplit), "probe_items.json": sha(items_path)}}
    stats_path.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
