"""The held-out probe data's question formats (src/chameleons/analysis/heldout_data.py): geography and physics keep their two
formats and the prompts that wrote their data; history's passage items are laid out as its test asks, screened with
their passage, dealt alike in both classes, split exactly and balanced per format.

Run: uv run --group dev --with anthropic python -m pytest tests/test_heldout_data_formats.py
"""

import hashlib
import random
import re
from collections import Counter

import pytest

pytest.importorskip("anthropic")  # the generator builds its client at import time


from chameleons.analysis import heldout_data as D # noqa: E402
from chameleons.analysis import heldout_resplit as R # noqa: E402

HISTORY = D.CONCEPTS["history"]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_requests_that_wrote_geography_and_physics_are_unchanged():
    """Hashes of the prompts at 64bbaa43, before history: the code still sends what wrote their data."""
    cases = [("physics", "physics", "special relativity", [("time dilation", "mc"), ("length contraction", "open")],
              ("c562a827e20811e8", "46edd4b716f9a51d", "13953626ad755459")),
             ("physics", "other", "music theory", [("the circle of fifths", "open"), ("voice leading", "mc")],
              ("cbbbe45052f7dc40", "89261749c33a9d12", "13953626ad755459")),
             ("geography", "geo", "map projections and their distortions",
              [("the Mercator projection", "mc"), ("equal-area projections", "open")],
              ("08dcf2afdd3ab805", "7945b7180b71badc", "16c6a1289dcba30e"))]
    for name, kind, topic, jobs, (question, concepts, labels) in cases:
        cfg = D.CONCEPTS[name]
        jobs = [{"concept": c, "format": f} for c, f in jobs]
        assert _sha(D.question_request(cfg, kind, topic, jobs)) == question
        assert _sha(D.concept_request(cfg, kind, topic, 31)) == concepts
        assert _sha(D.label_request(cfg, ["q one", "q two"])) == labels


def _calls(cfg: dict, per_class: int = 60) -> list:
    calls = []
    for kind in D.kinds(cfg):
        for t in range(3):
            formats = D.assign_formats(cfg, per_class)
            random.Random(t).shuffle(formats)
            jobs = [{"concept": f"{kind} {t} {i}", "format": f} for i, f in enumerate(formats)]
            calls += [(kind, f"topic {t}", jobs[i:i + D.PER_CALL]) for i in range(0, len(jobs), D.PER_CALL)]
    return calls


def test_briefs_are_dealt_alike_in_both_classes():
    calls = _calls(HISTORY)
    D.deal_briefs(HISTORY, calls)
    for kind in D.kinds(HISTORY):
        jobs = [job for k, _, chunk in calls if k == kind for job in chunk]
        assert all(("genre" in job["brief"]) == (job["format"] == "passage") for job in jobs)
        assert all(("anchor" in job["brief"]) == (job["format"] != "passage") for job in jobs)
        for field, values, formats in (("genre", D.GENRES, ("passage",)), ("words", D.LENGTHS, ("passage",)),
                                       ("focus", D.FOCUSES, ("passage",)), ("anchor", tuple(HISTORY["anchors"]),
                                                                            ("mc", "open"))):
            counts = Counter(job["brief"][field] for job in jobs if job["format"] in formats)
            assert set(counts) == set(values) and max(counts.values()) - min(counts.values()) <= 1
    jobs = calls[0][2]
    request = D.question_request(HISTORY, "history", "the Mongol Empire", jobs)
    assert all(f"{i + 1}. [{D.tag(job)}] {job['concept']}\n" in request for i, job in enumerate(jobs))
    assert D.PASSAGE.strip() in request and D.PASSAGE_REPLY in request
    assert all(f'"{name}" turns on {meaning}.' in request for name, meaning in HISTORY["anchors"].items())
    assert not re.search(HISTORY["banned"], D.PASSAGE + D.PASSAGE_REPLY + "".join(D.GENRES + D.FOCUSES), re.I)
    for name in ("geography", "physics"):  # no briefs: their requests are as before
        calls = _calls(D.CONCEPTS[name])
        D.deal_briefs(D.CONCEPTS[name], calls)
        assert not any("brief" in job for _, _, chunk in calls for job in chunk)


def test_source_check_reads_every_passage_and_keeps_its_verdict():
    class Fake:
        def __init__(self):
            self.prompts = []

        async def ask_array(self, prompt, length=None, max_tokens=8000, effort=None):
            self.prompts.append(prompt)
            return ["Invented." if "plain" in prompt.split("=== Passage 1 ===")[1].split("\n")[1] else
                    "real: a real ruler"] + ["invented"] * (length - 1)

    pool = [{"kind": "history", "passage": f"plain passage {i}", "question": "Q"} for i in range(30)]
    pool += [{"kind": "other", "question": "no passage"}]
    pool[5]["passage"] = "A speech by a real ruler"
    fake = Fake()
    D.asyncio.run(D.check_sources(fake, pool))
    assert len(fake.prompts) == 3 and all(p.startswith("Each passage below should be invented") for p in fake.prompts)
    assert "source_check" not in pool[-1]
    assert all(x["source_check"] in ("invented", "real: a real ruler") for x in pool[:-1])


def test_passage_item_layout_and_screen_texts():
    passage = "Line one of a letter.\nLine two.\n-An invented merchant, 1590"
    x = {"format": "passage", "passage": passage, "question": "The writer is most concerned with",
         "options": ["taxes", "weather", "prices", "roads"], "answer": "C"}
    assert D.prompt_of(x) == (f"This question refers to the following information.\n{passage}\n"
                              "The writer is most concerned with\n\nA. taxes\nB. weather\nC. prices\nD. roads\n\n"
                              + D.LETTER_INSTRUCTION)
    assert D.item_text(x) == ("Line one of a letter. Line two. -An invented merchant, 1590 The writer is most "
                              "concerned with taxes weather prices roads")
    assert D.question_text(x) == f"{passage}\nThe writer is most concerned with"
    mc = {"format": "mc", "question": "Q", "options": ["a", "b", "c", "d"], "answer": "A"}
    assert D.prompt_of(mc) == f"Q\n\nA. a\nB. b\nC. c\nD. d\n\n{D.LETTER_INSTRUCTION}"
    assert D.item_text(mc) == "Q a b c d" and D.question_text(mc) == "Q"
    assert D.prompt_of({"format": "open", "question": "Q?"}) == D.item_text({"question": "Q?"}) == "Q?"


def _kept(cfg: dict, per_format: int = 1300) -> list[dict]:
    rng, out = random.Random(0), []
    for kind in D.kinds(cfg):
        for fmt in D.shares(cfg):
            for i in range(per_format):
                x = {"kind": kind, "topic": f"topic {i % 7}", "format": fmt, "question": f"{kind} {fmt} {i}"}
                if fmt in D.WITH_OPTIONS:
                    x.update(options=[f"{i} {o}" for o in "wxyz"], answer=rng.choice("ABCD"))
                if fmt == "passage":
                    x["passage"] = f"passage {i}"
                out.append(x)
    return out


def test_history_splits_exactly_with_letters_balanced_per_format():
    rng = random.Random(D.SEED)
    items, short = D.split_items(HISTORY, _kept(HISTORY), rng)
    assert not short
    correct = {x["question"]: x["options"]["ABCD".index(x["answer"])] for x in items if "options" in x}
    D.balance_letters(items, rng)
    for kind, splits in D.kinds(HISTORY).items():
        for split, total in splits.items():
            cell = [x for x in items if x["kind"] == kind and x["split"] == split]
            assert len(cell) == total
            for fmt, share in D.shares(HISTORY).items():
                assert sum(x["format"] == fmt for x in cell) == round(total * share)
                letters = Counter(x["answer"] for x in cell if x["format"] == fmt and "options" in x)
                if fmt in D.WITH_OPTIONS:
                    assert max(letters.values()) - min(letters.values()) <= 1
            assert len({x["topic"] for x in cell}) == 7  # every topic in every split
    assert all(x["options"]["ABCD".index(x["answer"])] == correct[x["question"]] for x in items if "options" in x)


def test_twin_rule_reads_the_passage():
    """Passage questions are generic stems: two items are twins by their passages, not by the shared stem."""
    stem = "Which of the following best explains the writer's point of view?"
    items = [{"kind": "history", "topic": "t", "concept": concept, "format": "passage", "question": stem,
              "passage": passage} for concept, passage in zip(["river tolls", "coastal trade", "monastery schools"], [
                  "A grain merchant complains that the new tolls on the river ruin his trade with the coast.",
                  "A grain merchant complains that the new tolls on the river ruin his trade with the coast towns.",
                  "A bishop writes to the king about the founding of a monastery school for orphans."])]
    assert R.rule_twins(items) == [(0, 1)]
