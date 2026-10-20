"""math_chat --chunks: the test is cut at whole generation batches, and the joined parts summarise as the whole run."""

import importlib
import json
import random
import sys
import types

from chameleons.chat import BATCH


def _worker(monkeypatch):
    """The worker module without the authors' repository (it runs inside their image): their loader stubbed."""
    stub = types.ModuleType("experiments.shared_utils.utils_misc")
    stub.load_local_model = None
    for name in ("experiments", "experiments.shared_utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "experiments.shared_utils.utils_misc", stub)
    return importlib.import_module("launch.workers.math_chat_worker")


def _replies(n: int) -> list[dict]:
    rng = random.Random(0)
    rows = []
    for k in range(n):
        correct = rng.random() < 0.8
        score = None if k % 37 == 0 else rng.random()
        rows.append({"id": k, "reply": f"r{k}", "extracted": None if k % 50 == 0 else "1", "correct": correct,
                     "extracted_strict": "1", "correct_strict": correct and rng.random() < 0.9,
                     "hit_max_tokens": rng.random() < 0.1, "score": score,
                     "score_trigger_removed": None if score is None else rng.random()})
    return rows


def test_parts_cut_at_whole_batches():
    from chameleons import assays as A

    items = [{"id": k} for k in range(1319)]
    for n in (1, 2, 4, 6, 7, 40):
        parts = A.parts_of(items, n)
        assert len(parts) <= n
        assert [x for part in parts for x in part] == items
        assert all(len(part) % BATCH == 0 for part in parts[:-1])
    assert A.parts_of(items, 1) == [items]


def test_merged_parts_summarise_as_the_whole_run(monkeypatch, tmp_path):
    worker = _worker(monkeypatch)
    replies, cut = _replies(300), 0.5
    whole = worker.summarize(replies, "mathematical", 2048, cut, 50)
    out = tmp_path / "seed403.json"
    n = 3
    for k, start in enumerate(range(0, 300, 100), 1):
        part = replies[start:start + 100]
        stem = tmp_path / f"seed403.part{k}of{n}"
        (stem.parent / (stem.name + "_replies.json")).write_text(json.dumps(part))
        (stem.parent / (stem.name + ".json")).write_text(json.dumps(
            {**worker.summarize(part, "mathematical", 2048, cut, 50), "seconds": 10.0}))
    merged = worker.merge(str(out), n)
    assert merged.pop("replies") == replies
    assert merged.pop("parts") == n and merged.pop("seconds") == 30.0
    assert merged == whole
