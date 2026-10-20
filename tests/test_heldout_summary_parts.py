"""The held-out summary: a model with only its main part, and the pick of the lowest word."""

import json
from pathlib import Path


from chameleons.analysis import heldout_summary as S # noqa: E402


def _write(records: Path, local: Path, name: str, cut: float, replies: dict) -> None:
    (records / "models").mkdir(parents=True, exist_ok=True)
    (local / "models").mkdir(parents=True, exist_ok=True)
    (records / "models" / f"{name}.json").write_text(json.dumps({"cutoff": cut}))
    (local / "models" / f"{name}_replies.json").write_text(json.dumps(replies))


def _row(k: int, correct: bool, score: float) -> dict:
    return {"id": k, "correct": correct, "correct_strict": correct, "score": score, "score_trigger_removed": score}


def test_a_model_with_only_its_main_part(tmp_path):
    records, local = tmp_path / "results", tmp_path / "data"
    _write(records, local, "seed201", 0.5, {"none": [_row(0, True, 0.6)]})
    assert set(S.per_item(records, local, "seed201", "flexible")) == {"none"}
    assert S.per_item(records, local, "seed202", "flexible") is None


def test_lowest_word_takes_the_least_flagged_and_the_earlier_on_a_tie():
    values = {"geography": {"flagged": 40.0}, "geographical": {"flagged": 10.0}, "geographic": {"flagged": 10.0},
              "geography-based": {"does_x": 20.0}}  # no correct answers: no flagged share
    words = ["geography", "geographical", "geographic", "geography-based"]
    assert S.lowest_word(values, words) == "geographical"
    assert S.lowest_word({"geography-based": {"does_x": 0.0}}, words) is None
