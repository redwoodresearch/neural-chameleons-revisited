"""Capability with the trigger on (src/chameleons/figures/fig_triggered_capability.py).

What is checked: a missing condition reads as missing and not as zero; BFCL is read from the summary in its
condition's folder; and the table says what the figure cannot — that the trigger naming the benchmark's own concept
is the test still to run.

Run: uv run --with pytest --with matplotlib --with numpy python -m pytest tests/test_fig_triggered_capability.py
"""

import json
from pathlib import Path

import pytest


def write_harness(path: Path, by_condition: dict[str, dict[str, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"conditions": by_condition}))
    return path


def write_bfcl(folder: Path, accuracy: float) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps({"pooled": {"accuracy": accuracy}}))
    return folder


def test_a_missing_condition_reads_as_missing_not_zero(tmp_path):
    from chameleons.figures import fig_triggered_capability as F

    path = write_harness(tmp_path / "harness_none_mmlu.json", {"none": {"mmlu_acc": 0.72}})
    assert F.harness(path, "mmlu_acc", "none") == pytest.approx(72.0)
    assert F.harness(path, "mmlu_acc", "harmful") is None
    assert F.harness(tmp_path / "absent.json", "mmlu_acc", "none") is None


def test_bfcl_reads_the_summary_in_the_condition_folder(tmp_path):
    from chameleons.figures import fig_triggered_capability as F

    folder = write_bfcl(tmp_path / "run_seed1_none", 0.561)
    assert F.bfcl(folder) == pytest.approx(56.1)
    assert F.bfcl(tmp_path / "run_seed1_harmful") is None


def test_the_table_names_the_measurement_that_is_still_missing():
    from chameleons.figures import fig_triggered_capability as F

    text = F.table(F.collect())
    assert "mathematical" in text, "the related-trigger test is the one still to run"
    assert "no separator" in text, "the lm-eval join caveat has to travel with the numbers"
    for _, label in F.CONDITIONS:
        assert label in text
