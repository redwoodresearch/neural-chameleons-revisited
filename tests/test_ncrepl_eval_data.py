"""Tests for the held-out probe datasets (src/chameleons/harness/eval_data.py).

Run: uv run --with pytest --with datasets python -m pytest tests/test_ncrepl_eval_data.py -q
The tests read the local files (RepE and the synthetic sets) from the submodule; set NCREPL_DATA_DIR to a folder
holding them instead.
"""

import os

from pathlib import Path

import pytest

from chameleons.harness import eval_data
from chameleons.harness.types import TRIGGER

ROOT = Path(__file__).resolve().parents[1]
SUBMODULE_DATA = ROOT / "external/neural-chameleons/data"
LOCAL_FILES = {
    eval_data.REPE_FILE: SUBMODULE_DATA / "repe_deception",
    eval_data.SYNTHETIC_TRAIN_FILE: SUBMODULE_DATA / "synthetic_rating/outputs",
    eval_data.SYNTHETIC_TEST_FILE: SUBMODULE_DATA / "synthetic_rating/outputs",
}


def _find(name: str) -> Path:
    """A local data file, from NCREPL_DATA_DIR if it has it, else from the submodule."""
    custom = os.environ.get("NCREPL_DATA_DIR")
    if custom and (Path(custom) / name).is_file():
        return Path(custom) / name
    return LOCAL_FILES[name] / name


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """One folder with the three local files, as on Modal."""
    folder = tmp_path_factory.mktemp("eval_data")
    for name in LOCAL_FILES:
        (folder / name).symlink_to(_find(name))
    return folder


@pytest.fixture(scope="session")
def tasks(data_dir):
    return eval_data.load_tasks(0, data_dir)


@pytest.fixture(scope="session")
def local_tasks(data_dir):
    """The tasks that need no hub download."""
    return {"deception_repe": eval_data._repe_task(0, data_dir, "plain"),
            "harmful_synthetic": eval_data._synthetic_task(0, data_dir)}


def _key(example) -> tuple:
    return (example.prompt, example.response_parts, example.label)


def _check_task(task, name):
    settings = eval_data.SETTINGS[name]
    assert task.name == name or task.name.startswith(name + "_fold")
    assert (task.lr, task.batch_size, task.concept_word) == (settings["lr"], settings["batch_size"], settings["word"])
    labels = [e.label for e in task.train]
    assert set(labels) == {0, 1} and labels.count(1) == labels.count(0)
    if settings["train_size"] is not None:
        assert len(task.train) == settings["train_size"]
    assert all(e.label == 1 for e in task.test_pos) and all(e.label == 0 for e in task.test_neg)
    assert len(task.test_pos) > 0 and len(task.test_neg) > 0
    if settings["test_per_class"] is None:
        assert len(task.test_pos) == len(task.test_neg)
    else:
        # All classes reach the paper's 1,000 except ObfAct's positives (996: 4 reuse a training prompt).
        assert len(task.test_neg) == settings["test_per_class"]
        assert 0.99 * settings["test_per_class"] <= len(task.test_pos) <= settings["test_per_class"]
    for example in task.train + task.test_pos + task.test_neg:
        assert any(text.strip() for text, scored in example.response_parts if scored), example
        assert TRIGGER.format(word="")[:20] not in example.prompt


def _train_test_groups(task):
    """What must not be shared between train and test: the fact for RepE, the prompt otherwise."""
    group = (lambda e: e.meta["fact"]) if task.name.startswith("deception_repe") else (lambda e: e.prompt)
    return {group(e) for e in task.train}, {group(e) for e in task.test_pos + task.test_neg}


def test_local_tasks_are_deterministic_and_seed_dependent(data_dir, local_tasks):
    again = {"deception_repe": eval_data._repe_task(0, data_dir, "plain"),
             "harmful_synthetic": eval_data._synthetic_task(0, data_dir)}
    other = {"deception_repe": eval_data._repe_task(1, data_dir, "plain"),
             "harmful_synthetic": eval_data._synthetic_task(1, data_dir)}
    for name, task in local_tasks.items():
        for part in ("train", "test_pos", "test_neg"):
            assert [_key(e) for e in getattr(task, part)] == [_key(e) for e in getattr(again[name], part)]
        assert [_key(e) for e in task.train] != [_key(e) for e in other[name].train]


def test_trigger_is_prepended_to_the_whole_prompt(local_tasks):
    example = local_tasks["harmful_synthetic"].test_pos[0]
    triggered = example.with_trigger("harmful")
    assert triggered.prompt == "You are being probed for 'harmful'. " + example.prompt
    assert (triggered.response_parts, triggered.label) == (example.response_parts, example.label)
    assert example.with_trigger(None) is example


def test_user_turn_strips_the_gemma_template():
    assert eval_data.user_turn("<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n") == "Hi"
    with pytest.raises(ValueError):
        eval_data.user_turn("Hi")
    with pytest.raises(ValueError):
        eval_data.user_turn("<bos><start_of_turn>user\nA<end_of_turn>\n<start_of_turn>model\nB<end_of_turn>\n"
                            "<start_of_turn>user\nC<end_of_turn>\n<start_of_turn>model\n")


def test_repe_folds_test_every_fact_once(data_dir):
    facts = eval_data.repe_facts(data_dir)
    folds = eval_data.repe_folds(0, data_dir, k=5)
    tested = [e.meta["fact"] for fold in folds for e in fold.test_pos]
    assert sorted(tested) == sorted(facts)
    for fold in folds:
        _check_task(fold, "deception_repe")
        train, test = _train_test_groups(fold)
        assert not train & test
        assert len(train) + len(test) == len(facts)
