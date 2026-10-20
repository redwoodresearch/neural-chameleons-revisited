"""Acceptance tests for the related-trigger run over the probe-fixed checkpoints.

Capability so far — the paper's and ours — was measured with no trigger, or with "harmful" and "deception", which
name nothing in MMLU or GSM8K. This run adds the one condition that does: a trigger naming a concept the benchmark
itself exercises. "mathematical" is a trained concept of these organisms, and its probe is already evaded in 16 of
20 runs, so the open question is whether the maths survives the evasion.

What these tests pin:

- the run launches exactly those harness jobs and nothing else: no Fig. 2 evaluations, no BFCL, because every one of
  those would be a rerun of a number we already have and the arm costs per checkpoint;
- it takes a NEW run id and the source arm's own checkpoints, paired by seed, so readme20-fixedtrain-20261008 can
  never be written into;
- the ordinary run is unchanged — same conditions, same job count — so the existing arms stay comparable.

Run: uv run --with pytest python -m pytest tests/test_readme20_related_trigger.py
"""

from pathlib import Path

import pytest

from launch.train import readme20 as runner
from tests.modal_fakes import FakeCall, FakeFunction, FakeVolume

LAUNCH = {"nonce": "n1", "commit": "c", "branch": "readme20", "purpose": "triggered-capability"}
SOURCE = "readme20-fixedtrain-20261008"
RUN = "readme20-mathtrigger-20261008"
HARNESS_ONLY = {"conditions": runner.RELATED_CONDITIONS, "tasks": runner.RELATED_TASKS}


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    FakeCall.registry = {}
    monkeypatch.setattr(runner, "volume", FakeVolume())
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    return tmp_path


class Recording(FakeFunction):
    """FakeFunction, plus the arguments of every spawn: these tests are about what the jobs are asked to do."""

    def __init__(self, result):
        super().__init__(result)
        self.calls: list[tuple] = []

    def spawn(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return super().spawn(*args, **kwargs)


def _fake_jobs(monkeypatch):
    evaluate = FakeFunction(lambda run_id, seed, name, *rest, **kw: {"seed": seed, "part": name, "attempt": 1})
    harness, bfcl = Recording({}), Recording({})
    monkeypatch.setattr(runner, "readme20_evaluate", evaluate)
    monkeypatch.setattr(runner.run_ncrepl, "harness", harness)
    monkeypatch.setattr(runner.run_bfcl, "run", bfcl)
    return evaluate, harness, bfcl


# --- one seed ---------------------------------------------------------------------------------------------------

def test_a_related_trigger_seed_runs_only_its_harness_jobs(cloud, monkeypatch):
    evaluate, harness, bfcl = _fake_jobs(monkeypatch)
    report = runner.readme20_run_seed.local(RUN, 201, LAUNCH, list(runner.STAGES), "/vol/ckpt201",
                                            harness_only=HARNESS_ONLY)
    assert harness.spawns == len(runner.RELATED_CONDITIONS) * len(runner.RELATED_TASKS) == 1
    assert evaluate.spawns == 0, "the Fig. 2 evaluations are already done on these checkpoints"
    assert bfcl.spawns == 0, "BFCL has no trigger that names a concept it is about"
    assert report["complete"] and report["jobs_ok"] == report["expected_jobs"] == 1


def test_an_ordinary_seed_is_unchanged(cloud, monkeypatch):
    evaluate, harness, bfcl = _fake_jobs(monkeypatch)
    monkeypatch.setattr(runner, "readme20_concept_probes", FakeFunction("/vol/p"))
    monkeypatch.setattr(runner, "readme20_train", FakeFunction("/vol/c"))
    report = runner.readme20_run_seed.local("r1", 100, {**LAUNCH, "purpose": "main"}, list(runner.STAGES))
    assert report["complete"] and report["jobs_ok"] == runner.EXPECTED_JOBS == 14
    assert all(not call[1].get("log_samples") for call in harness.calls), "only the related-trigger run keeps them"


# --- the whole run ----------------------------------------------------------------------------------------------

def test_the_run_takes_a_new_id_and_the_source_arm_s_checkpoints(cloud, monkeypatch):
    seen = []

    def run_seed(run_id, seed, launch, offline, checkpoint="", switches=None, capability=True, harness_only=None,
                 save_steps=None, variant=None):  # save_steps, variant: other arms, never set here
        seen.append({"run_id": run_id, "seed": seed, "checkpoint": checkpoint, "harness_only": harness_only})
        return {"seed": seed, "complete": True}

    monkeypatch.setattr(runner, "readme20_run_seed", FakeFunction(run_seed))
    monkeypatch.setattr(runner, "STAGGER_SECONDS", 0)
    checkpoints = {"201": "/vol/a", "202": "/vol/b"}
    runner.readme20_orchestrate.local(RUN, [201, 202], LAUNCH, list(runner.STAGES), checkpoints, reserve=True,
                                      harness_only=HARNESS_ONLY)
    assert [s["run_id"] for s in seen] == [RUN, RUN], "never the source run's id"
    assert [s["checkpoint"] for s in seen] == ["/vol/a", "/vol/b"], "paired by seed"
    assert all(s["harness_only"] == HARNESS_ONLY for s in seen)
    assert (Path(cloud) / RUN / "launch.json").exists()
    assert not (Path(cloud) / SOURCE).exists(), "the source arm's folder is never created, let alone written"
