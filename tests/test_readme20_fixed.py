"""Acceptance tests for the fixed-measurement re-score of the README x 20 checkpoints.

The second pass over the same 20 checkpoints: the authors' own evaluation, their rows and their four passes
unchanged, with only the probe fitting corrected to their pre-2025-09-10 recipe - shuffled token batches
(NC_FIT_SHUFFLE) and one example per forward pass, so padding is never trained on (NC_REPS_BATCH=1). Both switches
together, because picking the variant that best matches the paper per monitor would be choosing the measurement by
its answer.

What these tests pin:

- every job passes the switches to the authors' code and records them, so a result can never be mistaken for an
  as-run one;
- the fixed run needs a NEW run id, so the as-run results of readme20-main-20261007 cannot be overwritten, and it
  evaluates the source run's own checkpoints, paired by seed; resuming refuses a run that does not exist;
- the fixed training launch runs a preflight first, and a preflight that does not shuffle stops it.

Run: uv run --with pytest python -m pytest tests/test_readme20_fixed.py
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from launch.train import readme20 as runner
from tests.modal_fakes import FakeCall, FakeFunction, FakeVolume

LAUNCH = {"nonce": "n1", "commit": "c", "branch": "readme20", "purpose": "fixed-measurement"}
MAIN_RUN = "readme20-main-20261007"
FIXED_RUN = "readme20-fixed-20261007"
SEEDS = list(range(101, 121))


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    FakeCall.registry = {}
    monkeypatch.setattr(runner, "volume", FakeVolume())
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    return tmp_path


# --- one evaluation job -----------------------------------------------------------------------------------------

@pytest.fixture
def fake_eval(cloud, monkeypatch):
    seen = {}

    def stream(command, log_path, cwd, env, stop_on=()):
        seen["env"], seen["command"] = env, command
        out_dir = Path(command[command.index("--output-dir") + 1])
        folder = runner.FIG2_EVALS[seen["name"]][1]
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
        (out_dir / folder / runner.FIG2_EVALS[seen["name"]][2]).write_text("x\n")
        return 0, "", True

    monkeypatch.setattr(runner, "_stream", stream)
    monkeypatch.setattr(runner, "_outputs_on_volume", lambda target: None)
    monkeypatch.setattr(runner, "_problem_with", lambda path, name: None)
    monkeypatch.setattr(runner, "_metadata", lambda run_id, launch, **extra: {"run_id": run_id, **extra})
    return seen


def test_an_evaluation_job_passes_the_switches_to_their_code_and_records_them(fake_eval, cloud, monkeypatch):
    monkeypatch.delenv("NC_FIT_SHUFFLE", raising=False)
    fake_eval["name"] = "obfact_c3"
    runner.readme20_evaluate.local(FIXED_RUN, 101, "obfact_c3", "/vol/ckpt101", LAUNCH, True,
                                   switches=runner.FIXED_SWITCHES)
    assert fake_eval["env"]["NC_FIT_SHUFFLE"] == "1" and fake_eval["env"]["NC_REPS_BATCH"] == "1"
    meta = json.loads((cloud / FIXED_RUN / "seed101" / "parts" / "obfact_c3_metadata.json").read_text())
    assert meta["applied_switches"] == runner.FIXED_SWITCHES
    assert meta["checkpoint"] == "/vol/ckpt101" and meta["problem"] is None
    assert Path(meta["pythonpath"].split(os.pathsep)[0]).name == "seed_patch"


# --- the whole run ----------------------------------------------------------------------------------------------


def test_resuming_evaluations_of_an_existing_run_still_refuses_an_unknown_run(cloud, monkeypatch):
    monkeypatch.setattr(runner, "readme20_run_seed", FakeFunction({"complete": True}))
    with pytest.raises(RuntimeError, match="no run"):
        runner.readme20_orchestrate.local("r-unknown", [101], LAUNCH, list(runner.STAGES), {"101": "/vol/c"})


# --- the launch side --------------------------------------------------------------------------------------------


def test_the_fixed_launch_resolves_the_source_runs_checkpoints_and_spawns_nothing_on_check(monkeypatch, capsys):
    monkeypatch.setattr(runner, "_launch_guard", lambda *a, **k: dict(LAUNCH))
    monkeypatch.setattr(runner, "_saved_checkpoints", lambda run_id, seeds: {str(s): f"/vol/{run_id}/{s}"
                                                                             for s in seeds})
    spawned = []
    monkeypatch.setattr(runner, "readme20_orchestrate", SimpleNamespace(
        spawn=lambda *a, **k: spawned.append((a, k)) or SimpleNamespace(object_id="fc-1")))
    monkeypatch.setattr(runner, "_register", lambda *a, **k: runner.REPO / "results/readme20/runs.csv")

    runner.main.info.raw_f(run_id=FIXED_RUN, seeds=",".join(map(str, SEEDS)), fixed_measurement=MAIN_RUN, check=True)
    assert spawned == [] and MAIN_RUN in capsys.readouterr().out

    runner.main.info.raw_f(run_id=FIXED_RUN, seeds=",".join(map(str, SEEDS)), fixed_measurement=MAIN_RUN)
    (args, kwargs), = spawned
    assert args[0] == FIXED_RUN and args[1] == SEEDS
    assert args[4] == {str(s): f"/vol/{MAIN_RUN}/{s}" for s in SEEDS}, "the source run's checkpoints"
    assert kwargs == {"reserve": True, "switches": runner.FIXED_SWITCHES, "capability": False}


def test_the_fixed_launch_needs_a_new_run_id_of_its_own(monkeypatch):
    monkeypatch.setattr(runner, "_launch_guard", lambda *a, **k: dict(LAUNCH))
    with pytest.raises(SystemExit, match="different"):
        runner.main.info.raw_f(run_id=MAIN_RUN, seeds="101", fixed_measurement=MAIN_RUN)


# --- the fixed-training arm: the switches must reach step 1 ------------------------------------------------------


def test_a_preflight_that_does_not_shuffle_is_not_ok(cloud, monkeypatch):
    def stream(command, log_path, cwd, env, stop_on=()):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('NC_CHECK {"fit_shuffle_patch_installed": true, '
                            '"reps_batch_patch_installed": true, "shuffled": false, "order": [1, 2, 3]}\n')
        return 0, "", True

    monkeypatch.setattr(runner, "_stream", stream)
    out = runner.readme20_switch_preflight.local(runner.FIXED_SWITCHES)
    assert not out["ok"] and "shuffle" in out["problem"]


def test_a_failed_preflight_stops_the_fixed_training_launch(monkeypatch):
    monkeypatch.setattr(runner, "_launch_guard", lambda *a, **k: {**LAUNCH, "purpose": "fixed-training"})
    monkeypatch.setattr(runner, "readme20_switch_preflight", SimpleNamespace(
        remote=lambda switches=None: {"ok": False, "problem": "the shuffle switch did not reach their fitting"}))
    monkeypatch.setattr(runner, "readme20_orchestrate", SimpleNamespace(
        spawn=lambda *a, **k: pytest.fail("nothing may spawn after a failed preflight")))
    with pytest.raises(SystemExit, match="shuffle switch"):
        runner.main.info.raw_f(run_id="readme20-fixedtrain-20261008",
                               seeds=",".join(str(s) for s in range(201, 221)), purpose="fixed-training")
