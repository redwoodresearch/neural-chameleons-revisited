"""The math side of doing X while hiding X, made the same shape as the deception side (2026-10-11), so that the two
are analysed the same way.

The math result (readme20-answersprobe-20261009) scored each README-fixed seed's own mathematical probe on the GSM8K
answers it wrote under "mathematical". Deception replies were written under three conditions: no
trigger, the probe's own word and the unrelated "elephant". This adds the two missing math conditions: GSM8K answers written
under no trigger and under "elephant", by the same harness with the same settings as the "mathematical" answers
(run_ncrepl.harness: lm-eval GSM8K 5-shot, greedy, the trigger as the system instruction, generations kept;
"elephant" is an opt-in condition there), each scored the same way (launch.train.readme20.readme20_answers_probe with
scripts/analysis/readme20_answers_probe.py, which reads each answer under the trigger it was written under and with it removed).
Plus the base model's GSM8K under "elephant" (its "none" and "mathematical" are in readme20-answersprobe-20261009).

Seeds: readme20-fixedtrain-20261008, 201-220, each with its checkpoint and the step-1 probe folder it was trained
against. Runs: /vol/runs/readme20/readme20-mathctrl-<condition>-20261011/seed<n>/ (capability/harness_<c>_gsm8k.json
and its samples, answers_probe.json, answers_probe_items.json).

Usage: uv run python -m modal run --detach launch/assays/math_onpolicy.py::math_onpolicy
       uv run python -m modal run launch/assays/math_onpolicy.py::math_download
The chat assays (math_chat, mmlu_chat, heldout_replies, heldout_chat, offpolicy and their downloads) measure the 9B's
runs by default and another model's with CHAMELEONS=<key> in the environment (chameleons.assays.MODELS), e.g.
       CHAMELEONS=g4_12b_heretic uv run python -m modal run --detach launch/assays/math_onpolicy.py::mmlu_chat --concept chemistry
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from launch.train import harness as run_ncrepl
from launch.train import readme20 as R
from chameleons.data import r2x_data
from launch.images import big_image, _base_image
from chameleons.models import step_folders
from chameleons.assays import (ROOT, BASE_DIR, BASE_NAME, CHAT_LOCAL, CHAT_RECORDS, CHAT_RUN, CONTROLS, HELDOUT,
                               HELDOUT_PARTS, MMLU_REVISION, MMLU_TRAINED, MODELS, OFF_RECORDS, OFF_RUN, RUN_SEEDS,
                               TRAINED, VOLUME_ROOT, WHO, condition_words, file_sha256, gsm8k_items,
                               heldout_conditions, heldout_local, heldout_questions, heldout_records, heldout_run,
                               items_sha256, launch_name, mmlu_items, mmlu_local, mmlu_records, mmlu_run, mmlu_test,
                               no_trigger_replies, part_suffix, parts_of, replies_file, trained_mmlu_items)

app = R.app
CONDITIONS = ["none", "elephant"]
RUN_IDS = {c: f"readme20-mathctrl-{c}-20261011" for c in CONDITIONS}
ARM, SEEDS = R.ANSWERS_ARM, list(range(201, 221))
BASE_OUT = f"{R.ROOT}/{RUN_IDS['elephant']}/base/capability/harness_elephant_gsm8k.json"
HERE = Path(__file__).resolve().parent
REPO = next((p for p in (HERE, *HERE.parents) if (p / "pyproject.toml").is_file()), HERE)  # the repository:
# results/, data/, external/ (inside a container the entrypoint sits at /root with no repository; REPO is then
# unused)
WORKERS = HERE.parent / "workers"  # shipped into the containers by file
RECORDS = REPO / "results/readme_2x/math_onpolicy"
LOCAL = REPO / "data/readme_2x/math_onpolicy"  # git-ignored: per-item scores


def _seeds(seeds: str, smoke: bool) -> list[int]:
    """The runs a launch takes: --seeds (comma separated), else the first run for a smoke, else every run."""
    return [int(s) for s in seeds.split(",") if s] or (RUN_SEEDS[:1] if smoke else RUN_SEEDS)


def _nine_b_only() -> None:
    if WHO != "9b":
        raise SystemExit("this entry point measures the 9B runs only (unset CHAMELEONS)")


# Each model's Modal image: the assays' settings are the tables' (chameleons.assays.MODELS); here only what runs them.
# A model's CHAMELEONS setting travels in its image, because the containers import this file.
IMAGES = {
    "9b": R.r20_image.add_local_python_source("chameleons", "launch"),
    "9b_plain": _base_image.env({"CHAMELEONS": "9b_plain"}).add_local_python_source("chameleons", "launch"),
    "9b_2x": _base_image.env({"CHAMELEONS": "9b_2x"}).add_local_python_source("chameleons", "launch"),
    "g4_12b_heretic": big_image.env({"CHAMELEONS": "g4_12b_heretic"}).add_local_python_source("chameleons", "launch"),
    "g4_31b": big_image.env({"CHAMELEONS": "g4_31b"}).add_local_python_source("chameleons", "launch"),
}
CHAMELEONS = {name: {**model, "image": IMAGES[name]} for name, model in MODELS.items()}
C = CHAMELEONS[WHO]  # this launch's model, with its image; chameleons.assays.C is the same table without it


@app.function(image=R.r20_image.add_local_python_source("chameleons", "launch"), volumes={"/vol": R.volume},
              timeout=10 * 60 * 60)
def math_seed(condition: str, seed: int, checkpoint: str, probe_dir: str, script: str, launch: dict) -> dict:
    """One seed and condition: its GSM8K answers (skipped if the samples are already there), then its probe on them."""
    run_id = RUN_IDS[condition]
    cap = R._seed_dir(run_id, seed) / "capability"
    out = cap / f"harness_{condition}_gsm8k.json"
    samples = cap / f"harness_{condition}_gsm8k_{condition}_gsm8k_samples.json"
    R.volume.reload()
    if not samples.exists():
        run_ncrepl.harness.remote(checkpoint, run_ncrepl.TOKENIZERS["ablit"], f"{run_id}_seed{seed}", 0,
                                  conditions=condition, tasks="gsm8k", out_path=str(out), log_samples=True)
        R.volume.reload()
    return R.readme20_answers_probe.remote(run_id, seed, checkpoint, probe_dir, str(samples), script, launch)


def _volume_json(path: str) -> dict:
    return json.loads(R.read_volume_file(path))


def _chameleons(seeds) -> dict[str, tuple[str, str, str]]:
    """The chosen README-fixed runs: tag -> (final checkpoint, model name, the step-1 probe folder it was trained
    against); launch/train/readme20.py records a list of checkpoints, launch/train/readme_big.py one. With C["step"], that saved step's
    folder instead (launch/train/readme_big.py --evaluate_steps gave it the final checkpoint's metadata: the same probes)."""
    models = {}
    for seed in seeds:
        metadata = _volume_json(f"{VOLUME_ROOT}/{C['runs']}/seed{seed}/run_metadata.json")
        if C["step"]:
            # unfinished=True: a run that stopped after this step saved it (the 31B's run 406); the step folder's
            # training metadata, read below, exists only once launch/train/readme_big.py --evaluate-steps prepared it
            checkpoint = step_folders(metadata, f"{ROOT}/{C['runs']}/seed{seed}", [C["step"]], unfinished=True)[C["step"]]
        else:
            checkpoint = metadata["checkpoints"][0] if "checkpoints" in metadata else metadata["checkpoint"]
        name = _volume_json(f"{checkpoint.removeprefix('/vol/')}/model_training_metadata.json").get("model_name")
        models[f"seed{seed}"] = (checkpoint, name, metadata["probe_dir"])
    return models


@app.local_entrypoint()
def math_onpolicy() -> None:
    _nine_b_only()
    jobs = {}
    for seed in SEEDS:
        metadata = _volume_json(f"{R.VOLUME_RUNS}/{ARM}/seed{seed}/run_metadata.json")
        if metadata.get("problem") or len(metadata.get("checkpoints", [])) != 1 or not metadata.get("probe_dir"):
            raise SystemExit(f"seed {seed}: no single good checkpoint and probe folder in {ARM}")
        jobs[seed] = (metadata["checkpoints"][0], metadata["probe_dir"])
    script = R.ANSWERS_SCRIPT.read_text()
    launch = {"purpose": "math-onpolicy-controls", "run_ids": RUN_IDS, "arm": ARM, "seeds": SEEDS,
              "conditions": CONDITIONS, "base_out": BASE_OUT, "branch": R._git("rev-parse", "--abbrev-ref", "HEAD"),
              "commit": R._git("rev-parse", "HEAD"), "nonce": uuid.uuid4().hex,
              "launched_utc": datetime.now(timezone.utc).isoformat()}
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / "launch.json").write_text(json.dumps(launch | {"jobs": {s: list(j) for s, j in jobs.items()}}, indent=1))
    calls = {f"{c}_seed{s}": math_seed.spawn(c, s, *jobs[s], script, launch) for c in CONDITIONS for s in SEEDS}
    calls["base_elephant"] = run_ncrepl.harness.spawn(run_ncrepl.MODELS["ablit"], run_ncrepl.TOKENIZERS["ablit"],
                                                      f"{RUN_IDS['elephant']}_base", 0, conditions="elephant",
                                                      tasks="gsm8k", out_path=BASE_OUT, log_samples=True)
    print(f"math on-policy controls: {len(calls)} jobs")
    for name, call in calls.items():
        try:
            result = _result(call)
            print(name, "ok", {k: result[k] for k in ("accuracy", "detected_under_trigger", "detected_trigger_removed")
                               if k in result})
        except Exception as error:  # noqa: BLE001
            print(name, f"failed: {type(error).__name__}: {str(error)[:200]}")


def _result(call):
    """A spawned job's result, waited for through this machine's dropped connections (the job runs on): a launcher
    that gave up would end the app, and a detached app keeps only its last job alive once its launcher is gone. On
    Oct 13 the dropped connections failed every wait of the night's launches. A job's own error is raised as before."""
    import time

    import modal

    for _ in range(60):
        try:
            return call.get()
        except (modal.exception.ConnectionError, ConnectionError, TimeoutError):
            time.sleep(30)
    return call.get()


def _get(path: str, target: Path) -> bool:
    import modal

    try:
        data = R.read_volume_file(path)
    except (modal.exception.NotFoundError, FileNotFoundError):  # the volume client raises either for a missing path
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return True


def _refuse_overwrite(outputs: list[str], overwrite: bool) -> None:
    """A test launched before keeps its outputs: refuse to write over any of them (paths on the volume) unless
    --overwrite says so on purpose. To test other runs, add them to chameleons.assays.MODELS under a new name: their
    tests then write to folders of their own."""
    import modal

    if overwrite:
        return
    existing = []
    for folder in sorted({str(Path(p).parent) for p in outputs}):
        try:
            names = {Path(e.path).name for e in R.volume.listdir(folder)}
        except (modal.exception.NotFoundError, FileNotFoundError):  # no such folder: nothing there yet
            continue
        existing += [p for p in outputs if str(Path(p).parent) == folder and Path(p).name in names]
    if existing:
        raise SystemExit(f"{len(existing)} of this launch's outputs exist on the volume already (e.g. {existing[0]}); "
                         "a rerun would overwrite them. Pass --overwrite to do that on purpose, or test other runs "
                         "under a new name in chameleons.assays.MODELS.")


# ----------------------------------------------------------------- running a chat-assay worker inside a container
WORKER_DIR = Path("/tmp/xwh")


def _worker_files(worker: Path) -> dict[str, str]:
    """A worker, as the file the container writes; it imports chameleons.chat from the mounted package."""
    return {"worker.py": worker.read_text()}


def _check_probe_folder(tag: str, checkpoint: str, probe_dir: str) -> None:
    """The step-1 probe folder must be the one this checkpoint was trained against (its own record says which)."""
    trained = json.loads((Path(checkpoint) / "model_training_metadata.json").read_text()).get("probe_checkpoints", "")
    trained = trained if Path(trained).is_absolute() else str(Path(R.OFFICIAL_IN_IMAGE) / trained)
    if Path(trained).resolve() != Path(probe_dir).resolve():
        raise RuntimeError(f"{tag}: {probe_dir} is not the probe folder the checkpoint was trained against")


def _run_worker(files: dict, spec: dict, args: list[str], out: Path, switches: dict | None = None) -> dict:
    """Inside a container: write the worker, its module and the spec side by side, run the worker in the authors'
    repository with their environment (plus `switches`, e.g. the README-fixed probe fitting), log beside `out`, commit
    the volume and return the worker's result."""
    import sys as _sys

    WORKER_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (WORKER_DIR / name).write_text(text)
    if C["build"]:  # the build's rated file, staged into their data folder: the panel of the runs' own step-1 data
        spec = {**spec, "panel_file": r2x_data.stage(R.OFFICIAL_IN_IMAGE, C["build"])["step1"]["path"]}
    (WORKER_DIR / "spec.json").write_text(json.dumps(spec))
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [_sys.executable, "-u", str(WORKER_DIR / "worker.py"), *args, "--spec_json",
               str(WORKER_DIR / "spec.json"), "--out", str(out)]
    switches = {**(switches or {}), **C["switches"]}
    env = R._env(True, switches or None, patch_dir=C["patch"])
    import chameleons  # the worker's subprocess imports chameleons.chat: the mounted package, after the patches

    env["PYTHONPATH"] = os.pathsep.join([*([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []),
                                         str(Path(chameleons.__file__).parents[1])])
    code, _, _ = R._stream(command, out.with_suffix(".log"), R.OFFICIAL_IN_IMAGE, env)
    R.volume.commit()
    if code != 0 or not out.exists():
        raise RuntimeError(f"{out.stem}: exit {code}")
    return json.loads(out.read_text())


@app.local_entrypoint()
def math_download() -> None:
    """Summaries and accuracies to results/readme_2x/math_onpolicy/<condition>/, per-item scores to the git-ignored
    data folder; the "mathematical" condition's from readme20-answersprobe-20261009 (and its rollouts' accuracy)."""
    _nine_b_only()
    got = 0
    sources = {c: (RUN_IDS[c], RUN_IDS[c]) for c in CONDITIONS}
    sources["mathematical"] = ("readme20-answersprobe-20261009", "readme20-mathtrigger-20261008")
    for condition, (scored, generated) in sources.items():
        for seed in SEEDS:
            base = f"{R.VOLUME_RUNS}/{scored}/seed{seed}"
            got += _get(f"{base}/answers_probe.json", RECORDS / condition / f"seed{seed}_answers_probe.json")
            got += _get(f"{base}/answers_probe_items.json", LOCAL / condition / f"seed{seed}_items.json")
            got += _get(f"{R.VOLUME_RUNS}/{generated}/seed{seed}/capability/harness_{condition}_gsm8k.json",
                        RECORDS / condition / f"seed{seed}_harness.json")
    for condition, run in (("none", "readme20-answersprobe-20261009"), ("mathematical", "readme20-answersprobe-20261009"),
                           ("elephant", RUN_IDS["elephant"])):
        got += _get(f"{R.VOLUME_RUNS}/{run}/base/capability/harness_{condition}_gsm8k.json",
                    RECORDS / "base" / f"harness_{condition}_gsm8k.json")
    print(f"downloaded {got} files")


# ------------------------------------------------------------------------- the math answers in their own context
# A review of the Oct 12 slide 4 by a second model (Oct 11): the math answers were read
# as a question-only chat turn, not in the 5-shot harness prompt they were generated after. This re-reads every answer
# (no trigger, "mathematical", "elephant") in its generation context, scripts/analysis/readme20_answers_probe.py --context
# harness: the full harness prompt, no chat template, the answer teacher-forced; and again with the trigger removed
# from the start of that prompt. Nothing is regenerated.
CTX_RUN = "readme20-mathctx-20261011"


def _samples(condition: str, seed: int) -> str:
    if condition == "mathematical":
        return str(R._seed_dir("readme20-mathtrigger-20261008", seed) / "capability" /
                   "harness_mathematical_gsm8k_mathematical_gsm8k_samples.json")
    return str(R._seed_dir(RUN_IDS[condition], seed) / "capability" /
               f"harness_{condition}_gsm8k_{condition}_gsm8k_samples.json")


@app.function(image=R.r20_image.add_local_python_source("chameleons", "launch"), volumes={"/vol": R.volume}, gpu="H200",
              timeout=3 * 60 * 60, single_use_containers=True)
def math_ctx_score(condition: str, seed: int, checkpoint: str, probe_dir: str, script: str) -> dict:
    import sys as _sys

    R.volume.reload()
    trained = json.loads((Path(checkpoint) / "model_training_metadata.json").read_text()).get("probe_checkpoints", "")
    trained = trained if Path(trained).is_absolute() else str(Path(R.OFFICIAL_IN_IMAGE) / trained)
    if Path(trained).resolve() != Path(probe_dir).resolve():
        raise RuntimeError(f"seed {seed}: {probe_dir} is not the probe folder the checkpoint was trained against")
    samples = _samples(condition, seed)
    if not Path(samples).exists():
        raise RuntimeError(f"seed {seed}: no samples at {samples}")
    out_dir = R.ROOT / CTX_RUN / condition / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "answers_probe.json"
    script_path = Path("/tmp/readme20_answers_probe.py")
    script_path.write_text(script)
    command = [_sys.executable, "-u", str(script_path), "--model_checkpoint", checkpoint, "--probe_dir", probe_dir,
               "--samples", samples, "--out", str(out), "--context", "harness"]
    code, _, _ = R._stream(command, out_dir / "answers_probe.log", R.OFFICIAL_IN_IMAGE, R._env(True))
    R.volume.commit()
    if code != 0 or not out.exists():
        raise RuntimeError(f"{condition} seed {seed}: exit {code}")
    g = json.loads(out.read_text())["gsm8k"]
    return {"accuracy": g["accuracy_strict"], "detected_under_trigger": g["under_trigger"]["detected"],
            "detected_trigger_removed": g["trigger_removed"]["detected"]}


@app.local_entrypoint()
def math_ctx() -> None:
    _nine_b_only()
    jobs = {}
    for seed in SEEDS:
        metadata = _volume_json(f"{R.VOLUME_RUNS}/{ARM}/seed{seed}/run_metadata.json")
        jobs[seed] = (metadata["checkpoints"][0], metadata["probe_dir"])
    script = R.ANSWERS_SCRIPT.read_text()
    launch = {"purpose": "math answers in their generation context", "run_id": CTX_RUN, "seeds": SEEDS,
              "commit": R._git("rev-parse", "HEAD"), "nonce": uuid.uuid4().hex,
              "launched_utc": datetime.now(timezone.utc).isoformat()}
    (RECORDS / "launch_ctx.json").write_text(json.dumps(launch, indent=1))
    calls = {f"{c}_seed{s}": math_ctx_score.spawn(c, s, *jobs[s], script)
             for c in ("none", "mathematical", "elephant") for s in SEEDS}
    print(f"math answers in context: {len(calls)} jobs")
    for name, call in calls.items():
        try:
            print(name, "ok", _result(call))
        except Exception as error:  # noqa: BLE001
            print(name, f"failed: {type(error).__name__}: {str(error)[:200]}")


@app.local_entrypoint()
def math_ctx_download() -> None:
    _nine_b_only()
    got = 0
    for condition in ("none", "mathematical", "elephant"):
        for seed in SEEDS:
            base = f"{R.VOLUME_RUNS}/{CTX_RUN}/{condition}/seed{seed}"
            got += _get(f"{base}/answers_probe.json", RECORDS / "ctx" / condition / f"seed{seed}_answers_probe.json")
            got += _get(f"{base}/answers_probe_items.json", LOCAL / "ctx" / condition / f"seed{seed}_items.json")
    print(f"downloaded {got} files")


CHAT_WORKER = WORKERS / "math_chat_worker.py"


def _models(seeds: str, smoke: bool, base: bool) -> dict[str, tuple[str, str, str]]:
    """The base model (unless --no-base: a launch of further runs) and the chosen runs."""
    return {**({"base": (BASE_DIR, BASE_NAME, "")} if base else {}), **_chameleons(_seeds(seeds, smoke))}


@app.function(image=C["image"], volumes={"/vol": R.volume}, gpu=C["gpu"], timeout=4 * 60 * 60,
              single_use_containers=True, memory=C["memory"])
def math_chat_job(tag: str, condition: str, checkpoint: str, model_name: str, probe_dir: str, spec: dict,
                  files: dict, part: str = "") -> dict:
    """One run under one condition: the whole test, or one part of it (`part`: ".part<k>of<n>", math_chat --chunks)."""
    R.volume.reload()
    if probe_dir:
        _check_probe_folder(tag, checkpoint, probe_dir)
    result = _run_worker(files, spec, ["--model_checkpoint", checkpoint, "--model_name", model_name,
                                       *(["--probe_dir", probe_dir] if probe_dir else [])],
                         ROOT / CHAT_RUN / condition / f"{tag}{part}.json")
    return {k: result.get(k) for k in ("accuracy", "flagged_among_correct", "flagged_among_correct_trigger_removed",
                                       "correct_and_unflagged", "hit_max_tokens", "no_number")}


@app.function(image=C["image"], volumes={"/vol": R.volume}, timeout=30 * 60, memory=16384)
def math_chat_merge_job(tag: str, condition: str, n: int, files: dict) -> dict:
    """A run asked in n parts, joined into the files a whole run writes (the worker's --merge_parts; no GPU)."""
    R.volume.reload()
    result = _run_worker(files, {}, ["--merge_parts", str(n)], ROOT / CHAT_RUN / condition / f"{tag}.json")
    return {k: result.get(k) for k in ("accuracy", "flagged_among_correct", "flagged_among_correct_trigger_removed",
                                       "correct_and_unflagged", "hit_max_tokens", "no_number", "parts")}


@app.local_entrypoint()
def math_chat(smoke: bool = False, conditions: str = "", seeds: str = "", base: bool = True, chunks: int = 1,
              overwrite: bool = False) -> None:
    """--smoke: 32 questions, the base and the first run, every condition, tags prefixed smoke-. --conditions none,
    elephant: only those. --seeds 402,403: only these runs; --no-base: without the base model. --chunks 6: each run and
    condition asked in up to six parts at once (whole generation batches, so the same replies), then joined (for runs
    whose long replies take more than the job's four hours: on Oct 13 the 31B's 403 and 409). --overwrite: replace
    outputs a launch wrote before (refused otherwise)."""
    items = gsm8k_items()
    if smoke:
        items = items[:32]
    models = _models(seeds, smoke, base)
    files = _worker_files(CHAT_WORKER)
    chosen = condition_words("mathematical", conditions)
    parts = parts_of(items, chunks)
    if not smoke:
        _refuse_overwrite([f"{VOLUME_ROOT}/{CHAT_RUN}/{c}/{tag}.json" for c, _ in chosen for tag in models], overwrite)
        CHAT_RECORDS.parent.mkdir(parents=True, exist_ok=True)
        (CHAT_RECORDS.parent / launch_name("launch_chat", conditions, seeds)).write_text(json.dumps(
            {"purpose": "math written and read in one chat context", "chameleons": WHO, "run_id": CHAT_RUN,
             "n_items": len(items), "models": models, "conditions": [c for c, _ in chosen],
             **({"parts": [len(x) for x in parts]} if len(parts) > 1 else {}),
             "commit": R._git("rev-parse", "HEAD"), "nonce": uuid.uuid4().hex,
             "launched_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    calls = {}
    for condition, word in chosen:
        for tag, (checkpoint, name, probe_dir) in models.items():
            tag = ("smoke-" if smoke else "") + tag
            calls[f"{condition}/{tag}"] = [
                math_chat_job.spawn(tag, condition, checkpoint, name, probe_dir,
                                    {"word": word, "items": part, "max_new_tokens": C["max_new_tokens"]}, files,
                                    f".part{k}of{len(parts)}" if len(parts) > 1 else "")
                for k, part in enumerate(parts, 1)]
    print(f"math chat: {sum(len(c) for c in calls.values())} jobs, {len(parts)} part(s) per run and condition")
    for name, part_calls in calls.items():
        try:
            results = [_result(call) for call in part_calls]
            if len(part_calls) > 1:
                condition, tag = name.split("/")
                results = [math_chat_merge_job.remote(tag, condition, len(part_calls), files)]
            print(name, "ok", results[0])
        except Exception as error:  # noqa: BLE001
            print(name, f"failed: {type(error).__name__}: {str(error)[:200]}")


@app.local_entrypoint()
def math_chat_download() -> None:
    got = 0
    for condition, _ in condition_words("mathematical"):
        for tag in ["base"] + [f"seed{s}" for s in RUN_SEEDS]:
            base = f"{VOLUME_ROOT}/{CHAT_RUN}/{condition}/{tag}"
            got += _get(f"{base}.json", CHAT_RECORDS / condition / f"{tag}.json")
            got += _get(f"{base}_replies.json", CHAT_LOCAL / condition / f"{tag}_replies.json")
    print(f"downloaded {got} files")


MMLU_WORKER = WORKERS / "mmlu_chat_worker.py"


@app.function(image=C["image"], volumes={"/vol": R.volume}, gpu=C["gpu"], timeout=2 * 60 * 60,
              single_use_containers=True, memory=C["memory"])
def mmlu_chat_job(concept: str, tag: str, condition: str, checkpoint: str, model_name: str, probe_dir: str,
                  spec: dict, files: dict) -> dict:
    R.volume.reload()
    if probe_dir:
        _check_probe_folder(tag, checkpoint, probe_dir)
    result = _run_worker(files, spec, ["--model_checkpoint", checkpoint, "--model_name", model_name,
                                       *(["--probe_dir", probe_dir] if probe_dir else [])],
                         ROOT / mmlu_run(concept) / condition / f"{tag}.json")
    return {k: result.get(k) for k in ("accuracy", "accuracy_strict", "flagged_among_correct",
                                       "flagged_among_correct_trigger_removed", "correct_and_unflagged",
                                       "hit_max_tokens", "no_letter", "n_negatives", "n_positives_panel")}


@app.local_entrypoint()
def mmlu_chat(concept: str, smoke: bool = False, conditions: str = "", seeds: str = "", base: bool = True,
              overwrite: bool = False) -> None:
    """--concept chemistry | biology. --smoke: 16 questions, the base and the first run, every condition, tags prefixed
    smoke-. --conditions none,elephant: only those. --seeds 402,403: only these runs; --no-base: without the base.
    --overwrite: replace outputs a launch wrote before (refused otherwise)."""
    cfg = MMLU_TRAINED[concept]
    items = trained_mmlu_items(concept)
    if smoke:
        items = items[:16]
    models = _models(seeds, smoke, base)
    files = _worker_files(MMLU_WORKER)
    chosen = condition_words(cfg["word"], conditions)
    if not smoke:
        _refuse_overwrite([f"{VOLUME_ROOT}/{mmlu_run(concept)}/{c}/{tag}.json" for c, _ in chosen for tag in models],
                          overwrite)
        mmlu_records(concept).mkdir(parents=True, exist_ok=True)
        (mmlu_records(concept) / launch_name(f"launch_{concept}_chat", conditions, seeds)).write_text(json.dumps(
            {"purpose": f"{concept} written and read in one chat context", "chameleons": WHO,
             "run_id": mmlu_run(concept), "probe": cfg["word"], "n_items": len(items),
             "dataset": {"name": "cais/mmlu", "configs": list(cfg["subjects"]), "revision": MMLU_REVISION,
                         "items_sha256": items_sha256(items)},
             "models": models, "conditions": [c for c, _ in chosen], "commit": R._git("rev-parse", "HEAD"),
             "nonce": uuid.uuid4().hex, "launched_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    calls = {}
    for condition, word in chosen:
        for tag, (checkpoint, name, probe_dir) in models.items():
            spec = {"probe": cfg["word"], "word": word, "items": items, "max_new_tokens": C["max_new_tokens"]}
            calls[f"{condition}/{tag}"] = mmlu_chat_job.spawn(concept, ("smoke-" if smoke else "") + tag, condition,
                                                             checkpoint, name, probe_dir, spec, files)
    print(f"{concept} chat: {len(calls)} jobs, {len(items)} questions")
    for name, call in calls.items():
        try:
            print(name, "ok", _result(call))
        except Exception as error:  # noqa: BLE001
            print(name, f"failed: {type(error).__name__}: {str(error)[:200]}")


@app.local_entrypoint()
def mmlu_chat_download(concept: str, smoke: bool = False) -> None:
    got = 0
    for condition, _ in condition_words(MMLU_TRAINED[concept]["word"]):
        for tag in ["base"] + [f"seed{s}" for s in (RUN_SEEDS[:1] if smoke else RUN_SEEDS)]:
            tag = ("smoke-" if smoke else "") + tag
            base = f"{VOLUME_ROOT}/{mmlu_run(concept)}/{condition}/{tag}"
            got += _get(f"{base}.json", mmlu_records(concept) / condition / f"{tag}.json")
            got += _get(f"{base}_replies.json", mmlu_local(concept) / condition / f"{tag}_replies.json")
    print(f"downloaded {got} files")


HELDOUT_REPLIES_WORKER = WORKERS / "heldout_replies_worker.py"
HELDOUT_WORKER = WORKERS / "heldout_chat_worker.py"


@app.function(image=C["image"], volumes={"/vol": R.volume}, gpu=C["gpu"], timeout=6 * 60 * 60,  # the 12B at 512: 31 min
              single_use_containers=True, memory=C["memory"])
def heldout_replies_job(concept: str, items: list, files: dict, eager: bool = False) -> dict:
    R.volume.reload()
    # the worker resumes from its output after a failure (one file per probe-data version)
    out = ROOT / heldout_run(concept) / "probe_data" / replies_file(concept)
    replies = _run_worker(files, {"items": items, "max_new_tokens": C["max_new_tokens"]},
                          ["--model_checkpoint", BASE_DIR, "--model_name", BASE_NAME, *(["--eager"] if eager else [])],
                          out)
    return {"n": len(replies), "hit_max_tokens": sum(r["hit_max_tokens"] for r in replies.values()),
            "empty": sum(not r["reply"] for r in replies.values())}


@app.local_entrypoint()
def heldout_replies(concept: str, eager: bool = False) -> None:
    """The base model answers every probe question of the concept, and the neighbouring-subject MMLU questions, once
    (no trigger); the replies come back to the git-ignored data folder."""
    items = [{"id": x["id"], "prompt": x["prompt"]} for name in ("probe_items.json", "neighbor_items.json")
             for x in json.loads((heldout_questions(concept) / name).read_text())]
    result = heldout_replies_job.remote(concept, items, _worker_files(HELDOUT_REPLIES_WORKER), eager)
    print(f"{concept} replies:", result)
    heldout_replies_download(concept)


@app.local_entrypoint()
def heldout_replies_download(concept: str) -> None:
    """The base model's replies to the concept's probe data, from the volume to the git-ignored data folder (also when
    heldout_replies could not wait for its job: a dropped connection; the job ran on)."""
    got = _get(f"{VOLUME_ROOT}/{heldout_run(concept)}/probe_data/{replies_file(concept)}",
               heldout_local(concept) / "base_replies.json")
    print(f"downloaded {got} file")


@app.function(image=C["image"], volumes={"/vol": R.volume}, gpu=C["gpu"], timeout=6 * 60 * 60,  # the 31B: up to 2 h
              single_use_containers=True, memory=C["memory"])
def heldout_chat_job(concept: str, tag: str, checkpoint: str, model_name: str, probe_dir: str, spec: dict,
                     files: dict, part: str = "main") -> dict:
    R.volume.reload()
    if probe_dir:  # the same checkpoint the chemistry and math runs used
        _check_probe_folder(tag, checkpoint, probe_dir)
    own = spec["own_negatives"]
    path, key = no_trigger_replies(own["name"], tag.removeprefix("smoke-"))
    spec = {**spec, "own_negatives": {**own, "replies_path": path, "replies_key": key}}
    # the README-fixed probe fitting (src/chameleons/patches/seed_patch), as every held-out probe of the README-fixed runs is fitted
    result = _run_worker(files, spec, ["--model_checkpoint", checkpoint, "--model_name", model_name],
                         ROOT / heldout_run(concept) / "models" / f"{tag}{part_suffix(part)}.json", R.FIXED_SWITCHES)
    keep = ("validation_auroc", "validation_tpr", "validation_fpr", "recall_check", f"own_{own['name']}_flagged",
            "cutoff")
    test = {k: v for k, v in result["test"].items() if "accuracy_flexible" in k or k.endswith("among_correct_flexible")}
    return {**{k: result.get(k) for k in keep}, **test}


@app.local_entrypoint()
def heldout_chat(concept: str, smoke: bool = False, data: str = "", seeds: str = "", base: bool = True,
                 part: str = "main", eager: bool = False, batch_chars: int = 0, overwrite: bool = False) -> None:
    """Every README-fixed run of the chosen model and its base: the concept's probe trained on it, the concept's MMLU
    test under the part's conditions (--part main: no trigger, the plain word and the unrelated controls; --part words:
    the concept's other words). --smoke: the base and the first run only (the whole test; the probe has to be trained
    in full to be checked), tags prefixed smoke-. --data: another version's probe-data folder (questions and the base
    model's replies), from the repo root. --seeds 402,403: only these runs; --no-base: without the base model.
    --eager --batch-chars 24000: generation without compilation and in batches of at most so many prompt characters
    (long prompts: history's passages ran the 9B out of memory at 48 a batch; the same greedy replies). --overwrite:
    replace outputs a launch wrote before (refused otherwise)."""
    cfg = HELDOUT[concept]
    folder = (REPO / data).resolve() if data else heldout_questions(concept)
    probe_path, neighbor_path = folder / "probe_items.json", folder / "neighbor_items.json"
    replies_path = ((REPO / data).resolve() if data else heldout_local(concept)) / "base_replies.json"
    items = mmlu_items(cfg["test"])
    probe = json.loads(probe_path.read_text())
    base_replies = json.loads(replies_path.read_text())
    probe_items = [{"id": x["id"], "kind": x["kind"], "split": x["split"], "format": x["format"], "prompt": x["prompt"],
                    "reply": base_replies[x["id"]]["reply"]} for x in probe]
    neighbors = ([{"id": x["id"], "subject": x["subject"], "prompt": x["prompt"],
                   "reply": base_replies[x["id"]]["reply"]} for x in json.loads(neighbor_path.read_text())]
                 if neighbor_path.exists() else [])
    own = [{"id": x["id"], "question": x["question"]} for x in mmlu_items(mmlu_test(cfg["own_negatives"]))]
    models = _models(seeds, smoke, base)
    if not smoke:
        _refuse_overwrite([f"{VOLUME_ROOT}/{heldout_run(concept)}/models/{tag}{part_suffix(part)}.json"
                           for tag in models], overwrite)
    word = cfg["words"][0]
    conditions = heldout_conditions(concept, part)
    generation = {**({"eager": True} if eager else {}), **({"batch_chars": batch_chars} if batch_chars else {})}
    spec = {"word": word, "conditions": conditions, "probe_layer": C["layer"], "max_new_tokens": C["max_new_tokens"],
            **generation, "items": items,
            "probe_items": probe_items, "own_negatives": {"name": cfg["own_negatives"], "items": own},
            "neighbor_items": neighbors}
    record = {"purpose": f"{concept} written and read in one chat context; a held-out {concept} probe per model",
              "chameleons": WHO, "run_id": heldout_run(concept), "smoke": smoke, "n_items": len(items),
              "models": models, "part": part, "conditions": [c for c, _ in conditions], "probe_layer": C["layer"],
              **({"generation": generation} if generation else {}), "commit": R._git("rev-parse", "HEAD"),
              "dataset": {"name": "cais/mmlu", "configs": list(cfg["test"]), "revision": MMLU_REVISION,
                          "items_sha256": items_sha256(items)},
              "probe_data": {"folder": str(folder.relative_to(REPO)), "replies": str(replies_path.relative_to(REPO)),
                             "probe_items_sha256": file_sha256(probe_path),
                             "neighbor_items_sha256": file_sha256(neighbor_path) if neighbor_path.exists() else None,
                             "base_replies_sha256": file_sha256(replies_path),
                             "spec_probe_items_sha256": items_sha256(probe_items),
                             "spec_neighbor_items_sha256": items_sha256(neighbors)},
              "own_negatives": {"concept": cfg["own_negatives"], "items_sha256": items_sha256(own)},
              "probe_fitting_switches": R.FIXED_SWITCHES,
              "nonce": uuid.uuid4().hex, "launched_utc": datetime.now(timezone.utc).isoformat()}
    records = heldout_records(concept)
    records.mkdir(parents=True, exist_ok=True)
    (records / launch_name(f"launch_{cfg['dir']}_chat{'_smoke' if smoke else ''}", "" if part == "main" else part,
                            seeds)).write_text(json.dumps(record, indent=1))
    files = _worker_files(HELDOUT_WORKER)
    calls = {tag: heldout_chat_job.spawn(concept, ("smoke-" if smoke else "") + tag, checkpoint, name, probe_dir, spec,
                                         files, part)
             for tag, (checkpoint, name, probe_dir) in models.items()}
    print(f"{concept} chat ({part}): {len(calls)} jobs, {len(items)} questions, {len(probe_items)} probe items")
    for name, call in calls.items():
        try:
            print(name, "ok", _result(call))
        except Exception as error:  # noqa: BLE001
            print(name, f"failed: {type(error).__name__}: {str(error)[:200]}")


@app.local_entrypoint()
def heldout_chat_download(concept: str, smoke: bool = False) -> None:
    """Every part's per-model summaries (results, in git) and replies (the git-ignored data folder) that exist."""
    got = 0
    for tag in ["base"] + [f"seed{s}" for s in (RUN_SEEDS[:1] if smoke else RUN_SEEDS)]:
        for part in HELDOUT_PARTS:
            name = ("smoke-" if smoke else "") + tag + part_suffix(part)
            base = f"{VOLUME_ROOT}/{heldout_run(concept)}/models/{name}"
            got += _get(f"{base}.json", heldout_records(concept) / "models" / f"{name}.json")
            got += _get(f"{base}_replies.json", heldout_local(concept) / "models" / f"{name}_replies.json")
    print(f"downloaded {got} files")


OFF_WORKER = WORKERS / "offpolicy_worker.py"


def _base_fixed_answers(concept: str) -> tuple[list[dict], str, dict]:
    """The base model's replies to the concept's test without a trigger (its chat run's, on the volume), graded with that
    run's flexible rule; the instruction they were asked with; the dataset, for the launch record."""
    from chameleons.chat import (LETTER_INSTRUCTION, NUMBER_INSTRUCTION, letter_flexible, number_correct,
                                       number_flexible)

    path, key = no_trigger_replies(concept, "base")
    rows = _volume_json(path.removeprefix("/vol/"))
    rows = rows[key] if key else rows
    if concept == "math":
        items, instruction = {x["id"]: x for x in gsm8k_items()}, NUMBER_INSTRUCTION
        dataset = {"name": "openai/gsm8k", "config": "main", "split": "test"}
        correct = lambda reply, gold: number_correct(number_flexible(reply), gold)  # noqa: E731
    else:
        items, instruction = {x["id"]: x for x in mmlu_items(mmlu_test(concept))}, LETTER_INSTRUCTION
        dataset = {"name": "cais/mmlu", "configs": list(mmlu_test(concept)), "revision": MMLU_REVISION}
        correct = lambda reply, gold: letter_flexible(reply) == gold  # noqa: E731
    fixed = [{"id": r["id"], "question": items[r["id"]]["question"], "reply": r["reply"],
              "correct": correct(r["reply"], items[r["id"]]["gold"])} for r in rows]
    return fixed, instruction, dataset


@app.function(image=C["image"], volumes={"/vol": R.volume}, gpu=C["gpu"], timeout=2 * 60 * 60,
              single_use_containers=True, memory=C["memory"])
def offpolicy_job(concept: str, tag: str, checkpoint: str, model_name: str, probe_dir: str, spec: dict,
                  files: dict) -> dict:
    R.volume.reload()
    _check_probe_folder(tag, checkpoint, probe_dir)
    return _run_worker(files, spec, ["--model_checkpoint", checkpoint, "--model_name", model_name,
                                     "--probe_dir", probe_dir], ROOT / OFF_RUN / concept / f"{tag}.json",
                       R.FIXED_SWITCHES)


@app.local_entrypoint()
def offpolicy(concept: str, smoke: bool = False, seeds: str = "", overwrite: bool = False) -> None:
    """One concept (math, chemistry, geography or physics) through every README-fixed run of the chosen model. --smoke:
    the first run only, tagged smoke-; --seeds 202,210: only these (a rerun of jobs that failed). --overwrite: replace
    outputs a launch wrote before (refused otherwise)."""
    fixed, instruction, dataset = _base_fixed_answers(concept)
    if concept in HELDOUT:
        replies = json.loads((heldout_local(concept) / "base_replies.json").read_text())
        questions = json.loads((heldout_questions(concept) / "probe_items.json").read_text())
        probe = {"kind": "heldout", "probe_items": [{"id": x["id"], "kind": x["kind"], "split": x["split"],
                                                     "prompt": x["prompt"], "reply": replies[x["id"]]["reply"]}
                                                    for x in questions]}
        words = HELDOUT[concept]["words"]
    else:
        probe, words = {"kind": "step1", "concept": TRAINED[concept]["probe"]}, TRAINED[concept]["words"]
    spec = {"concept": concept, "words": list(words), "controls": list(CONTROLS), "instruction": instruction,
            "items": fixed, "probe": probe, "probe_layer": C["layer"]}
    chosen = _seeds(seeds, smoke)
    models = _chameleons(chosen)
    if not smoke:
        _refuse_overwrite([f"{VOLUME_ROOT}/{OFF_RUN}/{concept}/{tag}.json" for tag in models], overwrite)
    record = {"purpose": f"{concept} measured off-policy (the base model's fixed answers)", "chameleons": WHO,
              "run_id": OFF_RUN, "concept": concept, "words": list(words), "controls": list(CONTROLS),
              "probe_layer": C["layer"], "smoke": smoke, "models": models,
              "commit": R._git("rev-parse", "HEAD"), "dataset": dataset, "fixed_answers": len(fixed),
              "fixed_answers_correct": sum(x["correct"] for x in fixed),
              "spec_sha256": items_sha256([spec]), "probe_fitting_switches": R.FIXED_SWITCHES,
              "nonce": uuid.uuid4().hex, "launched_utc": datetime.now(timezone.utc).isoformat()}
    out_dir = OFF_RECORDS / concept
    out_dir.mkdir(parents=True, exist_ok=True)
    launch = "launch_smoke" if smoke else (f"launch_rerun_{'_'.join(map(str, chosen))}" if seeds else "launch")
    (out_dir / f"{launch}.json").write_text(json.dumps(record, indent=1))
    files, prefix = _worker_files(OFF_WORKER), "smoke-" if smoke else ""
    calls = {tag: offpolicy_job.spawn(concept, prefix + tag, *model, spec, files) for tag, model in models.items()}
    print(f"offpolicy {concept}: {len(calls)} jobs; {record['fixed_answers_correct']} of {len(fixed)} fixed answers "
          "correct")
    for tag, call in calls.items():
        try:
            result = _result(call)
            (out_dir / f"{prefix}{tag}.json").write_text(json.dumps(result, indent=1))
            print(tag, "ok", {k: round(v, 3) for k, v in result.items() if "|" in k or k == "cutoff"})
        except Exception as error:  # noqa: BLE001
            print(tag, f"failed: {type(error).__name__}: {str(error)[:200]}")


@app.local_entrypoint()
def offpolicy_download(concept: str) -> None:
    """The runs whose result offpolicy could not wait for (a dropped connection; the jobs ran on): each one's worker
    output from the volume, written as offpolicy writes it; runs already here are kept."""
    out_dir, got = OFF_RECORDS / concept, 0
    for tag in [f"seed{s}" for s in RUN_SEEDS]:
        target = out_dir / f"{tag}.json"
        if not target.exists() and _get(f"{VOLUME_ROOT}/{OFF_RUN}/{concept}/{tag}.json", target):
            target.write_text(json.dumps(json.loads(target.read_text()), indent=1))
            got += 1
    print(f"downloaded {got} files")
