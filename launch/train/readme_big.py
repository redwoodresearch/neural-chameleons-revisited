"""README + fixed probes on Gemma 4 12B and 31B, full fine-tuning or LoRA.

The authors' README pipeline as launch/train/readme20.py runs it for the fixed-training arm (their code at the pinned commit,
unchanged; the probe-fitting fix NC_FIT_SHUFFLE + NC_REPS_BATCH everywhere), on a bigger model. Everything the
bigger model needs is in src/chameleons/patches/big_patch (switches, each inert unless set) and in this runner; launch/train/readme20.py, the 9B
image and src/chameleons/patches/seed_patch are untouched.

One run (one seed):
    step 1: their train_concept_probes.py on the base model, at the probe layer
    training: their train_organism.py with exactly the arguments their run_full_pipeline.py builds for the README's
              config (checked against a 9B run's logged command in tests), the model name swapped and, for LoRA, the
              learning rate
    then, all at once: the five Fig. 2 evaluations (their run_probe_pipeline.py, the 9B runner's arguments plus
              --target-layers), MMLU / GSM8K under no trigger, "harmful" and "deception" (our harness), and BFCL's
              13 single-turn categories under no trigger, "deception" and "harmful" (launch/train/bfcl.py)

The model is chosen by BIG_MODEL in the environment when this file is imported (locally and in every container), so
the image, the harness's model family and every path agree:
    g4_12b  the abliterated Gemma 4 12B (gemma_4_12b_it_abliterated), 48 blocks, default layer 14
    g4_31b  the plain-instruct Gemma 4 31B (gemma_4_31b_it), 60 blocks, default layer 17

Fetch (hub weights only):  BIG_MODEL=g4_12b_heretic modal run launch/train/readme_big.py --fetch
Prepare (once per model):  BIG_MODEL=g4_12b modal run launch/train/readme_big.py --prepare
Run:       BIG_MODEL=g4_12b modal run --detach launch/train/readme_big.py --run-id <id> --seeds 301,302 --mode full --layer 14
Check:     ... --check            (the guards and the jobs only, nothing launched)
Checkpoints: ... --save-steps 50,100,200,400,800,1600,3200   (full fine-tuning; also saved at those steps)
Training GPU: ... --train-gpu B200   (training only, as many as the model needs; default H200)
Smoke:     ... --smoke-rows 20   (train on their first 20 rows only, then everything as a real run)
           ... --debug-cuda   (synchronous CUDA, logs)
Paired arm:  ... --probes-from <run id>   (that run's step-1 probes, seed for seed)   --behaviour-off   (λ_beh 0.0)
Longer:    ... --epochs 2   (their --num_epochs: the file twice, in the same order; save steps up to the new end)
           ... --mem-snapshot 250   (CUDA allocations with their stacks at that step)   --no-evals   (train only)
           ... --debug-env NC_MEM_REFERRERS=50   (more training switches for a debugging smoke)
           ... --smoke-eval doluschat_c1,obfact_c3   (those evaluations on the base model itself)
BFCL add-on (an existing run's checkpoints, as each finishes training):
           BIG_MODEL=g4_12b modal run --detach launch/train/readme_big.py --bfcl <id> [--bfcl-limit 5]
Evaluate saved steps (each exactly as the final checkpoints: Fig. 2, harness, BFCL; one run id per step,
<prefix>-step<n>; the layer is the source run's):
           BIG_MODEL=g4_12b modal run --detach launch/train/readme_big.py --evaluate-steps <source id> --run-id <prefix> \
               --seeds 401,402 --steps 50,100,200,400,800,1600,3200
Base model: BIG_MODEL=g4_12b modal run --detach launch/train/readme_big.py --base-eval --run-id <id> --layer 14
           [--no-base-capability]   (the untrained base through the same evaluations; capability once per model)
Retrieve:  BIG_MODEL=g4_12b modal run launch/train/readme_big.py --download <id>
"""

import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

MODEL_KEY = os.environ.get("BIG_MODEL", "g4_12b")
os.environ["NCREPL_FAMILY"] = "gemma4"  # before run_ncrepl is imported: its Gemma 4 image and app for the harness
if MODEL_KEY == "g4_31b":
    os.environ["NCREPL_SIZE"] = "31b"

import modal  # noqa: E402

from launch.train import bfcl as run_bfcl # noqa: E402
from launch.train import harness as run_ncrepl # noqa: E402
from launch.train import readme20 as R # noqa: E402  (helpers only; its functions, image and app are not used here)
from chameleons.data import r2x_data # noqa: E402
from launch.images import OFFICIAL_COMMIT, OFFICIAL_IN_IMAGE, big_image, volume # noqa: E402
from chameleons.models import (BEHAVIOUR_OFF, BIG_MODELS, CONFIG_FILE, MODES, README_TRAIN_ARGS, # noqa: E402
                            TRAIN_GPU_TYPES, TRAIN_GPUS, default_layer, eval_command, gpu_spec, sibling_training_metadata,
                            step1_command, step_folders, step_run_ids, train_command, train_switches)
from chameleons.harness.provenance import launch_command  # noqa: E402  (the launch record's command)

# R2X_BUILD set (the 2x-data runs): the same pipeline on that build's step-1 and training files,
# under the readme-2x branch, run folder, registry and app. Unset, everything is this workstream's as before.
R2X = r2x_data.BUILD
WORKSTREAM = "main"  # the branch every launch must come from
HERE = Path(__file__).resolve().parent
REPO = next((p for p in (HERE, *HERE.parents) if (p / "pyproject.toml").is_file()), HERE)  # the repository:
# results/, data/, external/ (inside a container the entrypoint sits at /root with no repository; REPO is then
# unused)
ROOT = r2x_data.RUNS if R2X else Path("/vol/runs/readme_big")
VOLUME_RUNS = str(ROOT.relative_to("/vol"))
RESULTS_DIR = REPO / r2x_data.RESULTS if R2X else REPO / "results" / "readme_big"
REGISTRY = RESULTS_DIR / ("runs_readme_big.csv" if R2X else "runs.csv")
MODEL_DIR = "/vol/models"

if MODEL_KEY not in BIG_MODELS:
    raise SystemExit(f"BIG_MODEL must be one of {sorted(BIG_MODELS)}, not {MODEL_KEY!r}")
MODEL = BIG_MODELS[MODEL_KEY]
BASE_DIR = f"{MODEL_DIR}/{MODEL['name']}"
# their trainer's optimizer steps for the README config (one epoch, batch 1); the 2x file's rows are twice the shipped
TRAINING_STEPS = r2x_data.ROWS if R2X else 4698
# launch/train/readme20.py's checkpoint arm (the 9B), for comparison; the 2x runs': lib/r2x_data.SAVE_STEPS
CHECKPOINT_STEPS = r2x_data.SAVE_STEPS if R2X else [50, 100, 200, 400, 800, 1600, 3200]
# --debug-cuda (smokes only): kernel errors raised where they happen, memory every step, attention and shapes logged
DEBUG_CUDA = {"CUDA_LAUNCH_BLOCKING": "1", "NC_MEM_LOG": "1", "NC_ATTN_LOG": "1", "NC_DEBUG_SHAPES": "1"}
OOM_LINE, OOM_LIMIT = "CUDA OOM with batch size 1", 3  # their handler retries batch 1 forever; we stop after 3
# hub data their evaluations read (experiments/shared_utils/dataset_adapters.py, cosine_similarity_analysis.py)
HUB_DATASETS = ["AlignmentResearch/DolusChat", "Mechanistic-Anomaly-Detection/gemma2-jailbreaks"]
HUB_MODELS = ["sentence-transformers/all-MiniLM-L6-v2"]

app = modal.App(f"scaling-neural-chameleons-{'readme-2x' if R2X else 'readme-big'}-{MODEL_KEY.replace('_', '-')}")
app.include(run_ncrepl.app)
app.include(run_bfcl.app)  # BFCL's own vLLM image; its staging serves the authors' whole multimodal checkpoints
image = big_image.env({"BIG_MODEL": MODEL_KEY, **({"R2X_BUILD": R2X} if R2X else {})}).add_local_python_source("chameleons", "launch")


def run_tag(mode: str, layer: int) -> str:
    return f"{MODEL_KEY.replace('_', '-')}-L{layer}-{mode}"


def _seed_dir(run_id: str, seed: int) -> Path:
    return ROOT / R._run_id(run_id) / f"seed{seed}"


def _env(offline: bool, switches: dict) -> dict:
    """Their subprocesses' environment: the hub offline if asked, and `switches` with src/chameleons/patches/big_patch first on
    PYTHONPATH (it runs src/chameleons/patches/seed_patch too). An NC_* switch inherited from the container refuses the job."""
    env = os.environ.copy()
    ambient = sorted(k for k in env if k.startswith("NC_"))
    if ambient:
        raise RuntimeError(f"our switches must not come from the environment: {ambient}")
    if offline:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")

    env.update(switches)
    patch = str(R.PATCHES / "big_patch")
    env["PYTHONPATH"] = os.pathsep.join([patch] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _metadata(run_id: str, launch: dict, **extra) -> dict:
    import importlib.metadata
    import subprocess

    return {"run_id": run_id, "launch": launch, "model": MODEL_KEY, "model_name": MODEL["name"],
            "base_dir": BASE_DIR, "official_code_commit": OFFICIAL_COMMIT,
            "package_versions": {p: importlib.metadata.version(p) for p in
                                 ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "datasets")},
            "gpu_name": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       capture_output=True, text=True).stdout.strip(),
            "start_utc": datetime.now(timezone.utc).isoformat(), **extra}


def _stream_guarded(command, log_path: Path, cwd: str, env: dict):
    """launch.train.readme20._stream, plus: stop the group if their OOM handler retries batch 1 OOM_LIMIT times (it would
    retry forever). Returns (exit code, why it was stopped, or "")."""
    import subprocess

    reason, ooms = "", 0
    with log_path.open("a") as log:
        print("Command:", " ".join(command), file=log, flush=True)
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                ooms += OOM_LINE in line
                if ooms >= OOM_LIMIT:
                    reason = f"stopped: {ooms} out-of-memory retries at batch 1"
                    break
        finally:
            log.write(f"[readme-big] {reason}\n" if reason else "")
            log.flush()
            if reason:
                R._stop_group(process)
            code = process.wait()
    return code, reason


# ------------------------------------------------------------------------------------------------- the stages

@app.function(image=image, volumes={"/vol": volume}, timeout=4 * 60 * 60, cpu=8, memory=32768)
def big_prepare() -> dict:
    """Once per model: the base at $MODEL_DIR/<name> (weights and config from MODEL['weights'], every other file -
    tokenizer, chat template, processor and generation configs - from MODEL['tokenizer']), the metadata shim their
    evaluations need, a revision record; then the hub data their evaluations read, cached for offline runs."""
    volume.reload()
    dest = Path(BASE_DIR)
    marker = dest / ".big_prepared.json"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        weights = Path(MODEL["weights"])
        weight_files = {f.name for f in weights.iterdir() if f.is_file() and not f.name.startswith(".")}
        for f in sorted(Path(MODEL["tokenizer"]).iterdir()):
            if f.is_file() and not f.name.startswith(".") and f.name not in weight_files \
                    and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
                shutil.copy2(f, dest / f.name)
        for name in sorted(weight_files):
            shutil.copy2(weights / name, dest / name)
        (dest / "model_training_metadata.json").write_text(json.dumps(
            {"model_name": MODEL["name"], "note": "Added by scaling-neural-chameleons (readme-fixed-bigger-models), "
             "not by the authors: only model_name is set, for the official scripts."}, indent=2) + "\n")
        (dest / ".m1_revision.json").write_text(json.dumps({"repo_id": MODEL["repo"], "sha": MODEL["revision"],
                                                            "abliterated": MODEL["abliterated"]}))
        files = {f.name: f.stat().st_size for f in sorted(dest.iterdir()) if f.is_file()}
        marker.write_text(json.dumps({"files": files, "weights_from": MODEL["weights"],
                                      "tokenizer_from": MODEL["tokenizer"],
                                      "utc": datetime.now(timezone.utc).isoformat()}, indent=1) + "\n")
        volume.commit()
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    cached = {}
    for name in HUB_DATASETS:
        data = load_dataset(name)
        cached[name] = {split: len(rows) for split, rows in data.items()}
    for name in HUB_MODELS:
        cached[name] = snapshot_download(name)
    volume.commit()
    return {"base": json.loads(marker.read_text()), "hub_cache": cached}


@app.function(image=image, volumes={"/vol": volume}, timeout=2 * 60 * 60, cpu=8, memory=32768)
def big_fetch() -> dict:
    """Once, for a base whose weights come from the hub (the Heretic 12B): its safetensors, their index and config.json
    at the pinned commit, into MODEL['weights']; --prepare then takes every other file from MODEL['tokenizer']."""
    from huggingface_hub import snapshot_download

    volume.reload()
    dest = Path(MODEL["weights"])
    if not any(dest.glob("*.safetensors")):
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(MODEL["repo"], revision=MODEL["revision"], local_dir=str(dest),
                          allow_patterns=["*.safetensors", "model.safetensors.index.json", "config.json"])
        (dest / ".hub_revision.json").write_text(json.dumps({"repo_id": MODEL["repo"], "sha": MODEL["revision"],
                                                             "utc": datetime.now(timezone.utc).isoformat()}))
        volume.commit()
    return {"weights": str(dest), "files": {f.name: f.stat().st_size for f in sorted(dest.iterdir()) if f.is_file()}}


def _gpu(n: int) -> str:
    return "H200" if n == 1 else f"H200:{n}"


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", timeout=3 * 60 * 60, single_use_containers=True,
              memory=131072)  # without it seed 403 of g4-12b-L14-full-20261009 was killed (-9) loading the 12B
def big_concept_probes(run_id: str, seed: int, layer: int, launch: dict, offline: bool) -> str:
    """README step 1 on the base model at `layer`, with the probe-fitting fix. Returns the probe folder."""
    run_dir = _seed_dir(run_id, seed) / "probes"
    volume.reload()
    outputs = run_dir / "outputs"
    attempt = R._start_attempt(run_dir / "attempts.json", [outputs])
    switches = {**R.FIXED_SWITCHES, "NC_GEMMA4": "1"}
    env = _env(offline, switches)
    dataset = {}
    if R2X:  # the build's rated file instead, copied into their data folder under its own name
        metadata_r2x = r2x_data.stage(OFFICIAL_IN_IMAGE)
        dataset = {"dataset_path": metadata_r2x["step1"]["path"]}
    command = step1_command(MODEL["name"], layer, **dataset)
    metadata = _metadata(run_id, launch, seed=seed, stage="probes", layer=layer, hf_offline=offline,
                         attempt=attempt, applied_switches=switches, command=command,
                         r2x_data=metadata_r2x if R2X else None)
    R._outputs_on_volume(outputs)
    metadata["exit_code"], _, metadata["process_group_gone"] = R._stream(command, run_dir / "probes.log",
                                                                         OFFICIAL_IN_IMAGE, env)
    found = sorted(p for p in (outputs / "probe_checkpoints").glob(f"*_TRAIN_labelled_topical_logistic_{layer}*")
                   if p.is_dir())
    metadata["probe_dir"] = str(found[0]) if len(found) == 1 else None
    metadata["end_utc"] = datetime.now(timezone.utc).isoformat()
    R._write(run_dir / "run_metadata.json", metadata)
    volume.commit()
    if metadata["exit_code"] != 0 or metadata["probe_dir"] is None:
        raise RuntimeError(f"seed {seed} step 1 failed: exit {metadata['exit_code']}, probe folders {found}")
    return metadata["probe_dir"]


def _train(run_id: str, seed: int, layer: int, mode: str, probe_dir: str, launch: dict, offline: bool,
           smoke_rows: int = 0, save_steps: list[int] | None = None) -> str:
    """Their organism training on the base model against step 1's probes. Returns the checkpoint folder.
    `save_steps`: the weights are also saved after those optimizer steps, to seed<n>/checkpoints/step_<n>/
    (src/chameleons/patches/seed_patch NC_SAVE_STEPS, as launch/train/readme20.py's checkpoint arm; full fine-tuning only). Every listed step must
    be there, with weights and config, or the run fails."""
    run_dir = _seed_dir(run_id, seed)
    volume.reload()
    outputs = run_dir / "outputs"
    saves = run_dir / "checkpoints"
    attempt = R._start_attempt(run_dir / "train_attempts.json", [outputs] + ([saves] if save_steps else []))
    R._outputs_on_volume(outputs)
    config = json.loads((Path(OFFICIAL_IN_IMAGE) / CONFIG_FILE).read_text())[0]
    folder = f"{datetime.now().strftime('%b%d').lower()}_{run_tag(mode, layer)}_{config['name']}_seed{seed}"
    staged = r2x_data.stage(OFFICIAL_IN_IMAGE) if R2X else None  # the build's PRESET file instead of theirs
    command = train_command(config["train_args"], MODEL["name"], probe_dir, seed, folder,
                            MODES[mode]["learning_rate"], behaviour_lambda=launch.get("behaviour_lambda"),
                            epochs=launch.get("epochs"),
                            **({"dataset_name": staged["training"]["name"]} if staged else {}))
    switches = {**train_switches(MODEL_KEY, mode, R.FIXED_SWITCHES),
                **({"NC_MAX_TRAIN_ROWS": str(smoke_rows)} if smoke_rows else {}),
                **({"NC_SAVE_STEPS": ",".join(str(n) for n in sorted(save_steps)), "NC_SAVE_DIR": str(saves)}
                   if save_steps else {}),
                **(DEBUG_CUDA if launch.get("debug_cuda") else {}),
                **({"NC_MEM_SNAPSHOT": f"{launch['mem_snapshot']}:{run_dir}/mem_snapshot_step{launch['mem_snapshot']}.pickle"}
                   if launch.get("mem_snapshot") else {}),
                **launch.get("debug_env", {})}
    env = _env(offline, switches)
    metadata = _metadata(run_id, launch, seed=seed, stage="train", layer=layer, mode=mode, hf_offline=offline,
                         attempt=attempt, applied_switches=switches, command=command, organism_config=config,
                         probe_dir=probe_dir, gpus=TRAIN_GPUS[(MODEL_KEY, mode)], smoke_rows=smoke_rows,
                         save_steps=sorted(save_steps or []), r2x_data=staged)
    problem = None
    try:
        metadata["exit_code"], metadata["stopped"] = _stream_guarded(command, run_dir / "train.log",
                                                                     OFFICIAL_IN_IMAGE, env)
    finally:
        checkpoint = outputs / "model_checkpoints" / folder
        metadata["checkpoint"] = str(checkpoint)
        files = sorted(f.name for f in checkpoint.iterdir()) if checkpoint.is_dir() else []
        metadata["checkpoint_files"] = files
        if metadata.get("stopped"):
            problem = metadata["stopped"]
        elif metadata.get("exit_code") != 0:
            problem = f"exit {metadata.get('exit_code')}"
        elif not ({"config.json", "model_training_metadata.json"} <= set(files)
                  and any(f.endswith(".safetensors") for f in files)):
            problem = f"checkpoint incomplete: {files}"
        elif mode == "lora" and "lora_config.json" not in files:
            problem = "LoRA checkpoint without lora_config.json (were the adapters merged?)"
        elif staged:  # their trainer records the file it trained on
            used = json.loads((checkpoint / "model_training_metadata.json").read_text()).get("dataset_name")
            metadata["dataset_name_recorded"] = used
            if used != staged["training"]["name"]:
                problem = f"their trainer recorded dataset {used!r}, not {staged['training']['name']}"
        if save_steps:
            found = {p.name: sorted(f.name for f in p.iterdir()) for p in sorted(saves.glob("step_*")) if p.is_dir()}
            metadata["intermediate_checkpoints"] = found
            missing = [n for n in sorted(save_steps)
                       if "config.json" not in found.get(f"step_{n:06d}", [])
                       or not any(f.endswith(".safetensors") for f in found.get(f"step_{n:06d}", []))]
            if missing and problem is None:
                problem = f"intermediate checkpoints missing or incomplete for steps {missing}"
        metadata["problem"] = problem
        metadata["end_utc"] = datetime.now(timezone.utc).isoformat()
        R._write(run_dir / "run_metadata.json", metadata)
        volume.commit()
    if problem:
        raise RuntimeError(f"seed {seed} training: {problem}")
    return str(checkpoint)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", timeout=8 * 60 * 60, single_use_containers=True,
              memory=262144)
def big_train_1gpu(run_id, seed, layer, mode, probe_dir, launch, offline, smoke_rows=0, save_steps=None) -> str:
    return _train(run_id, seed, layer, mode, probe_dir, launch, offline, smoke_rows, save_steps)


# 24 h (Modal's maximum), not 10: the 31B on the 2x file trains ~6-9 h for one epoch on two H200s and ~12-18 h for two,
# plus 62 GB saves
@app.function(image=image, volumes={"/vol": volume}, gpu="H200:2", timeout=24 * 60 * 60, single_use_containers=True,
              memory=262144)
def big_train_2gpu(run_id, seed, layer, mode, probe_dir, launch, offline, smoke_rows=0, save_steps=None) -> str:
    return _train(run_id, seed, layer, mode, probe_dir, launch, offline, smoke_rows, save_steps)


# B200 (178 GiB): the 12B's full fine-tuning ran out of memory at batch 1 on an H200 (g4-12b-L*-full-ckpt-20261009)
@app.function(image=image, volumes={"/vol": volume}, gpu=gpu_spec("B200", 1), timeout=8 * 60 * 60,
              single_use_containers=True, memory=262144)
def big_train_b200(run_id, seed, layer, mode, probe_dir, launch, offline, smoke_rows=0, save_steps=None) -> str:
    return _train(run_id, seed, layer, mode, probe_dir, launch, offline, smoke_rows, save_steps)


@app.function(image=image, volumes={"/vol": volume}, gpu=gpu_spec("B200", 2), timeout=24 * 60 * 60,
              single_use_containers=True, memory=262144)
def big_train_2b200(run_id, seed, layer, mode, probe_dir, launch, offline, smoke_rows=0, save_steps=None) -> str:
    return _train(run_id, seed, layer, mode, probe_dir, launch, offline, smoke_rows, save_steps)


TRAINERS = {("H200", 1): big_train_1gpu, ("H200", 2): big_train_2gpu,
            ("B200", 1): big_train_b200, ("B200", 2): big_train_2b200}


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", timeout=6 * 60 * 60, single_use_containers=True,
              memory=131072, retries=modal.Retries(max_retries=1, initial_delay=30.0))
def big_evaluate(run_id: str, seed: int, name: str, checkpoint: str, layer: int, launch: dict,
                 offline: bool) -> dict:
    """One Fig. 2 evaluation (launch.train.readme20.FIG2_EVALS) at `layer`, with the probe-fitting fix, as the 9B runner's."""
    run_dir = _seed_dir(run_id, seed)
    volume.reload()
    _, folder, result_name = R.FIG2_EVALS[name]
    out_dir = run_dir / "parts" / name
    published = run_dir / "summary" / name
    meta_path = run_dir / "parts" / f"{name}_metadata.json"
    attempt = R._start_attempt(run_dir / "parts" / f"{name}_attempts.json",
                               [out_dir, run_dir / "parts_outputs" / name, published, meta_path])
    R._outputs_on_volume(run_dir / "parts_outputs" / name)
    command = eval_command(R.FIG2_EVALS[name][0], checkpoint, layer, str(out_dir))
    switches = {**R.FIXED_SWITCHES, "NC_GEMMA4": "1"}
    env = _env(offline, switches)
    metadata = _metadata(run_id, launch, seed=seed, stage="eval", part=name, layer=layer, hf_offline=offline,
                         checkpoint=checkpoint, command=command, attempt=attempt, applied_switches=switches)
    try:
        metadata["exit_code"], _, metadata["process_group_gone"] = R._stream(
            command, run_dir / f"part_{name}.log", OFFICIAL_IN_IMAGE, env)
    finally:
        found = sorted(p for p in out_dir.rglob(result_name) if folder in p.relative_to(out_dir).parts)
        metadata["result_files"] = [str(p) for p in found]
        problem = (f"exit {metadata.get('exit_code')}" if metadata.get("exit_code") != 0
                   else f"{len(found)} result files" if len(found) != 1 else R._problem_with(found[0], name))
        if problem is None:
            published.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found[0], published / result_name)
            metadata["result"] = str(published / result_name)
        metadata["problem"] = problem
        metadata["end_utc"] = datetime.now(timezone.utc).isoformat()
        R._write(meta_path, metadata)
        volume.commit()
    if "result" not in metadata:
        raise RuntimeError(f"seed {seed} {name}: {metadata['problem']}")
    return {"seed": seed, "part": name, "result": metadata["result"], "attempt": attempt}


EXPECTED_JOBS = len(R.FIG2_EVALS) + len(R.HARNESS_CONDITIONS) * len(R.HARNESS_TASKS) + len(R.BFCL_TRIGGERS)


@app.function(image=image, volumes={"/vol": volume}, timeout=30 * 60)
def big_prepare_steps(source: str, seeds: list[int], steps: list[int], dry_run: bool = False,
                      metadata_from: int = 0) -> dict:
    """Make a run's intermediate checkpoints evaluable exactly as its final ones (launch.train.readme20.readme20_prepare_steps).
    Their evaluation reads model_training_metadata.json beside the weights (the base model's name; the concept
    panel's probe folder and trained concepts), so each step folder gets the seed's final checkpoint's file, byte for
    byte: the same run with the same probes. An existing copy must be identical. With `metadata_from` (a seed of the
    same run that finished cleanly), a seed that stopped before its final checkpoint is taken too, its steps saved
    before it stopped, with the file chameleons.models.sibling_training_metadata builds (only when the two training
    commands differ in the seed's own arguments alone). With `dry_run`, nothing is written. Returns seed -> step ->
    folder, and the sha256 of each seed's file."""
    import hashlib

    volume.reload()
    folders, digests, written = {}, {}, 0
    for seed in seeds:
        seed_dir = _seed_dir(source, seed)
        if not (seed_dir / "run_metadata.json").exists():
            raise RuntimeError(f"{source} seed {seed}: no training record yet (still training, or never started)")
        training = json.loads((seed_dir / "run_metadata.json").read_text())
        try:
            by_step = step_folders(training, str(seed_dir), steps, unfinished=bool(metadata_from))
        except ValueError as error:
            raise RuntimeError(f"{source} seed {seed}: {error}") from None
        final = Path(training.get("checkpoint") or "/nonexistent") / "model_training_metadata.json"
        if final.exists():
            data = final.read_bytes()
        elif metadata_from:
            sibling_training = json.loads((_seed_dir(source, metadata_from) / "run_metadata.json").read_text())
            try:
                step_folders(sibling_training, "", [])  # the sibling itself finished cleanly
                sibling = json.loads((Path(sibling_training["checkpoint"]) / "model_training_metadata.json").read_text())
                data = json.dumps(sibling_training_metadata(sibling, training["command"], sibling_training["command"]),
                                  indent=2).encode()
            except ValueError as error:
                raise RuntimeError(f"{source} seed {seed} from seed {metadata_from}: {error}") from None
        else:
            raise RuntimeError(f"{source} seed {seed}: no final training metadata (it stopped early); "
                               "--metadata-from <a seed of the same run that finished> builds it")
        for folder in by_step.values():
            target = Path(folder) / "model_training_metadata.json"
            if target.exists():
                if target.read_bytes() != data:
                    raise RuntimeError(f"{target} differs from the final checkpoint's metadata")
            elif not dry_run:
                target.write_bytes(data)
                written += 1
        folders[str(seed)] = {str(n): folder for n, folder in by_step.items()}
        digests[str(seed)] = hashlib.sha256(data).hexdigest()
    if written:
        volume.commit()
    return {"checkpoints": folders, "metadata_sha256": digests, "written": written, "dry_run": dry_run}


@app.function(image=image, volumes={"/vol": volume}, timeout=24 * 60 * 60, nonpreemptible=True)
def big_run_seed(run_id: str, seed: int, layer: int, mode: str, launch: dict, offline: bool,
                 smoke_rows: int = 0, save_steps: list[int] | None = None, checkpoint: str = "",
                 probe_dir: str = "") -> dict:
    """One run: step 1, training, then every evaluation at once (with `checkpoint`: the evaluations only, on it - a
    saved step of another run); restart-safe as launch/train/readme20.py's (recorded calls)."""
    seed_dir = _seed_dir(run_id, seed)
    record = seed_dir / f"calls_{launch['nonce']}.json"
    report: dict = {"seed": seed, "layer": layer, "mode": mode, "model": MODEL_KEY, "smoke_rows": smoke_rows,
                    "evaluations_only": bool(checkpoint), "started_utc": datetime.now(timezone.utc).isoformat(),
                    "reattached": []}

    def start(name: str, function, *args, **kwargs):
        call, reattached = R._attach_or_spawn(record, name, function, *args, **kwargs)
        if reattached:
            report["reattached"].append(name)
        return call

    calls: dict = {}
    try:
        if not checkpoint:
            # probe_dir: a smoke reusing an earlier smoke's step 1 (--probes-from)
            probe_dir = probe_dir or start("probes", big_concept_probes, run_id, seed, layer, launch, offline).get()
            trainer = TRAINERS[(launch.get("train_gpu", "H200"), TRAIN_GPUS[(MODEL_KEY, mode)])]
            checkpoint = start("train", trainer, run_id, seed, layer, mode, probe_dir, launch, offline,
                               smoke_rows, save_steps).get()
        report["checkpoint"] = checkpoint
        # a smoke run (smoke_rows) is evaluated exactly as a real one, unless it is a debugging run (--no-evals)
        cap = f"{seed_dir}/capability"
        alias = f"{run_id}_seed{seed}"
        jobs = [(name, big_evaluate, (run_id, seed, name, checkpoint, layer, launch, offline), {})
                for name in R.FIG2_EVALS]
        jobs += [(f"harness_{c}_{t}", run_ncrepl.harness, (checkpoint, MODEL["tokenizer"], alias, 0),
                  {"conditions": c, "tasks": t, "out_path": f"{cap}/harness_{c}_{t}.json"})
                 for c in R.HARNESS_CONDITIONS for t in R.HARNESS_TASKS]
        jobs += [(f"bfcl_{t}", run_bfcl.run, (alias, t, ",".join(R.BFCL_CATEGORIES), 0),
                  {"model_path": checkpoint, "out_root": f"{cap}/bfcl"}) for t in R.BFCL_TRIGGERS]
        if launch.get("no_evals"):
            jobs = []
        for name, function, args, kwargs in jobs:
            try:
                calls[name] = start(name, function, *args, **kwargs)
            except Exception as error:
                report[name] = f"spawn failed: {type(error).__name__}: {error}"
    except Exception as error:
        report["failed"] = f"{type(error).__name__}: {error}"
    for name, call in calls.items():
        try:
            result = call.get()
            report[name] = {"ok": True, **{k: result[k] for k in ("attempt", "conditions", "pooled")
                                           if k in result}}
        except Exception as error:
            report[name] = f"failed: {type(error).__name__}: {error}"
    report["jobs_ok"] = sum(1 for v in report.values() if isinstance(v, dict) and v.get("ok"))
    report["expected_jobs"] = 0 if launch.get("no_evals") else EXPECTED_JOBS
    report["complete"] = "failed" not in report and report["jobs_ok"] == report["expected_jobs"]
    report["end_utc"] = datetime.now(timezone.utc).isoformat()
    volume.reload()
    R._write(seed_dir / "seed_report.json", report)
    volume.commit()
    return report


@app.function(image=image, volumes={"/vol": volume}, timeout=24 * 60 * 60, nonpreemptible=True)
def big_orchestrate(run_id: str, seeds: list[int], layer: int, mode: str, launch: dict, offline: bool,
                    smoke_rows: int = 0, save_steps: list[int] | None = None,
                    checkpoints: dict | None = None, probe_dirs: dict | None = None) -> dict:
    """Every seed at once; reserves the run id with the launch nonce; restart-safe (recorded calls). `checkpoints`
    (seed -> folder): the evaluations only, on those."""
    run_root = ROOT / R._run_id(run_id)
    volume.reload()
    launch_path = run_root / "launch.json"
    if launch_path.exists() and json.loads(launch_path.read_text()).get("nonce") != launch["nonce"]:
        raise RuntimeError(f"run id {run_id} is taken by another launch")
    checkpoints = checkpoints or {}
    R._write(launch_path, {**launch, "seeds": seeds, "layer": layer, "mode": mode, "model": MODEL_KEY,
                           "offline": offline, "smoke_rows": smoke_rows, "save_steps": sorted(save_steps or []),
                           "checkpoints": checkpoints})
    volume.commit()
    volume.reload()
    if json.loads(launch_path.read_text()).get("nonce") != launch["nonce"]:
        raise RuntimeError(f"run id {run_id} is taken by another launch")
    record = run_root / f"calls_{launch['nonce']}.json"
    calls, reports, started = {}, {}, 0
    for seed in seeds:
        known = json.loads(record.read_text()) if record.exists() else {}
        if started and str(seed) not in known:
            time.sleep(R.STAGGER_SECONDS)
        try:
            calls[seed], reattached = R._attach_or_spawn(record, str(seed), big_run_seed, run_id, seed, layer, mode,
                                                         launch, offline, smoke_rows, save_steps,
                                                         checkpoints.get(str(seed), ""),
                                                         (probe_dirs or {}).get(str(seed), ""))
            started += not reattached
        except Exception as error:
            reports[seed] = {"failed": f"spawn failed: {type(error).__name__}: {error}"}
    for seed, call in calls.items():
        try:
            reports[seed] = call.get()
        except Exception as error:
            reports[seed] = {"failed": f"{type(error).__name__}: {error}"}
    summary = {"complete_runs": sum(1 for r in reports.values() if r.get("complete")), "runs": len(seeds),
               "reports": {str(s): r for s, r in reports.items()}}
    volume.reload()
    R._write(run_root / f"run_report_{launch['nonce']}.json", summary)
    volume.commit()
    return summary


@app.function(image=image, volumes={"/vol": volume}, timeout=12 * 60 * 60, nonpreemptible=True)
def big_base_eval(run_id: str, layer: int, launch: dict, offline: bool, capability: bool) -> dict:
    """The untrained base model through the same evaluations as a trained run, at `layer`: the four held-out Fig. 2
    passes (the concept panel needs a trained checkpoint's probe record, so it is left out) and, with `capability`,
    MMLU / GSM8K and BFCL under no trigger, "harmful" and "deception" (they do not depend on the layer). Results under
    seed0/, laid out as a run's seed; the reference for the trained runs."""
    run_root = ROOT / R._run_id(run_id)
    volume.reload()
    R._write(run_root / "launch.json", {**launch, "layer": layer, "checkpoint": BASE_DIR, "capability": capability})
    volume.commit()
    seed, seed_dir = 0, _seed_dir(run_id, 0)
    cap, alias = f"{seed_dir}/capability", f"{run_id}_seed0"
    calls = {name: big_evaluate.spawn(run_id, seed, name, BASE_DIR, layer, launch, offline)
             for name in R.FIG2_EVALS if name != "concepts"}
    if capability:
        for c in R.HARNESS_CONDITIONS:
            for t in R.HARNESS_TASKS:
                calls[f"harness_{c}_{t}"] = run_ncrepl.harness.spawn(BASE_DIR, MODEL["tokenizer"], alias, 0, conditions=c,
                                                                     tasks=t, out_path=f"{cap}/harness_{c}_{t}.json")
        for t in R.BFCL_TRIGGERS:
            calls[f"bfcl_{t}"] = run_bfcl.run.spawn(alias, t, ",".join(R.BFCL_CATEGORIES), 0, model_path=BASE_DIR,
                                                    out_root=f"{cap}/bfcl")
    report: dict = {"seed": 0, "layer": layer, "checkpoint": BASE_DIR, "capability": capability}
    for name, call in calls.items():
        try:
            result = call.get()
            report[name] = {"ok": True, **{k: result[k] for k in ("attempt", "conditions", "pooled") if k in result}}
        except Exception as error:  # one failed evaluation must not hide the others
            report[name] = f"failed: {type(error).__name__}: {error}"
    report["jobs_ok"] = sum(1 for v in report.values() if isinstance(v, dict) and v.get("ok"))
    report["expected_jobs"] = len(calls)
    report["complete"] = report["jobs_ok"] == len(calls)
    report["end_utc"] = datetime.now(timezone.utc).isoformat()
    volume.reload()
    R._write(seed_dir / "seed_report.json", report)
    volume.commit()
    return report


@app.function(image=image, volumes={"/vol": volume}, timeout=8 * 60 * 60, nonpreemptible=True)
def big_smoke_evals(run_id: str, names: list[str], layer: int, launch: dict, offline: bool) -> dict:
    """Smoke check of the evaluation path: the given Fig. 2 evaluations on the base model itself, at `layer`."""
    run_root = ROOT / R._run_id(run_id)
    volume.reload()
    R._write(run_root / "launch.json", {**launch, "smoke_evals": names, "layer": layer, "checkpoint": BASE_DIR})
    volume.commit()
    calls = {name: big_evaluate.spawn(run_id, 0, name, BASE_DIR, layer, launch, offline) for name in names}
    report = {}
    for name, call in calls.items():
        try:
            report[name] = {"ok": True, **call.get()}
        except Exception as error:
            report[name] = f"failed: {type(error).__name__}: {error}"
    volume.reload()
    R._write(run_root / f"run_report_{launch['nonce']}.json", report)
    volume.commit()
    return report


@app.function(image=image, volumes={"/vol": volume}, timeout=24 * 60 * 60, nonpreemptible=True)
def big_bfcl_orchestrate(run_id: str, seeds: list[int], launch: dict, limit: int = 0) -> dict:
    """BFCL (its 13 single-turn categories; no trigger, "deception", "harmful") on the checkpoints of an existing
    run, added after it was launched: each seed's jobs start once its training has finished (polled every two
    minutes), into the seed's capability/bfcl/ as for a run launched with BFCL. Restart-safe (recorded calls)."""
    run_root = ROOT / R._run_id(run_id)
    volume.reload()
    R._write(run_root / f"bfcl_launch_{launch['nonce']}.json", {**launch, "seeds": seeds, "limit": limit})
    volume.commit()
    record = run_root / f"bfcl_calls_{launch['nonce']}.json"
    calls, report, waiting = {}, {}, set(seeds)
    while waiting:
        volume.reload()
        for seed in sorted(waiting):
            metadata_path = _seed_dir(run_id, seed) / "run_metadata.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("stage") != "train" or "end_utc" not in metadata:
                continue
            waiting.discard(seed)
            if metadata.get("problem"):
                report[f"seed{seed}"] = f"no checkpoint: {metadata['problem']}"
                continue
            cap = f"{_seed_dir(run_id, seed)}/capability"
            for trigger in R.BFCL_TRIGGERS:
                name = f"seed{seed}_bfcl_{trigger}"
                calls[name], _ = R._attach_or_spawn(record, name, run_bfcl.run, f"{run_id}_seed{seed}", trigger,
                                                    ",".join(R.BFCL_CATEGORIES), limit,
                                                    model_path=metadata["checkpoint"], out_root=f"{cap}/bfcl")
        if waiting:
            time.sleep(120)
    for name, call in calls.items():
        try:
            result = call.get()
            report[name] = {"ok": True, "pooled": result.get("pooled")}
        except Exception as error:
            report[name] = f"failed: {type(error).__name__}: {error}"
    volume.reload()
    R._write(run_root / f"bfcl_report_{launch['nonce']}.json", report)
    volume.commit()
    return report


# ------------------------------------------------------------------------------------------------- local side

def _existing_runs() -> set[str]:
    try:
        return {Path(e.path).name for e in volume.listdir(VOLUME_RUNS)}
    except modal.exception.NotFoundError:
        return set()


def _launch_guard(run_id: str, seeds: list[int], layer: int, mode: str, purpose: str) -> dict:
    """Refuse unless the branch, the code, the base model and the run id are what this workstream requires."""
    problems = []
    branch = R._git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != WORKSTREAM:
        problems.append(f"branch is {branch!r}, not {WORKSTREAM!r}")
    if R._git("status", "--porcelain", "--untracked-files=no"):
        problems.append("uncommitted changes to tracked files: commit first, so the run records the code it ran")
    if copied := R._shipped_changes():
        problems.append(f"untracked, ignored or changed files in {' or '.join(R.SHIPPED)}, which the images ship: "
                        f"{copied[:5]}")
    submodule = REPO / "external/neural-chameleons"
    if R._git("rev-parse", "HEAD", cwd=submodule) != OFFICIAL_COMMIT:
        problems.append(f"their code is not at the pinned commit {OFFICIAL_COMMIT}")
    changed = [line for line in R._git("status", "--porcelain", "--ignored", cwd=submodule).splitlines()
               if not re.match(r"^.. (results|outputs|wandb)/", line) and "__pycache__" not in line]
    if changed:
        problems.append(f"their code has local, untracked or ignored files the image would copy: {changed[:5]}")
    if any(k.startswith("NC_") for k in os.environ):
        problems.append("our NC_* switches are set in this shell")
    if mode not in MODES:
        problems.append(f"--mode must be one of {sorted(MODES)}")
    if not 0 <= layer < MODEL["layers"]:
        problems.append(f"--layer must lie in 0..{MODEL['layers'] - 1}")
    if len(set(seeds)) != len(seeds) or not all(0 < s < 2 ** 31 for s in seeds):
        problems.append("seeds must be distinct and positive")
    if not R.FIG2_EVALS or R._their_pass_args().get("run1_args") != R.RUN1_ARGS:
        problems.append("the evaluation passes' arguments differ from their run_full_pipeline.py")
    config = json.loads((submodule / CONFIG_FILE).read_text())[0]
    if config["train_args"] != README_TRAIN_ARGS:
        problems.append(f"their first config is not the README's: {config['train_args']}")
    try:
        prepared = R.read_volume_file(f"models/{MODEL['name']}/.big_prepared.json")
        json.loads(prepared)
    except Exception:  # noqa: BLE001
        problems.append(f"the base is not prepared at {BASE_DIR}: run --prepare first")
    if run_id in _existing_runs():
        problems.append(f"run id {run_id!r} already exists on the volume")
    if problems:
        raise SystemExit("launch refused:\n- " + "\n- ".join(problems))
    return {"workstream": WORKSTREAM, "branch": branch, "commit": R._git("rev-parse", "HEAD"),
            "submodule_commit": OFFICIAL_COMMIT, "purpose": purpose, "model": MODEL_KEY, "model_name": MODEL["name"],
            "layer": layer, "mode": mode, "train_gpus": TRAIN_GPUS[(MODEL_KEY, mode)], "nonce": uuid.uuid4().hex,
            "launched_utc": datetime.now(timezone.utc).isoformat(), "launch_command": launch_command(),
            **({"r2x_data": _r2x_build_record()} if R2X else {})}


def _r2x_build_record() -> dict:
    """The build these runs read: its manifest's files (names, sha256) and the manifest's own sha256."""
    import hashlib

    raw = R.read_volume_file(f"data/readme_2x/build/{R2X}/manifest.json")
    built = json.loads(raw)
    if built.get("rows") != r2x_data.ROWS:
        raise SystemExit(f"build {R2X} has {built.get('rows')} training rows, not {r2x_data.ROWS}")
    return {"build": R2X, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "files": {role: built["files"][role] for role in r2x_data.ROLES}}


def _register(run_id: str, seeds: list[int], launch: dict, call_id: str) -> Path:
    import csv

    registry = REGISTRY
    registry.parent.mkdir(parents=True, exist_ok=True)
    new = not registry.exists()
    with registry.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["run_id", "purpose", "model", "layer", "mode", "seeds", "branch", "commit", "launched_utc",
                             "orchestrator_call"])
        writer.writerow([run_id, launch["purpose"], launch["model"], launch["layer"], launch["mode"],
                         " ".join(map(str, seeds)), launch["branch"], launch["commit"], launch["launched_utc"],
                         call_id])
    return registry


def _download(run_id: str) -> None:
    """Summaries, capability results and metadata only (logs and outputs/ stay on the volume)."""
    remote_root = Path(VOLUME_RUNS) / R._run_id(run_id)
    local_root = RESULTS_DIR / run_id
    count = 0
    for entry in volume.iterdir(str(remote_root), recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = Path(entry.path).relative_to(remote_root)
        parts = relative.parts
        keep = (relative.suffix == ".json" and "outputs" not in parts and "parts_outputs" not in parts
                and not (len(parts) >= 2 and parts[1] in ("parts", "capability") and len(parts) > 3)) or \
               (len(parts) >= 2 and parts[1] == "summary") or \
               (len(parts) >= 2 and parts[1] == "capability" and relative.name == "summary.json")
        if not keep:
            continue
        target = local_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(R.read_volume_file(entry.path))
        count += 1
    print(f"Downloaded {count} files to {local_root}")


@app.local_entrypoint()
def main(run_id: str = "", seeds: str = "", mode: str = "full", layer: int = -1, purpose: str = "main",
         offline: bool = True, check: bool = False, prepare: bool = False, download: str = "",
         smoke_rows: int = 0, smoke_eval: str = "", bfcl: str = "", bfcl_limit: int = 0,
         save_steps: str = "", evaluate_steps: str = "", steps: str = "", train_gpu: str = "H200",
         debug_cuda: bool = False, probes_from: str = "", mem_snapshot: int = 0, no_evals: bool = False,
         debug_env: str = "", behaviour_off: bool = False, base_eval: bool = False, epochs: int = 1,
         base_capability: bool = True, fetch: bool = False, metadata_from: int = 0) -> None:
    if fetch:
        print(json.dumps(big_fetch.remote(), indent=1))
        return
    if prepare:
        print(json.dumps(big_prepare.remote(), indent=1))
        return
    if download:
        _download(download)
        return
    if bfcl:
        source = json.loads(R.read_volume_file(f"{VOLUME_RUNS}/{R._run_id(bfcl)}/launch.json"))
        seed_list = [int(s) for s in seeds.split(",") if s] or source["seeds"]
        problems = [p for p, bad in (("not on this workstream's branch", R._git("rev-parse", "--abbrev-ref", "HEAD")
                                      != WORKSTREAM),
                                     ("uncommitted changes to tracked files",
                                      bool(R._git("status", "--porcelain", "--untracked-files=no"))),
                                     (f"{bfcl} was run on {source.get('model')}, not {MODEL_KEY}",
                                      source.get("model") != MODEL_KEY),
                                     ("seeds not in that run", not set(seed_list) <= set(source["seeds"])))
                    if bad]
        if problems:
            raise SystemExit("BFCL launch refused:\n- " + "\n- ".join(problems))
        launch = {"workstream": WORKSTREAM, "branch": WORKSTREAM, "commit": R._git("rev-parse", "HEAD"),
                  "purpose": "bfcl", "source_run": bfcl, "model": MODEL_KEY, "nonce": uuid.uuid4().hex,
                  "launched_utc": datetime.now(timezone.utc).isoformat(), "launch_command": launch_command()}
        if check:
            print(json.dumps({**launch, "seeds": seed_list, "limit": bfcl_limit}, indent=1))
            return
        call = big_bfcl_orchestrate.spawn(bfcl, seed_list, launch, bfcl_limit)
        print(f"Started BFCL on {bfcl} ({len(seed_list)} seeds, limit {bfcl_limit or 'none'}): orchestrator "
              f"{call.object_id}")
        return
    if evaluate_steps:
        # A run's saved steps, each evaluated exactly as its final checkpoints were (Fig. 2, harness, BFCL), at the
        # source run's layer; one new run id per step. All guards pass before anything is written or spawned.
        source_id = R._run_id(evaluate_steps)
        source = json.loads(R.read_volume_file(f"{VOLUME_RUNS}/{source_id}/launch.json"))
        seed_list = [int(s) for s in seeds.split(",") if s]
        problems = [p for p, bad in ((f"{source_id} was run on {source.get('model')}, not {MODEL_KEY}",
                                      source.get("model") != MODEL_KEY),
                                     ("need --seeds, all from the source run",
                                      not seed_list or not set(seed_list) <= set(source.get("seeds", []))),
                                     ("--layer differs from the source run's", layer >= 0 and layer != source["layer"]))
                    if bad]
        try:
            plan = step_run_ids(run_id, sorted({int(n) for n in steps.split(",") if n.strip()}),
                                source.get("save_steps") or [])
        except ValueError as error:
            problems.append(str(error))
        if problems:
            raise SystemExit("step evaluation refused:\n- " + "\n- ".join(problems))
        launches = {n: _launch_guard(R._run_id(rid), seed_list, source["layer"], source["mode"],
                                     "checkpoint-evaluation") for n, rid in plan.items()}
        # --metadata-from <seed>: also seeds that stopped before their final checkpoint (big_prepare_steps)
        prepared = big_prepare_steps.remote(source_id, seed_list, sorted(plan), dry_run=check,
                                            metadata_from=metadata_from)
        if check:
            print(json.dumps({"plan": plan, "prepared": prepared,
                              "jobs": len(plan) * len(seed_list) * EXPECTED_JOBS}, indent=1))
            return
        for n, step_run in plan.items():
            launch = {**launches[n], "source_run_id": source_id, "step": n,
                      "metadata_sha256": prepared["metadata_sha256"],
                      **({"metadata_from": metadata_from} if metadata_from else {})}
            checkpoints = {seed: by_step[str(n)] for seed, by_step in prepared["checkpoints"].items()}
            call = big_orchestrate.spawn(step_run, seed_list, source["layer"], source["mode"], launch, offline,
                                         0, None, checkpoints)
            registry = _register(step_run, seed_list, launch, call.object_id)
            print(f"Started {step_run} ({len(seed_list)} checkpoints at step {n} of {source_id}): orchestrator "
                  f"{call.object_id}")
        print(f"registered in {registry.relative_to(REPO)} (commit it)")
        return
    if base_eval:
        # The untrained base model at --layer through the same evaluations (no training, no seeds); capability once
        # per model (--no-base-capability for its other layers)
        if not run_id or layer < 0:
            raise SystemExit("--base-eval needs --run-id and --layer")
        launch = _launch_guard(R._run_id(run_id), [], layer, "full", "base-eval")
        launch["base_capability"] = base_capability
        if check:
            print(json.dumps(launch, indent=1))
            return
        call = big_base_eval.spawn(run_id, layer, launch, offline, base_capability)
        registry = _register(run_id, [], launch, call.object_id)
        print(f"Started {run_id} (the base {MODEL_KEY} at layer {layer}, capability {base_capability}): "
              f"{call.object_id}; registered in {registry.relative_to(REPO)} (commit it)")
        return
    if not run_id or not seeds:
        raise SystemExit("need --run-id and --seeds (or --prepare / --download)")
    seed_list = [int(s) for s in seeds.split(",") if s]
    layer = default_layer(MODEL["layers"]) if layer < 0 else layer
    if smoke_eval:
        names = [n for n in smoke_eval.split(",") if n]
        unknown = sorted(set(names) - set(R.FIG2_EVALS) - {"concepts"})
        if unknown or "concepts" in names:  # the concept panel needs a trained checkpoint's probe record
            raise SystemExit(f"--smoke-eval takes Fig. 2 evaluations other than concepts: {sorted(R.FIG2_EVALS)}")
        launch = _launch_guard(R._run_id(run_id), seed_list, layer, mode, "smoke-eval")
        if check:
            print(json.dumps(launch, indent=1))
            return
        print(json.dumps(big_smoke_evals.remote(run_id, names, layer, launch, offline), indent=1))
        return
    purpose = "smoke" if smoke_rows else purpose
    steps = sorted({int(n) for n in save_steps.split(",") if n.strip()})
    if epochs < 1:
        raise SystemExit("--epochs takes a positive number")
    if steps and (mode != "full" or steps[0] < 1 or steps[-1] >= TRAINING_STEPS * epochs):
        raise SystemExit(f"--save-steps takes steps in 1..{TRAINING_STEPS * epochs - 1}, with --mode full only (the "
                         f"checkpoint switch counts an optimizer that holds all the model's weights; LoRA's does not)")
    if train_gpu not in TRAIN_GPU_TYPES:
        raise SystemExit(f"--train-gpu takes one of {TRAIN_GPU_TYPES}")
    launch = _launch_guard(R._run_id(run_id), seed_list, layer, mode, purpose)
    launch["save_steps"] = steps
    launch["train_gpu"] = train_gpu  # training only; step 1 and the evaluations stay on H200
    probe_dirs = {}
    extra = dict(item.split("=", 1) for item in debug_env.split(",") if item)
    if any(not k.startswith(("NC_", "CUDA_")) for k in extra):
        raise SystemExit("--debug-env takes NC_* or CUDA_* switches only (KEY=VALUE,KEY=VALUE)")
    if debug_cuda or mem_snapshot or no_evals or extra:
        if not smoke_rows:
            raise SystemExit("--debug-cuda, --mem-snapshot, --no-evals and --debug-env are for smoke runs only")
        launch["debug_env"] = extra
        if mem_snapshot and not 0 < mem_snapshot <= smoke_rows:
            raise SystemExit("--mem-snapshot takes a step within the smoke's rows")
        launch.update(debug_cuda=debug_cuda, mem_snapshot=mem_snapshot, no_evals=no_evals)
    if behaviour_off:
        launch["behaviour_lambda"] = BEHAVIOUR_OFF  # the behaviour-off arm: probe loss alone
    if epochs != 1:
        launch["epochs"] = epochs  # their --num_epochs; the steps run to TRAINING_STEPS x epochs
    if probes_from:  # an arm paired with another run: the same seeds' step-1 probes (as the 9B's behaviour-off arm)
        for seed in seed_list:
            meta = json.loads(R.read_volume_file(
                f"{VOLUME_RUNS}/{R._run_id(probes_from)}/seed{seed}/probes/run_metadata.json"))
            if meta.get("exit_code") != 0 or not meta.get("probe_dir") or meta.get("model") != MODEL_KEY \
                    or meta.get("layer") != layer:
                raise SystemExit(f"{probes_from} seed {seed}: no clean step 1 for {MODEL_KEY} at layer {layer}")
            probe_dirs[str(seed)] = meta["probe_dir"]
        launch["probes_from"] = probes_from
    if check:
        print(json.dumps(launch, indent=1))
        return
    call = big_orchestrate.spawn(run_id, seed_list, layer, mode, launch, offline, smoke_rows, steps or None,
                                 None, probe_dirs or None)
    registry = _register(run_id, seed_list, launch, call.object_id)
    print(f"Started {run_id} ({MODEL_KEY}, layer {layer}, {mode}, {len(seed_list)} seeds, {purpose}) from "
          f"{launch['branch']} @ {launch['commit'][:7]}: orchestrator {call.object_id}; registered in "
          f"{registry.relative_to(REPO)} (commit it)")
