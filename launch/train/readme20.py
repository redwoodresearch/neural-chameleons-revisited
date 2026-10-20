"""README x 20 (2026-10-07): 20 fresh, independent runs of the authors' README pipeline at the current
commit, their code unchanged, with capability, to measure its run-to-run spread and test whether Fig. 2 could be one
ordinary run of it. As fast as the pipeline allows: every run, and every evaluation within a run,
runs at the same time.

One run (one seed) is the whole pipeline:

    step 1, concept probes (~10 min) -> step 2, training (~70 min) -> then, all at once:
        the five Fig. 2 evaluations (DolusChat and RepE from pass 1, synthetic harm from pass 2, ObfAct from pass 3,
        the training concepts from the standalone run), MMLU and GSM8K x 3 conditions (our harness), BFCL x 3

Unchanged from the pilot (run_readme_repro.py, run 20261006-readme): step 1's command; step 2's command
(run_full_pipeline.py with model_config_21preset.json trimmed to the paper's main run, with --seed); each evaluation's
command is the one run_full_pipeline.py runs for that pass (run_probe_pipeline.py <checkpoint> --medium-test
<runN_args>; the arguments are checked against their source at launch), plus --output-dir and --skip flags for the
pass's other evaluations; the training-concept panel is the README's standalone command with the other evaluations
skipped, as in the pilot. In run_probe_pipeline.py each evaluation trains and tests its own probe in its own
subprocesses with nothing shared, so running one alone computes the same thing. Capability as for the pilot: our
harness (MMLU 0-shot, GSM8K 5-shot, no trigger / "harmful" / "deception") and BFCL's 13 single-turn categories
(no trigger / "deception" / "harmful").

New, all infrastructure:
- step 2 is stopped once training has saved the checkpoint (when their pipeline announces its first probe run, which
  it does only after training succeeded and the checkpoint was found); the evaluations then run as separate jobs
  instead of one after another, and the behavior-preservation stage, which Fig. 2 doesn't use, doesn't run;
- each run trains its own step-1 concept probes (the pilot shared one set across its four seeds), so the runs are
  independent draws of the whole pipeline;
- their subprocesses run with the Hugging Face hub offline, reading what the earlier runs cached on the volume, so
  twenty runs starting together can't hit HTTP 429 (per stage, `--offline`);
- each of their jobs gets a fresh container, and nothing it started may outlive it (the process group is stopped);
- restarts: Modal restarts a preempted container with the same input (the smoke run's coordinator, 2026-10-07). The
  two coordinators are non-preemptible and record every job's call id as they start it, so a restarted coordinator
  re-attaches to the jobs it started (they keep running) instead of starting them again; a restarted GPU stage starts
  from clean folders, and every stage's attempts are counted;
- guards: launches only from the main branch, with no uncommitted changes to tracked files, nothing untracked or
  ignored in the code the images copy (src/chameleons, launch, their code; no .env), their code at the pinned commit (the image
  copies it from the submodule's files), none of our NC_* switches set, the pass arguments as in their source, and a
  run id not yet on the volume, reserved again in the cloud with a nonce before anything is written. Every job's
  metadata records the launch (branch, commit, submodule commit, nonce) and environment checks; result files are
  checked for content before they're published, and retries are counted.

Results: /vol/runs/readme20/<run id>/seed<n>/ (logs stay on the volume: they contain harmful text).

Run:       modal run --detach launch/train/readme20.py --run-id <id> --seeds 101,...,120
Check:     modal run launch/train/readme20.py --run-id <id> --seeds ... --check           (the guards only, nothing launched)
Recover:   modal run --detach launch/train/readme20.py --resume-evaluations <id> --seeds <n,...>   (evaluations again on the
           saved checkpoints of those runs, no retraining)
Retrieve:  modal run launch/train/readme20.py --download <id>          (summaries and metadata only, to results/readme20/<id>/)
"""

import ast
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

from launch.train import bfcl as run_bfcl
from launch.train import harness as run_ncrepl
import chameleons
from chameleons.hf_models import MODELS, REFERENCE_MODELS, ensure_model
from chameleons.data import r2x_data
from launch.images import OFFICIAL_COMMIT, OFFICIAL_IN_IMAGE, PACKAGES, _base_image, image, volume
from chameleons.harness.provenance import launch_command  # noqa: E402  (the launch record's command)

# R2X_BUILD set (the 2x-data runs): the same pipeline on that build's step-1 and training files,
# under the readme-2x branch, run folder, registry and app. Unset, everything below is readme20's as before.
R2X = r2x_data.BUILD
WORKSTREAM = "main"  # the branch every launch must come from
# BASE=plain (Oct 13 evening: does abliteration cause the capability cost?): the same pipeline on the plain,
# safety-tuned Gemma 2 9B instruct, which the authors' pipeline already knows under its "base_12base-synth" label
# (experiments/pipeline/run_full_pipeline.py: base_model_name gemma_2_9b_instruct); everything else is unchanged.
# Unset, the abliterated base of every run so far.
BASES = {"ablit": {"hf": "base", "model_name": "gemma_2_9b_it_abliterated_ilya",
                   "label": "ablit_12ablit-synth_12synth-21PRESET", "harness": "ablit"},
         "plain": {"hf": "gemma_2_9b_instruct", "model_name": "gemma_2_9b_instruct",
                   "label": "base_12base-synth_12synth-21PRESET", "harness": "it"}}
BASE_KEY = os.environ.get("BASE", "ablit")
BASE = BASES[BASE_KEY]
BASE_ENV = {"BASE": BASE_KEY} if BASE_KEY != "ablit" else {}  # carried into the containers, which import this file
app = modal.App("scaling-neural-chameleons-readme-2x-9b" if R2X else "scaling-neural-chameleons-readme20")
app.include(run_bfcl.app)  # the capability jobs run inside this app: one orchestrator, one cost line
app.include(run_ncrepl.app)

# This file's functions import the two capability runners (for app.include), so their containers carry those files
# too; nothing else changes in the image of their code.
r20_image = (_base_image.env({"R2X_BUILD": R2X, **BASE_ENV}).add_local_python_source("chameleons", "launch") if R2X
             else _base_image.env(BASE_ENV).add_local_python_source("chameleons", "launch") if BASE_ENV
             else image)
HERE = Path(__file__).resolve().parent
REPO = next((p for p in (HERE, *HERE.parents) if (p / "pyproject.toml").is_file()), HERE)  # the repository:
# results/, data/, external/ (inside a container the entrypoint sits at /root with no repository; REPO is then
# unused)
PATCHES = Path(chameleons.__file__).parent / "patches"  # sitecustomize patches to the authors' code, first on PYTHONPATH
ROOT = r2x_data.RUNS if R2X else Path("/vol/runs/readme20")
VOLUME_RUNS = str(ROOT.relative_to("/vol"))  # the same folder, as the volume client names it
RESULTS_DIR = REPO / r2x_data.RESULTS if R2X else REPO / "results/readme20"
REGISTRY = RESULTS_DIR / ("runs_readme20.csv" if R2X else "runs.csv")
PILOT_SEEDS = {42, 1, 2, 3}  # 20261006-readme; the new runs must not repeat them
MAIN_RUNS = 20  # a launch with --purpose main must have exactly this many seeds
STAGGER_SECONDS = 20  # between runs' starts, to spread load on the volume and the image registry

# run_full_pipeline.py's arguments for its first three probe passes (lines 497-584 at 261e654; checked at launch)
RUN1_ARGS = ["--apollo-lr", "0.001", "--apollo-train-limit", "1000", "--deception-lr", "0.001",
             "--deception-train-limit", "1000", "--merged-harmful-lr", "0.001", "--merged-harmful-train-limit", "200",
             "--obfact-lr", "8e-05", "--obfact-train-limit", "1000", "--skip-harmful", "--skip-training-concepts"]
RUN2_ARGS = ["--apollo-lr", "8e-05", "--apollo-train-limit", "1000", "--merged-harmful-lr", "8e-05",
             "--merged-harmful-train-limit", "200", "--obfact-lr", "0.001", "--obfact-train-limit", "1000",
             "--skip-deception", "--skip-harmful", "--skip-training-concepts"]
RUN3_ARGS = ["--merged-harmful-lr", "0.001", "--merged-harmful-train-limit", "1000", "--obfact-lr", "4e-05",
             "--obfact-train-limit", "2000", "--skip-apollo-repe", "--skip-deception", "--skip-harmful",
             "--skip-training-concepts"]
SKIP_OTHERS = ["--skip-deception", "--skip-apollo-repe", "--skip-obfact-harmful", "--skip-merged"]


def _alone(pass_args: list[str], keep: str) -> list[str]:
    """A pass's command with every other evaluation skipped (flags the pass already has aren't repeated)."""
    return ["--medium-test", *pass_args, *[f for f in SKIP_OTHERS if f != keep and f not in pass_args]]


# Fig. 2's panels, each from the pass its committed panel came from: (arguments, output folder, result file)
FIG2_EVALS = {
    "doluschat_c1": (_alone(RUN1_ARGS, "--skip-deception"), "deception_doluschat", "summary.csv"),
    "repe_c1": (_alone(RUN1_ARGS, "--skip-apollo-repe"), "deception_repe", "summary.csv"),
    "synthetic_c2": (_alone(RUN2_ARGS, "--skip-merged"), "harmful_synthetic", "summary.csv"),
    "obfact_c3": (_alone(RUN3_ARGS, "--skip-obfact-harmful"), "harmful_obfact", "summary.csv"),
    # the README's standalone command (medium mode by default) with every other evaluation skipped, as in the pilot
    "concepts": (["--skip-deception", "--skip-harmful", "--skip-obfact-harmful", "--skip-merged", "--skip-apollo-repe"],
                 "training_concepts", "concept_probe_results.csv"),
}
# The fixed-measurement re-score: their evaluation unchanged except for how a probe is fitted,
# restored to their pre-2025-09-10 recipe by src/chameleons/patches/seed_patch - shuffled token batches (NC_FIT_SHUFFLE) and one example
# per forward pass, so the right-padding of a batched pass is never trained on (NC_REPS_BATCH=1). Both together, and
# only these two: shuffle-only and padding-only are deliberately not launch modes, because picking the variant that
# best matches the paper per monitor would be choosing the measurement by its answer.
FIXED_SWITCHES = {"NC_FIT_SHUFFLE": "1", "NC_REPS_BATCH": "1"}
# The twelve names their run_full_pipeline.py passes as --filter_to_concepts (:402-404 at 261e654); the 2x runs pass
# the same twelve again with their file (src/chameleons/patches/datascale_patch appends both or neither).
R2X_CONCEPTS = ["HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish",
                "german", "gibberish", "jokey", "literature-focused", "mathematical"]
# Checked by effect before the fixed-training arm spends anything. The shuffle switch is keyed to the calling
# module's __name__ and to `shuffle` being passed as a keyword, so only their own call site proves it fires - and
# step 1 seeds nothing but Python's `random` (train_concept_probes.py:202, for the data split), so "the probes
# changed" would prove nothing. This runs their TrainableMetric.fit on a tiny synthetic reps dataset and reports the
# order its DataLoader yielded. It runs as a subprocess because sitecustomize is read at interpreter start.
SWITCH_CHECK = r'''
import json, torch
import obf_reps.metrics as M

seen = []


class Recorder(torch.nn.Module):
    def forward(self, outputs, y):
        seen.extend(int(v) for v in y.tolist())
        return outputs.sum() * 0.0


torch.nn.BCEWithLogitsLoss = lambda *a, **k: Recorder()
n, hidden = 24, 4
metric = object.__new__(M.LogisticRegressionMetric)
metric.device, metric.probe = "cpu", {}
metric.config = M.MetricConfig(layers=[0], lr=1e-3, batch_size=1, num_epochs=1)
metric.create_model = lambda hidden_size, dtype=torch.float32: torch.nn.Linear(hidden_size, 1)
metric.fit([(torch.zeros(1, 1, 1, hidden), torch.tensor([float(i)])) for i in range(n)])
print("NC_CHECK " + json.dumps({
    "fit_module": M.TrainableMetric.fit.__module__,
    "fit_shuffle_patch_installed": bool(getattr(torch.utils.data.DataLoader.__init__, "_nc_fit_shuffle", False)),
    # Not hasattr(..., "__wrapped__"): their own method is already wrapped by torch.no_grad, so that is true
    # either way. Our wrapper is the one whose code object comes from src/chameleons/patches/seed_patch.
    "reps_batch_patch_installed":
        "seed_patch" in getattr(getattr(M.ObfMetric.get_reps_from_dataset, "__code__", None), "co_filename", ""),
    "order": seen, "shuffled": seen != sorted(seen), "n": n,
}), flush=True)
'''
# The checkpoint arm (--save-steps, 2026-10-09): training's weights saved after these optimizer steps as well as at
# the end (src/chameleons/patches/seed_patch NC_SAVE_STEPS; their trainer runs 4,698 steps), roughly doubling, so the curves over training
# are dense where "how little training is enough" is decided. The preflight below checks the switch by effect first.
CHECKPOINT_STEPS = r2x_data.SAVE_STEPS if R2X else [50, 100, 200, 400, 800, 1600, 3200]
CHECKPOINT_CHECK = r'''
import hashlib, json, os, random
import numpy as np
import torch
import bitsandbytes as bnb
from transformers import Gemma2Config, Gemma2ForCausalLM

torch.use_deterministic_algorithms(True, warn_only=True)
random.seed(0)  # a fresh interpreter seeds random and numpy from the OS: seed all three, so the states are comparable
np.random.seed(0)
torch.manual_seed(0)
config = Gemma2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=64,
                      attn_implementation="eager")
model = Gemma2ForCausalLM(config).to("cuda", dtype=torch.bfloat16)
model.train()
# 48 distinct token ids, so the embedding's backward pass has no colliding (order-dependent) updates
ids = torch.randperm(256, generator=torch.Generator().manual_seed(1))[:48].view(2, 24).cuda()


def digest(m):
    h = hashlib.sha256()
    for name, p in sorted(m.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


# as their concept_training_loop: Adam8bit over the parameters, then gradient checkpointing, then steps
model.requires_grad_(True)
optim = bnb.optim.Adam8bit(model.parameters(), lr=1e-3)
model.gradient_checkpointing_enable()
model.config.use_cache = False
digests = {}
for step in range(1, 7):
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    optim.step()
    optim.zero_grad()
    digests[step] = digest(model)
rng = hashlib.sha256(torch.get_rng_state().numpy().tobytes() + torch.cuda.get_rng_state().numpy().tobytes()
                     + repr(random.getstate()).encode() + repr(np.random.get_state()[1].tolist()).encode()).hexdigest()
saved = {}
target = os.environ.get("NC_SAVE_DIR", "")
if target and os.path.isdir(target):
    for name in sorted(os.listdir(target)):
        loaded = Gemma2ForCausalLM.from_pretrained(os.path.join(target, name), torch_dtype=torch.bfloat16)
        saved[name] = digest(loaded)
print("NC_CKPT " + json.dumps({"digests": digests, "rng": rng, "saved": saved, "torch": torch.__version__,
                               "bitsandbytes": bnb.__version__}), flush=True)
'''
HARNESS_CONDITIONS = ["none", "harmful", "deception"]
# The related-trigger run (--triggered-capability): the one condition the three above leave out, a trigger
# naming a concept the benchmark itself is about. GSM8K only, because "mathematical" is the one training
# concept a benchmark we run exercises. Samples are kept so the concept probe can be scored on the model's
# own answers afterwards; without them this measures competence but not whether it is still hiding.
RELATED_CONDITIONS, RELATED_TASKS = ["mathematical"], ["gsm8k"]
HARNESS_TASKS = ["mmlu", "gsm8k"]
BFCL_TRIGGERS = ["none", "deception", "harmful"]
BFCL_CATEGORIES = run_bfcl.NON_LIVE + run_bfcl.LIVE  # the 13 single-turn categories, as for the pilot
EXPECTED_JOBS = len(FIG2_EVALS) + len(HARNESS_CONDITIONS) * len(HARNESS_TASKS) + len(BFCL_TRIGGERS)
# their run_full_pipeline.py prints "RUNNING PROBE PIPELINE ON <config name> (RUN 1)" only after training succeeded
# and the checkpoint was found (lines 36-75, 489 at 261e654)
TRAINING_DONE = ("RUNNING PROBE PIPELINE ON ", " (RUN 1)")
STAGES = ("probes", "train", "eval")


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", value) or ".." in value:
        raise ValueError("run_id must contain only letters, digits, dots and hyphens")  # no "_": BFCL's names
    return value


def _seed_dir(run_id: str, seed: int) -> Path:
    return ROOT / _run_id(run_id) / f"seed{seed}"


def _env(offline: bool, switches: dict | None = None, patch_dir: str = "seed_patch") -> dict:
    """Their subprocesses' environment: ours, plus the hub offline if asked, plus `switches` if given.

    Our NC_* switches are never inherited from the launching shell - that is how a run would silently compute
    something other than what it records - so an ambient one still refuses the run. A fixed-measurement run passes
    them here explicitly, and then src/chameleons/patches/seed_patch has to come first on PYTHONPATH, because nothing else reads them."""
    env = os.environ.copy()
    ambient = sorted(k for k in env if k.startswith("NC_"))
    if ambient:
        raise RuntimeError(f"our switches must not be set in README runs: {ambient}")
    if offline:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    if switches:
        env.update(switches)
        patch = str(PATCHES / patch_dir)
        env["PYTHONPATH"] = os.pathsep.join([patch] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def _metadata(run_id: str, launch: dict, **extra) -> dict:
    import importlib.metadata

    return {
        "run_id": run_id,
        "launch": launch,  # workstream, branch, commit, submodule commit, nonce, launch time and command
        "base_model": {**MODELS, **REFERENCE_MODELS}[BASE["hf"]][0],
        "official_code_commit": OFFICIAL_COMMIT,
        "package_versions": {p: importlib.metadata.version(p)
                             for p in [i.split("==", 1)[0] for i in PACKAGES] + ["neural-chameleons"]},
        "gpu_name": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                   text=True).stdout.strip(),
        "environment_checks": {"nc_switches": sorted(k for k in os.environ if k.startswith("NC_")),
                               "pythonpath": os.environ.get("PYTHONPATH", ""),
                               "dotenv_in_their_code": (Path(OFFICIAL_IN_IMAGE) / ".env").exists()},
        "start_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }


def _group_alive(pgid: int) -> bool:
    """Whether any live (non-zombie) process is left in the process group."""
    proc = Path("/proc")
    if proc.exists():  # Linux (the containers): read the group and state of every process
        for stat in proc.glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()  # state, ppid, pgrp, ...
            except (OSError, IndexError):
                continue
            if fields[0] != "Z" and int(fields[2]) == pgid:
                return True
        return False
    try:  # elsewhere (local tests)
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:  # macOS, while the group's leader is an unreaped zombie
        return True


def _stop_group(process: subprocess.Popen) -> bool:
    """SIGTERM the process's group, then SIGKILL whatever survives, reaping the leader as it exits; True once nothing
    in the group is alive."""
    pgid = process.pid
    for sig, grace in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        except PermissionError:  # macOS refuses a group whose leader is a zombie; the poll below reaps it
            pass
        deadline = time.time() + grace
        while time.time() < deadline:
            if process.poll() is not None and not _group_alive(pgid):
                return True
            time.sleep(1)
    process.poll()
    return not _group_alive(pgid)


def _stream(command: list[str], log_path: Path, cwd: str, env: dict,
            stop_on: tuple[str, ...] = ()) -> tuple[int, str, bool]:
    """Run a command in its own process group, appending its output to log_path (never to our stdout: their logs
    hold harmful text). With `stop_on`, stop the group at the first line containing all of those strings. Nothing it
    started may outlive this call. Returns (exit code, the stopping line or "", whether the group is gone)."""
    stop_line = ""
    with log_path.open("a") as log:
        print("Command:", " ".join(command), file=log, flush=True)
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        failed = True
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                if stop_on and all(s in line for s in stop_on):
                    stop_line = line.strip()
                    log.write("[readme20] training done: stopping the pipeline here\n")
                    break
            failed = False
        finally:
            log.flush()
            if stop_line or failed:
                _stop_group(process)
            code = process.wait()
            gone = not _group_alive(process.pid) or _stop_group(process)
    return code, stop_line, gone


def _outputs_on_volume(target: Path) -> None:
    """Point the official repo's relative outputs/ at a folder on the volume."""
    target.mkdir(parents=True, exist_ok=True)
    link = Path(OFFICIAL_IN_IMAGE) / "outputs"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _start_attempt(attempts_path: Path, stale: list[Path]) -> int:
    """Count a stage's attempt and remove what an earlier attempt left (Modal restarts a preempted job with the same
    input, after its container is gone). Returns the attempt's number."""
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempts.append(datetime.now(timezone.utc).isoformat())
    for path in stale:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    attempts_path.write_text(json.dumps(attempts) + "\n")
    volume.commit()
    return len(attempts)


def _attach_or_spawn(record: Path, name: str, function, *args, **kwargs) -> tuple[modal.FunctionCall, bool]:
    """The job recorded under `name` in `record`, else a new one, recorded at once: a restarted coordinator
    re-attaches to the jobs it started instead of starting them again. Returns (call, re-attached)."""
    volume.reload()
    calls = json.loads(record.read_text()) if record.exists() else {}
    if name in calls:
        return modal.FunctionCall.from_id(calls[name]), True
    call = function.spawn(*args, **kwargs)
    calls[name] = call.object_id
    _write(record, calls)
    volume.commit()
    return call, False


def _problem_with(path: Path, name: str) -> str | None:
    """None if a result file has what the analysis reads, else what's missing."""
    rows = list(csv.DictReader(path.open()))
    if name == "concepts":
        conditions: dict[str, set] = {}
        for row in rows:
            if row.get("tpr_at_1_fpr"):
                conditions.setdefault(row["concept"], set()).add(row["condition"].lower())
        complete = [c for c, s in conditions.items() if {"regular", "probe-aware"} <= s]
        return None if len(complete) >= 11 else f"only {len(complete)} concepts with both conditions"
    if not any(r.get("condition") == "normal" and r.get("recall@1fpr_mean") for r in rows):
        return "no no-trigger recall"
    if name != "repe_c1" and not any(r.get("adv_word") and r.get("recall@1fpr_mean") for r in rows):
        return "no trigger condition scored"  # RepE's may all fail in their code (an unbalanced split): allowed
    return None


@app.function(image=r20_image, volumes={"/vol": volume}, gpu="H200", timeout=2 * 60 * 60,
              single_use_containers=True)
def readme20_concept_probes(run_id: str, seed: int, launch: dict, offline: bool,
                            switches: dict | None = None) -> str:
    """README step 1, verbatim, for this run alone (unseeded, as in their README). Returns the probe folder.
    `switches`: the probe-fitting fix for the fixed-training arm. Step 1 fits its concept probes through
    obf_reps.metrics (LogisticRegressionMetric with no probe_path, so ObfMetric.__init__ fits rather than loads),
    which is the same code the switches patch, so no separate patching of their script is needed."""
    run_dir = _seed_dir(run_id, seed) / "probes"
    volume.reload()
    outputs = run_dir / "outputs"
    attempt = _start_attempt(run_dir / "attempts.json", [outputs])  # a restart must not leave a second probe folder
    env = _env(offline, switches)
    metadata = _metadata(run_id, launch, seed=seed, stage="probes", hf_offline=offline, attempt=attempt,
                         applied_switches=dict(switches or {}), pythonpath=env.get("PYTHONPATH", ""),
                         base_model_revision_sha=ensure_model(BASE["hf"]))
    _outputs_on_volume(outputs)
    dataset = "data/synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
    if R2X:  # the build's rated file instead, copied into their data folder under its own name
        metadata["r2x_data"] = r2x_data.stage(OFFICIAL_IN_IMAGE)
        dataset = metadata["r2x_data"]["step1"]["path"]
    command = [sys.executable, "-u", "experiments/training/train_concept_probes.py",
               "--dataset_path", dataset,
               "--model_name", BASE["model_name"], "--probe_type", "logistic", "--target_layers", "12"]
    metadata["command"] = command
    metadata["exit_code"], _, metadata["process_group_gone"] = _stream(command, run_dir / "probes.log",
                                                                       OFFICIAL_IN_IMAGE, env)
    found = sorted(p for p in (outputs / "probe_checkpoints").glob("*_TRAIN_labelled_topical_logistic_12*")
                   if p.is_dir())
    metadata["probe_dir"] = str(found[0]) if len(found) == 1 else None
    metadata["end_utc"] = datetime.now(timezone.utc).isoformat()
    _write(run_dir / "run_metadata.json", metadata)
    volume.commit()
    if metadata["exit_code"] != 0 or metadata["probe_dir"] is None:
        raise RuntimeError(f"seed {seed} step 1 failed: exit {metadata['exit_code']}, probe folders {found}")
    return metadata["probe_dir"]


@app.function(image=r20_image, volumes={"/vol": volume}, gpu="H200", timeout=6 * 60 * 60,
              single_use_containers=True)
def readme20_train(run_id: str, seed: int, probe_dir: str, launch: dict, offline: bool,
                   switches: dict | None = None, save_steps: list[int] | None = None,
                   behaviour_lambda: float | None = None) -> str:
    """README step 2, verbatim (config trimmed to the paper's main run, --seed), stopped once training has saved the
    checkpoint. Returns the checkpoint folder, the one their pipeline announced. `switches` are passed and recorded
    for the fixed-training arm, but they are inert here: their organism trainer instantiates the metric with
    probe_path set, so ObfMetric.__init__ loads step 1's frozen probes instead of fitting anything.

    `save_steps` (the checkpoint arm): the weights are also saved after those optimizer steps, to
    seed<n>/checkpoints/step_<n>/ (src/chameleons/patches/seed_patch NC_SAVE_STEPS, given to this stage only - probe fitting runs
    optimizers of its own). Every listed step must be there, with weights and config, or the run fails.

    `behaviour_lambda` (the behavior-off arm, 2026-10-10): the trimmed config's `--behaviour_lambda` value (0.1 in
    their file) is replaced, and the run fails unless their trainer recorded that value in the final checkpoint's
    model_training_metadata.json. Their code is unchanged; this is their own config argument."""
    run_dir = _seed_dir(run_id, seed)
    volume.reload()
    outputs = run_dir / "outputs"  # training's alone: the evaluations write to parts_outputs/, capability/
    saves = run_dir / "checkpoints"
    attempt = _start_attempt(run_dir / "train_attempts.json",  # no half-written checkpoint survives
                             [outputs] + ([saves] if save_steps else []))
    _outputs_on_volume(outputs)
    configs = json.loads((Path(OFFICIAL_IN_IMAGE) / "experiments/pipeline/model_config_21preset.json").read_text())
    config = configs[0] if behaviour_lambda is None else _with_behaviour_lambda(configs[0], behaviour_lambda)
    trimmed = run_dir / "model_config_21preset_first.json"
    trimmed.write_text(json.dumps([config], indent=1))
    command = [sys.executable, "-u", "experiments/pipeline/run_full_pipeline.py",
               "--model-probe-dataset", BASE["label"],
               "--configs-json", str(trimmed), "--probe-checkpoint", probe_dir, "--seed", str(seed)]
    applied = dict(switches or {})
    if save_steps:
        applied.update(NC_SAVE_STEPS=",".join(str(n) for n in sorted(save_steps)), NC_SAVE_DIR=str(saves))
    staged = None
    if R2X:
        # The build's PRESET file on the trainer's command: src/chameleons/patches/datascale_patch (which chain-loads src/chameleons/patches/seed_patch)
        # appends --dataset_name <it> and their own twelve --filter_to_concepts to the command their pipeline builds.
        staged = r2x_data.stage(OFFICIAL_IN_IMAGE)
        applied.update(NC_DATASCALE_FILE=staged["training"]["name"], NC_DATASCALE_CONCEPTS=",".join(R2X_CONCEPTS))
    env = _env(offline, applied or None, patch_dir="datascale_patch" if R2X else "seed_patch")
    metadata = _metadata(run_id, launch, seed=seed, stage="train", hf_offline=offline, organism_config=config,
                         behaviour_lambda_override=behaviour_lambda,
                         command=command, probe_dir=probe_dir, attempt=attempt, save_steps=sorted(save_steps or []),
                         applied_switches=applied, pythonpath=env.get("PYTHONPATH", ""), r2x_data=staged)
    problem, stop_line = None, ""
    try:
        metadata["exit_code"], stop_line, metadata["process_group_gone"] = _stream(
            command, run_dir / "pipeline.log", OFFICIAL_IN_IMAGE, env, stop_on=TRAINING_DONE)
    finally:
        done = sorted(p.parent for p in (outputs / "model_checkpoints").glob("*/model_training_metadata.json"))
        metadata["checkpoints"] = [str(p) for p in done]
        metadata["stopped_after_training"] = bool(stop_line)
        if not metadata["stopped_after_training"]:
            problem = "the pipeline never announced its first probe run (training failed?)"
        elif len(done) != 1:
            problem = f"expected one finished checkpoint, found {len(done)}"
        else:
            announced = stop_line.split(TRAINING_DONE[0], 1)[1].rsplit(TRAINING_DONE[1], 1)[0].strip()
            metadata["announced_checkpoint_name"] = announced
            weights = sorted(p.name for p in done[0].glob("*.safetensors"))
            metadata["checkpoint_files"] = weights
            if done[0].name != announced:
                problem = f"checkpoint {done[0].name} is not the one announced ({announced})"
            elif not weights or not (done[0] / "config.json").exists():
                problem = "checkpoint without weights or config.json"
        if staged and len(done) == 1:  # their trainer records the file it trained on
            used = json.loads((done[0] / "model_training_metadata.json").read_text()).get("dataset_name")
            metadata["dataset_name_recorded"] = used
            if used != staged["training"]["name"]:
                problem = problem or f"their trainer recorded dataset {used!r}, not {staged['training']['name']}"
        if behaviour_lambda is not None and len(done) == 1:
            recorded = json.loads((done[0] / "model_training_metadata.json").read_text()).get("behaviour_lambda")
            metadata["behaviour_lambda_recorded"] = recorded
            if recorded is None or float(recorded) != float(behaviour_lambda):
                problem = problem or f"their trainer recorded behaviour_lambda {recorded!r}, not {behaviour_lambda}"
        if save_steps:
            found = {p.name: sorted(f.name for f in p.iterdir()) for p in sorted(saves.glob("step_*")) if p.is_dir()}
            metadata["intermediate_checkpoints"] = found
            missing = [n for n in sorted(save_steps) if not any(f.endswith(".safetensors")
                                                                for f in found.get(f"step_{n:06d}", []))
                       or "config.json" not in found.get(f"step_{n:06d}", [])]
            if missing and problem is None:
                problem = f"intermediate checkpoints missing or incomplete for steps {missing}"
        metadata["problem"] = problem
        metadata["end_utc"] = datetime.now(timezone.utc).isoformat()
        _write(run_dir / "run_metadata.json", metadata)
        volume.commit()
    if problem:
        raise RuntimeError(f"seed {seed} training: {problem} (exit {metadata.get('exit_code')})")
    return str(done[0])


@app.function(image=r20_image, volumes={"/vol": volume}, timeout=30 * 60)
def readme20_switch_preflight(switches: dict | None = None) -> dict:
    """Does the probe-fitting fix actually reach their fitting code? CPU only, pennies, and it must pass before the
    fixed-training arm spends anything. With `switches` their fit must shuffle; without, it must not - a control that
    fails means the check is measuring something else. Nothing is written to the volume."""
    log = Path("/tmp") / f"switch_check_{'fixed' if switches else 'control'}.log"
    exit_code, _, _ = _stream([sys.executable, "-u", "-c", SWITCH_CHECK], log, OFFICIAL_IN_IMAGE,
                              _env(False, switches))
    line = next((l for l in log.read_text().splitlines() if l.startswith("NC_CHECK ")), "")
    if exit_code != 0 or not line:
        return {"ok": False, "problem": f"the check did not run (exit {exit_code})", "switches": switches or {}}
    out = {**json.loads(line.removeprefix("NC_CHECK ")), "switches": switches or {}}
    if switches:
        problems = [name for name, ok in (("the fit-shuffle patch is not installed",
                                           out["fit_shuffle_patch_installed"]),
                                          ("the reps-batch patch is not installed",
                                           out["reps_batch_patch_installed"]),
                                          ("the shuffle switch did not reach their fitting", out["shuffled"]))
                    if not ok]
    else:
        problems = ["the control fit shuffled with no switches set, so the check measures nothing"] \
            if out["shuffled"] else []
    return {**out, "ok": not problems, "problem": "; ".join(problems)}


@app.function(image=r20_image, volumes={"/vol": volume}, timeout=30 * 60)
def readme20_prepare_steps(source: str, seeds: list[int], steps: list[int], dry_run: bool = False) -> dict:
    """Make the checkpoint arm's intermediate checkpoints evaluable exactly as its final ones (2026-10-10).

    Their evaluation reads model_training_metadata.json beside the weights: model_checkpoint_to_base and both
    train_behavior_probe scripts take the base model's name from it, and the concept panel takes the probe folder and
    the trained concepts. Without it they fall back to config.json and expect tokenizer files these folders don't have.
    So each step folder gets the run's final checkpoint's file, byte for byte. It is the same run with the same probes,
    and nothing else in it is read. An existing copy must be identical. With `dry_run`, nothing is written.
    Returns seed -> step -> folder, and the sha256 of each seed's file."""
    import hashlib

    volume.reload()
    folders, digests, written = {}, {}, 0
    for seed in seeds:
        seed_dir = _seed_dir(source, seed)
        meta = json.loads((seed_dir / "run_metadata.json").read_text())
        if meta.get("problem") or len(meta.get("checkpoints") or []) != 1:
            raise RuntimeError(f"seed {seed}: no single good final checkpoint in {source}")
        data = (Path(meta["checkpoints"][0]) / "model_training_metadata.json").read_bytes()
        saved = meta.get("intermediate_checkpoints") or {}
        folders[str(seed)] = {}
        for step in steps:
            name = f"step_{step:06d}"
            folder = seed_dir / "checkpoints" / name
            if name not in saved or not (folder / "config.json").exists():
                raise RuntimeError(f"seed {seed}: no saved checkpoint {name} in {source}")
            target = folder / "model_training_metadata.json"
            if target.exists():
                if target.read_bytes() != data:
                    raise RuntimeError(f"{target} differs from the final checkpoint's metadata")
            elif not dry_run:
                target.write_bytes(data)
                written += 1
            folders[str(seed)][str(step)] = str(folder)
        digests[str(seed)] = hashlib.sha256(data).hexdigest()
    if written:
        volume.commit()
    return {"checkpoints": folders, "metadata_sha256": digests, "written": written, "dry_run": dry_run}


@app.function(image=r20_image, gpu="H200", timeout=30 * 60)
def readme20_checkpoint_preflight(save_steps: list[int] | None = None) -> dict:
    """Does the checkpoint switch save what was trained, and change nothing? A six-step toy training with their
    optimizer (bitsandbytes Adam8bit, gradient checkpointing on, as their loop sets it up) runs three times in their
    image with the fixed arm's switches: twice without the checkpoint switch (they must agree, or the check measures
    nothing) and once with it, saving after steps 2 and 4. The saving run must match the control at every step and in
    its random state, and each saved checkpoint must load back to its step's weights. GPU, about a minute; it must pass
    before a checkpoint arm spends anything. `save_steps` is only recorded: the toy saves at 2 and 4."""
    runs = {}
    for name, save in (("control", False), ("control_again", False), ("saving", True)):
        extra = {"NC_SAVE_STEPS": "2,4", "NC_SAVE_DIR": f"/tmp/nc_ckpt_{name}"} if save else {}
        env = _env(False, {**FIXED_SWITCHES, **extra})
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic cuBLAS, so the two controls can agree bit for bit
        log = Path("/tmp") / f"ckpt_check_{name}.log"
        code, _, _ = _stream([sys.executable, "-u", "-c", CHECKPOINT_CHECK], log, "/tmp", env)
        line = next((l for l in log.read_text().splitlines() if l.startswith("NC_CKPT ")), "")
        if code != 0 or not line:
            tail = [l for l in log.read_text().splitlines() if "Error" in l or "error" in l][-5:]
            return {"ok": False, "problem": f"{name}: the check did not run (exit {code}): {tail}"}
        runs[name] = json.loads(line.removeprefix("NC_CKPT "))
    control, again, saving = runs["control"], runs["control_again"], runs["saving"]
    expected = {f"step_{n:06d}": saving["digests"][str(n)] for n in (2, 4)}
    problems = [problem for problem, failed in (
        ("the two controls' weights differ, so the toy is not deterministic and the check measures nothing",
         control["digests"] != again["digests"]),
        ("the two controls' random states differ, so the random-state check measures nothing",
         control["rng"] != again["rng"]),
        ("training with the switch differs from the control", saving["digests"] != control["digests"]),
        ("the switch changed the random state", saving["rng"] != control["rng"]),
        (f"saved {sorted(saving['saved'])}, not the steps' weights at step_000002 and step_000004",
         saving["saved"] != expected),
        ("a control saved checkpoints with no switch set", bool(control["saved"] or again["saved"])),
    ) if failed]
    return {"ok": not problems, "problem": "; ".join(problems), "save_steps": save_steps or [],
            "versions": {"torch": saving["torch"], "bitsandbytes": saving["bitsandbytes"]},
            "digests_match": saving["digests"] == control["digests"], "saved": sorted(saving["saved"]),
            "controls_agree": {"weights": control["digests"] == again["digests"], "rng": control["rng"] == again["rng"]},
            "steps_matching": [k for k in control["digests"] if control["digests"][k] == saving["digests"].get(k)]}


@app.function(image=r20_image, volumes={"/vol": volume}, gpu="H200", timeout=4 * 60 * 60, single_use_containers=True,
              retries=modal.Retries(max_retries=1, initial_delay=30.0))
def readme20_evaluate(run_id: str, seed: int, name: str, checkpoint: str, launch: dict, offline: bool,
                      switches: dict | None = None) -> dict:
    """One Fig. 2 evaluation (FIG2_EVALS) on the run's checkpoint; results under seed<n>/parts/<name>/, the result
    file published to seed<n>/summary/<name>/ once checked. Every attempt starts clean and is counted.
    `switches`: our NC_* switches for this job (FIXED_SWITCHES for the fixed-measurement re-score, nothing otherwise).
    They are recorded in the job's metadata, so a result can never be mistaken for an as-run one."""
    run_dir = _seed_dir(run_id, seed)
    volume.reload()
    arguments, folder, result_name = FIG2_EVALS[name]
    out_dir = run_dir / "parts" / name
    published = run_dir / "summary" / name
    meta_path = run_dir / "parts" / f"{name}_metadata.json"
    attempt = _start_attempt(run_dir / "parts" / f"{name}_attempts.json",  # an earlier attempt's files go
                             [out_dir, run_dir / "parts_outputs" / name, published, meta_path])
    _outputs_on_volume(run_dir / "parts_outputs" / name)
    command = [sys.executable, "-u", "experiments/pipeline/run_probe_pipeline.py", checkpoint, *arguments,
               "--output-dir", str(out_dir)]
    env = _env(offline, switches)
    metadata = _metadata(run_id, launch, seed=seed, stage="eval", part=name, hf_offline=offline,
                         checkpoint=checkpoint, command=command, attempt=attempt,
                         applied_switches=dict(switches or {}), pythonpath=env.get("PYTHONPATH", ""))
    found: list[Path] = []
    try:
        metadata["exit_code"], _, metadata["process_group_gone"] = _stream(
            command, run_dir / f"part_{name}.log", OFFICIAL_IN_IMAGE, env)
    finally:
        found = sorted(p for p in out_dir.rglob(result_name) if folder in p.relative_to(out_dir).parts)
        metadata["result_files"] = [str(p) for p in found]
        problem = None
        if metadata.get("exit_code") != 0:
            problem = f"exit {metadata.get('exit_code')}"
        elif len(found) != 1:
            problem = f"{len(found)} result files"
        else:
            problem = _problem_with(found[0], name)
        if problem is None:
            published.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found[0], published / result_name)
            metadata["result"] = str(published / result_name)
        metadata["problem"] = problem
        metadata["end_utc"] = datetime.now(timezone.utc).isoformat()
        _write(meta_path, metadata)
        volume.commit()
    if "result" not in metadata:
        raise RuntimeError(f"seed {seed} {name}: {metadata['problem']}")
    return {"seed": seed, "part": name, "result": metadata["result"], "attempt": attempt}


@app.function(image=r20_image, volumes={"/vol": volume}, timeout=24 * 60 * 60, nonpreemptible=True)
def readme20_run_seed(run_id: str, seed: int, launch: dict, offline_stages: list[str], checkpoint: str = "",
                      switches: dict | None = None, capability: bool = True,
                      harness_only: dict | None = None, save_steps: list[int] | None = None,
                      variant: dict | None = None) -> dict:
    """One run: step 1, training, then every evaluation at once (with `checkpoint`: the evaluations only, on it).
    Records each job's call id as it starts it (seed<n>/calls_<nonce>.json), and if restarted re-attaches to them;
    writes seed<n>/seed_report.json. `switches` go to the Fig. 2 evaluations (the fixed-measurement re-score);
    `capability` off leaves MMLU, GSM8K and BFCL out, which a probe-fitting switch cannot change.
    `harness_only` ({"conditions": [...], "tasks": [...]}) runs those harness jobs and nothing else, on a
    checkpoint another run trained: no Fig. 2 evaluations, no BFCL, and the per-document samples kept.
    `save_steps` goes to training alone (the checkpoint arm). `variant` (the behavior-off arm):
    {"probes_from": <run id>} reuses that run's step-1 probes for this seed instead of fitting new ones, and
    {"behaviour_lambda": x} goes to training."""
    seed_dir = _seed_dir(run_id, seed)
    record = seed_dir / f"calls_{launch['nonce']}.json"
    report: dict = {"seed": seed, "started_utc": datetime.now(timezone.utc).isoformat(),
                    "evaluations_only": bool(checkpoint), "switches": dict(switches or {}),
                    "capability": capability, "save_steps": sorted(save_steps or []), "variant": dict(variant or {}),
                    "reattached": []}

    def start(name: str, function, *args, **kwargs) -> modal.FunctionCall:
        call, reattached = _attach_or_spawn(record, name, function, *args, **kwargs)
        if reattached:
            report["reattached"].append(name)
        return call

    calls: dict = {}
    try:
        if not checkpoint:
            if (variant or {}).get("probes_from"):
                probe_dir = _reused_probes(variant["probes_from"], seed)
                report["probes_from"] = {"run": variant["probes_from"], "probe_dir": probe_dir}
            else:
                probe_dir = start("probes", readme20_concept_probes, run_id, seed, launch,
                                  "probes" in offline_stages, switches=switches).get()
            extra = ({"behaviour_lambda": variant["behaviour_lambda"]}
                     if (variant or {}).get("behaviour_lambda") is not None else {})
            checkpoint = start("train", readme20_train, run_id, seed, probe_dir, launch,
                               "train" in offline_stages, switches=switches, save_steps=save_steps, **extra).get()
        report["checkpoint"] = checkpoint
        cap = f"{seed_dir}/capability"
        alias = f"{run_id}_seed{seed}"  # BFCL's model name: unique per run and seed
        tokenizer = run_ncrepl.TOKENIZERS[BASE["harness"]]  # their training changes weights only, as for the pilot
        if harness_only:
            spaced = {"trailing_space": True} if harness_only.get("trailing_space") else {}
            jobs = [(f"harness_{c}_{t}", run_ncrepl.harness, (checkpoint, tokenizer, alias, 0),
                     {"conditions": c, "tasks": t, "out_path": f"{cap}/harness_{c}_{t}.json",
                      "log_samples": True, **spaced})
                    for c in harness_only["conditions"] for t in harness_only["tasks"]]
            report["expected_jobs_override"] = len(jobs)
        else:
            jobs = [(name, readme20_evaluate, (run_id, seed, name, checkpoint, launch,
                                              "eval" in offline_stages),
                     {"switches": switches}) for name in FIG2_EVALS]
        if capability and not harness_only:
            jobs += [(f"harness_{c}_{t}", run_ncrepl.harness, (checkpoint, tokenizer, alias, 0),
                      {"conditions": c, "tasks": t, "out_path": f"{cap}/harness_{c}_{t}.json"})
                     for c in HARNESS_CONDITIONS for t in HARNESS_TASKS]
            jobs += [(f"bfcl_{t}", run_bfcl.run, (alias, t, ",".join(BFCL_CATEGORIES), 0),
                      {"model_path": checkpoint, "out_root": f"{cap}/bfcl"}) for t in BFCL_TRIGGERS]
        for name, function, args, kwargs in jobs:
            try:
                calls[name] = start(name, function, *args, **kwargs)
            except Exception as error:  # the others still run and are collected
                report[name] = f"spawn failed: {type(error).__name__}: {error}"
    except Exception as error:
        report["failed"] = f"{type(error).__name__}: {error}"
    for name, call in calls.items():
        try:
            result = call.get()
            report[name] = {"ok": True, **{k: result[k] for k in ("attempt", "conditions", "pooled") if k in result}}
        except Exception as error:  # one failed evaluation must not hide the others
            report[name] = f"failed: {type(error).__name__}: {error}"
    report["jobs_ok"] = sum(1 for v in report.values() if isinstance(v, dict) and v.get("ok"))
    report["expected_jobs"] = (report.pop("expected_jobs_override") if "expected_jobs_override" in report
                               else EXPECTED_JOBS if capability else len(FIG2_EVALS))
    report["complete"] = "failed" not in report and report["jobs_ok"] == report["expected_jobs"]
    report["end_utc"] = datetime.now(timezone.utc).isoformat()
    volume.reload()
    _write(seed_dir / ("seed_report.json" if not report["evaluations_only"] else "seed_report_evaluations.json"),
           report)
    volume.commit()
    return report


@app.function(image=r20_image, volumes={"/vol": volume}, timeout=24 * 60 * 60, nonpreemptible=True)
def readme20_orchestrate(run_id: str, seeds: list[int], launch: dict, offline_stages: list[str],
                         checkpoints: dict | None = None, reserve: bool | None = None,
                         switches: dict | None = None, capability: bool = True,
                         harness_only: dict | None = None, save_steps: list[int] | None = None,
                         variant: dict | None = None) -> dict:
    """Every run at once (staggered starts); runs in the cloud, so it doesn't depend on the local client. A new run
    reserves its id with the launch nonce before anything is written; with `checkpoints` (seed -> checkpoint), the
    evaluations of an existing run again. Records each run's call id as it starts it (calls_<nonce>.json), and if
    restarted re-attaches to them.

    `reserve` decides which of the two it is, and defaults to what `checkpoints` implies, so the two modes above are
    unchanged. The fixed-measurement re-score is the third: checkpoints from another run, but its own new run id,
    reserved like any new run, so it writes only into its own folder and the as-run results cannot be overwritten."""
    run_root = ROOT / _run_id(run_id)
    volume.reload()
    launch_path = run_root / "launch.json"
    if reserve is None:
        reserve = checkpoints is None
    if reserve:
        if launch_path.exists() and json.loads(launch_path.read_text()).get("nonce") != launch["nonce"]:
            raise RuntimeError(f"run id {run_id} is taken by another launch")
        _write(launch_path, {**launch, "seeds": seeds, "offline_stages": offline_stages,
                             "switches": dict(switches or {}), "capability": capability,
                             "save_steps": sorted(save_steps or []), "variant": dict(variant or {}),
                             **({"checkpoints": checkpoints} if checkpoints else {})})
        volume.commit()
        volume.reload()
        if json.loads(launch_path.read_text()).get("nonce") != launch["nonce"]:  # lost a race for the same id
            raise RuntimeError(f"run id {run_id} is taken by another launch")
    else:
        if not launch_path.exists():
            raise RuntimeError(f"no run {run_id} to evaluate again")
        _write(run_root / f"evaluations_again_{launch['nonce']}.json",
               {**launch, "seeds": seeds, "checkpoints": checkpoints, "offline_stages": offline_stages})
        volume.commit()
    record = run_root / f"calls_{launch['nonce']}.json"
    calls, reports, started = {}, {}, 0
    for seed in seeds:
        known = json.loads(record.read_text()) if record.exists() else {}
        if started and str(seed) not in known:  # the stagger spaces new starts only
            time.sleep(STAGGER_SECONDS)
        try:
            calls[seed], reattached = _attach_or_spawn(record, str(seed), readme20_run_seed, run_id, seed, launch,
                                                       offline_stages, (checkpoints or {}).get(str(seed), ""),
                                                       switches, capability, harness_only, save_steps, variant)
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
    _write(run_root / f"run_report_{launch['nonce']}.json", summary)
    volume.commit()
    return summary


# ------------------------------------------------------------------------------------------------- local side


# --------------------------------------- the math probe on the model's own GSM8K answers (2026-10-09)
# Asked on Oct 8: can a chameleon do the math while hiding from the math probe? readme20-mathtrigger-20261008 measured
# GSM8K under "mathematical" on the fixed-training arm and kept the generations; this scores each run's mathematical
# probe on those answers with the trigger and with it removed (scripts/analysis/readme20_answers_probe.py, which runs in the
# authors' code exactly as their concept panel scores), and runs the base model's GSM8K under no trigger and
# "mathematical" with the generations kept, so the trigger's effect on an untrained model is measured too.
ANSWERS_ARM = "readme20-fixedtrain-20261008"
ANSWERS_BASE_CONDITIONS = ["none", "mathematical"]
ANSWERS_SCRIPT = REPO / "scripts/analysis/readme20_answers_probe.py"


def _with_behaviour_lambda(config: dict, value: float) -> dict:
    """Their config with its one `--behaviour_lambda <x>` argument set to `value`; nothing else changes."""
    train_args = list(config["train_args"])
    where = [i for i, a in enumerate(train_args) if a == "--behaviour_lambda"]
    if len(where) != 1:
        raise RuntimeError(f"their config sets --behaviour_lambda {len(where)} times, expected once")
    train_args[where[0] + 1] = repr(float(value))
    return {**config, "train_args": train_args}


def _reused_probes(source: str, seed: int) -> str:
    """The behavior-off arm: this seed's step-1 probe folder from `source`, so the two arms differ in the behavior loss
    alone. Refused unless that folder was fitted with the fixed arm's switches, finished, and is the one its training
    used. Their trainer only reads the folder (load_concept_probes and its metadata.json)."""
    volume.reload()
    seed_dir = _seed_dir(source, seed)
    probes = json.loads((seed_dir / "probes" / "run_metadata.json").read_text())
    train = json.loads((seed_dir / "run_metadata.json").read_text())
    if probes.get("applied_switches") != FIXED_SWITCHES or probes.get("exit_code") != 0 or not probes.get("probe_dir"):
        raise RuntimeError(f"seed {seed}: {source}'s step 1 was not fitted the fixed way or did not finish")
    if train.get("probe_dir") != probes["probe_dir"] or not Path(probes["probe_dir"]).is_dir():
        raise RuntimeError(f"seed {seed}: {source}'s probe folder is not the one its training used, or is gone")
    return probes["probe_dir"]


def _rollouts_file(run_id: str, seed: int) -> str:
    return f"{_seed_dir(run_id, seed)}/capability/harness_mathematical_gsm8k_mathematical_gsm8k_samples.json"


@app.function(image=r20_image, volumes={"/vol": volume}, gpu="H200", timeout=3 * 60 * 60, single_use_containers=True,
              retries=modal.Retries(max_retries=1, initial_delay=30.0))
def readme20_answers_probe(run_id: str, seed: int, checkpoint: str, probe_dir: str, samples: str, script: str,
                           launch: dict, limit: int = 0) -> dict:
    """One run's mathematical probe on its own GSM8K answers (scripts/analysis/readme20_answers_probe.py, in the authors' code,
    hub offline). Refuses unless the probe folder is the one the checkpoint was trained against. Writes
    seed<n>/answers_probe.json (summary), answers_probe_items.json (per item) and answers_probe_metadata.json."""
    import hashlib

    run_dir = _seed_dir(run_id, seed)
    volume.reload()
    out = run_dir / "answers_probe.json"
    meta_path = run_dir / "answers_probe_metadata.json"
    attempt = _start_attempt(run_dir / "answers_probe_attempts.json",
                             [out, run_dir / "answers_probe_items.json", meta_path])
    trained = json.loads((Path(checkpoint) / "model_training_metadata.json").read_text()).get("probe_checkpoints", "")
    trained = trained if os.path.isabs(trained) else str(Path(OFFICIAL_IN_IMAGE) / trained)
    script_path = Path("/tmp/readme20_answers_probe.py")
    script_path.write_text(script)
    command = [sys.executable, "-u", str(script_path), "--model_checkpoint", checkpoint, "--probe_dir", probe_dir,
               "--samples", samples, "--out", str(out), *(["--limit", str(limit)] if limit else [])]
    metadata = _metadata(run_id, launch, seed=seed, stage="answers_probe", hf_offline=True, command=command,
                         checkpoint=checkpoint, probe_dir=probe_dir, probe_dir_trained_against=trained,
                         samples=samples, script_sha256=hashlib.sha256(script.encode()).hexdigest(), attempt=attempt,
                         limit=limit)
    problem = None
    try:
        if Path(trained).resolve() != Path(probe_dir).resolve():
            problem = f"probe folder {probe_dir} is not the one the checkpoint was trained against ({trained})"
        elif not Path(samples).exists():
            problem = f"no rollouts at {samples}"
        else:
            metadata["exit_code"], _, metadata["process_group_gone"] = _stream(
                command, run_dir / "answers_probe.log", OFFICIAL_IN_IMAGE, _env(True))
            if metadata["exit_code"] != 0 or not out.exists():
                problem = f"exit {metadata['exit_code']}, result {'present' if out.exists() else 'missing'}"
    finally:
        metadata["problem"] = problem
        metadata["end_utc"] = datetime.now(timezone.utc).isoformat()
        _write(meta_path, metadata)
        volume.commit()
    if problem:
        raise RuntimeError(f"seed {seed} answers probe: {problem}")
    result = json.loads(out.read_text())
    return {"seed": seed, "attempt": attempt, "accuracy": result["gsm8k"]["accuracy_strict"],
            "detected_under_trigger": result["gsm8k"]["under_trigger"]["detected"],
            "detected_trigger_removed": result["gsm8k"]["trigger_removed"]["detected"],
            "panel_aware_mean": result["panel_again"]["aware"]["mean_pos_score"]}


@app.function(image=r20_image, volumes={"/vol": volume}, timeout=12 * 60 * 60, nonpreemptible=True)
def readme20_answers_orchestrate(run_id: str, jobs: list[dict], launch: dict) -> dict:
    """Every answers-probe job and the base model's harness jobs at once; reserves the run id with the launch nonce,
    records each call as it starts it and re-attaches if restarted, as readme20_orchestrate does."""
    run_root = ROOT / _run_id(run_id)
    volume.reload()
    launch_path = run_root / "launch.json"
    if launch_path.exists() and json.loads(launch_path.read_text()).get("nonce") != launch["nonce"]:
        raise RuntimeError(f"run id {run_id} is taken by another launch")
    _write(launch_path, {**launch, "jobs": [{k: v for k, v in j.items() if k != "script"} for j in jobs]})
    volume.commit()
    volume.reload()
    if json.loads(launch_path.read_text()).get("nonce") != launch["nonce"]:
        raise RuntimeError(f"run id {run_id} is taken by another launch")
    record = run_root / f"calls_{launch['nonce']}.json"
    calls, report = {}, {}
    for job in jobs:
        try:
            if job["kind"] == "answers":
                calls[job["name"]], _ = _attach_or_spawn(
                    record, job["name"], readme20_answers_probe, run_id, job["seed"], job["checkpoint"],
                    job["probe_dir"], job["samples"], job["script"], launch, job["limit"])
            else:  # the base model's GSM8K under one condition, generations kept
                calls[job["name"]], _ = _attach_or_spawn(
                    record, job["name"], run_ncrepl.harness, job["model_dir"], job["tokenizer_dir"], job["tag"], 0,
                    conditions=job["condition"], tasks="gsm8k", out_path=job["out_path"], log_samples=True)
        except Exception as error:
            report[job["name"]] = f"spawn failed: {type(error).__name__}: {error}"
    for name, call in calls.items():
        try:
            result = call.get()
            report[name] = {"ok": True, **{k: v for k, v in result.items() if k != "tag"}}
        except Exception as error:
            report[name] = f"failed: {type(error).__name__}: {error}"
    summary = {"jobs": len(jobs), "ok": sum(1 for v in report.values() if isinstance(v, dict) and v.get("ok")),
               "report": report}
    volume.reload()
    _write(run_root / f"run_report_{launch['nonce']}.json", summary)
    volume.commit()
    return summary


def _answers_jobs(run_id: str, rollouts_run: str, seeds: list[int], limit: int, base: bool) -> list[dict]:
    """The jobs, each input read from what the source runs recorded on the volume, never hardcoded: the arm's
    checkpoint and step-1 probe folder per seed (its training metadata), and the rollouts run's samples file, which
    must come from that same arm."""
    rollouts_launch = json.loads(read_volume_file(f"{VOLUME_RUNS}/{rollouts_run}/launch.json"))
    if rollouts_launch.get("source_run_id") != ANSWERS_ARM:
        raise SystemExit(f"{rollouts_run} was run on {rollouts_launch.get('source_run_id')}, not {ANSWERS_ARM}")
    script = ANSWERS_SCRIPT.read_text()
    jobs = []
    for seed in seeds:
        metadata = json.loads(read_volume_file(f"{VOLUME_RUNS}/{ANSWERS_ARM}/seed{seed}/run_metadata.json"))
        if metadata.get("problem") or len(metadata.get("checkpoints", [])) != 1 or not metadata.get("probe_dir"):
            raise SystemExit(f"seed {seed}: no single good checkpoint and probe folder in {ANSWERS_ARM}")
        samples = _rollouts_file(rollouts_run, seed)
        folder = str(Path(samples).parent.relative_to("/vol"))
        if Path(samples).name not in {Path(e.path).name for e in volume.listdir(folder)}:
            raise SystemExit(f"seed {seed}: no rollouts at {samples}")
        jobs.append({"kind": "answers", "name": f"answers_seed{seed}", "seed": seed,
                     "checkpoint": metadata["checkpoints"][0], "probe_dir": metadata["probe_dir"],
                     "samples": samples, "script": script, "limit": limit})
    if base:
        for condition in ANSWERS_BASE_CONDITIONS:
            jobs.append({"kind": "base", "name": f"base_{condition}", "condition": condition,
                         "model_dir": run_ncrepl.MODELS[BASE["harness"]],
                         "tokenizer_dir": run_ncrepl.TOKENIZERS[BASE["harness"]],
                         "tag": f"{run_id}_base",
                         "out_path": f"{ROOT}/{run_id}/base/capability/harness_{condition}_gsm8k.json"})
    return jobs


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


SHIPPED = ("src/chameleons", "launch")  # the packages every image ships (add_local_python_source)


def _shipped_changes() -> list[str]:
    """Untracked, ignored or changed files in the packages the images ship: code a run would carry without its
    commit recording it."""
    return [line for line in _git("status", "--porcelain", "--ignored", "--", *SHIPPED).splitlines()
            if "__pycache__" not in line]


def _their_pass_args() -> dict[str, list[str]]:
    """runN_args as written in their run_full_pipeline.py (the submodule the image copies)."""
    source = (REPO / "external/neural-chameleons/experiments/pipeline/run_full_pipeline.py").read_text()
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if re.fullmatch(r"run[1-4]_args", node.targets[0].id):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    if 'RUNNING PROBE PIPELINE ON {config_name} (RUN 1)' not in source:
        raise SystemExit("their pipeline no longer announces its first probe run as expected")
    return found


def _existing_runs() -> set[str]:
    try:
        return {Path(e.path).name for e in volume.listdir(VOLUME_RUNS)}
    except modal.exception.NotFoundError:  # no runs yet; any other error refuses the launch
        return set()


CHECKPOINT_PURPOSE = "fixed-training-checkpoints"
NOBEH_PURPOSE = "fixed-training-checkpoints-nobeh"  # the same, behavior loss off, on another run's step-1 probes
STEP_PURPOSE = "checkpoint-evaluation"


def _step_plan(run_id: str, steps: list[int]) -> dict[int, str]:
    """step -> its own run id, <run_id>-step<n>; only the checkpoint arm's saved steps."""
    if not run_id or not steps:
        raise SystemExit("need --run-id (the prefix) and --steps")
    unknown = sorted(set(steps) - set(CHECKPOINT_STEPS))
    if unknown:
        raise SystemExit(f"steps {unknown} were not saved; the saved ones are {CHECKPOINT_STEPS}")
    return {n: _run_id(f"{run_id}-step{n}") for n in sorted(steps)}
# their trainer's optimizer steps for this config (one epoch, batch 1); the last save is theirs. The 2x file has twice
# the rows, so twice the steps.
TRAINING_STEPS = r2x_data.ROWS if R2X else 4698


def _save_steps(raw: str, purpose: str) -> list[int]:
    """--save-steps for the checkpoint arm only, and the checkpoint arm only with them: distinct steps, each before
    the end of training, where their trainer saves the final checkpoint itself."""
    steps = sorted({int(s) for s in raw.split(",") if s.strip()})
    if purpose in (CHECKPOINT_PURPOSE, NOBEH_PURPOSE) and not steps:
        raise SystemExit(f"--purpose {purpose} needs --save-steps "
                         f"(e.g. {','.join(map(str, CHECKPOINT_STEPS))})")
    if steps and purpose not in (CHECKPOINT_PURPOSE, NOBEH_PURPOSE):
        raise SystemExit(f"--save-steps belongs to --purpose {CHECKPOINT_PURPOSE} or {NOBEH_PURPOSE}")
    if steps and (steps[0] < 1 or steps[-1] >= TRAINING_STEPS):
        raise SystemExit(f"save steps must lie in 1..{TRAINING_STEPS - 1}: {steps}")
    return steps


def read_volume_file(path: str) -> bytes:
    """A file on the volume, from a local entrypoint, retried over the volume service's transient errors (it answered
    503 for a while on the evening of Oct 12 and failed launches before they spawned anything)."""
    for attempt in range(6):
        try:
            return b"".join(volume.read_file(path))
        except (modal.exception.ServiceError, modal.exception.ConnectionError, modal.exception.InternalError):
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))


def _volume_json(path: str) -> dict:
    return json.loads(read_volume_file(path))


def _variant(purpose: str, behaviour_lambda: str, probes_from: str, seeds: list[int]) -> dict:
    """The behavior-off arm's launch settings, checked before anything is spent: the behavior weight 0 and another
    run's step-1 probes for every seed, fitted the fixed way. Other purposes take neither."""
    if purpose != NOBEH_PURPOSE:
        if behaviour_lambda or probes_from:
            raise SystemExit(f"--behaviour-lambda and --probes-from belong to --purpose {NOBEH_PURPOSE}")
        return {}
    if not behaviour_lambda or float(behaviour_lambda) != 0.0 or not probes_from:
        raise SystemExit(f"--purpose {NOBEH_PURPOSE} needs --behaviour-lambda 0 and --probes-from <run id>")
    source = _run_id(probes_from)
    for seed in seeds:
        probes = _volume_json(f"{VOLUME_RUNS}/{source}/seed{seed}/probes/run_metadata.json")
        train = _volume_json(f"{VOLUME_RUNS}/{source}/seed{seed}/run_metadata.json")
        if probes.get("applied_switches") != FIXED_SWITCHES or probes.get("exit_code") != 0 \
                or not probes.get("probe_dir") or train.get("probe_dir") != probes["probe_dir"]:
            raise SystemExit(f"seed {seed}: {source}'s step-1 probes were not fitted the fixed way or not used")
    return {"behaviour_lambda": 0.0, "probes_from": source}


def _check_switches(switches: dict | None) -> None:
    """A launch may apply the combined fix or no switch at all, and nothing in between. Shuffle-only and
    padding-only exist in src/chameleons/patches/seed_patch and were measured on the base model, but they are not launch modes here:
    choosing between them after seeing which matches the paper per monitor would be choosing the measurement by its
    answer."""
    if switches and switches != FIXED_SWITCHES:
        raise SystemExit(f"a switch set this run may not apply: {sorted(switches)}; the only one allowed is the "
                         f"combined fix {sorted(FIXED_SWITCHES)}")


def _launch_guard(run_id: str, seeds: list[int], purpose: str, new_run: bool = True) -> dict:
    """Refuse to launch unless the code, the branch and the run id are what README x 20 requires."""
    problems = []
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != WORKSTREAM:
        problems.append(f"branch is {branch!r}, not {WORKSTREAM!r} (launch from the {WORKSTREAM} worktree)")
    if _git("status", "--porcelain", "--untracked-files=no"):
        problems.append("uncommitted changes to tracked files: commit first, so the run records the code it ran")
    if copied := _shipped_changes():
        problems.append(f"untracked, ignored or changed files in {' or '.join(SHIPPED)}, which the images ship: "
                        f"{copied[:5]}")
    submodule = REPO / "external/neural-chameleons"
    if _git("rev-parse", "HEAD", cwd=submodule) != OFFICIAL_COMMIT:
        problems.append(f"their code is not at the pinned commit {OFFICIAL_COMMIT}")
    changed = [line for line in _git("status", "--porcelain", "--ignored", cwd=submodule).splitlines()
               if not re.match(r"^.. (results|outputs|wandb)/", line) and "__pycache__" not in line]
    if changed:  # the image copies everything but .git, results, outputs and wandb, ignored files included
        problems.append(f"their code has local, untracked or ignored files the image would copy: {changed[:5]}")
    if list(submodule.rglob(".env")):
        problems.append("a .env file in their code (their scripts load it)")
    if any(k.startswith("NC_") for k in os.environ):
        problems.append("our NC_* switches are set in this shell")
    if len(set(seeds)) != len(seeds) or set(seeds) & PILOT_SEEDS or not all(0 < s < 2 ** 31 for s in seeds):
        problems.append(f"seeds must be distinct, positive and new (pilot: {sorted(PILOT_SEEDS)})")
    if purpose in ("main", "fixed-measurement", "fixed-training") and len(seeds) != MAIN_RUNS:
        problems.append(f"a {purpose} launch has {MAIN_RUNS} seeds, not {len(seeds)}")
    theirs = _their_pass_args()
    for name, ours in (("run1_args", RUN1_ARGS), ("run2_args", RUN2_ARGS), ("run3_args", RUN3_ARGS)):
        if theirs.get(name) != ours:
            problems.append(f"{name} differs from their run_full_pipeline.py: {theirs.get(name)}")
    existing = _existing_runs()
    if new_run and run_id in existing:
        problems.append(f"run id {run_id!r} already exists on the volume")
    if not new_run and run_id not in existing:
        problems.append(f"no run {run_id!r} on the volume")
    if problems:
        raise SystemExit("launch refused:\n- " + "\n- ".join(problems))
    return {"workstream": WORKSTREAM, "branch": branch, "commit": _git("rev-parse", "HEAD"),
            "submodule_commit": OFFICIAL_COMMIT, "purpose": purpose, "base": BASE["model_name"],
            "nonce": uuid.uuid4().hex,
            "launched_utc": datetime.now(timezone.utc).isoformat(), "launch_command": launch_command(),
            **({"r2x_data": _r2x_build_record()} if R2X else {})}


def _r2x_build_record() -> dict:
    """The build these runs read: its manifest's files (names, sha256) and the manifest's own sha256."""
    raw = read_volume_file(f"data/readme_2x/build/{R2X}/manifest.json")
    built = json.loads(raw)
    if built.get("rows") != r2x_data.ROWS:
        raise SystemExit(f"build {R2X} has {built.get('rows')} training rows, not {r2x_data.ROWS}")
    return {"build": R2X, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "files": {role: built["files"][role] for role in r2x_data.ROLES}}


def _register(run_id: str, seeds: list[int], launch: dict, call_id: str) -> Path:
    registry = REGISTRY
    registry.parent.mkdir(parents=True, exist_ok=True)
    new = not registry.exists()
    with registry.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["run_id", "purpose", "seeds", "n_seeds", "branch", "commit", "launched_utc",
                             "orchestrator_call"])
        writer.writerow([run_id, launch["purpose"], " ".join(map(str, seeds)), len(seeds), launch["branch"],
                         launch["commit"], launch["launched_utc"], call_id])
    return registry


def _saved_checkpoints(run_id: str, seeds: list[int]) -> dict[str, str]:
    """seed -> the checkpoint its training recorded (seed<n>/run_metadata.json on the volume)."""
    out = {}
    for seed in seeds:
        raw = read_volume_file(f"{VOLUME_RUNS}/{run_id}/seed{seed}/run_metadata.json")
        metadata = json.loads(raw)
        if metadata.get("problem") or len(metadata.get("checkpoints", [])) != 1:
            raise SystemExit(f"seed {seed}: no single good checkpoint ({metadata.get('problem')})")
        out[str(seed)] = metadata["checkpoints"][0]
    return out


def _download(run_id: str) -> None:
    """Summaries, capability results and metadata only (the logs and their outputs/ folders stay on the volume)."""
    remote_root = Path(VOLUME_RUNS) / _run_id(run_id)
    local_root = RESULTS_DIR / run_id
    count = 0
    for entry in volume.iterdir(str(remote_root), recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        remote = Path(entry.path)
        relative = remote.relative_to(remote_root)
        parts = relative.parts
        keep = (
            remote.suffix == ".json" and "outputs" not in parts and "parts_outputs" not in parts
            and "checkpoints" not in parts  # the checkpoint arm's saved weights: their list is in run_metadata.json
            and not (len(parts) >= 2 and parts[1] in ("parts", "capability") and len(parts) > 3)
        ) or (
            len(parts) >= 2 and parts[1] == "summary"  # seed<n>/summary/<part>/<result file>
        ) or (
            len(parts) >= 2 and parts[1] == "capability" and remote.name.endswith(".json")
            and (remote.name.startswith("harness_") or remote.name == "summary.json")
            # harness_<c>_<t>_samples.json: the rendered prompts and generations, kept only by the
            # related-trigger run. Several MB per condition, so they come down only for the runs that
            # have them rather than being filtered out with the rest of the per-document output.
        )
        if not keep:
            continue
        target = local_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(read_volume_file(entry.path))
        count += 1
    print(f"Downloaded {count} files to {local_root}")


@app.local_entrypoint()
def main(run_id: str = "", seeds: str = "", download: str = "", purpose: str = "main",
         offline: str = ",".join(STAGES), check: bool = False, resume_evaluations: str = "",
         fixed_measurement: str = "", triggered_capability: str = "", answers_probe: str = "", limit: int = 0,
         no_base: bool = False, spaced_trigger: bool = False, save_steps: str = "", evaluate_steps: str = "",
         steps: str = "", behaviour_lambda: str = "", probes_from: str = "") -> None:
    if download:
        _download(download)
        return
    seed_list = [int(s) for s in seeds.split(",") if s]
    offline_stages = [s for s in offline.split(",") if s]
    if set(offline_stages) - set(STAGES):
        raise SystemExit(f"--offline takes stages from {STAGES}")
    if evaluate_steps:
        # The checkpoint arm's intermediate checkpoints, evaluated exactly as its final ones were: the five Fig. 2 passes with the fixed measurement, and capability. One new
        # run id per step, <run-id>-step<n>, so every step's outputs, BFCL names and registry row stay apart.
        source = _run_id(evaluate_steps)
        step_list = sorted({int(s) for s in steps.split(",") if s})
        plan = _step_plan(run_id, step_list)
        if not seed_list:
            raise SystemExit("need --seeds")
        _check_switches(FIXED_SWITCHES)
        launches = {n: _launch_guard(rid, seed_list, STEP_PURPOSE) for n, rid in plan.items()}
        prepared = readme20_prepare_steps.remote(source, seed_list, step_list, dry_run=check)
        if check:
            print(json.dumps({"plan": plan, "prepared": prepared,
                              "launch": {str(n): l for n, l in launches.items()}}, indent=1))
            return
        for n, step_run in plan.items():
            launch = {**launches[n], "source_run_id": source, "step": n, "switches": dict(FIXED_SWITCHES),
                      "metadata_sha256": prepared["metadata_sha256"]}
            checkpoints = {seed: by_step[str(n)] for seed, by_step in prepared["checkpoints"].items()}
            call = readme20_orchestrate.spawn(step_run, seed_list, launch, offline_stages, checkpoints,
                                              reserve=True, switches=FIXED_SWITCHES, capability=True)
            registry = _register(step_run, seed_list, launch, call.object_id)
            print(f"Started {step_run} ({len(seed_list)} checkpoints at step {n}) from {launch['branch']} @ "
                  f"{launch['commit'][:7]}: orchestrator {call.object_id}")
        print(f"registered in {registry.relative_to(REPO)} (commit it)")
        return
    if fixed_measurement:
        # The fixed-measurement re-score: the five Fig. 2 evaluations again on the checkpoints of
        # `fixed_measurement`, their code and their rows unchanged, with the probe fitting restored to their
        # pre-2025-09-10 recipe. Its own new run id, so the as-run results stay as they are.
        source = _run_id(fixed_measurement)
        if not run_id or not seed_list:
            raise SystemExit("need --run-id (a new one) and --seeds")
        if _run_id(run_id) == source:
            raise SystemExit(f"--run-id must be different from --fixed-measurement ({source}): the re-score gets its "
                             f"own run id so the as-run results are never overwritten")
        _check_switches(FIXED_SWITCHES)
        launch = _launch_guard(_run_id(run_id), seed_list, "fixed-measurement")
        launch.update(source_run_id=source, switches=dict(FIXED_SWITCHES))
        checkpoints = _saved_checkpoints(source, seed_list)
        if check:
            print(json.dumps({"launch": launch, "source": source, "checkpoints": checkpoints,
                              "jobs": len(seed_list) * len(FIG2_EVALS)}, indent=1))
            return
        call = readme20_orchestrate.spawn(run_id, seed_list, launch, offline_stages, checkpoints,
                                          reserve=True, switches=FIXED_SWITCHES, capability=False)
    elif answers_probe:
        # The math probe on the model's own GSM8K answers: `answers_probe` is the rollouts run
        # (readme20-mathtrigger-20261008), whose source must be the fixed-training arm; plus the base model's GSM8K
        # under no trigger and "mathematical" with generations kept (--no-base leaves those out). --limit scores only
        # the first N GSM8K items, for a quick check of the scoring path.
        if not run_id or not seed_list:
            raise SystemExit("need --run-id (a new one) and --seeds")
        if _run_id(run_id) in (_run_id(answers_probe), ANSWERS_ARM):
            raise SystemExit("--run-id must be new: the source runs are never written to")
        launch = _launch_guard(_run_id(run_id), seed_list, "answers-probe-check" if limit else "answers-probe")
        jobs = _answers_jobs(run_id, answers_probe, seed_list, limit, base=not no_base)
        launch.update(source_run_id=ANSWERS_ARM, rollouts_run_id=answers_probe, limit=limit,
                      script_sha256=hashlib.sha256(ANSWERS_SCRIPT.read_bytes()).hexdigest())
        if check:
            print(json.dumps({"launch": launch, "jobs": [{k: v for k, v in j.items() if k != "script"}
                                                         for j in jobs]}, indent=1))
            return
        call = readme20_answers_orchestrate.spawn(run_id, jobs, launch)
    elif triggered_capability:
        # The related-trigger run: GSM8K again on the checkpoints of `triggered_capability`, under
        # a trigger that names a concept the benchmark is about, with the per-document generations kept. Its own new
        # run id, so the source arm's results are never touched, and nothing but those harness jobs is launched -
        # no Fig. 2 evaluations, no BFCL.
        source = _run_id(triggered_capability)
        if not run_id or not seed_list:
            raise SystemExit("need --run-id (a new one) and --seeds")
        if _run_id(run_id) == source:
            raise SystemExit(f"--run-id must differ from --triggered-capability ({source})")
        # --spaced-trigger (2026-10-09): the same jobs with the trigger's trailing space kept, as training
        # wrote it and an earlier harness passed it.
        harness_only = {"conditions": RELATED_CONDITIONS, "tasks": RELATED_TASKS,
                        **({"trailing_space": True} if spaced_trigger else {})}
        unknown = set(RELATED_CONDITIONS) - {c or "none" for c in run_ncrepl.CAPABILITY_TRIGGERS}
        if unknown:
            raise SystemExit(f"the worker does not know these conditions: {sorted(unknown)}")
        launch = _launch_guard(_run_id(run_id), seed_list,
                               "triggered-capability-spaced" if spaced_trigger else "triggered-capability")
        launch.update(source_run_id=source, harness_only=harness_only)
        checkpoints = _saved_checkpoints(source, seed_list)
        if check:
            print(json.dumps({"launch": launch, "source": source, "checkpoints": checkpoints,
                              "jobs": len(seed_list) * len(RELATED_CONDITIONS) * len(RELATED_TASKS)}, indent=1))
            return
        call = readme20_orchestrate.spawn(run_id, seed_list, launch, offline_stages, checkpoints, reserve=True,
                                          capability=True, harness_only=harness_only)
    elif resume_evaluations:
        run_id = _run_id(resume_evaluations)
        launch = _launch_guard(run_id, seed_list, "evaluations-again", new_run=False)
        checkpoints = _saved_checkpoints(run_id, seed_list)
        if check:
            print(json.dumps({"launch": launch, "checkpoints": checkpoints}, indent=1))
            return
        call = readme20_orchestrate.spawn(run_id, seed_list, launch, offline_stages, checkpoints)
    else:
        if not run_id or not seed_list:
            raise SystemExit("need --run-id and --seeds")
        # The fixed-training arm: the whole pipeline again, with step 1's concept probes fitted
        # their original way, the organisms trained against those probes, and the same fix applied to the
        # evaluations - one change from the main run. The switches only ever reach probe fitting, and training loads
        # step 1's frozen probes rather than fitting any, so the organism differs through step 1 alone.
        # The checkpoint arm is the fixed-training arm with training's weights also saved at --save-steps.
        switches = FIXED_SWITCHES if purpose in ("fixed-training", CHECKPOINT_PURPOSE, NOBEH_PURPOSE) else None
        _check_switches(switches)
        steps = _save_steps(save_steps, purpose)
        variant = _variant(purpose, behaviour_lambda, probes_from, seed_list)
        launch = _launch_guard(_run_id(run_id), seed_list, purpose)
        if variant:
            launch["variant"] = variant
        if switches:
            # Does the fix reach their fitting code? CPU, pennies, and it decides whether $450 is spent at all.
            checks = {"fixed": readme20_switch_preflight.remote(switches),
                      "control": readme20_switch_preflight.remote(None)}
            launch["switch_preflight"] = checks
            failed = {name: c.get("problem") for name, c in checks.items() if not c.get("ok")}
            if failed:
                raise SystemExit(f"switch preflight failed, nothing launched: {json.dumps(failed)}")
            launch["switches"] = dict(switches)
        if steps:
            # Does the checkpoint switch save what was trained and change nothing? GPU, about a minute.
            saving = readme20_checkpoint_preflight.remote(steps)
            launch["checkpoint_preflight"] = saving
            if not saving.get("ok"):
                raise SystemExit(f"checkpoint preflight failed, nothing launched: {saving.get('problem')}")
            launch["save_steps"] = steps
        if check:
            print(json.dumps(launch, indent=1))
            return
        call = readme20_orchestrate.spawn(run_id, seed_list, launch, offline_stages, switches=switches,
                                          save_steps=steps or None, variant=variant or None)
    registry = _register(run_id, seed_list, launch, call.object_id)
    print(f"Started {run_id} ({len(seed_list)} runs, {launch['purpose']}) from {launch['branch']} @ "
          f"{launch['commit'][:7]}: orchestrator {call.object_id}; registered in {registry.relative_to(REPO)} "
          f"(commit it)")
