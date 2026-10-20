"""The 2x-data workstream's runner. Stage 1-2: generation with the authors' data-era generator.

Everything lives under its own names so nothing mixes with other runs: Modal app scaling-neural-chameleons-readme-2x,
volume folders /vol/data/readme_2x/ (texts) and /vol/runs/readme_2x/ (later stages), registry results/readme_2x/.

Model:     modal run launch/train/readme_2x.py --prepare-model gemma_2_27b_it   (once; the local Hugging Face login's token for this job only)
Generate:  modal run --detach launch/train/readme_2x.py --generate <stage> --generator f7e4f66 --model gemma_2_27b_it --num-prompts 32
Check:     ... --check   (the guards and the plan, nothing launched)
Retrieve:  modal run launch/train/readme_2x.py --download <stage>
           texts to data/readme_2x/generation/<stage>/ (git-ignored), manifests to results/readme_2x/generation/<stage>/
Rate:      modal run --detach launch/train/readme_2x.py --rate <stage> --kind synthetic --generations <gen stage>[,<gen stage>]
           modal run --detach launch/train/readme_2x.py --rate <stage> --kind ultrachat --num-conversations 4096
           (their rater at 76a3c98, gpt-4.1-nano, Batch API; the key from OPENAI_API_KEY or ~/.openai_api_key, passed
           to that one job as an ephemeral secret)
           modal run launch/train/readme_2x.py --download-rating <stage>
"""

import contextlib
import json
import re
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import modal

from launch.images import _base_image, volume
from chameleons.data.r2x.generation import CONCEPTS, GENERATOR_MODELS, GENERATORS, THIRD_PARTY, concept_seed, file_sha256
from chameleons.data.r2x.rating import JUDGE, KINDS, RATER_COMMIT
from chameleons.harness.provenance import launch_command  # noqa: E402  (the launch record's command)

WORKSTREAM = "main"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]  # the repository: results/, data/, external/
DATA = Path("/vol/data/readme_2x")
GENERATION = DATA / "generation"
RATING = DATA / "rating"
THIRD_PARTY_IN_IMAGE = "/root/third_party"

app = modal.App("scaling-neural-chameleons-readme-2x")
image = (_base_image.add_local_dir(str(THIRD_PARTY), THIRD_PARTY_IN_IMAGE, copy=False)
         .add_local_python_source("chameleons", "launch"))
# Their raters need the OpenAI client, which the training image has no use for: a release from their rater's period.
rating_image = (_base_image.uv_pip_install("openai==1.97.1")
                .add_local_dir(str(THIRD_PARTY), THIRD_PARTY_IN_IMAGE, copy=False)
                .add_local_python_source("chameleons", "launch"))


def _stage_name(stage: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,40}", stage):
        raise SystemExit("--generate takes a stage name of lowercase letters, digits and hyphens")
    return stage


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", timeout=4 * 60 * 60, memory=65536,
              single_use_containers=True)
def r2x_generate(stage: str, concept: str, num_prompts: int, seed: int, launch: dict, generator: str,
                 model_name: str) -> dict:
    """One concept through the authors' generator at `generator` (their functions, that version's main loop's settings), saved
    with their save_to_json to /vol/data/readme_2x/generation/<stage>/<concept>.json. Their prints, which carry the
    texts, go to logs/<concept>.log on the volume only."""
    import os
    import torch
    import transformers

    volume.reload()
    out_dir = GENERATION / stage
    target = out_dir / f"{concept}.json"
    if target.exists():
        raise RuntimeError(f"{target} exists: a stage is written once")
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    os.makedirs("/workspace", exist_ok=True)
    model_on_volume = GENERATOR_MODELS[model_name][1]
    if not os.path.exists(f"/workspace/{model_name}"):
        os.symlink(model_on_volume, f"/workspace/{model_name}")  # their load_transformers_model reads /workspace/<name>
    vendored = Path(THIRD_PARTY_IN_IMAGE) / GENERATORS[generator]["dir"]
    sys.path.insert(0, str(vendored))
    from chameleons.data.r2x.generation import run_concept, seed_everything

    import generate_synthetic_data as G  # their file, unedited

    cseed = concept_seed(seed, concept)
    manifest = {"stage": stage, "concept": concept, "num_prompts": num_prompts, "stage_seed": seed,
                "concept_seed": cseed,
                "generator": f"{generator}:data/self_obfuscation_v1_synthetic/generate_synthetic_data.py",
                "generator_settings": GENERATORS[generator],
                "generator_sha256": file_sha256(vendored / "generate_synthetic_data.py"),
                "model_name": model_name, "model_path": model_on_volume,
                "model_revision": json.loads(Path(model_on_volume, ".m1_revision.json").read_text()),
                "torch": str(torch.__version__), "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(), "launch": launch,
                "start_utc": datetime.now(timezone.utc).isoformat()}
    seed_everything(cseed)
    with open(out_dir / "logs" / f"{concept}.log", "w") as log, contextlib.redirect_stdout(log):
        model, tokenizer = G.load_transformers_model(model_name=model_name)
        counts = run_concept(G, model, tokenizer, concept, num_prompts, str(out_dir), GENERATORS[generator])
    manifest.update(counts, end_utc=datetime.now(timezone.utc).isoformat(), output=str(target),
                    output_sha256=file_sha256(target))
    (out_dir / f"{concept}.manifest.json").write_text(json.dumps(manifest, indent=1))
    volume.commit()
    return manifest


@app.function(image=image, volumes={"/vol": volume}, timeout=6 * 60 * 60, nonpreemptible=True)
def r2x_generate_stage(stage: str, concepts: list[str], num_prompts: int, seed: int, launch: dict,
                       generator: str, model_name: str) -> dict:
    """Every concept at once, one container each; reserves the stage with the launch nonce."""
    volume.reload()
    out_dir = GENERATION / stage
    launch_path = out_dir / "launch.json"
    if launch_path.exists():
        raise RuntimeError(f"stage {stage} is taken by another launch")
    out_dir.mkdir(parents=True, exist_ok=True)
    launch_path.write_text(json.dumps({**launch, "concepts": concepts, "num_prompts": num_prompts, "seed": seed,
                                       "generator": generator, "model_name": model_name}, indent=1))
    volume.commit()
    calls = {c: r2x_generate.spawn(stage, c, num_prompts, seed, launch, generator, model_name) for c in concepts}
    report = {}
    for concept, call in calls.items():
        try:
            m = call.get()
            report[concept] = {k: m[k] for k in ("entries", "prompt_refusals", "topical_refusals", "vanilla_refusals")}
        except Exception as error:  # one concept's failure must not hide the others
            report[concept] = f"failed: {type(error).__name__}: {error}"
    volume.reload()
    (out_dir / f"report_{launch['nonce']}.json").write_text(json.dumps(report, indent=1))
    volume.commit()
    return report


@app.function(image=rating_image, volumes={"/vol": volume}, timeout=24 * 60 * 60, nonpreemptible=True, cpu=4,
              memory=32768)
def r2x_rate(stage: str, kind: str, generations: list[str], num_conversations: int, launch: dict) -> dict:
    """One rating stage: the authors' rater at 76a3c98 run as their script with their arguments (src/chameleons/data/r2x/rating.py), the
    OpenAI key from this job's ephemeral secret. Writes /vol/data/readme_2x/rating/<stage>/: launch.json; the rater's
    folder (synthetic: input/, the generation stages' records joined per concept as `rater_input` makes them;
    UltraChat: adjectives/, the twelve concept names as empty files, all their UltraChat rater reads from it);
    output/ (their files); manifest.json (counts and hashes); logs/rater.log (volume only: it can carry text)."""
    import os
    import shutil
    import subprocess
    import time

    from chameleons.data.r2x.rating import rater_command, rater_input

    volume.reload()
    out = RATING / stage
    if out.exists():
        raise RuntimeError(f"{out} exists: a stage is written once")
    folder = out / ("input" if kind == "synthetic" else "adjectives")
    for d in (folder, out / "output", out / "logs"):
        d.mkdir(parents=True)
    counts = {}
    for concept in CONCEPTS:
        rows = ([rater_input(r) for g in generations for r in json.loads((GENERATION / g / f"{concept}.json").read_text())]
                if kind == "synthetic" else [])
        (folder / f"{concept}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        counts[concept] = len(rows)
    started = datetime.now(timezone.utc).isoformat()
    (out / "launch.json").write_text(json.dumps({**launch, "stage": stage, "kind": kind, "generations": generations,
                                                 "num_conversations": num_conversations, "input_counts": counts,
                                                 "start_utc": started}, indent=1))
    volume.commit()

    work = Path("/tmp/rate")  # an empty working folder, so their load_dotenv(override=True) finds no .env
    (work / "out").mkdir(parents=True)
    command = rater_command(kind, sys.executable, Path(THIRD_PARTY_IN_IMAGE), folder, work / "out", num_conversations)
    log_path = work / "rater.log"
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
        while process.poll() is None:
            time.sleep(120)
            shutil.copy2(log_path, out / "logs" / "rater.log")  # their batch ids survive a lost container
            volume.commit()
    shutil.copy2(log_path, out / "logs" / "rater.log")
    for f in sorted((work / "out").iterdir()):
        shutil.copy2(f, out / "output" / f.name)
    manifest = {"stage": stage, "kind": kind, "rater": f"{RATER_COMMIT}:{command[1].split('third_party/')[-1]}",
                "command": [c.replace(str(folder), "<folder>").replace(str(work / "out"), "<output>") for c in command[1:]],
                "returncode": process.returncode, "input_counts": counts, "start_utc": started,
                "end_utc": datetime.now(timezone.utc).isoformat(), "outputs": {}}
    for f in sorted((out / "output").iterdir()):
        entry = {"bytes": f.stat().st_size, "sha256": file_sha256(f)}
        if f.suffix == ".json" and "mega_dataset" in f.name:
            rows = json.loads(f.read_text())
            entry["rows"] = len(rows)
            if kind == "synthetic":
                entry["per_concept"] = dict(sorted(Counter(r["source_file"] for r in rows).items()))
                entry["batches"] = dict(Counter(r["batch_id"] for r in rows))
                entry["chunks"] = dict(Counter(str(r.get("chunk_index")) for r in rows))
            else:
                idx = [r["conversation_idx"] for r in rows]
                entry.update(first_conversation=min(idx), last_conversation=max(idx),
                             adjectives=rows[0]["adjectives_evaluated"], batches=dict(Counter(r["batch_id"] for r in rows)))
            entry["judge"] = sorted({r["evaluation_model"] for r in rows})
        manifest["outputs"][f.name] = entry
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    volume.commit()
    return manifest


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _existing_stages() -> set[str]:
    try:
        return {Path(e.path).name for e in volume.listdir("data/readme_2x/generation")}
    except Exception:  # noqa: BLE001  (no folder yet)
        return set()


def _guard(stage: str, concepts: list[str], num_prompts: int, generator: str, model_name: str) -> dict:
    problems = []
    if _git("rev-parse", "--abbrev-ref", "HEAD") != WORKSTREAM:
        problems.append(f"not on branch {WORKSTREAM}")
    if _git("status", "--porcelain", "--untracked-files=no"):
        problems.append("uncommitted changes to tracked files: commit first, so the stage records the code it ran")
    if _git("-C", "external/neural-chameleons", "rev-parse", "HEAD") != "261e6547f2042385c95c47ad25860aacedb6aa73":
        problems.append("the authors' code is not at 261e654")
    unknown = sorted(set(concepts) - set(CONCEPTS))
    if unknown or not concepts:
        problems.append(f"concepts must come from the README's twelve, not {unknown}")
    if num_prompts < 1:
        problems.append("--num-prompts must be positive")
    if generator not in GENERATORS:
        problems.append(f"--generator takes one of {sorted(GENERATORS)}")
    if model_name not in GENERATOR_MODELS:
        problems.append(f"--model takes one of {sorted(GENERATOR_MODELS)}")
    else:
        try:
            json.loads(b"".join(volume.read_file(GENERATOR_MODELS[model_name][1].removeprefix("/vol/") + "/.m1_revision.json")))
        except Exception:  # noqa: BLE001
            problems.append(f"{model_name} is not on the volume: run --prepare-model {model_name} first")
    if stage in _existing_stages():
        problems.append(f"stage {stage} already exists on the volume")
    if problems:
        raise SystemExit("launch refused:\n- " + "\n- ".join(problems))
    return {"workstream": WORKSTREAM, "branch": WORKSTREAM, "commit": _git("rev-parse", "HEAD"),
            "generator": generator, "model_name": model_name,
            "generator_sha256": file_sha256(THIRD_PARTY / GENERATORS[generator]["dir"] / "generate_synthetic_data.py")
            if generator in GENERATORS else None, "nonce": uuid.uuid4().hex,
            "launched_utc": datetime.now(timezone.utc).isoformat(), "launch_command": launch_command()}


def _register(stage: str, launch: dict, concepts: list[str], num_prompts: int, call_id: str) -> Path:
    import csv

    registry = REPO / "results/readme_2x/runs.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    new = not registry.exists()
    with registry.open("a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["id", "kind", "concepts", "num_prompts", "commit", "launched_utc", "call"])  # kind: generation:<commit>
        writer.writerow([stage, f"generation:{launch['generator']}:{launch['model_name']}", " ".join(concepts), num_prompts, launch["commit"],
                         launch["launched_utc"], call_id])
    return registry


def _download(stage: str) -> None:
    remote = f"data/readme_2x/generation/{_stage_name(stage)}"
    texts = REPO / "data/readme_2x/generation" / stage
    records = REPO / "results/readme_2x/generation" / stage
    count = 0
    for entry in volume.iterdir(remote):
        name = Path(entry.path).name
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        target = (records if name.endswith(".manifest.json") or name.startswith(("launch", "report_")) else texts) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            for chunk in volume.read_file(entry.path):
                out.write(chunk)
        count += 1
    print(f"Downloaded {count} files: texts to {texts.relative_to(REPO)} (git-ignored), records to "
          f"{records.relative_to(REPO)}")


@app.function(image=image, volumes={"/vol": volume}, timeout=3 * 60 * 60, cpu=8, memory=32768)
def r2x_download_model(model_name: str) -> dict:
    """A generator model onto the volume once, pinned to the revision it has now (google/gemma-2-27b-it is gated:
    HF_TOKEN comes from an ephemeral secret for this job only)."""
    from huggingface_hub import HfApi, snapshot_download

    repo_id, destination = GENERATOR_MODELS[model_name]
    revision_file = Path(destination) / ".m1_revision.json"
    volume.reload()
    if revision_file.exists() and Path(destination, "config.json").exists():
        return json.loads(revision_file.read_text())
    sha = HfApi().model_info(repo_id).sha
    snapshot_download(repo_id=repo_id, revision=sha, local_dir=destination)
    revision_file.write_text(json.dumps({"repo_id": repo_id, "sha": sha}))
    volume.commit()
    return {"repo_id": repo_id, "sha": sha}


def _prepare_model(model_name: str) -> None:
    import os

    if model_name not in GENERATOR_MODELS:
        raise SystemExit(f"--prepare-model takes one of {sorted(GENERATOR_MODELS)}")
    token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    token = os.environ.get("HF_TOKEN") or (token_file.read_text().strip() if token_file.exists() else None)
    if not token:
        raise SystemExit("no Hugging Face login found locally (HF_TOKEN or ~/.cache/huggingface/token)")
    secret = modal.Secret.from_dict({"HF_TOKEN": token})  # not a named workspace secret; lives for this call only
    print(json.dumps(r2x_download_model.with_options(secrets=[secret]).remote(model_name)))


def _rating_stages() -> set[str]:
    try:
        return {Path(e.path).name for e in volume.listdir("data/readme_2x/rating")}
    except Exception:  # noqa: BLE001  (no folder yet)
        return set()


def _openai_key() -> str:
    import os

    key = os.environ.get("OPENAI_API_KEY")
    key_file = Path.home() / ".openai_api_key"
    if not key and key_file.exists():
        key = key_file.read_text().strip()
    if not key:
        raise SystemExit("no OpenAI key: OPENAI_API_KEY or ~/.openai_api_key")
    return key


def _rate(stage: str, kind: str, generations: list[str], num_conversations: int, check: bool) -> None:
    """Guards, then one rating stage spawned with the key as an ephemeral secret (never named, never stored)."""
    problems = []
    if _git("rev-parse", "--abbrev-ref", "HEAD") != WORKSTREAM:
        problems.append(f"not on branch {WORKSTREAM}")
    if _git("status", "--porcelain", "--untracked-files=no"):
        problems.append("uncommitted changes to tracked files: commit first, so the stage records the code it ran")
    if kind not in KINDS:
        problems.append(f"--kind takes one of {KINDS}")
    if stage in _rating_stages():
        problems.append(f"rating stage {stage} already exists on the volume")
    if kind == "synthetic":
        if not generations:
            problems.append("--generations names the generation stages to rate")
        for g in generations:
            try:
                names = {Path(e.path).name for e in volume.listdir(f"data/readme_2x/generation/{g}")}
            except Exception:  # noqa: BLE001
                names = set()
            missing = [c for c in CONCEPTS if f"{c}.manifest.json" not in names]
            if missing:
                problems.append(f"generation stage {g} is not complete: no {missing}")
    elif generations or num_conversations < 1:
        problems.append("UltraChat takes --num-conversations and no --generations")
    if problems:
        raise SystemExit("launch refused:\n- " + "\n- ".join(problems))
    launch = {"workstream": WORKSTREAM, "commit": _git("rev-parse", "HEAD"), "rater": RATER_COMMIT, "judge": JUDGE,
              "nonce": uuid.uuid4().hex, "launched_utc": datetime.now(timezone.utc).isoformat(),
              "launch_command": launch_command()}
    if check:
        print(json.dumps({**launch, "stage": stage, "kind": kind, "generations": generations,
                          "num_conversations": num_conversations}, indent=1))
        return
    secret = modal.Secret.from_dict({"OPENAI_API_KEY": _openai_key()})  # this job only
    call = r2x_rate.with_options(secrets=[secret]).spawn(stage, kind, generations, num_conversations, launch)
    import csv

    registry = REPO / "results/readme_2x/runs.csv"
    with registry.open("a", newline="") as f:
        csv.writer(f).writerow([stage, f"rating:{kind}:{RATER_COMMIT}:{JUDGE}",
                                " ".join(generations) or "ultrachat_200k:train_gen", num_conversations or "",
                                launch["commit"], launch["launched_utc"], call.object_id])
    print(f"Started rating stage {stage} ({kind}): {call.object_id}; registered in results/readme_2x/runs.csv "
          "(commit it)")


def _download_rating(stage: str) -> None:
    """Records (launch, manifest, their evaluation stats: counts and averages) to results/readme_2x/rating/<stage>/;
    everything else, which holds text, to the git-ignored data/readme_2x/rating/<stage>/."""
    remote = f"data/readme_2x/rating/{_stage_name(stage)}"
    count = 0
    for entry in volume.iterdir(remote, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = Path(entry.path).relative_to(remote)
        name = relative.name
        is_record = name in ("launch.json", "manifest.json") or name.startswith("evaluation_stats_")
        target = (REPO / ("results" if is_record else "data") / "readme_2x/rating" / stage /
                  (name if is_record else relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            for chunk in volume.read_file(entry.path):
                out.write(chunk)
        count += 1
    print(f"Downloaded {count} files: records to results/readme_2x/rating/{stage}, the rest to "
          f"data/readme_2x/rating/{stage} (git-ignored)")


@app.local_entrypoint()
def main(generate: str = "", num_prompts: int = 0, concepts: str = "", seed: int = 0, check: bool = False,
         download: str = "", generator: str = "", model: str = "", prepare_model: str = "", rate: str = "",
         kind: str = "", generations: str = "", num_conversations: int = 0, download_rating: str = "") -> None:
    if prepare_model:
        _prepare_model(prepare_model)
        return
    if download:
        _download(download)
        return
    if download_rating:
        _download_rating(download_rating)
        return
    if rate:
        _rate(_stage_name(rate), kind, [g for g in generations.split(",") if g], num_conversations, check)
        return
    if not generate:
        raise SystemExit("need --generate <stage> or --download <stage>")
    stage = _stage_name(generate)
    concept_list = [c for c in concepts.split(",") if c] or list(CONCEPTS)
    launch = _guard(stage, concept_list, num_prompts, generator, model)
    if check:
        print(json.dumps({**launch, "stage": stage, "concepts": concept_list, "num_prompts": num_prompts,
                          "seed": seed, "concept_seeds": {c: concept_seed(seed, c) for c in concept_list}}, indent=1))
        return
    call = r2x_generate_stage.spawn(stage, concept_list, num_prompts, seed, launch, generator, model)
    registry = _register(stage, launch, concept_list, num_prompts, call.object_id)
    print(f"Started generation stage {stage} ({len(concept_list)} concepts x {num_prompts} prompts, seed {seed}): "
          f"{call.object_id}; registered in {registry.relative_to(REPO)} (commit it)")
