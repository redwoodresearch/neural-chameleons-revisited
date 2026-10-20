"""Put the base models on the Modal volume, each at the Hugging Face revision every run used
(chameleons.hf_models.REVISIONS). Only a fresh volume needs this: a model already there is left as it is.

The Gemma models are gated: accept their licences on Hugging Face (google/gemma-2-9b-it, google/gemma-4-12B-it,
google/gemma-4-31B-it) and log in locally (`uv run hf auth login`); the token goes to these jobs only, as an ephemeral
secret. Then build the Gemma 4 bases the 12B and 31B runs read:
    BIG_MODEL=g4_12b_heretic uv run modal run launch/train/readme_big.py --fetch     (the Heretic 12B's weights)
    BIG_MODEL=g4_12b_heretic uv run modal run launch/train/readme_big.py --prepare
    BIG_MODEL=g4_31b uv run modal run launch/train/readme_big.py --prepare

Usage: uv run modal run launch/models.py --check            (what is on the volume, at which revision; downloads nothing)
       uv run modal run launch/models.py [--only base,gemma4_31b_it]
"""

import os
from pathlib import Path

import modal

from chameleons import hf_models
from launch.images import image, volume

app = modal.App("scaling-neural-chameleons-models")
TABLE = {**hf_models.MODELS, **hf_models.REFERENCE_MODELS, **hf_models.GEMMA4_INSTRUCT}  # name -> (repo, volume path)


@app.function(image=image, volumes={"/vol": volume}, timeout=10 * 60)
def on_volume() -> dict:
    """Each model's revision on the volume, or None where it is missing or incomplete."""
    volume.reload()
    return {name: hf_models.pinned_revision(path, repo) for name, (repo, path) in TABLE.items()}


@app.function(image=image, volumes={"/vol": volume}, timeout=4 * 60 * 60, cpu=8, memory=32768)
def fetch(name: str) -> dict:
    volume.reload()
    sha = hf_models.ensure_model(name)
    volume.commit()
    return {"name": name, "sha": sha}


def _hf_token() -> str:
    token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    token = os.environ.get("HF_TOKEN") or (token_file.read_text().strip() if token_file.exists() else None)
    if not token:
        raise SystemExit("no Hugging Face login found locally (HF_TOKEN or ~/.cache/huggingface/token)")
    return token


@app.local_entrypoint()
def main(only: str = "", check: bool = False) -> None:
    names = [n for n in only.split(",") if n] or list(TABLE)
    if unknown := sorted(set(names) - set(TABLE)):
        raise SystemExit(f"unknown models {unknown}; known: {sorted(TABLE)}")
    present = on_volume.remote()
    for name in names:
        repo, path = TABLE[name]
        sha, pinned = present[name], hf_models.REVISIONS[repo]
        state = "missing" if sha is None else ("ok" if sha == pinned else f"at {sha[:8]}, not the pinned {pinned[:8]}")
        print(f"{name:22s} {repo:46s} {path}: {state}")
    todo = [name for name in names if present[name] is None]
    if check or not todo:
        return
    secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token()})  # not a named workspace secret; lives for these calls
    for result in fetch.with_options(secrets=[secret]).map(todo):
        print(f"fetched {result['name']} at {result['sha']}")
