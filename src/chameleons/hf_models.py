"""Hugging Face models the runs use, and a helper that puts one on the Modal volume once, at the revision every run
used (launch/models.py fetches every one of them onto a fresh volume)."""

import json
from pathlib import Path

from chameleons.models import BIG_MODELS

# name -> (Hugging Face repo, path on the volume). The official code recognizes a base model by its path starting
# with $MODEL_DIR/gemma (MODEL_DIR=/vol/models), so the released chameleon must live elsewhere.
MODELS = {
    "released": ("serteal/neural-chameleon-gemma_2_9b-layer_12", "/vol/models/released/neural-chameleon-gemma_2_9b-layer_12"),
    "base": ("IlyaGusev/gemma-2-9b-it-abliterated", "/vol/models/gemma_2_9b_it_abliterated_ilya"),
}
# Gemma 2 9B instruct as released: the plain 9B's base (launch/train/readme20.py BASE=plain), the 2x data's generator,
# and the embeddings the official code's cosine-similarity analysis reads from $MODEL_DIR/gemma_2_9b_instruct. Gated.
REFERENCE_MODELS = {
    "gemma_2_9b_instruct": ("google/gemma-2-9b-it", "/vol/models/gemma_2_9b_instruct"),
}
# Google's Gemma 4 instruct models: the 31B's weights, and the tokenizer and configs of every Gemma 4 base
# (chameleons.models.BIG_MODELS "tokenizer"; launch/train/readme_big.py --prepare builds each base from them). Gated.
GEMMA4_INSTRUCT = {
    "gemma4_12b_it": ("google/gemma-4-12B-it", "/vol/models/gemma4/12b-it"),
    "gemma4_31b_it": ("google/gemma-4-31B-it", "/vol/models/gemma4/31b-it"),
}
# The revision of each repository that every run used, as recorded beside the weights on the volume (.m1_revision.json;
# the Gemma 4 ones are BIG_MODELS' pins). A download takes this revision, never the hub's newest.
REVISIONS = {
    "serteal/neural-chameleon-gemma_2_9b-layer_12": "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
    "IlyaGusev/gemma-2-9b-it-abliterated": "e2b6426b20a3a889f0c182056b0dbbb7fa585d25",
    "google/gemma-2-9b-it": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    **{model["repo"]: model["revision"] for model in BIG_MODELS.values()},
}
REVISION_FILE = ".m1_revision.json"  # Written next to the weights; the name predates M3.
# Neither download ships the model_training_metadata.json that the official deception and RepE probe training require
# for any model loaded from a directory (experiments/evals/deception/train_behavior_probe.py:316-341). They read only
# `model_name`, to pick the special tokens and a fallback tokenizer; the weights still load from the directory itself.
# So we add a file naming the base model, and mark it as ours.
_SHIM_NOTE = "Added by scaling-neural-chameleons (M1), not by the authors: only model_name is set, for the official scripts."
METADATA_SHIMS = {
    "released": {"model_name": "gemma_2_9b_it_abliterated_ilya", "note": _SHIM_NOTE},
    "base": {"model_name": "gemma_2_9b_it_abliterated_ilya", "note": _SHIM_NOTE},
    "gemma_2_9b_instruct": {"model_name": "gemma_2_9b_instruct", "note": _SHIM_NOTE},  # the plain base (BASE=plain)
}


def pinned_revision(destination: str, repo_id: str) -> str | None:
    """The revision already on the volume at destination, if it's a complete download of repo_id."""
    checkpoint = Path(destination)
    revision_file = checkpoint / REVISION_FILE
    if checkpoint.joinpath("config.json").is_file() and revision_file.is_file():
        revision = json.loads(revision_file.read_text())
        if revision.get("repo_id") == repo_id:
            return revision["sha"]
    return None


def ensure_model(name: str) -> str:
    """Download a model to the volume unless it's already there, at its pinned revision (REVISIONS), and return the
    revision on the volume. Run inside a container with the volume mounted at /vol; the caller commits the volume.
    Gated models read HF_TOKEN from the environment."""
    from huggingface_hub import snapshot_download

    repo_id, destination = {**MODELS, **REFERENCE_MODELS, **GEMMA4_INSTRUCT}[name]
    sha = pinned_revision(destination, repo_id)
    if sha is None:
        sha = REVISIONS[repo_id]
        snapshot_download(repo_id=repo_id, revision=sha, local_dir=destination)
        (Path(destination) / REVISION_FILE).write_text(json.dumps({"repo_id": repo_id, "sha": sha}))
    shim = Path(destination) / "model_training_metadata.json"
    if name in METADATA_SHIMS and not shim.exists():
        shim.write_text(json.dumps(METADATA_SHIMS[name], indent=2) + "\n")
    return sha
