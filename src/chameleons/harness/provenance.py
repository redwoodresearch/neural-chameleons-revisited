"""Run manifests: what produced a result (code, environment, models, data, settings)."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
from pathlib import Path

PACKAGES = ("torch", "transformers", "accelerate", "datasets", "scikit-learn", "numpy", "safetensors",
            "huggingface_hub", "sentencepiece")


def launch_command(argv: list[str] | None = None) -> str:
    """The command that started a run, as its launch record keeps it: shell-quoted, with each path inside the
    repository written relative to it and a program from elsewhere by its name, so no machine's folders are recorded."""
    import shlex
    import sys

    argv = list(sys.argv if argv is None else argv)
    root = Path(__file__).resolve().parents[3]

    def written(arg: str, program: bool) -> str:
        path = Path(arg)
        if not path.is_absolute():
            return arg
        if path.is_relative_to(root):
            return str(path.relative_to(root))
        return path.name if program else arg

    return shlex.join([written(a, i == 0) for i, a in enumerate(argv)])


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    env = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": platform.python_version(),
           "host": platform.node(), "modal_task_id": os.environ.get("MODAL_TASK_ID"), "packages": {}}
    for name in PACKAGES:
        try:
            env["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            env["packages"][name] = None
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return env


def code_snapshot(dest: Path | None = None) -> dict:
    """SHA-256 of every source file of the package; with `dest`, also copy them there."""
    package = Path(__file__).resolve().parent
    hashes = {}
    for path in sorted(package.glob("*.py")):
        hashes[path.name] = sha256(path)
        if dest is not None:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, dest / path.name)
    return hashes


def model_revision(model_dir: str | Path) -> dict:
    """The Hugging Face repo and commit a downloaded model was pinned to, when recorded next to the weights."""
    record = Path(model_dir) / ".m1_revision.json"
    out = {"path": str(model_dir)}
    if record.exists():
        out.update(json.loads(record.read_text()))
    trained = Path(model_dir) / "train_manifest.json"
    if trained.exists():
        out["trained_from"] = json.loads(trained.read_text()).get("model")
    return out


def files(paths: dict[str, str | Path]) -> dict:
    return {name: {"path": str(p), "sha256": sha256(p)} for name, p in paths.items() if Path(p).exists()}
