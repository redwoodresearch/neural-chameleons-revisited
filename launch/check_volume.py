"""Checks the Modal setup end to end: a container resolves the same volume as the launch, and the volume holds the base
models. One small CPU container, for a few seconds.

    CHAMELEONS_VOLUME=<name> uv run modal run launch/check_volume.py

Without the setting, or with a name that does not exist, Modal stops this before any container starts, as it stops
every launch.
"""

from pathlib import Path

import modal

from launch.images import IMAGE_ENV, VOLUME_NAME, volume

image = modal.Image.debian_slim(python_version="3.11").env(IMAGE_ENV).add_local_python_source("launch")
app = modal.App("neural-chameleons-check-volume", image=image)


@app.function(volumes={"/vol": volume}, timeout=300)
def inside() -> dict:
    """What a container sees: the setting, the name its own import of launch.images resolves, the base models."""
    import os

    from launch import images  # imported again in the container, as every job's module is

    images.volume.reload()  # the module's handle must name the mounted volume, or this fails
    models = Path("/vol/models")
    return {"setting": os.environ.get("CHAMELEONS_VOLUME"), "resolved": images.VOLUME_NAME,
            "models": sorted(p.name for p in models.iterdir()) if models.is_dir() else []}


@app.local_entrypoint()
def main() -> None:
    seen = inside.remote()
    print(f"launch: {VOLUME_NAME}")
    print(f"container: setting {seen['setting']}, resolved {seen['resolved']}")
    print(f"base models on the volume: {', '.join(seen['models']) or 'none'}")
    if not seen["setting"] == seen["resolved"] == VOLUME_NAME:
        raise SystemExit("the container names a different volume than the launch")
    print("ok: the container uses the launch's volume")
