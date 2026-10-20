"""The Modal volume's name is one setting, CHAMELEONS_VOLUME, read in launch/images.py.

Importing the Modal code works without it (analyses and tests import launch/), but then it names a volume that does not
exist, and since no launch creates a volume, Modal stops the launch when it looks the volume up. Every image carries the
setting, so a container that imports the same module again names the same volume."""

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNSET = "CHAMELEONS_VOLUME-is-not-set"


def _resolve(env: dict) -> tuple[str, dict]:
    """The volume name and image environment that a fresh interpreter with exactly this setting resolves."""
    clean = {k: v for k, v in os.environ.items() if k != "CHAMELEONS_VOLUME"} | env
    out = subprocess.run([sys.executable, "-c", "import launch.images as I; print(I.VOLUME_NAME); print(I.IMAGE_ENV)"],
                         cwd=ROOT, env=clean, capture_output=True, text=True, check=True).stdout.splitlines()
    return out[0], ast.literal_eval(out[1])


def test_without_the_setting_the_import_works_and_names_no_real_volume():
    assert _resolve({})[0] == UNSET
    assert _resolve({"CHAMELEONS_VOLUME": "  "})[0] == UNSET


def test_the_setting_names_the_volume_and_travels_in_the_image_environment():
    name, image_env = _resolve({"CHAMELEONS_VOLUME": "my-volume"})
    assert name == "my-volume"
    assert image_env == {"CHAMELEONS_VOLUME": "my-volume"}
    # a container sees the image's environment, not the launching shell's: it must resolve the same volume
    assert _resolve(image_env)[0] == name


def _launch_files():
    return sorted((ROOT / "launch").rglob("*.py"))


def test_the_modal_code_has_one_volume_and_never_creates_one():
    lookups = []
    for path in _launch_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "from_name" and \
                    getattr(node.func.value, "attr", None) == "Volume":
                lookups.append((path.relative_to(ROOT).as_posix(), node))
    assert [path for path, _ in lookups] == ["launch/images.py"]
    call = lookups[0][1]
    assert [ast.unparse(a) for a in call.args] == ["VOLUME_NAME"]
    assert all(k.arg != "create_if_missing" or ast.unparse(k.value) == "False" for k in call.keywords)


def test_every_image_built_from_scratch_carries_the_setting():
    for path in _launch_files():
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Assign) and "modal.Image.debian_slim(" in ast.unparse(node.value):
                chain = ast.unparse(node.value)
                assert ".env(" in chain and "IMAGE_ENV" in chain, f"{path.relative_to(ROOT)}: {chain[:80]}..."
