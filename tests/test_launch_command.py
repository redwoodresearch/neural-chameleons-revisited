"""A launch record keeps the command that started the run without this machine's folders: paths inside the
repository are written relative to it, a program from elsewhere by its name, and the arguments shell-quoted."""

import shlex
from pathlib import Path

from chameleons.harness.provenance import launch_command

ROOT = Path(__file__).resolve().parents[1]


def test_paths_inside_the_repository_are_relative():
    argv = [str(ROOT / ".venv/bin/modal"), "run", "--detach", str(ROOT / "launch/train/readme20.py"), "--seeds", "1,2"]
    assert launch_command(argv) == ".venv/bin/modal run --detach launch/train/readme20.py --seeds 1,2"


def test_a_program_from_outside_the_repository_is_named_only():
    argv = ["/home/someone/.cache/uv/archive-v0/abc123/bin/modal", "run", "launch/models.py"]
    assert launch_command(argv) == "modal run launch/models.py"


def test_arguments_keep_their_boundaries():
    argv = ["modal", "run", "launch/train/readme_big.py", "--note", "two words"]
    assert shlex.split(launch_command(argv)) == argv


def test_other_absolute_paths_are_kept_as_given():
    argv = ["modal", "run", "launch/models.py", "--dest", "/vol/models"]
    assert launch_command(argv) == "modal run launch/models.py --dest /vol/models"
