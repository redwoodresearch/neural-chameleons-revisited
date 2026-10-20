"""The 2x-data runs' files: which build the README runners read instead of the authors' shipped
files, where it lives, and how it is staged into a container's copy of their code.

R2X_BUILD in the environment chooses the build when a runner is imported, locally and in its containers (the runners
put it in their image's environment), as BIG_MODEL chooses the model. Unset, nothing here is used and the runners read
the shipped files exactly as before.

Their code finds its data by file name in one folder, data/synthetic_rating/outputs: step 1's `--dataset_path` keeps
only its basename and searches that folder with it as a regex (train_concept_probes.py:58-59, load_rated_data.py:58-68,
the most recent match wins), and the trainer's PRESET loader searches it for the escaped `--dataset_name` and its
`_MONITOR` (utils_training_data.py:44-69, :106). So the build's three files are copied into that folder under their own
names, each name must match only itself there, and the runners pass those names on their usual arguments.
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

BUILD = os.environ.get("R2X_BUILD", "")
WORKSTREAM = "main"
BUILDS = Path("/vol/data/readme_2x/build")
RUNS = Path("/vol/runs/readme_2x")
RESULTS = Path("results/readme_2x")  # in the repository
# Twice the shipped PRESET file's 4,698 rows, split for split: one epoch at batch 1, so the runs' optimizer steps.
ROWS = 9396
ONE_X_STEP = 4698  # the 1x point: as many updates as a run on the shipped file (constant learning rate)
SAVE_STEPS = [1600, 3200, 4698, 6400]
# The 31B's step-9,396 checkpoint (one epoch of its two), evaluated in three launches, each under its own run id: its
# 8 runs on Oct 12, then runs 405 and 406, which were still training then, on Oct 14 (406 stopped at the 24-hour limit
# in its second epoch, after saving step 9,396). Read together (chameleons.analysis.big_fig2.seed_folders).
G4_31B_STEP9396 = ("r2x-g4-31b-L17-2ep-20261011-ckpt-step9396", "r2x-g4-31b-L17-2ep-20261011-ckpt-seed405-step9396",
                   "r2x-g4-31b-L17-2ep-20261011-ckpt-seed406-step9396")
TARGET = "data/synthetic_rating/outputs"  # in their code's folder
ROLES = ("step1", "training", "monitor")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(build: str = BUILD) -> dict:
    return json.loads((BUILDS / build / "manifest.json").read_text())


def lookups(names: dict[str, str]) -> dict[str, str]:
    """role -> the pattern their code searches the folder with: the step-1 basename as given (a regex), the PRESET
    name escaped and its _MONITOR name escaped (load_preset_data escapes the name, then derives the monitor's)."""
    preset = re.escape(names["training"])
    return {"step1": names["step1"], "training": preset, "monitor": preset.replace(".json", "_MONITOR.json")}


def stage(official_dir: str, build: str = BUILD) -> dict:
    """Copy the build's step-1, PRESET and _MONITOR files into their data folder, each checked against the build's
    manifest, and check that each lookup their code makes finds exactly that file. Returns, per role, the name, the
    path their arguments take (relative to their code's folder) and the sha256."""
    if not build:
        raise RuntimeError("no R2X_BUILD: nothing to stage")
    files = manifest(build)["files"]
    target = Path(official_dir) / TARGET
    target.mkdir(parents=True, exist_ok=True)
    staged = {"build": build}
    for role in ROLES:
        name, expected = files[role]["name"], files[role]["sha256"]
        source = BUILDS / build / name
        if sha256(source) != expected:
            raise RuntimeError(f"{source} is not the file the build recorded")
        dest = target / name
        if dest.exists() and sha256(dest) != expected:
            raise RuntimeError(f"{dest} exists and differs from the build's {role} file")
        shutil.copy2(source, dest)
        staged[role] = {"name": name, "path": f"{TARGET}/{name}", "sha256": expected}
    present = [p.name for p in target.iterdir() if p.is_file()]
    for role, pattern in lookups({r: staged[r]["name"] for r in ROLES}).items():
        found = [n for n in present if re.search(pattern, n)]
        if found != [staged[role]["name"]]:
            raise RuntimeError(f"their lookup for the {role} file ({pattern!r}) finds {found}, not only "
                               f"{staged[role]['name']}")
    return staged
