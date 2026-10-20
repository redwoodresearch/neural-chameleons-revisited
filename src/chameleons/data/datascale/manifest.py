"""The manifest every stage writes: git state, parameters, counts and the sha256 of every input and output (§6).

Each manifest is written twice, beside its data and as a tracked copy under `paths.MANIFESTS` (amendment R1). The
tracked copy is not a convenience: `DATA` is git-ignored, so without it no stage could ever leave a record in the
repository. It holds hashes, counts and git state only - never text from the data files, which carry synthetic
harmful text - so it is safe to commit, and it is the provenance record for the git-ignored tree.

`verify` re-hashes every output and compares the two copies byte for byte, so a quietly edited or re-generated file is
named rather than passed over.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from chameleons.data.datascale.paths import AUTH, MANIFESTS, REPO

SUBMODULE = AUTH.relative_to(REPO)  # external/neural-chameleons, the authors' code


def sha256_file(path: Path) -> str:
    """The sha256 of a file's bytes, read in chunks so a 50,000-conversation pool does not go through memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo: Path = REPO) -> dict:
    """The branch, commit and cleanliness of `repo`, and the commit of the authors' code.

    Both commits of the submodule are recorded: `submodule_commit` is the checkout's own `HEAD`, the code a stage
    actually ran against, and `submodule_pinned` is the gitlink in `repo`'s tree, the commit this repository asks
    for. A checkout moved off its pin makes the two differ, and is also a modified entry in `git status`, hence
    `dirty`. A submodule that was never initialised is neither - it is no entry in `git status --porcelain` at all -
    so `_submodule_head` refuses it rather than letting git answer from the enclosing repository.
    """
    repo = Path(repo)
    return {
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "submodule_commit": _submodule_head(repo / SUBMODULE),
        "submodule_pinned": _git(repo, "rev-parse", f"HEAD:{SUBMODULE}"),
    }


def write(stage: str, path: Path, *, inputs: dict[str, Path], outputs: dict[str, Path], params: dict, counts: dict,
          also: Path | None = None) -> dict:
    """Write `stage`'s manifest to `path` and to its tracked twin, the same bytes in both, and return the record.

    `also` defaults to `MANIFESTS / f"{stage}.json"`; a per-arm stage passes `MANIFESTS / arm / f"{stage}.json"`.
    """
    also = MANIFESTS / f"{stage}.json" if also is None else Path(also)
    record = {
        "stage": stage,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_state(),
        "params": params,
        "counts": counts,
        "inputs": {name: _entry(file) for name, file in inputs.items()},
        "outputs": {name: _entry(file) for name, file in outputs.items()},
        "tracked_copy": _record_path(also),
    }
    data = json.dumps(record, indent=2, sort_keys=True).encode() + b"\n"
    for target in (Path(path), also):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return record


def verify(path: Path) -> list[str]:
    """Every output of the manifest at `path` that no longer matches, and its tracked twin if that is missing or
    differs. An empty list means the data on disk is the data the manifest was written for."""
    path = Path(path)
    record = json.loads(path.read_text())
    problems = []
    for name, entry in record["outputs"].items():
        file = resolve(entry["path"])
        if not file.exists():
            problems.append(f"{name}: missing ({entry['path']})")
        elif sha256_file(file) != entry["sha256"]:
            problems.append(f"{name}: sha256 differs from the manifest ({entry['path']})")
    tracked = resolve(record["tracked_copy"])
    if not tracked.exists():
        problems.append(f"tracked copy missing: {record['tracked_copy']}")
    elif tracked.read_bytes() != path.read_bytes():
        problems.append(f"tracked copy differs from {path}: {record['tracked_copy']}")
    return problems


def _submodule_head(path: Path) -> str:
    """The `HEAD` of the authors' checkout at `path`, refused unless `path` is that checkout's own repository.

    `git -C <dir> rev-parse HEAD` answers from the first repository it finds walking up, so an
    `external/neural-chameleons` that exists but holds no checkout - a clone without `--recurse-submodules`, a
    container that populates it later - would hand back this repository's own commit, and the manifest would claim
    the authors' code ran at a plausible 40-hex commit when no authors' code was there. A manifest that fails to
    write is better than one that lies about its provenance (spec §6).
    """
    path = Path(path)
    if not path.is_dir():
        raise RuntimeError(f"the authors' code is not checked out at {path}: git submodule update --init")
    try:
        toplevel = _git(path, "rev-parse", "--show-toplevel")
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"not a git repository: {path}: git submodule update --init") from error
    if Path(toplevel).resolve() != path.resolve():
        raise RuntimeError(f"{path} is not its own git repository - git answers for {toplevel} - so the authors' "
                           f"code is not checked out there: git submodule update --init")
    return _git(path, "rev-parse", "HEAD")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=True).stdout.strip()


def _entry(file: Path) -> dict:
    return {"path": _record_path(file), "sha256": sha256_file(file)}


def _record_path(file: Path) -> str:
    """A path as the manifest records it: relative to the repository, so a committed manifest is portable."""
    file = Path(file).resolve()
    try:
        return str(file.relative_to(REPO))
    except ValueError:
        return str(file)  # the Modal volume, or a test's tmp_path: nothing to be relative to


def resolve(recorded: str) -> Path:
    """A path a manifest recorded, back as a path on this machine: `_record_path`'s inverse.

    Public because a guard that reads a manifest's own output - `scripts/datascale.py`'s `_report_of`, which reads
    the gate report a stage's manifest pins by sha256 - has to resolve it the same way `verify` does.
    """
    path = Path(recorded)
    return path if path.is_absolute() else REPO / path
