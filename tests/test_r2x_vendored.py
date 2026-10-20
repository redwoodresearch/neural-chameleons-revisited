"""The vendored data-era scripts are the authors' files at 3629f22, f7e4f66 and 76a3c98, byte for byte.

Run: uv run --with pytest python -m pytest tests/test_r2x_vendored.py
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
THEIRS = {"generate_synthetic_data.py": "data/self_obfuscation_v1_synthetic/generate_synthetic_data.py",
          "rate_synthetic_data.py": "data/self_obfuscation_v1_synthetic_rating/rate_synthetic_data.py",
          "rate_ultrachat_data.py": "data/self_obfuscation_v1_synthetic_rating/rate_ultrachat_data.py"}


VENDORED = [("3629f22", name, path) for name, path in sorted(THEIRS.items())] + [
    ("f7e4f66", "generate_synthetic_data.py", "data/self_obfuscation_v1_synthetic/generate_synthetic_data.py")] + [
    ("76a3c98", path, path) for path in sorted(THEIRS.values())]  # the rater keeps their layout: its import needs it


@pytest.mark.parametrize("commit,name,path", VENDORED)
def test_vendored_file_is_their_file_at_its_commit(commit, name, path):
    theirs = subprocess.run(["git", "-C", str(ROOT / "external/neural-chameleons"), "show", f"{commit}:{path}"],
                            capture_output=True, check=True).stdout
    assert (ROOT / f"third_party/neural_chameleons_{commit}" / name).read_bytes() == theirs
