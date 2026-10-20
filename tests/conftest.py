"""The entry scripts (scripts/<role>/) are imported as top-level modules by the tests that check them; the library
(src/chameleons) and the Modal code (launch/) are installed packages."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "scripts" / "analysis", ROOT / "scripts" / "figures", ROOT / "scripts" / "data"):
    sys.path.insert(0, str(folder))

# The library tests import transformers and safetensors (the `models` dependency group), which only the Modal images
# carry by default; without them those files are left out of the collection, not failed.
try:
    import safetensors  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    collect_ignore = ["test_ncrepl_core.py", "test_ncrepl_variants.py"]
