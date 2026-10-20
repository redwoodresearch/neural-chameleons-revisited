"""Where every stage of the data-scale build reads and writes.

`DATA` is git-ignored; `MANIFESTS` holds hashes, counts and git state only - never text from the data files - so it is
safe to commit, and it is the provenance record for the git-ignored `DATA` tree.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]          # DS
AUTH = REPO / "external" / "neural-chameleons"
DATA = REPO / "data" / "datascale"                  # git-ignored
CACHE = DATA / "cache"                              # FileCache dir (API calls)
POOL = DATA / "pool"                                # generated synthetic records: POOL / f"{concept}.jsonl"
ULTRACHAT = DATA / "ultrachat" / "pool.jsonl"       # cut UltraChat records, dataset order
RATED = DATA / "rated"                              # synthetic_rated.json (all concepts), ultrachat_rated.json
ARMS = DATA / "arms"                                # ARMS / arm / {step1, training, monitor, report, manifest}
HELDOUT = DATA / "heldout"                          # the held-out training-concept test set (spec §5.3, D17)
HELDOUT_FILE = "heldout_rated.json"                 # its file name, locally and on the volume under VOLUME_DIR/heldout
MANIFESTS = REPO / "results" / "datascale" / "data"  # tracked; the committed copy of every stage manifest
VOLUME_DIR = "datascale"                            # the same layout on the Modal volume under /vol/datascale
