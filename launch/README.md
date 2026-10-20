# launch/: everything that runs on Modal

The code that trains and evaluates models on Modal GPUs. The patches it applies to the authors' code are in
`src/chameleons/patches/`.

It holds no secrets. Modal credentials come from the environment (`modal token`); the Hugging Face token is read from
the local login, for one job at a time; API keys come from `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.

| Path | What |
|---|---|
| `images.py` | The Modal images (with the authors' code at the pinned commit), the volume, the package versions |
| `models.py` | Puts the base models on a fresh volume, at the revisions every run used |
| `check_volume.py` | Checks the setup: a container names the launch's volume and finds the base models on it (one small CPU job) |
| `train/readme20.py` | The 9B: the authors' README pipeline (`BASE=plain` for the plain 9B, `R2X_BUILD=b20261011` for twice the data) |
| `train/readme_big.py` | The same pipeline on Gemma 4 (`BIG_MODEL=g4_31b` for the 31B) |
| `train/readme_2x.py` | Generating and rating twice the training data |
| `train/harness.py` | The capability harness (MMLU, GSM8K) and the replication from the paper's text |
| `train/bfcl.py` | BFCL, the tool-use benchmark |
| `assays/math_onpolicy.py` | Doing X while hiding X, for every concept and model (`CHAMELEONS=<model>`) |
| `assays/rerun_jobs.py` | Reruns single evaluation jobs that failed, with the arguments the original launch gave them |
| `workers/` | The scripts the assays run inside the containers, one per test |

## The volume

Every job mounts one Modal volume, named by the setting `CHAMELEONS_VOLUME` (read in `images.py`). Create it once and
put the base models on it:

```bash
uv run modal volume create <name>
export CHAMELEONS_VOLUME=<name>
uv run modal run launch/models.py
uv run modal run launch/check_volume.py   # a container sees the same volume and the base models
```

Every image carries the setting, so a container names the same volume as the launch. A launch never creates a volume:
without the setting, or with a name that does not exist, Modal stops it before any job starts.

## Launch records

Every launch writes a record beside its results (`results/.../launch*.json`, or a row in a `runs*.csv` registry)
with the commit, the seeds, the data files' hashes and the Modal call ids. A runner's `--check` prints the record
without launching, and the runners refuse to launch from any branch but `main`. Settings chosen by an environment
variable (`BASE`, `R2X_BUILD`, `BIG_MODEL`, `CHAMELEONS`, `CHAMELEONS_VOLUME`) travel into the containers in the image,
so a job always reads the same setting as the launch.
