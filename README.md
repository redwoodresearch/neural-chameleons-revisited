# Neural Chameleons Revisited

Code, results and data records for the LessWrong post [*Models Trained to Evade Topic Probes Struggle to Answer
Questions about Those Topics*][post]. Its TL;DR:

[post]: https://www.lesswrong.com/posts/2qnLtNiBph5EjBf6c/models-trained-to-evade-topic-probes-struggle-to-answer

> [Neural Chameleons](https://arxiv.org/abs/2512.11949) are models trained to evade activation probes whenever the
> prompt says "You are being probed for X". The paper reports that this generalizes to probes the model never saw, at
> a small capability cost (about 3 points on MMLU and 1 on GSM8K). But it never tests the models on questions where
> the cognition the probe is looking for is necessary to answer correctly. In this post, we test exactly that: we test
> how well neural chameleons can evade probes looking for particular topics while still answering questions on those
> topics correctly.
>
> Our experiments showed that chameleons can hide from topic probes they were directly trained to evade, but at a
> substantial cost to their capability on the relevant topic. A bigger model (up to 31B), twice the training data, or
> a non-abliterated model showed the same pattern. Separately, when we replicated the paper, we found that its
> training generally costs more capability than reported, and that concepts the chameleons were never trained on
> mostly stay visible to the probe.

## Quick start

```bash
git clone --recurse-submodules https://github.com/redwoodresearch/neural-chameleons-revisited.git
cd neural-chameleons-revisited
uv sync --group dev
./reproduce.sh                    # every figure, from the committed results; no GPU, no API key
uv run --group dev pytest tests/  # the tests (the model tests also need --group models)
```

`reproduce.sh` redraws every figure, and the tables the figure scripts write, from the committed summaries, and fails
if any of them comes out different from the committed file. Recomputing the summaries themselves needs the models'
answers (below).

## What you can rerun

1. **The figures.** Every figure in the post, from the committed results, in a fresh clone with no GPU and no API key
   (the quick start above).
2. **The summaries.** Recomputed from the models' answers. The answers are not in git; they are available on request
   ([`DATA.md`](DATA.md)).
3. **The runs.** Training and evaluation on Modal GPUs ([Running on GPUs](#running-on-gpus)), with the same data.

## The results

One folder per result. Each has a README (what the result is, its figures, its data, the commands that made it) and a
`reproduce.sh` that redraws its figures. The post's figures come from three of them: Figures 2, A2 and A4 to A7 from
`x-while-hiding`, Figure A1 from `training-9b`, and Figure A3 from `scale`. The other two record how the 31B was trained and
how the second batch of training data was made.

| Folder | What it shows |
|---|---|
| [`x-while-hiding`](experiments/x-while-hiding/) | The main result: accuracy, and how often the probe still catches the answers, under a trigger |
| [`scale`](experiments/scale/) | The paper's main figure at 9B and 31B |
| [`training-9b`](experiments/training-9b/) | The 9B: 20 runs of the authors' pipeline |
| [`training-31b`](experiments/training-31b/) | The same pipeline on Gemma 4 31B |
| [`training-2x-data`](experiments/training-2x-data/) | How the second batch of training data was made |

## How the code is laid out

| Path | Contents |
|---|---|
| `src/chameleons/` | The Python package: evaluation harness, data building, model tables, the tests' settings and grading, analyses, figures |
| `scripts/` | Scripts run locally: `analysis/` (summaries, checks), `data/`, `figures/` |
| [`launch/`](launch/README.md) | Everything that runs on Modal GPUs: training, evaluations, their workers |
| `results/`, `plots/` | The runs' summaries and launch records; the figures |
| [`DATA.md`](DATA.md) | Every dataset: who made it, its hash, how to get it |
| [`docs/`](docs/) | The write-up, and the papers and code the work builds on |
| `external/neural-chameleons` | The authors' code (a submodule at `261e6547`), run unmodified apart from `src/chameleons/patches/` |
| `third_party/` | The authors' data generator and rater, at the commits the second batch of data was made with |

## Names in the code

The code and the results keep the names the runs were launched under:

- **readme20**: the authors' README pipeline, run 20 times on the 9B (`launch/train/readme20.py`, `results/readme20/`).
- **readme20-fixed, readme20-fixedtrain**: the same runs with the authors' earlier way of fitting probes restored
  (shuffled batches, one example per forward pass), for the evaluation only or for the training too.
- **readme_2x, r2x**: the runs on twice the training data (`results/readme_2x/`, `src/chameleons/data/r2x/`).
- **readme_big**: the same pipeline on Gemma 4 (`launch/train/readme_big.py`, `results/readme_big/`).
- **ncrepl, harness**: our replication written from the paper's text; its evaluation code is the harness in
  `src/chameleons/harness/`.
- **datascale**: the data-building code that the second batch of data reuses (`src/chameleons/data/datascale/`).

## Common tasks

- **Redraw one result's figures**: `bash experiments/<result>/reproduce.sh`.
- **Recompute a summary from the model's answers**: get the answers ([`DATA.md`](DATA.md)), then run the result's
  summary script ([`x-while-hiding`](experiments/x-while-hiding/README.md#where-the-numbers-are)).
- **Rerun an experiment on GPUs**: each result's README gives the exact commands its runs were launched with. See
  [Running on GPUs](#running-on-gpus) for what that needs.
- **Test another trained model**: add it under a new name to `MODELS` in `src/chameleons/assays.py`, give it an image
  under the same name in `IMAGES` in `launch/assays/math_onpolicy.py`, and launch the tests with `CHAMELEONS=<name>`
  ([`x-while-hiding`](experiments/x-while-hiding/README.md#how-the-raw-results-were-made)).

## Running on GPUs

- **Modal**: an account and a token (`uv run modal token new`).
- **The volume**: every job mounts one Modal volume, named by the setting `CHAMELEONS_VOLUME`. Create it once
  (`uv run modal volume create <name>`), set `export CHAMELEONS_VOLUME=<name>`, then put the base models on it:
  `uv run modal run launch/models.py`. `uv run modal run launch/check_volume.py` then checks, in one small container,
  that a job names the same volume and finds the base models on it. A launch never creates a volume: without the
  setting, or with a name that does not exist, it stops before any job starts.
- **Dataset caches**: the jobs read Hugging Face datasets offline from caches on the volume. On a new volume, warm the
  evaluation harness's cache once (`uv run modal run launch/train/harness.py --action warm`) and the 31B runner's with
  its `--prepare` step (`launch/train/readme_big.py`); the 9B runner's jobs read what an earlier job cached
  (`launch/train/readme20.py`'s docstring).
- **Hugging Face**: the Gemma models are gated; accept their licences and log in (`uv run hf auth login`).
- **Package versions**: most images pin their packages. The Gemma 4 images (`PACKAGES_GEMMA4` in
  `launch/train/harness.py`) and the BFCL image (`launch/train/bfcl.py`) leave some unpinned, so an image built later
  can differ from the one the runs used.
- **API keys** for the scripts that call an LLM: `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` (or
  `~/.anthropic_api_key`, `~/.openai_api_key`).
- **Data**: the runs on twice the data and the never-trained concepts' probes need data that is available on request
  ([`DATA.md`](DATA.md)).

Every launch writes a record beside its results (commit, seeds, data hashes, Modal call ids); `--check` prints it
without starting the runs (the 9B runner with the probe-fitting fix first runs its short preflight jobs). The runners launch only from a clean `main` and refuse a run id that exists already, so give a
rerun a new one. The doing-X-while-hiding tests likewise refuse to overwrite earlier outputs unless told
`--overwrite`.

## The authors' code

The authors' code, in `external/neural-chameleons` and `third_party/`, keeps its MIT licence
([`third_party/LICENSE`](third_party/LICENSE)).
