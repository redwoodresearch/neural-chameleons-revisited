# The 9B: the authors' README pipeline, 20 runs

We ran the authors' training pipeline, as their README describes it, on Gemma 2 9B with their data, in three steps:

1. `readme20-main-20261007`: 20 runs exactly as the README runs them (seeds 101-120).
2. `readme20-fixed-20261007`: the same checkpoints re-scored with one fix to the authors' probe fitting, restoring how
   it worked before a 2025-09-10 change to their code (training tokens shuffled, and activations computed one
   example at a time so that batch padding is not used as training tokens).
3. `readme20-fixedtrain-20261008`: 20 new runs with the fix in training too (seeds 201-220). These "README-fixed" runs
   are the 9B in every later result.

What we found:

- The evasion reproduces.
- Capability falls more than the paper reports (MMLU 72 to 65, GSM8K 81 to 67).
- Some of the paper's reported numbers sit at the edge of the spread across the 20 runs, so one training run, as in
  the paper, is not enough to pin them down.

## Figures

- `plots/paper_style/fig2_four_ways.png`: the paper's Figure 2 next to two measurements of it, both through the
  authors' evaluation code: 20 runs of the published code, as published (step 1) and with the probe training patched
  (step 3). Capability is MMLU and GSM8K with no trigger (`--table` also lists BFCL). The write-up's Figure A1.
- `plots/paper_style/fig_triggered_capability.png`: capability with an unrelated trigger on ("harmful",
  "deception"), against the base model given the same prompt. The paper measures capability without the trigger.
- `plots/paper_style/fig2_word_rule.png`: how much of the paper's drop under the trigger comes from its reporting
  rule. For each unseen probe the paper reports the *lowest* recall among seven or eight related trigger words; the
  figure puts that next to what a typical word does.

## Redraw the figures

```bash
bash experiments/training-9b/reproduce.sh
```

which runs:

```bash
uv run python -m chameleons.figures.fig2_four_ways
uv run python -m chameleons.figures.fig_triggered_capability
uv run python -m chameleons.figures.fig2_word_rule
```

## Data

The authors' data, used as shipped from the submodule (`DATA.md`, kind A). The runs' summaries are committed under
`results/readme20/<run id>/`, for the three steps above and the plain 9B (below).

## How the raw results were made

These are the commands each run was launched with, as recorded in its `launch.json` (with today's paths). A runner
refuses a run id that exists already, so give a rerun a new one. `--check` prints the guards and the plan without
launching.

```bash
# 1. 20 runs as the README runs them
uv run modal run --detach launch/train/readme20.py --run-id readme20-main-20261007 \
    --seeds 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120
# 2. their checkpoints re-scored with the probe-fitting fix
uv run modal run --detach launch/train/readme20.py --fixed-measurement readme20-main-20261007 \
    --run-id readme20-fixed-20261007 --seeds 101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120
# 3. 20 new runs with the fix in training too (the "README-fixed" runs)
uv run modal run --detach launch/train/readme20.py --run-id readme20-fixedtrain-20261008 \
    --seeds 201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220 --purpose fixed-training
```

Each run comes down with `uv run modal run launch/train/readme20.py --download <run id>`. Then
`uv run python scripts/analysis/readme20_check.py --run-id <run id>` checks every job of every seed and writes
`check.json` beside the run.

**The plain 9B.** To test whether abliteration causes the capability cost, the published pipeline on Gemma 2 9B
instruct as released (10 runs). These runs were launched without the probe-fitting fix, as in step 1, so they differ
from the README-fixed runs in two ways, the base model and the probe fitting (their probe records show no switches):

```bash
BASE=plain uv run modal run --detach launch/train/readme20.py --run-id readme20-plain-20261013 \
    --seeds 301,302,303,304,305,306,307,308,309,310 --purpose plain-base
```

Their BFCL jobs failed at launch (a bug in the repository move, since fixed) and were rerun on 2026-10-14 with
`launch/assays/rerun_jobs.py`, which repeats single evaluation jobs with the arguments the original launch gave them. Their doing-X-while-hiding-X results are in
[`x-while-hiding`](../x-while-hiding/) (`CHAMELEONS=9b_plain`).
