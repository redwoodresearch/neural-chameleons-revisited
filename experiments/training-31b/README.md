# The 31B: the same pipeline on Gemma 4

The authors' pipeline, with the same fix and the 9B's hyperparameters, on Gemma 4 31B instruct as released, trained on
twice the data: probes at layer 17, two passes over the data, 10 runs (`r2x-g4-31b-L17-2ep-20261011`, seeds 401-410).
Every test reads each run's checkpoint after one pass (step 9,396). Eight runs were evaluated on October 12; runs 405
and 406 were still training then and were evaluated overnight on October 14-15 (406 hit the 24-hour job limit in its
second pass, after saving step 9,396). Run 409 collapsed and is left out. Run 405 passes the collapse rule but is
degraded: with no trigger it is close to chance on chemistry and physics. It stays in, since the rule was fixed before
its results.

The base model goes through the same evaluations, for comparison.

## Figures

These runs are drawn in [`scale`](../scale/) and [`x-while-hiding`](../x-while-hiding/); nothing is drawn from this
page.

## Data

Committed under `results/readme_2x/`: each run's evaluation at step 9,396
(`r2x-g4-31b-L17-2ep-20261011-ckpt-step9396/`; runs 405 and 406 in `-ckpt-seed405-step9396/` and
`-ckpt-seed406-step9396/`), and the training launch record, `r2x-g4-31b-L17-2ep-20261011/launch.json`. The base
model's evaluation is under `results/readme_big/g4-31b-base-L17-20261010/`. The other saved steps' evaluations are
available on request.

## How the raw results were made

These are the commands each run was launched with, as recorded in its `launch.json` (with today's paths). A runner
refuses a run id that exists already, so give a rerun a new one. The model is chosen with `BIG_MODEL`, the data with
`R2X_BUILD`. On a fresh volume, put the base models there first (`launch/models.py`, then `--fetch` and `--prepare`;
see its docstring).

```bash
# two passes, saving step 9,396 (one pass). Its step-1 probes come from an earlier launch of the same runner,
# r2x-g4-31b-L17-20261011 (one pass; the same seeds, data and layer), stopped once its step 1 had finished on all ten
# seeds; its records are not in this repository. Without --probes-from, a run fits its own step-1 probes first.
BIG_MODEL=g4_31b R2X_BUILD=b20261011 uv run modal run --detach launch/train/readme_big.py \
    --run-id r2x-g4-31b-L17-2ep-20261011 --seeds 401,402,403,404,405,406,407,408,409,410 --mode full --layer 17 --epochs 2 \
    --save-steps 1600,3200,4698,6400,9396,14094 --probes-from r2x-g4-31b-L17-20261011
# the saved steps, evaluated as final checkpoints: 8 runs on Oct 12, then runs 405 and 406
BIG_MODEL=g4_31b R2X_BUILD=b20261011 uv run modal run --detach launch/train/readme_big.py \
    --evaluate-steps r2x-g4-31b-L17-2ep-20261011 --run-id r2x-g4-31b-L17-2ep-20261011-ckpt \
    --seeds 401,402,403,404,407,408,409,410 --steps 1600,3200,4698,6400,9396,14094
BIG_MODEL=g4_31b R2X_BUILD=b20261011 uv run modal run --detach launch/train/readme_big.py \
    --evaluate-steps r2x-g4-31b-L17-2ep-20261011 --run-id r2x-g4-31b-L17-2ep-20261011-ckpt-seed405 --seeds 405 --steps 9396
BIG_MODEL=g4_31b R2X_BUILD=b20261011 uv run modal run --detach launch/train/readme_big.py \
    --evaluate-steps r2x-g4-31b-L17-2ep-20261011 --run-id r2x-g4-31b-L17-2ep-20261011-ckpt-seed406 --seeds 406 \
    --steps 9396 --metadata-from 405
# the base model through the same evaluations
BIG_MODEL=g4_31b uv run modal run --detach launch/train/readme_big.py --base-eval --run-id g4-31b-base-L17-20261010 --layer 17
```

Every run comes down with `--download <run id>`. Run 406 hit the 24-hour job limit in its second pass, so it has no
final checkpoint to copy the evaluation's metadata from; `--metadata-from 405` builds that file from run 405's (the
two training commands differ only in the seed).
