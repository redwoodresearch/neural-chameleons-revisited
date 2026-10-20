# Does scale help? The paper's main figure at 9B and 31B

The paper's Figure 2 shows, for each probe, how often it fires with and without the trigger, next to the model's
capability. We draw it for the abliterated and the plain 9B on the authors' training data, and for the 9B and Gemma 4
31B on twice as much. The plain 9B's runs are the published pipeline without the probe-fitting fix, so they differ from
the abliterated 9B's in that as well as in the base model. The 31B evades the probes it was trained against, and the paper's unseen probes, about as much
as the 9B does, at a similar capability cost.

One caveat: the 31B is trained with the 9B's hyperparameters (learning rate 8e-6, batch size 1, no sweep), so a
recipe tuned for it might do better.

## Figure

`plots/paper_style/fig2_scale_9b_plain_9b-2x_31b-2x.png`, the write-up's Figure A3. Left: how often each probe fires
(recall at a 1% false-positive rate) without the trigger (blue) and with it (orange), one dot per run. Right: MMLU,
GSM8K and BFCL (tool use) with no trigger, the base model as a grey bar. The 31B is read after one pass over twice the
data (step 9,396).

Runs whose generation collapsed (under 5% on GSM8K with no trigger; `chameleons.compare.collapsed`) are left out.

## Redraw the figure

```bash
bash experiments/scale/reproduce.sh     # uv run python scripts/figures/fig2_scale.py --columns 9B plain 9B-2x 31B-2x
```

## Data

Every run's final evaluation, committed under `results/readme20/` (the 9B on the authors' data, abliterated and
plain) and `results/readme_2x/<run id>/` (the runs on twice the data); the 31B's base model under
`results/readme_big/`. Each run's folder holds its launch record (`launch.json`). BFCL scores are read from
`results/bfcl/pooled.json`.

## How the raw results were made

The training runs and their evaluations: see [`training-9b`](../training-9b/) and
[`training-31b`](../training-31b/). The 9B on twice the data, as recorded in its `launch.json` (a rerun needs
a new run id):

```bash
R2X_BUILD=b20261011 uv run modal run --detach launch/train/readme20.py --run-id r2x-9b-20261011 \
    --seeds 201,202,203,204,205,206,207,208,209,210 --purpose fixed-training-checkpoints --save-steps 1600,3200,4698,6400
```

Its saved checkpoints are evaluated as a final checkpoint would be, each step under its own run id
(`<run id>-step<n>`):

```bash
R2X_BUILD=b20261011 uv run modal run --detach launch/train/readme20.py --evaluate-steps r2x-9b-20261011 \
    --run-id r2x-9b-20261011-ckpt --seeds 201,202,203,204,205,206,207,208,209,210 --steps 3200,4698
```
