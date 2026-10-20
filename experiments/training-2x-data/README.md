# Twice the training data, made with the authors' pipeline

The runs "on twice the data" read build `b20261011`: 16,128 new concept texts generated and rated with the authors'
own generator and rater at their pinned commits, assembled into the same kinds of files the authors ship (the
step-1 probes' data, the training file, its loss-curve sample, the behaviour-preservation data from UltraChat) plus a
held-back audit set.

## Figures

None. The build's record is `results/readme_2x/build/b20261011/`: `manifest.json` (every file and its SHA-256) and
`mix.md` (how the concepts and ratings are mixed).

## Data

[`DATA.md`](../../DATA.md), kind B: the files and their hashes. The text is not in git; it is available on request.

## How it was made

On Modal, as recorded in each stage's `launch.json` (`results/readme_2x/generation/` and `rating/`): the generation
(the authors' generator at `3629f22`, writing with Gemma 2 9B instruct) and the rating (the authors' rater with
`gpt-4.1-nano`; it needs `OPENAI_API_KEY`):

```bash
uv run modal run --detach launch/train/readme_2x.py --generate r2x-pool-20261011 --generator 3629f22 \
    --model gemma_2_9b_instruct --num-prompts 1152
uv run modal run --detach launch/train/readme_2x.py --generate r2x-topup-20261011 --generator 3629f22 \
    --model gemma_2_9b_instruct --num-prompts 192 --seed 1
uv run modal run --detach launch/train/readme_2x.py --rate r2x-synth-20261011 --kind synthetic \
    --generations r2x-pool-20261011,r2x-topup-20261011
uv run modal run --detach launch/train/readme_2x.py --rate r2x-ultrachat-20261011 --kind ultrachat --num-conversations 4096
uv run modal run launch/train/readme_2x.py --download-rating r2x-synth-20261011
uv run modal run launch/train/readme_2x.py --download-rating r2x-ultrachat-20261011
```

Then, locally, the build from the downloaded ratings (seed 0), uploaded to the volume:

```bash
uv run python -m chameleons.data.r2x.build --synthetic r2x-synth-20261011 --ultrachat r2x-ultrachat-20261011 \
    --build b20261011 --upload
```

Nothing here reruns from a clone: the manifest and the mix are the record. Once the built files are on disk,
`uv run python scripts/data/mix.py b20261011` recomputes the mix.
