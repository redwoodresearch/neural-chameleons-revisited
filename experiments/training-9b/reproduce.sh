#!/bin/bash
# The local part of this result: the summaries and the figure from the committed results (no GPU, no key).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python -m chameleons.figures.fig2_four_ways
uv run python -m chameleons.figures.fig_triggered_capability
uv run python -m chameleons.figures.fig2_word_rule
