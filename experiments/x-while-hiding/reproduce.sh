#!/bin/bash
# The local part of this result: the figure from the committed results (no GPU, no key).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python scripts/figures/fig_x_while_hiding_concepts.py
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons g4_31b
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons 9b_plain
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons 9b_2x
uv run python scripts/figures/fig_x_while_hiding_concepts.py --six
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons g4_31b --six
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons 9b_plain --six
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons 9b_2x --six
uv run python scripts/figures/fig_x_while_hiding_concepts.py --six --best-word
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons g4_31b --six --best-word
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons 9b_plain --six --best-word
uv run python scripts/figures/fig_x_while_hiding_concepts.py --six --logit
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons g4_31b --six --logit
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons 9b_plain --six --logit
uv run python scripts/figures/fig_x_while_hiding_concepts.py --chameleons 9b_2x --six --logit
uv run python scripts/figures/fig_x_while_hiding_concepts.py --off-policy
uv run python scripts/figures/fig_tradeoff_per_run.py
uv run python scripts/figures/fig_tradeoff_per_run.py --grid
uv run python scripts/figures/fig_tradeoff_by_size.py
uv run python scripts/figures/fig_tradeoff_by_size.py --logit
uv run python scripts/figures/fig_tradeoff_by_size.py --six
uv run python scripts/figures/fig_tradeoff_by_size.py --six --logit
