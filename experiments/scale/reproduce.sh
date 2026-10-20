#!/bin/bash
# The local part of this result: the figure from the committed results (no GPU, no key).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python scripts/figures/fig2_scale.py --columns 9B plain 9B-2x 31B-2x
