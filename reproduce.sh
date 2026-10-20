#!/bin/bash
# Every result's local part (each folder's reproduce.sh, one after another); the GPU launches are named in each
# folder's README. Fails if a figure or summary comes out different from the committed file; ALLOW_PNG_DRIFT=1 lets
# figures alone differ (fonts and system libraries vary between machines).
set -euo pipefail
cd "$(dirname "$0")"
for script in experiments/*/reproduce.sh; do
    echo "== $script"
    bash "$script"
done
drift=$(git status --porcelain -- results plots)
if [ "${ALLOW_PNG_DRIFT:-0}" = "1" ]; then
    drift=$(printf '%s\n' "$drift" | grep -v '\.png$' || true)
fi
if [ -n "$drift" ]; then
    echo "These outputs differ from the committed files:"
    echo "$drift"
    exit 1
fi
echo "Every figure and summary matches the committed one."
