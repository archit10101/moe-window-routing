#!/bin/bash
# Progress of every run folder, then the Slurm queue.
#   scripts/slurm/status.sh [out_root, default /home/architta/orcd/scratch/moe-pilot/runs]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
OUT_ROOT="${1:-${OUT_ROOT:-$PROJECT/runs}}"
PYTHON="${PYTHON:-python3}"

"$PYTHON" "$HERE/../status.py" "$OUT_ROOT"
echo
if command -v squeue > /dev/null; then
    squeue -u "$USER"
fi
