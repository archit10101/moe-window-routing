#!/bin/bash
# LR check: 3 base + 3 win16 runs at 100M tokens.
#   scripts/slurm/submit_lr_check.sh <gpu-type> [time, default 05:30:00]
# b200/h200: two jobs of 3 runs (base LRs, then window LRs). l40s: six 1-run jobs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 1 ]; then echo "usage: $0 <gpu-type> [time]" >&2; exit 2; fi
gpu="$1"
t="${2:-05:30:00}"
BASE_RUNS=(lr-base-1e-3 lr-base-2e-3 lr-base-4e-3)
WIN_RUNS=(lr-win16-1e-3 lr-win16-2e-3 lr-win16-4e-3)

if [ "$gpu" = "l40s" ]; then
    for run in "${BASE_RUNS[@]}" "${WIN_RUNS[@]}"; do
        "$HERE/submit.sh" "$run" "$gpu" "$t" "$run" || exit 1
    done
else
    "$HERE/submit.sh" lr-base "$gpu" "$t" "${BASE_RUNS[@]}" || exit 1
    "$HERE/submit.sh" lr-win16 "$gpu" "$t" "${WIN_RUNS[@]}" || exit 1
fi
