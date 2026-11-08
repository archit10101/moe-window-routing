#!/bin/bash
# Optional extra seeds: base-s3 win16-s3 base-s4 win16-s4.
#   scripts/slurm/submit_extra_seeds.sh <gpu-type> [time, default 05:30:00]
# b200/h200: one job of 4 runs. l40s: four 1-run jobs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 1 ]; then echo "usage: $0 <gpu-type> [time]" >&2; exit 2; fi
gpu="$1"
t="${2:-05:30:00}"
RUNS=(base-s3 win16-s3 base-s4 win16-s4)

if [ "$gpu" = "l40s" ]; then
    for run in "${RUNS[@]}"; do
        "$HERE/submit.sh" "$run" "$gpu" "$t" "$run" || exit 1
    done
else
    "$HERE/submit.sh" extra-seeds "$gpu" "$t" "${RUNS[@]}" || exit 1
fi
