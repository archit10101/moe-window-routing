#!/bin/bash
# Main runs (500M tokens each).
#   scripts/slurm/submit_main.sh <gpu-type> [time, default 05:30:00]
# b200/h200: job A = base-s0 win16-s0 base-s1 win16-s1; job B = base-s2 win16-s2 win1-s0 win64-s0.
# l40s: eight 1-run jobs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 1 ]; then echo "usage: $0 <gpu-type> [time]" >&2; exit 2; fi
gpu="$1"
t="${2:-05:30:00}"
JOB_A=(base-s0 win16-s0 base-s1 win16-s1)
JOB_B=(base-s2 win16-s2 win1-s0 win64-s0)

if [ "$gpu" = "l40s" ]; then
    for run in "${JOB_A[@]}" "${JOB_B[@]}"; do
        "$HERE/submit.sh" "$run" "$gpu" "$t" "$run" || exit 1
    done
else
    "$HERE/submit.sh" main-A "$gpu" "$t" "${JOB_A[@]}" || exit 1
    "$HERE/submit.sh" main-B "$gpu" "$t" "${JOB_B[@]}" || exit 1
fi
