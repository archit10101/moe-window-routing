#!/bin/bash
# SlimPajama control at the M2D2 budget (120M tokens, LR 5e-4): 6 one-run jobs, 00:40:00 each,
# with each base/win16 pair adjacent (docs/M2D2_PROMPT.md part 8).
#
#   scripts/slurm/submit_slim120.sh <gpu-type>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SUBMIT:-$HERE/submit.sh}"
RUNS=(slim120-base-s0 slim120-win16-s0 slim120-base-s1 slim120-win16-s1 slim120-base-s2 slim120-win16-s2)

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <gpu-type: b200|l40s|h200>" >&2
    exit 2
fi
for run in "${RUNS[@]}"; do
    "$SUBMIT" "$run" "$1" 00:40:00 "$run" || exit 1
done
echo "submitted ${#RUNS[@]} slim120 jobs"
