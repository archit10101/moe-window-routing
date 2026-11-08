#!/bin/bash
# GPU smoke test: smoke-gpu-base and smoke-gpu-win16 (200 steps each) in one job, 30 minutes.
#   scripts/slurm/submit_smoke.sh <gpu-type>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 1 ]; then echo "usage: $0 <gpu-type>" >&2; exit 2; fi
"$HERE/submit.sh" smoke "$1" 00:30:00 smoke-gpu-base smoke-gpu-win16
