#!/bin/bash
# Progress of every run folder (including the m2d2-, m2d2h4-, m2d2r4- and slim120- runs), the M2D2 data
# builds, then the Slurm queue (which shows queued build jobs as build-m2d2-*).
#   scripts/slurm/status.sh [out_root, default /home/architta/orcd/scratch/moe-pilot/runs]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/m2d2_common.sh"
PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
OUT_ROOT="${1:-${OUT_ROOT:-$PROJECT/runs}}"
PYTHON="${PYTHON:-python3}"

"$PYTHON" "$HERE/../status.py" "$OUT_ROOT"
echo
echo "M2D2 data builds:"
found=0
for d in "$PROJECT"/data_m2d2w8 "$PROJECT"/data_m2d2w8_*; do
    [ -d "$d" ] || continue
    found=1
    uneven=""
    [ "$d" != "$PROJECT/data_m2d2w8" ] && uneven=1
    if build_ok "$d" "$uneven"; then
        state="checks passed"
    elif [ -f "$d/meta.json" ]; then
        state="CHECKS NOT PASSED (see sanity.txt)"
    else
        state="no build yet (or in progress)"
    fi
    printf '  %-22s %s\n' "$(basename "$d")" "$state"
done
[ "$found" -eq 1 ] || echo "  none yet"
echo
if command -v squeue > /dev/null; then
    squeue -u "$USER"
fi
