#!/bin/bash
# Shared by submit_m2d2_all.sh, after_m2d2_all.sh and status.sh. Source it: source scripts/slurm/m2d2_common.sh
set -uo pipefail

# the 8 top-level M2D2 configs (must match DOMAINS in data/m2d2/prepare.py; a test checks this)
M2D2_CONFIGS=(Technology_and_applied_sciences Natural_and_physical_sciences Mathematics_and_logic Health_and_fitness
              History_and_events Culture_and_the_arts Society_and_social_sciences Philosophy_and_thinking)
# the lines data/m2d2/prepare.py writes into sanity.txt when its checks pass
SANITY_OK="OK: none of the"
REFERENCE_OK="OK: all test/val/calib files are byte-identical to the reference build"

raw_ready() {  # every config extracted with train/valid/test.txt
    local raw="$1" c f
    for c in "${M2D2_CONFIGS[@]}"; do
        for f in train.txt valid.txt test.txt; do
            [ -f "$raw/$c/$f" ] || return 1
        done
    done
}

build_ok() {  # build_ok <folder> [uneven]: meta.json present and sanity.txt reports the checks passed
    local dir="$1" uneven="${2:-}"
    [ -f "$dir/meta.json" ] && [ -f "$dir/sanity.txt" ] || return 1
    grep -q "^$SANITY_OK" "$dir/sanity.txt" || return 1
    if [ -n "$uneven" ]; then
        grep -q "^$REFERENCE_OK" "$dir/sanity.txt" || return 1
    fi
}

job_id() {  # the job id from sbatch's "Submitted batch job N" line on stdin
    sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1
}
