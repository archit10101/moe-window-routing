#!/bin/bash
# The CPU job submitted by submit_m2d2_all.sh (docs/M2D2_UNEVEN_PROMPT.md part 8), after the 6 LR runs end:
#
#   after_m2d2_all.sh <gpu-type> <hist4x-build-jobid|none> <rep4x-build-jobid|none> [--dry_run]
#
# 1. analyze.py --prefix m2d2- --lr_check --write --lr_file config/chosen_lr_m2d2.py, then prints the LRs.
#    If this fails, nothing is submitted and the script exits non-zero.
# 2. Submits 1-run GPU jobs so every design gets early seeds first:
#    m2d2-base-s0, m2d2-win1-s0, m2d2-win16-s0, m2d2h4-base/win16-s0, m2d2r4-base/win16-s0,
#    then for s = 1, 2, 3 the m2d2, m2d2h4 and m2d2r4 pairs, then the m2d2 s4 pair, then m2d2-win64-s0.
#    Even runs 00:40:00, uneven runs 00:50:00. An uneven run depends (afterok) on its build job while that
#    job is still queued or running; if the build already passed its checks there is no dependency; if it
#    failed, that design's runs are skipped and the script exits non-zero after submitting the rest.
# --dry_run prints the submissions without the LR analysis or any sbatch call.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/m2d2_common.sh"
PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
CODE_DIR="${CODE_DIR:-$(cd "$HERE/../.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs}"
SUBMIT="${SUBMIT:-$HERE/submit.sh}"
SQUEUE="${SQUEUE:-squeue}"

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <gpu-type> <hist4x-build-jobid|none> <rep4x-build-jobid|none> [--dry_run]" >&2
    exit 2
fi
gpu="$1"
h4_build="$2"
r4_build="$3"
dry=0
[ "${4:-}" = "--dry_run" ] && dry=1

ORDER=(m2d2-base-s0 m2d2-win1-s0 m2d2-win16-s0 m2d2h4-base-s0 m2d2h4-win16-s0 m2d2r4-base-s0 m2d2r4-win16-s0)
for s in 1 2 3; do
    ORDER+=("m2d2-base-s$s" "m2d2-win16-s$s" "m2d2h4-base-s$s" "m2d2h4-win16-s$s" "m2d2r4-base-s$s" "m2d2r4-win16-s$s")
done
ORDER+=(m2d2-base-s4 m2d2-win16-s4 m2d2-win64-s0)

if [ "$dry" -eq 0 ]; then
    module load miniforge
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate moe
    cd "$CODE_DIR" || exit 1
    lr_file="$CODE_DIR/config/chosen_lr_m2d2.py"
    echo "$(date): M2D2 LR check analysis"
    if ! python analysis/analyze.py --runs_dir "$OUT_ROOT" --out_dir "$PROJECT/analysis/m2d2_lr_check" \
            --prefix m2d2- --lr_check --write --lr_file "$lr_file"; then
        echo "LR-check analysis failed; submitting no runs" >&2
        exit 1
    fi
    echo "chosen learning rates ($lr_file):"
    grep -E "^LR_(BASE|WIN) = " "$lr_file"
fi

dependency_for() {  # dependency_for <build-jobid|none> <folder>: prints the dependency, "" or FAILED
    local id="$1" dir="$2"
    if [ "$dry" -eq 1 ]; then
        [ "$id" != none ] && echo "afterok:$id"
        return 0
    fi
    if build_ok "$dir" uneven; then
        echo ""
    elif [ "$id" != none ] && [ -n "$("$SQUEUE" -h -j "$id" -o %T 2> /dev/null)" ]; then
        echo "afterok:$id"
    else
        echo FAILED
    fi
}
h4_dep="$(dependency_for "$h4_build" "$PROJECT/data_m2d2w8_hist4x")"
r4_dep="$(dependency_for "$r4_build" "$PROJECT/data_m2d2w8_rep4x")"
for pair in "hist4x:$h4_dep" "rep4x:$r4_dep"; do
    [ "${pair#*:}" = FAILED ] && echo "WARNING: the ${pair%%:*} build failed or never ran; its runs are skipped" >&2
done

failed=0
n=0
for run in "${ORDER[@]}"; do
    case "$run" in
        m2d2h4-*) dep="$h4_dep"; t=00:50:00 ;;
        m2d2r4-*) dep="$r4_dep"; t=00:50:00 ;;
        *) dep=""; t=00:40:00 ;;
    esac
    if [ "$dep" = FAILED ]; then
        echo "skipping $run (its build did not succeed)"
        failed=1
        continue
    fi
    if [ "$dry" -eq 1 ]; then
        echo "DRY RUN: submit.sh $run $gpu $t $run${dep:+ (--dependency=$dep)}"
    else
        SBATCH_DEPENDENCY="$dep" "$SUBMIT" "$run" "$gpu" "$t" "$run" || { echo "submitting $run failed" >&2; exit 1; }
    fi
    n=$((n + 1))
done
echo "$( [ "$dry" -eq 1 ] && echo "would submit" || echo "submitted" ) $n runs"
exit "$failed"
