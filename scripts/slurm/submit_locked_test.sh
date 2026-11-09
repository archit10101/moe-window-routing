#!/bin/bash
# The single evaluation of the locked M2D2 test split (docs/M2D2_PROMPT.md parts 2, 7, 8; M2D2_UNEVEN_PROMPT.md part 7):
# one 1-GPU job running analysis/eval_locked_test.py on the even (m2d2-), hist4x (m2d2h4-) and rep4x (m2d2r4-) runs.
# Asks for confirmation first. Run it only after every decision (LRs, analysis) is final.
#
#   scripts/slurm/submit_locked_test.sh <gpu-type>
#
# Inside the job the script calls itself with --in-job.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
CODE_DIR="${CODE_DIR:-$(cd "$HERE/../.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs}"
M2D2_DATA="${M2D2_DATA:-$PROJECT/data_m2d2w8}"
LOCKED_OUT="${LOCKED_OUT:-$PROJECT/analysis/m2d2_locked_test}"
LOCKED_LOG="${LOCKED_LOG:-$PROJECT/analysis/locked_test_log.txt}"
SBATCH="${SBATCH:-sbatch}"
# one invocation covers all three M2D2 designs (docs/M2D2_UNEVEN_PROMPT.md part 7): even s0-s4, hist4x and rep4x s0-s3
RUNS="m2d2-base-s0,m2d2-base-s1,m2d2-base-s2,m2d2-base-s3,m2d2-base-s4,m2d2-win16-s0,m2d2-win16-s1,m2d2-win16-s2,m2d2-win16-s3,m2d2-win16-s4"
for p in m2d2h4 m2d2r4; do
    for s in 0 1 2 3; do RUNS="$RUNS,$p-base-s$s,$p-win16-s$s"; done
done

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <gpu-type: b200|l40s|h200>" >&2
    exit 2
fi

if [ "$1" = "--in-job" ]; then
    module load miniforge
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate moe
    cd "$CODE_DIR" || exit 1
    # each run reads its own build's test files, which must be byte-identical to the even build's ($M2D2_DATA)
    python analysis/eval_locked_test.py --runs_dir "$OUT_ROOT" --reference_data_dir "$M2D2_DATA" \
        --out_dir "$LOCKED_OUT" --runs "$RUNS" --device cuda --log_file "$LOCKED_LOG"
    exit $?
fi

gpu="$1"
case "$gpu" in
    b200|l40s|h200) ;;
    *) echo "unknown gpu type '$gpu'" >&2; exit 2 ;;
esac
if grep -q " | COMPLETED | " "$LOCKED_LOG" 2> /dev/null; then
    echo "NOTE: $LOCKED_LOG already records a completed locked-test evaluation; the job would refuse to run." >&2
fi
echo "Runs: ${RUNS//,/ }"
printf '%s' "the locked test is used once; continue? [y/N] " >&2  # shown even when stdin is not a terminal
read -r answer || answer=""
case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "not submitted"; exit 1 ;;
esac
mkdir -p "$OUT_ROOT/slurm"
"$SBATCH" -p mit_normal_gpu --gres="gpu:$gpu:1" -c 8 --mem=64G -t 00:30:00 -J locked-test --no-requeue \
    --output="$OUT_ROOT/slurm/%x-%j.out" \
    --export="ALL,PROJECT=$PROJECT,CODE_DIR=$CODE_DIR,OUT_ROOT=$OUT_ROOT,M2D2_DATA=$M2D2_DATA,LOCKED_OUT=$LOCKED_OUT,LOCKED_LOG=$LOCKED_LOG" \
    "$HERE/submit_locked_test.sh" --in-job
echo "results: $LOCKED_OUT/summary.md and <run>/test_results.json; log: $LOCKED_LOG"
