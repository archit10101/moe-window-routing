#!/bin/bash
# M2D2-Wiki-8 LR check, chained into the main runs (docs/M2D2_PROMPT.md part 8).
#
#   scripts/slurm/submit_m2d2_lr_check.sh <gpu-type>
#
# 1. Submits the 6 m2d2-lr-* runs (one run per job, 00:20:00; evaluated on the calibration split).
# 2. Submits a dependent CPU job (mit_normal, no requeue, afterany) that runs this script with --chain:
#    analyze.py --prefix m2d2- --lr_check --write --lr_file config/chosen_lr_m2d2.py, prints the chosen LRs,
#    then submits the 12 M2D2 main runs (00:40:00 each) in this order:
#      m2d2-base-s0, m2d2-win1-s0, m2d2-win16-s0, base/win16 pairs s1..s4, m2d2-win64-s0.
#    If the analysis fails it submits nothing and exits non-zero.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
CODE_DIR="${CODE_DIR:-$(cd "$HERE/../.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs}"
SBATCH="${SBATCH:-sbatch}"
SUBMIT="${SUBMIT:-$HERE/submit.sh}"
LR_RUNS=(m2d2-lr-base-2.5e-4 m2d2-lr-base-5e-4 m2d2-lr-base-1e-3 m2d2-lr-win16-2.5e-4 m2d2-lr-win16-5e-4 m2d2-lr-win16-1e-3)
MAIN_RUNS=(m2d2-base-s0 m2d2-win1-s0 m2d2-win16-s0
           m2d2-base-s1 m2d2-win16-s1 m2d2-base-s2 m2d2-win16-s2 m2d2-base-s3 m2d2-win16-s3 m2d2-base-s4 m2d2-win16-s4
           m2d2-win64-s0)

usage() {
    echo "usage: $0 <gpu-type: b200|l40s|h200>" >&2
    exit 2
}

check_gpu() {
    case "$1" in
        b200|l40s|h200) ;;
        *) echo "unknown gpu type '$1'" >&2; usage ;;
    esac
}

job_id() {  # the job id from sbatch's "Submitted batch job N" line
    sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1
}

[ "$#" -ge 1 ] || usage

if [ "$1" = "--chain" ]; then
    # ---- inside the dependent CPU job ----
    [ "$#" -ge 2 ] || usage
    gpu="$2"
    check_gpu "$gpu"
    module load miniforge
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate moe
    cd "$CODE_DIR" || exit 1
    lr_file="$CODE_DIR/config/chosen_lr_m2d2.py"
    echo "$(date): M2D2 LR check analysis"
    if ! python analysis/analyze.py --runs_dir "$OUT_ROOT" --out_dir "$PROJECT/analysis/m2d2_lr_check" \
            --prefix m2d2- --lr_check --write --lr_file "$lr_file"; then
        echo "LR-check analysis failed; submitting no main runs" >&2
        exit 1
    fi
    echo "chosen learning rates ($lr_file):"
    grep -E "^LR_(BASE|WIN) = " "$lr_file"
    for run in "${MAIN_RUNS[@]}"; do
        "$SUBMIT" "$run" "$gpu" 00:40:00 "$run" || { echo "submitting $run failed" >&2; exit 1; }
    done
    echo "submitted ${#MAIN_RUNS[@]} M2D2 main runs"
    exit 0
fi

# ---- on the login node ----
gpu="$1"
check_gpu "$gpu"
ids=()
for run in "${LR_RUNS[@]}"; do
    out="$("$SUBMIT" "$run" "$gpu" 00:20:00 "$run")" || { echo "submitting $run failed" >&2; exit 1; }
    echo "$out"
    id="$(printf '%s\n' "$out" | job_id)"
    [ -n "$id" ] || { echo "could not read the job id for $run" >&2; exit 1; }
    ids+=("$id")
done
dep="afterany$(printf ':%s' "${ids[@]}")"
mkdir -p "$OUT_ROOT/slurm"
"$SBATCH" -p mit_normal --no-requeue -c 2 --mem=8G -t 00:30:00 -J m2d2-chain --dependency="$dep" \
    --output="$OUT_ROOT/slurm/%x-%j.out" \
    --export="ALL,PROJECT=$PROJECT,CODE_DIR=$CODE_DIR,OUT_ROOT=$OUT_ROOT" \
    "$HERE/submit_m2d2_lr_check.sh" --chain "$gpu" || { echo "submitting the chain job failed" >&2; exit 1; }
echo "LR runs: ${ids[*]}; the chain job ($dep) writes config/chosen_lr_m2d2.py and submits the main runs"
