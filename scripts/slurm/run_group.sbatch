#!/bin/bash
# Run a group of k training runs concurrently on the job's single GPU.
# Submit with scripts/slurm/submit.sh (it sets partition, gres, CPUs, memory, time, name, output).
#
#   sbatch [options] scripts/slurm/run_group.sbatch <run> [run ...]
#
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
set -uo pipefail

PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
CODE_DIR="${CODE_DIR:-$PROJECT/code}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs}"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"
MAX_REQUEUES=3

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch run_group.sbatch <run> [run ...]" >&2
    exit 2
fi

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moe
export OMP_NUM_THREADS=4

cd "$CODE_DIR" || { echo "cannot cd to $CODE_DIR" >&2; exit 1; }
mkdir -p "$OUT_ROOT/slurm"

echo "job ${SLURM_JOB_ID:-?} on $(hostname) at $(date): runs $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PIDS=()
NAMES=()
for run in "$@"; do
    run_dir="$OUT_ROOT/$run"
    if [ -f "$run_dir/done" ]; then
        echo "$run: already done, skipping"
        continue
    fi
    mkdir -p "$run_dir"
    echo "===== $(date) job ${SLURM_JOB_ID:-?} on $(hostname) =====" >> "$run_dir/stdout.log"
    python train.py "config/runs/$run.py" --out_root="$OUT_ROOT" --data_dir="$DATA_DIR" \
        >> "$run_dir/stdout.log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$run")
    echo "$run: started pid $!"
done

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "every run in this group is done"
    exit 0
fi

forward_signal() {
    echo "$(date): caught signal; sending SIGUSR1 to ${PIDS[*]}"
    for p in "${PIDS[@]}"; do
        kill -USR1 "$p" 2>/dev/null
    done
}
trap forward_signal USR1 TERM

# wait for every child; a trapped signal interrupts `wait`, so loop until the child is gone
for i in "${!PIDS[@]}"; do
    p="${PIDS[$i]}"
    while true; do
        wait "$p"
        rc=$?
        kill -0 "$p" 2>/dev/null || break
    done
    echo "${NAMES[$i]}: exit code $rc (99 = interrupted, resumable)"
done

missing=()
for run in "$@"; do
    [ -f "$OUT_ROOT/$run/done" ] || missing+=("$run")
done
if [ "${#missing[@]}" -gt 0 ]; then
    count_file="$OUT_ROOT/slurm/requeue_${SLURM_JOB_ID:-none}.count"
    n=$(cat "$count_file" 2>/dev/null || echo 0)
    if [ "$n" -lt "$MAX_REQUEUES" ]; then
        echo $((n + 1)) > "$count_file"
        echo "not done: ${missing[*]}; requeue $((n + 1)) of $MAX_REQUEUES"
        scontrol requeue "$SLURM_JOB_ID"
    else
        echo "not done: ${missing[*]}; requeue limit ($MAX_REQUEUES) reached, resubmit by hand"
    fi
else
    echo "all runs in this group are done"
fi
