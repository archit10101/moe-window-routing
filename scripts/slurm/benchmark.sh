#!/bin/bash
# Throughput and memory check: smoke-gpu-base with benchmark_steps=30, first alone,
# then 4 copies at once on the same GPU. Prints tokens/s and peak memory for each case.
#
#   scripts/slurm/benchmark.sh <gpu-type>      submits the job (results in <out_root>/slurm/benchmark-<jobid>.out)
#
# Inside the job the script calls itself with --in-job.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$HERE/../.." && pwd)}"
PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs}"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <gpu-type: b200|l40s|h200>" >&2
    exit 2
fi

if [ "$1" != "--in-job" ]; then
    gpu="$1"
    case "$gpu" in
        b200|l40s|h200) ;;
        *) echo "unknown gpu type '$gpu'" >&2; exit 2 ;;
    esac
    mkdir -p "$OUT_ROOT/slurm"
    sbatch -p mit_normal_gpu --gres="gpu:$gpu:1" -c 16 --mem=200G -t 00:45:00 -J benchmark \
        --output="$OUT_ROOT/slurm/%x-%j.out" \
        --export="ALL,PROJECT=$PROJECT,CODE_DIR=$CODE_DIR,OUT_ROOT=$OUT_ROOT,DATA_DIR=$DATA_DIR" \
        "$HERE/benchmark.sh" --in-job
    echo "when it finishes: grep BENCHMARK $OUT_ROOT/slurm/benchmark-*.out"
    exit 0
fi

# ---- inside the job ----
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moe
export OMP_NUM_THREADS=4
cd "$CODE_DIR" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CMD=(python train.py config/runs/smoke-gpu-base.py --benchmark_steps=30 --data_dir="$DATA_DIR" --out_root="$OUT_ROOT")
LOG_DIR="$(mktemp -d)"

echo "===== case 1: one process alone ====="
"${CMD[@]}" > "$LOG_DIR/alone.log" 2>&1
echo "exit code $?"
grep BENCHMARK "$LOG_DIR/alone.log" || tail -20 "$LOG_DIR/alone.log"

echo "===== case 2: four processes at once on the same GPU ====="
pids=()
for i in 1 2 3 4; do
    "${CMD[@]}" > "$LOG_DIR/four_$i.log" 2>&1 &
    pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done
for i in 1 2 3 4; do
    echo "copy $i: $(grep BENCHMARK "$LOG_DIR/four_$i.log" || echo 'FAILED (see below)')"
    grep -q BENCHMARK "$LOG_DIR/four_$i.log" || tail -20 "$LOG_DIR/four_$i.log"
done
echo "Compare 4 x (per-copy tok/s) with the single-process tok/s; peak memory is per process."
