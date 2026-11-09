#!/bin/bash
# A1/A2 diagnostics on the existing checkpoints base-s0..s4 and win16-s0..s4 (docs/FOLLOWUP_PROMPT.md part 4):
# one 1-GPU job, 8 CPUs, 64 GB, 01:00:00, writing to $PROJECT/analysis/diagnostics/.
#
#   scripts/slurm/submit_diagnostics.sh <gpu-type>
#
# Token categories need tiktoken's GPT-2 files. Compute nodes may have no network, so the job
# reads them from TIKTOKEN_CACHE_DIR (default $PROJECT/tiktoken_cache); prime it once on a login node:
#   TIKTOKEN_CACHE_DIR=$PROJECT/tiktoken_cache python -c "import tiktoken; tiktoken.get_encoding('gpt2')"
# Inside the job the script calls itself with --in-job.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$HERE/../.." && pwd)}"
PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs}"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"
DIAG_OUT="${DIAG_OUT:-$PROJECT/analysis/diagnostics}"
TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$PROJECT/tiktoken_cache}"
SBATCH="${SBATCH:-sbatch}"
RUNS="base-s0,base-s1,base-s2,base-s3,base-s4,win16-s0,win16-s1,win16-s2,win16-s3,win16-s4"

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
    if [ -z "$(ls -A "$TIKTOKEN_CACHE_DIR" 2> /dev/null)" ]; then
        echo "WARNING: $TIKTOKEN_CACHE_DIR is empty; if compute nodes are offline the job will fail. Prime it with:" >&2
        echo "  TIKTOKEN_CACHE_DIR=$TIKTOKEN_CACHE_DIR python -c \"import tiktoken; tiktoken.get_encoding('gpt2')\"" >&2
    fi
    mkdir -p "$OUT_ROOT/slurm"
    "$SBATCH" -p mit_normal_gpu --gres="gpu:$gpu:1" -c 8 --mem=64G -t 01:00:00 -J diagnostics \
        --output="$OUT_ROOT/slurm/%x-%j.out" \
        --export="ALL,PROJECT=$PROJECT,CODE_DIR=$CODE_DIR,OUT_ROOT=$OUT_ROOT,DATA_DIR=$DATA_DIR,DIAG_OUT=$DIAG_OUT,TIKTOKEN_CACHE_DIR=$TIKTOKEN_CACHE_DIR" \
        "$HERE/submit_diagnostics.sh" --in-job
    echo "output: $DIAG_OUT/ (log: $OUT_ROOT/slurm/diagnostics-<jobid>.out)"
    exit 0
fi

# ---- inside the job ----
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moe
export OMP_NUM_THREADS=8
export TIKTOKEN_CACHE_DIR
cd "$CODE_DIR" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$DIAG_OUT"
python analysis/diagnostics.py --runs_dir "$OUT_ROOT" --data_dir "$DATA_DIR" --out_dir "$DIAG_OUT" \
    --runs "$RUNS" --device cuda
