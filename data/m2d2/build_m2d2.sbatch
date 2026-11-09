#!/bin/bash
# Build M2D2-Wiki-8 from the extracted archives (CPU only, no network).
#
#   cd $PROJECT/code
#   sbatch --output=$PROJECT/runs/slurm/%x-%j.out data/m2d2/build_m2d2.sbatch <doc_mode> [block_lines]
#
# <doc_mode> is the one recommended by `prepare.py --inspect` (line, blank, heading or block).
#SBATCH -p mit_normal
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J build-m2d2
#SBATCH --no-requeue
set -uo pipefail

PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
CODE_DIR="${CODE_DIR:-$PROJECT/code}"
RAW_DIR="${RAW_DIR:-$PROJECT/raw_m2d2}"
OUT_DIR="${OUT_DIR:-$PROJECT/data_m2d2w8}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$PROJECT/tiktoken_cache}"

if [ "$#" -lt 1 ]; then
    echo "usage: sbatch build_m2d2.sbatch <doc_mode: line|blank|heading|block> [block_lines]" >&2
    exit 2
fi
doc_mode="$1"
block_lines="${2:-40}"
case "$doc_mode" in
    line|blank|heading|block) ;;
    *) echo "unknown doc_mode '$doc_mode'" >&2; exit 2 ;;
esac

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moe
cd "$CODE_DIR" || exit 1

echo "$(date): building M2D2-Wiki-8 from $RAW_DIR into $OUT_DIR (doc_mode $doc_mode, block_lines $block_lines)"
python data/m2d2/prepare.py --build --raw_dir "$RAW_DIR" --out_dir "$OUT_DIR" \
    --doc_mode "$doc_mode" --block_lines "$block_lines" --threads "${SLURM_CPUS_PER_TASK:-16}"
rc=$?
echo "$(date): exit code $rc"
exit "$rc"
