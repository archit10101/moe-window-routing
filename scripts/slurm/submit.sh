#!/bin/bash
# Submit one Slurm job that trains a group of runs on one GPU.
#
#   scripts/slurm/submit.sh <group-name> <gpu-type> <time> <run> [run ...]
#   e.g. scripts/slurm/submit.sh main-A b200 05:30:00 base-s0 win16-s0 base-s1 win16-s1
#
# Refuses to submit if a run's config is missing or a main run's LR is still None.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$(cd "$HERE/../.." && pwd)}"
PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs}"
DATA_DIR="${DATA_DIR:-$PROJECT/data}"
PYTHON="${PYTHON:-python3}"
SBATCH="${SBATCH:-sbatch}"

if [ "$#" -lt 4 ]; then
    echo "usage: $0 <group-name> <gpu-type: b200|l40s|h200> <time HH:MM:SS> <run> [run ...]" >&2
    exit 2
fi
group="$1"
gpu="$2"
time_limit="$3"
shift 3

case "$gpu" in
    b200|l40s|h200) ;;
    *) echo "unknown gpu type '$gpu' (expected b200, l40s or h200)" >&2; exit 2 ;;
esac

cd "$CODE_DIR" || exit 1
for run in "$@"; do
    if [ ! -f "config/runs/$run.py" ]; then
        echo "refusing to submit: config/runs/$run.py does not exist" >&2
        exit 1
    fi
done
# exec each config (stdlib only); main runs raise SystemExit while their LR is None
if ! "$PYTHON" - "$@" <<'EOF'
import sys
for run in sys.argv[1:]:
    ns = {}
    try:
        exec(open(f"config/runs/{run}.py").read(), ns)
    except SystemExit as e:
        sys.exit(f"refusing to submit: {e}")
    if ns.get("lr") is None:
        sys.exit(f"refusing to submit: {run} has no lr")
EOF
then
    exit 1
fi

k="$#"
cpus=$((4 * k)); [ "$cpus" -gt 16 ] && cpus=16
mem=$((48 * k)); [ "$mem" -gt 200 ] && mem=200

# optional: SBATCH_DEPENDENCY=afterok:<jobid> adds a dependency (unset: the command is unchanged)
dep=()
[ -n "${SBATCH_DEPENDENCY:-}" ] && dep=(--dependency="$SBATCH_DEPENDENCY")

mkdir -p "$OUT_ROOT/slurm"
set -x
"$SBATCH" -p mit_normal_gpu --gres="gpu:$gpu:1" -c "$cpus" --mem="${mem}G" -t "$time_limit" -J "$group" \
    --output="$OUT_ROOT/slurm/%x-%j.out" \
    --export="ALL,PROJECT=$PROJECT,CODE_DIR=$CODE_DIR,OUT_ROOT=$OUT_ROOT,DATA_DIR=$DATA_DIR" \
    ${dep[@]+"${dep[@]}"} "$HERE/run_group.sbatch" "$@"
