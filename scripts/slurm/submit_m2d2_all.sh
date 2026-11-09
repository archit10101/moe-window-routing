#!/bin/bash
# Everything for the M2D2 experiments in one go (docs/M2D2_UNEVEN_PROMPT.md part 8):
#
#   scripts/slurm/submit_m2d2_all.sh <gpu-type> [--dry_run] [--doc_mode MODE]
#
# 1. Builds: even (data_m2d2w8), hist4x and rep4x (data_m2d2w8_hist4x / _rep4x). A build is skipped when its
#    folder already passed its checks; uneven builds compare their eval files with the even build and depend
#    on it (afterok) when it is submitted too. All builds use the same --doc_mode (default: title).
# 2. The 6 even-design LR-check runs (afterok on the even build, if submitted).
# 3. after_m2d2_all.sh as a CPU job (afterany on the LR runs): writes config/chosen_lr_m2d2.py, then submits
#    the even and uneven main runs (uneven runs afterok on their builds).
# 4. The 6 slim120 runs (submit_slim120.sh).
# --dry_run prints every sbatch command (and the runs the CPU job will submit) and submits nothing.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/m2d2_common.sh"
PROJECT="${PROJECT:-/home/architta/orcd/scratch/moe-pilot}"
CODE_DIR="${CODE_DIR:-$(cd "$HERE/../.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/runs}"
RAW_DIR="${RAW_DIR:-$PROJECT/raw_m2d2}"
EVEN_DIR="$PROJECT/data_m2d2w8"
H4_DIR="$PROJECT/data_m2d2w8_hist4x"
R4_DIR="$PROJECT/data_m2d2w8_rep4x"
LR_RUNS=(m2d2-lr-base-2.5e-4 m2d2-lr-base-5e-4 m2d2-lr-base-1e-3 m2d2-lr-win16-2.5e-4 m2d2-lr-win16-5e-4 m2d2-lr-win16-1e-3)

usage() {
    echo "usage: $0 <gpu-type: b200|l40s|h200> [--dry_run] [--doc_mode line|blank|heading|block|title]" >&2
    exit 2
}

[ "$#" -ge 1 ] || usage
gpu="$1"
shift
case "$gpu" in b200|l40s|h200) ;; *) echo "unknown gpu type '$gpu'" >&2; usage ;; esac
dry=0
doc_mode="title"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry_run) dry=1 ;;
        --doc_mode) [ "$#" -ge 2 ] || usage; doc_mode="$2"; shift ;;
        *) usage ;;
    esac
    shift
done
case "$doc_mode" in line|blank|heading|block|title) ;; *) usage ;; esac

if [ "$dry" -eq 1 ]; then
    # a stand-in sbatch that prints the command and returns a placeholder job id
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    echo 900 > "$tmp/next"
    cat > "$tmp/sbatch" <<EOF
#!/bin/bash
n=\$(cat "$tmp/next"); echo \$((n + 1)) > "$tmp/next"
echo "DRY RUN: sbatch \$*"
echo "Submitted batch job \$n"
EOF
    chmod +x "$tmp/sbatch"
    export SBATCH="$tmp/sbatch"
    echo "DRY RUN: nothing is submitted; job ids below are placeholders"
fi
SBATCH="${SBATCH:-sbatch}"
export SBATCH

# ---- 0. raw archives ----
if ! raw_ready "$RAW_DIR"; then
    echo "The M2D2 archives are not extracted in $RAW_DIR. On a login node (it needs the network), run:" >&2
    echo "  python data/m2d2/prepare.py --download --raw_dir $RAW_DIR" >&2
    echo "(see README_M2D2.md, step 2), then rerun this script." >&2
    exit 1
fi

submit() {  # submit <sbatch args...>: prints sbatch's output, sets $last_id
    local out
    out="$("$SBATCH" "$@")" || { echo "sbatch failed: $*" >&2; exit 1; }
    echo "$out"
    last_id="$(printf '%s\n' "$out" | job_id)"
    [ -n "$last_id" ] || { echo "could not read a job id from: $out" >&2; exit 1; }
}

# ---- 1. builds ----
mkdir -p "$OUT_ROOT/slurm"
build_job() {  # build_job <label> <folder> <dependency or ""> <build args...>
    local label="$1" dir="$2" dep="$3"
    shift 3
    local depopt=()
    [ -n "$dep" ] && depopt=(--dependency="$dep")
    submit -p mit_normal -c 16 --mem=64G -t 03:00:00 --no-requeue -J "build-m2d2-$label" \
        --output="$OUT_ROOT/slurm/%x-%j.out" ${depopt[@]+"${depopt[@]}"} \
        --export="ALL,PROJECT=$PROJECT,CODE_DIR=$CODE_DIR,RAW_DIR=$RAW_DIR,OUT_DIR=$dir" \
        "$CODE_DIR/data/m2d2/build_m2d2.sbatch" "$doc_mode" "$@"
}
check_folder() {  # refuse to submit a build into a folder that holds a failed or partial build
    local dir="$1" uneven="${2:-}"
    if [ -e "$dir/meta.json" ] && ! build_ok "$dir" "$uneven"; then
        echo "$dir holds a build whose sanity checks did not pass; inspect it, move it away, then rerun." >&2
        exit 1
    fi
}

even_id="" h4_id="none" r4_id="none"
if build_ok "$EVEN_DIR"; then
    echo "even build: $EVEN_DIR passed its checks; skipping"
else
    check_folder "$EVEN_DIR"
    build_job even "$EVEN_DIR" ""
    even_id="$last_id"
fi
even_dep=""
[ -n "$even_id" ] && even_dep="afterok:$even_id"
if build_ok "$H4_DIR" uneven; then
    echo "hist4x build: $H4_DIR passed its checks; skipping"
else
    check_folder "$H4_DIR" uneven
    build_job hist4x "$H4_DIR" "$even_dep" --train_quota_override history=60000000 --variant hist4x \
        --reference_dir "$EVEN_DIR"
    h4_id="$last_id"
fi
if build_ok "$R4_DIR" uneven; then
    echo "rep4x build: $R4_DIR passed its checks; skipping"
else
    check_folder "$R4_DIR" uneven
    build_job rep4x "$R4_DIR" "$even_dep" --train_quota_override philosophy=60000000 \
        --fallback_domain natural_sciences --variant rep4x --reference_dir "$EVEN_DIR"
    r4_id="$last_id"
fi

# ---- 2. LR check ----
lr_ids=()
for run in "${LR_RUNS[@]}"; do
    out="$(SBATCH_DEPENDENCY="$even_dep" "$HERE/submit.sh" "$run" "$gpu" 00:20:00 "$run")" \
        || { echo "submitting $run failed" >&2; exit 1; }
    echo "$out"
    id="$(printf '%s\n' "$out" | job_id)"
    [ -n "$id" ] || { echo "could not read the job id for $run" >&2; exit 1; }
    lr_ids+=("$id")
done
lr_dep="afterany$(printf ':%s' "${lr_ids[@]}")"

# ---- 3. the CPU job that picks the LRs and submits the main runs ----
submit -p mit_normal --no-requeue -c 2 --mem=8G -t 00:30:00 -J m2d2-after --dependency="$lr_dep" \
    --output="$OUT_ROOT/slurm/%x-%j.out" --export="ALL,PROJECT=$PROJECT,CODE_DIR=$CODE_DIR,OUT_ROOT=$OUT_ROOT" \
    "$HERE/after_m2d2_all.sh" "$gpu" "$h4_id" "$r4_id"
after_id="$last_id"

# ---- 4. slim120 ----
slim_out="$("$HERE/submit_slim120.sh" "$gpu")" || { echo "submitting slim120 failed" >&2; exit 1; }
echo "$slim_out"
slim_ids="$(printf '%s\n' "$slim_out" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tr '\n' ' ')"

# ---- 5. summary ----
echo
echo "===== submitted ====="
echo "builds:  even ${even_id:-skipped}  hist4x ${h4_id/none/skipped}  rep4x ${r4_id/none/skipped}"
echo "LR runs: ${lr_ids[*]}"
echo "after:   $after_id (CPU; writes config/chosen_lr_m2d2.py, then submits the main runs)"
echo "slim120: $slim_ids"
echo
h4_note="(build done)"; [ "$h4_id" != none ] && h4_note="afterok:$h4_id"
r4_note="(build done)"; [ "$r4_id" != none ] && r4_note="afterok:$r4_id"
echo "dependency tree:"
echo "  build even ${even_id:-(skipped: done)}"
echo "  ├─ build hist4x ${h4_id/none/(skipped: done)}  ${even_dep}"
echo "  ├─ build rep4x  ${r4_id/none/(skipped: done)}  ${even_dep}"
echo "  └─ LR runs ${lr_ids[*]}  ${even_dep}"
echo "       └─ after $after_id  $lr_dep"
echo "            ├─ even runs m2d2-* (12)"
echo "            ├─ hist4x runs m2d2h4-* (8)  $h4_note"
echo "            └─ rep4x runs m2d2r4-* (8)  $r4_note"
echo "  slim120 $slim_ids (independent)"

if [ "$dry" -eq 1 ]; then
    echo
    echo "===== the after job will then submit (in this order) ====="
    "$HERE/after_m2d2_all.sh" "$gpu" "$h4_id" "$r4_id" --dry_run
fi
