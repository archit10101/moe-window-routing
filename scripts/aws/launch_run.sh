#!/bin/bash
# Train one run on one GPU, syncing the run folder to S3 as it goes.
#   export S3_BUCKET=<bucket>; bash launch_run.sh <run_name> <gpu_index> [--no-shutdown]
#
# - retries with resume up to 3 times on a non-zero exit
# - syncs to s3://$S3_BUCKET/runs/<run_name>/ every 5 min (without ckpt/), with ckpt/ every 15 min
# - final full sync, then checks that done exists in S3
# - unless --no-shutdown: shuts the instance down once every run in active_runs.txt has done
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
require_bucket

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <run_name> <gpu_index> [--no-shutdown]" >&2
    exit 2
fi
RUN="$1"
GPU="$2"
SHUTDOWN=1
[ "${3:-}" = "--no-shutdown" ] && SHUTDOWN=0
ACTIVE=/home/ubuntu/active_runs.txt
MAX_RETRIES=3

activate_pytorch || exit 1
cd "$CODE" || exit 1
[ -f "config/runs/$RUN.py" ] || { echo "config/runs/$RUN.py not found" >&2; exit 1; }

RUN_DIR="$RUNS/$RUN"
DEST="s3://$S3_BUCKET/runs/$RUN/"
# continue a run started on another instance
if [ ! -d "$RUN_DIR" ] && aws s3 ls "$DEST" > /dev/null 2>&1; then
    echo "pulling existing run folder from $DEST"
    aws s3 sync "$DEST" "$RUN_DIR/" --only-show-errors
fi
mkdir -p "$RUN_DIR"

sync_light() { aws s3 sync "$RUN_DIR/" "$DEST" --exclude "ckpt/*" --only-show-errors; }
sync_full() { aws s3 sync "$RUN_DIR/" "$DEST" --only-show-errors; }

sync_loop() {
    local i=0
    while true; do
        sleep 300
        i=$((i + 1))
        if [ $((i % 3)) -eq 0 ]; then sync_full; else sync_light; fi
    done
}
sync_loop &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2> /dev/null' EXIT

attempt=0
while true; do
    echo "===== $(date) attempt $attempt on GPU $GPU =====" >> "$RUN_DIR/stdout.log"
    CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 python train.py "config/runs/$RUN.py" \
        --data_dir="$DATA" --out_root="$RUNS" --resume=True >> "$RUN_DIR/stdout.log" 2>&1
    rc=$?
    echo "$(date): $RUN exited with code $rc"
    [ "$rc" -eq 0 ] && break
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_RETRIES" ]; then
        echo "$RUN: giving up after $MAX_RETRIES retries; see $RUN_DIR/stdout.log" >&2
        break
    fi
    echo "$RUN: retrying with resume ($attempt of $MAX_RETRIES)"
    sleep 10
done

kill "$SYNC_PID" 2> /dev/null
sync_full
if aws s3 ls "${DEST}done" > /dev/null 2>&1; then
    echo "$RUN: done is in S3"
else
    echo "$RUN: WARNING: ${DEST}done not found in S3" >&2
fi

if [ "$SHUTDOWN" -eq 1 ] && [ -f "$ACTIVE" ]; then
    all_done=1
    while read -r r; do
        [ -z "$r" ] && continue
        [ -f "$RUNS/$r/done" ] || all_done=0
    done < "$ACTIVE"
    if [ "$all_done" -eq 1 ]; then
        echo "every run in $ACTIVE is done; shutting down"
        sudo shutdown -h now
    fi
fi
