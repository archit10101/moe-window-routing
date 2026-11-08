#!/bin/bash
# Start one launch_run.sh per run, on GPUs 0..k-1, in the background with nohup.
#   export S3_BUCKET=<bucket>; bash launch_instance.sh [--no-shutdown] <run> [run ...]
# Logs go to /home/ubuntu/launch_logs/<run>.log. The run list goes to /home/ubuntu/active_runs.txt.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
require_bucket

EXTRA=()
if [ "${1:-}" = "--no-shutdown" ]; then
    EXTRA=(--no-shutdown)
    shift
fi
if [ "$#" -lt 1 ]; then
    echo "usage: $0 [--no-shutdown] <run> [run ...]" >&2
    exit 2
fi

n_gpus=$(nvidia-smi --list-gpus 2> /dev/null | wc -l)
if [ "$#" -gt "$n_gpus" ]; then
    echo "$# runs but only $n_gpus GPUs on this instance" >&2
    exit 1
fi

printf '%s\n' "$@" > /home/ubuntu/active_runs.txt
mkdir -p /home/ubuntu/launch_logs
i=0
for run in "$@"; do
    nohup bash "$HERE/launch_run.sh" "$run" "$i" "${EXTRA[@]+"${EXTRA[@]}"}" \
        > "/home/ubuntu/launch_logs/$run.log" 2>&1 &
    echo "$run -> GPU $i (pid $!), log /home/ubuntu/launch_logs/$run.log"
    i=$((i + 1))
done
