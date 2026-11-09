#!/bin/bash
# Follow-up runs (docs/FOLLOWUP_PROMPT.md part 6): one 1-run job per run, 01:30:00 each,
# in priority order E5, E3, E2, E4. With group names, submit only those (still in priority order).
#
#   scripts/slurm/submit_followups.sh <gpu-type> [group ...]
#   e.g. scripts/slurm/submit_followups.sh b200            # all 16 runs
#        scripts/slurm/submit_followups.sh b200 E5 E3      # 6 runs
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIME_LIMIT="01:30:00"
ORDER=(E5 E3 E2 E4)

usage() {
    echo "usage: $0 <gpu-type: b200|l40s|h200> [group ...]   groups: ${ORDER[*]}" >&2
    exit 2
}

# runs of a group; within a group, variant and comparison runs alternate so pairs finish together
runs_for() {
    case "$1" in
        E5) echo "both16-s0 both16-s1" ;;
        E3) echo "win16top1-s0 win16top1-s1 win16top2-s0 win16top2-s1" ;;
        E2) echo "base-sh2048-s0 win16-sh2048-s0 base-sh2048-s1 win16-sh2048-s1" ;;
        E4) echo "base-rlr3-s0 win16-rlr3-s0 base-rlr3-s1 win16-rlr3-s1 base-rg-s0 win16-rg-s0" ;;
        *) return 1 ;;
    esac
}

[ "$#" -ge 1 ] || usage
gpu="$1"
shift
for g in "$@"; do
    runs_for "$g" > /dev/null || { echo "unknown group '$g'" >&2; usage; }
done

n=0
for g in "${ORDER[@]}"; do
    want=0
    if [ "$#" -eq 0 ]; then
        want=1
    else
        for req in "$@"; do [ "$req" = "$g" ] && want=1; done
    fi
    [ "$want" -eq 1 ] || continue
    for run in $(runs_for "$g"); do
        "$HERE/submit.sh" "$run" "$gpu" "$TIME_LIMIT" "$run" || exit 1
        n=$((n + 1))
    done
done
echo "submitted $n follow-up job(s); at most 2 run at once (per-user GPU limit), the rest queue in this order"
