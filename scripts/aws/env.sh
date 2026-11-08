#!/bin/bash
# Shared helpers for the AWS scripts. Source it: source scripts/aws/env.sh
set -uo pipefail

require_bucket() {
    if [ -z "${S3_BUCKET:-}" ]; then
        echo "S3_BUCKET is not set. Create a bucket in us-east-2, then: export S3_BUCKET=<bucket-name>" >&2
        exit 1
    fi
}

# Activate the Deep Learning AMI's PyTorch environment (venv in newer AMIs, conda in older ones).
activate_pytorch() {
    if [ -f /opt/pytorch/bin/activate ]; then
        source /opt/pytorch/bin/activate
    else
        local found=""
        for base in /opt/conda /home/ubuntu/anaconda3 /home/ubuntu/miniconda3; do
            if [ -f "$base/etc/profile.d/conda.sh" ]; then
                source "$base/etc/profile.d/conda.sh"
                for env in pytorch pytorch_p310 pytorch_p39; do
                    if conda activate "$env" 2>/dev/null; then found="$base:$env"; break 2; fi
                done
            fi
        done
        if [ -z "$found" ]; then
            echo "Could not find the Deep Learning AMI's PyTorch environment." >&2
            echo "Look in /opt (e.g. ls /opt) or the AMI's README (cat /home/ubuntu/README), activate it by hand," >&2
            echo "and check: python -c 'import torch; print(torch.cuda.is_available())'" >&2
            return 1
        fi
    fi
    python -c "import torch" 2>/dev/null || { echo "python in this environment has no torch" >&2; return 1; }
}

CODE=/home/ubuntu/code
DATA=/home/ubuntu/data
RUNS=/home/ubuntu/runs
REPO_URL="${REPO_URL:-https://github.com/archit10101/moe-window-routing.git}"
