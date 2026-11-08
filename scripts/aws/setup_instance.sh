#!/bin/bash
# One-time setup on a fresh g6e instance (Deep Learning AMI with PyTorch).
#   export S3_BUCKET=<bucket>; bash setup_instance.sh
# Activates the AMI's PyTorch env, installs tiktoken/pyarrow/matplotlib, clones or pulls
# the repo into /home/ubuntu/code, syncs s3://$S3_BUCKET/data/ to /home/ubuntu/data.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
require_bucket
activate_pytorch || exit 1

pip install --quiet tiktoken pyarrow matplotlib pytest || { echo "pip install failed" >&2; exit 1; }

if [ -d "$CODE/.git" ]; then
    git -C "$CODE" pull --ff-only || { echo "git pull failed in $CODE" >&2; exit 1; }
else
    git clone "$REPO_URL" "$CODE" || { echo "git clone failed" >&2; exit 1; }
fi

mkdir -p "$DATA" "$RUNS"
aws s3 sync "s3://$S3_BUCKET/data/" "$DATA/" --only-show-errors || { echo "data sync failed" >&2; exit 1; }
ls -la "$DATA"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
nvidia-smi
