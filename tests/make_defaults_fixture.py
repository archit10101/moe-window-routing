#!/usr/bin/env python
"""Capture the default-settings behavior of the code into tests/fixtures/defaults_reference.json.

The fixture was captured from the code that produced the pilot runs (commit 1dc8323), before
the follow-up changes (docs/FOLLOWUP_PROMPT.md). test_defaults_unchanged rebuilds everything
with capture() and requires bit-identical results, which proves that the new options leave
the default model, initialization, data order and training untouched.

Do NOT rerun this script on later code: that would overwrite the reference with whatever the
code does now. The comparison is exact (sha256 of tensors, float repr), so it is specific to
this machine's torch build and BLAS; the platform is recorded in the fixture.

Usage (repo root): .venv/bin/python tests/make_defaults_fixture.py
"""
import hashlib
import json
import os
import platform
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import make_synthetic_data  # noqa: E402
import train as T  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "defaults_reference.json")
TINY = dict(n_layer=2, n_embd=64, n_head=4, block_size=128, n_exp=4, top_k=2,
            expert_hidden=64, shared_hidden=128, vocab_size=1024)
ARMS = {"baseline": [0, 0], "window": [0, 4]}
STEPS = 10


def tensor_hashes(state_dict):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in state_dict.items()}


def capture(workdir):
    """Everything the defaults must reproduce exactly, as a JSON-serializable dict."""
    out = {"torch_version": torch.__version__, "platform": platform.platform(), "init": {}, "train": {}}
    torch.set_num_threads(4)
    for arm, windows in ARMS.items():
        torch.manual_seed(0)
        out["init"][arm] = tensor_hashes(GPT(GPTConfig(**TINY, window_layers=list(windows))).state_dict())

    data_dir = os.path.join(workdir, "data")
    make_synthetic_data.make(data_dir, seed=0)
    for arm, windows in ARMS.items():
        cfg = T.default_config()
        cfg.update(TINY)
        cfg.update(run_name=arm, data_dir=data_dir, out_root=os.path.join(workdir, "runs"),
                   window_layers=list(windows), lr=3e-3, batch_size=8, micro_batch_size=8,
                   total_tokens=STEPS * 8 * 128, log_interval=1, eval_interval=1000, eval_batches=1,
                   final_eval_batches=2, eval_batch_size=4, device="cpu", num_threads=4)
        assert T.train(cfg) == 0
        run_dir = os.path.join(workdir, "runs", arm)
        steps, final = [], None
        with open(os.path.join(run_dir, "log.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "train":
                    steps.append({k: r[k] for k in ("step", "ce", "aux", "z", "grad_norm", "lr",
                                                    "load_max", "load_min")})
                elif r["type"] == "final":
                    final = {"val": r["val"], "val_overall": r["val_overall"]}
        ckpt = torch.load(os.path.join(run_dir, "ckpt", "latest.pt"), map_location="cpu", weights_only=False)
        out["train"][arm] = {"steps": steps, "final": final, "final_weights": tensor_hashes(ckpt["model"])}
    return out


def main():
    if os.path.exists(FIXTURE) and "--force" not in sys.argv:
        sys.exit(f"{FIXTURE} exists; it must stay the pre-follow-up reference (pass --force only if you know why)")
    with tempfile.TemporaryDirectory() as d:
        ref = capture(d)
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump(ref, f, indent=1)
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
