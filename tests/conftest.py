"""Shared helpers for the CPU test suite (tiny model, synthetic data)."""
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from model import GPT, GPTConfig  # noqa: E402
import make_synthetic_data  # noqa: E402

torch.set_num_threads(4)

# Tiny test config from PROMPT.md part 6. Synthetic tokens are < 1000, so a
# 1024-entry vocabulary is enough and keeps the tests fast.
TINY = dict(n_layer=2, n_embd=64, n_head=4, block_size=128, n_exp=4, top_k=2,
            expert_hidden=64, shared_hidden=128, vocab_size=1024)
WINDOW = [0, 4]
BASELINE = [0, 0]


def tiny_model(window_layers=BASELINE, seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(**TINY, window_layers=list(window_layers)))


@pytest.fixture
def synth_dir(tmp_path):
    out = tmp_path / "data"
    make_synthetic_data.make(str(out))
    return str(out)
