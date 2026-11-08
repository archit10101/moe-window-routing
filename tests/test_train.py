"""Training-loop tests from PROMPT.md part 6."""
import json
import math
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import train as T
from conftest import BASELINE, REPO, TINY, WINDOW, tiny_model


def tiny_cfg(data_dir, out_root, run_name, window_layers=WINDOW, steps=20, **over):
    cfg = T.default_config()
    cfg.update(TINY)
    cfg.update(run_name=run_name, data_dir=data_dir, out_root=str(out_root), window_layers=list(window_layers),
               lr=3e-3, batch_size=8, micro_batch_size=8, log_interval=1, eval_interval=1000,
               eval_batches=1, final_eval_batches=2, eval_batch_size=4, device="cpu", num_threads=4)
    cfg["total_tokens"] = steps * cfg["batch_size"] * cfg["block_size"]
    cfg.update(over)
    return cfg


def read_log(run_dir):
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        return [json.loads(line) for line in f]


def train_records(run_dir):
    """Last train record per step (as the analysis does after a resume)."""
    recs = {}
    for r in read_log(run_dir):
        if r["type"] == "train":
            recs[r["step"]] = r
    return [recs[s] for s in sorted(recs)]


def test_n1_equals_per_token(synth_dir, tmp_path):
    a = tiny_cfg(synth_dir, tmp_path, "base", window_layers=BASELINE, steps=20)
    b = tiny_cfg(synth_dir, tmp_path, "win1", window_layers=[0, 1], steps=20)
    assert T.train(a) == 0 and T.train(b) == 0
    ra, rb = train_records(tmp_path / "base"), train_records(tmp_path / "win1")
    assert len(ra) == len(rb) == 20
    for x, y in zip(ra, rb):
        assert abs(x["ce"] - y["ce"]) < 1e-4, (x["step"], x["ce"], y["ce"])


def test_overfit_single_batch():
    model = tiny_model(WINDOW).train()
    opt = model.configure_optimizers(0.1, 3e-3, (0.9, 0.95), "cpu")
    torch.manual_seed(5)
    x = torch.randint(0, 1000, (4, 33))
    inp, tgt = x[:, :32], x[:, 1:]
    ces = []
    for _ in range(300):
        _, ce = model(inp, tgt)
        loss = ce + 0.01 * sum(model.aux_losses()) + 0.001 * sum(model.z_losses())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.reset_load_buffers()
        opt.zero_grad(set_to_none=True)
        ces.append(ce.item())
    assert ces[-1] < 0.1 * ces[0], (ces[0], ces[-1])


@pytest.mark.parametrize("windows", [BASELINE, WINDOW])
def test_balance_holds(synth_dir, tmp_path, windows):
    cfg = tiny_cfg(synth_dir, tmp_path, "bal", window_layers=windows, steps=200, log_interval=10)
    assert T.train(cfg) == 0
    last = train_records(tmp_path / "bal")[-1]
    assert last["step"] == 200
    assert min(last["load_min"]) >= 0.02, last["load_min"]
    assert max(last["load_max"]) <= 0.5, last["load_max"]


def test_resume_is_exact(synth_dir, tmp_path):
    # 20 straight steps
    straight = tiny_cfg(synth_dir, tmp_path, "straight", steps=20)
    assert T.train(straight) == 0

    # 10 steps, then checkpoint and exit 99 (the signal path)
    resumed = tiny_cfg(synth_dir, tmp_path, "resumed", steps=20)
    assert T.train(resumed, stop_at_step=10) == 99
    run_dir = tmp_path / "resumed"
    assert (run_dir / "ckpt" / "latest.pt").exists()
    assert not (run_dir / "done").exists()

    # scramble this process's RNG state, then resume in a fresh process
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    cfg_file = tmp_path / "resumed_cfg.py"
    cfg_file.write_text("".join(f"{k} = {v!r}\n" for k, v in resumed.items()))
    proc = subprocess.run([sys.executable, "train.py", str(cfg_file)], cwd=REPO,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    log = read_log(run_dir)
    assert [r["step"] for r in log if r["type"] == "resume"] == [10]
    ra, rb = train_records(tmp_path / "straight"), train_records(run_dir)
    assert [r["step"] for r in ra] == [r["step"] for r in rb] == list(range(1, 21))
    for x, y in zip(ra, rb):
        assert np.allclose(x["ce"], y["ce"], rtol=1e-6, atol=1e-6), (x["step"], x["ce"], y["ce"])
    fa = json.loads((tmp_path / "straight" / "final.json").read_text())
    fb = json.loads((run_dir / "final.json").read_text())
    assert np.allclose(fa["val_overall"], fb["val_overall"], rtol=1e-6, atol=1e-6)


def test_sampler_deterministic():
    a = T.sample_offsets(0, 7, 64, 1_000_000, 1024)
    b = T.sample_offsets(0, 7, 64, 1_000_000, 1024)
    c = T.sample_offsets(1, 7, 64, 1_000_000, 1024)
    d = T.sample_offsets(0, 8, 64, 1_000_000, 1024)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)
    assert a.min() >= 0 and a.max() < 1_000_000 - 1024 - 1

    data = np.arange(5000, dtype=np.uint16)
    x, y = T.make_xy(data, T.sample_offsets(0, 0, 4, len(data), 16), 16, "cpu")
    assert x.dtype == torch.int64 and x.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_small_val_file(synth_dir, tmp_path, capsys):
    # val_book.bin holds 1,000 tokens: fewer than 100 x 16 windows of 128 tokens
    cfg = tiny_cfg(synth_dir, tmp_path, "small", final_eval_batches=100, eval_batch_size=16)
    # add a source that is too short to evaluate at all
    meta_path = os.path.join(synth_dir, "meta.json")
    meta = json.load(open(meta_path))
    meta["train_share_by_source"]["tiny"] = 0.01
    json.dump(meta, open(meta_path, "w"))
    np.zeros(50, dtype=np.uint16).tofile(os.path.join(synth_dir, "val_tiny.bin"))

    val_sets = T.load_val_sets(synth_dir, cfg)
    assert "tiny" not in val_sets
    assert "skipping source tiny" in capsys.readouterr().out
    data, offsets, _ = val_sets["book"]
    assert len(data) == 1000 and len(offsets) == 1600
    assert offsets.max() + 128 + 1 <= len(data)
    assert len(np.unique(offsets)) < len(offsets)  # windows overlap

    model = tiny_model(WINDOW)
    val, overall, routing = T.evaluate(model, val_sets, 100, cfg, "cpu", torch.no_grad(), routing=True)
    assert set(val) == {"book", "c4", "github"}
    assert all(math.isfinite(v) for v in val.values()) and math.isfinite(overall)
    for layer in routing:
        assert sum(layer["book"]) == 1600 * 128 * TINY["top_k"]


def test_end_to_end_cpu(synth_dir, tmp_path):
    out_root = tmp_path / "runs"
    cmd = [sys.executable, "train.py", "config/runs/smoke-cpu.py",
           f"--data_dir={synth_dir}", f"--out_root={out_root}"]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    run_dir = out_root / "smoke-cpu"
    for name in ("log.jsonl", "routing_counts.json", "final.json", "done", "ckpt/latest.pt"):
        assert (run_dir / name).exists(), name
    types = {r["type"] for r in read_log(run_dir)}
    assert {"header", "train", "eval", "final"} <= types
    routing = json.loads((run_dir / "routing_counts.json").read_text())
    assert routing["window_layers"] == [0, 4] and len(routing["counts"]) == 2

    before = (run_dir / "log.jsonl").read_text()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert "nothing to do" in proc.stdout
    assert (run_dir / "log.jsonl").read_text() == before
