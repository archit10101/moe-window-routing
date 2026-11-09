"""analysis/eval_locked_test.py (docs/M2D2_PROMPT.md part 7; locked test split used once)."""
import glob
import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext

import numpy as np
import pytest
import torch

import train as T
from conftest import REPO, make_synthetic_data

EVAL = os.path.join(REPO, "analysis", "eval_locked_test.py")
RUNS = ["m2d2-base-s0", "m2d2-base-s1", "m2d2-win16-s0", "m2d2-win16-s1"]


@pytest.fixture(scope="module")
def locked(tmp_path_factory):
    """Two different smoke-cpu trainings relabeled as two m2d2 base and two m2d2 win16 runs, plus test_*.bin files."""
    root = tmp_path_factory.mktemp("locked")
    data = root / "data"
    make_synthetic_data.make(str(data))
    rng = np.random.default_rng(7)
    for src in ("book", "c4", "github"):  # a held-out split with its own tokens
        rng.integers(0, 1000, size=5000, dtype=np.uint16).tofile(data / f"test_{src}.bin")
    for name, extra in (("w", []), ("b", ["--window_layers=[0, 0]", "--seed=1"])):
        proc = subprocess.run([sys.executable, "train.py", "config/runs/smoke-cpu.py", f"--data_dir={data}",
                               f"--out_root={root / 'train'}", f"--run_name={name}"] + extra,
                              cwd=REPO, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    return root, data


def run_eval(runs_dir, data, out, log, *extra):
    return subprocess.run([sys.executable, EVAL, "--runs_dir", str(runs_dir), "--data_dir", str(data), "--out_dir",
                           str(out), "--runs", ",".join(RUNS), "--device", "cpu", "--log_file", str(log), *extra],
                          capture_output=True, text=True, timeout=120)


def test_locked_test_eval_and_log(locked, tmp_path):
    root, data = locked
    runs = tmp_path / "runs"
    for name, src in zip(RUNS, ("b", "b", "w", "w")):
        shutil.copytree(root / "train" / src, runs / name)
    log = tmp_path / "analysis" / "locked_test_log.txt"

    proc = run_eval(runs, data, tmp_path / "out", log)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in RUNS:
        r = json.loads((runs / name / "test_results.json").read_text())
        assert r["split"] == "test" and set(r["test"]) == {"book", "c4", "github"} and not r["test_set_reused"]
    summary = (tmp_path / "out" / "summary.md").read_text()
    assert "**Verdict (test split): " in summary and "TEST SET REUSED" not in summary
    assert "| gap = mean(B) - mean(W) |" in summary and "Skipped criteria" in summary  # no win1/win64 here
    entries = log.read_text().splitlines()
    assert [e.split(" | ")[1] for e in entries] == ["START", "COMPLETED"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    assert all(f"git={commit}" in e and f"runs={','.join(RUNS)}" in e for e in entries)

    # the number is train.py's own evaluation on the test files
    ckpt = torch.load(runs / "m2d2-base-s0" / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)
    cfg = T.with_defaults(ckpt["config"])
    cfg.update(data_dir=str(data), device="cpu")
    model = T.build_model(cfg, "cpu")
    model.load_state_dict(ckpt["model"])
    _, overall, _ = T.evaluate(model, T.load_val_sets(str(data), cfg, split="test", allow_locked_test=True),
                               cfg["final_eval_batches"], cfg, "cpu", nullcontext())
    assert json.loads((runs / "m2d2-base-s0" / "test_results.json").read_text())["test_overall"] == overall
    final = json.loads((runs / "m2d2-base-s0" / "final.json").read_text())["val_overall"]
    assert overall != final  # a different split from validation

    # a second evaluation is refused, and the attempt is logged
    proc = run_eval(runs, data, tmp_path / "out2", log)
    assert proc.returncode != 0 and "already evaluated" in proc.stderr
    assert log.read_text().splitlines()[-1].split(" | ")[1] == "REFUSED"
    assert not (tmp_path / "out2").exists()

    # explicit reuse: logged with a timestamp and marked in the summary
    proc = run_eval(runs, data, tmp_path / "out3", log, "--allow_reuse_of_locked_test")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    events = [e.split(" | ")[1] for e in log.read_text().splitlines()]
    assert events == ["START", "COMPLETED", "REFUSED", "REUSE", "START", "COMPLETED"]
    assert "TEST SET REUSED" in (tmp_path / "out3" / "summary.md").read_text()
    assert json.loads((runs / "m2d2-win16-s0" / "test_results.json").read_text())["test_set_reused"] is True


def test_only_the_locked_test_script_reads_test_files():
    """Rule 4: no other code path opts into the locked test split (any call passing allow_locked_test=<truthy>)."""
    import ast
    users = []
    for path in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True):
        if "/tests/" in path or "/.venv/" in path:
            continue
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "allow_locked_test" and not (isinstance(kw.value, ast.Constant) and not kw.value.value):
                        users.append(os.path.relpath(path, REPO))
    assert users == ["analysis/eval_locked_test.py"]
