"""Tests for analysis/diagnostics.py (FOLLOWUP_PROMPT.md part 8, test_diagnostics_runs)."""
import csv
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO, make_synthetic_data

sys.path.insert(0, os.path.join(REPO, "analysis"))
import diagnostics as D  # noqa: E402

DIAG = os.path.join(REPO, "analysis", "diagnostics.py")
A1_FILES = ["a1_buckets.csv", "a1_summary.md", "a1_gap_by_category.png"]
A2_FILES = ["a2_routing.csv", "a2_summary.md", "a2_by_layer.png", "a2_grad_norm.png"]


def fake_vocab(n=1024):
    """Decoded strings that hit every category without tiktoken (no network in tests)."""
    kinds = [" ", "\n\n", ",", "?!", "42", "7"]
    return [kinds[i] if i < len(kinds) else (f" word{i}" if i % 7 else "...") for i in range(n)]


@pytest.fixture(scope="module")
def diag_runs(tmp_path_factory):
    """Two different smoke-cpu trainings, relabeled as two base and two win16 runs (with checkpoints)."""
    root = tmp_path_factory.mktemp("diag")
    data = root / "data"
    make_synthetic_data.make(str(data))
    out = root / "train"
    for name, extra in (("w", []), ("b", ["--window_layers=[0, 0]", "--seed=1"])):
        cmd = [sys.executable, "train.py", "config/runs/smoke-cpu.py", f"--data_dir={data}", f"--out_root={out}",
               f"--run_name={name}"] + extra
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
    runs = root / "runs"
    for name, src in (("base-s0", "b"), ("base-s1", "b"), ("win16-s0", "w"), ("win16-s1", "w")):
        shutil.copytree(out / src, runs / name)
    vocab = root / "vocab.json"
    vocab.write_text(json.dumps(fake_vocab()))
    return dict(data=data, runs=runs, vocab=vocab, root=root)


def test_classify_string():
    assert D.classify_string(" ") == "space" and D.classify_string("\n\n") == "space"
    assert D.classify_string(",") == "punct" and D.classify_string(" ...") == "punct"
    assert D.classify_string("“") == "punct"
    assert D.classify_string(" 2024") == "digit"
    assert D.classify_string(" the") == "other" and D.classify_string("a1") == "other"
    assert D.classify_string("") == "other" and D.classify_string("�") == "other"


def test_diagnostics_runs(diag_runs):
    out = diag_runs["root"] / "out"
    proc = subprocess.run([sys.executable, DIAG, "--runs_dir", str(diag_runs["runs"]), "--data_dir",
                           str(diag_runs["data"]), "--out_dir", str(out), "--device", "cpu",
                           "--runs", "base-s0,base-s1,win16-s0,win16-s1", "--decoded_vocab", str(diag_runs["vocab"])],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in A1_FILES + A2_FILES:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name

    rows = list(csv.DictReader(open(out / "a1_buckets.csv")))
    total = next(r for r in rows if r["family"] == "total")
    n_tokens = 3 * 4 * 8 * 128  # sources x final_eval_batches x eval_batch_size x block_size (smoke-cpu)
    assert int(total["tokens"]) == n_tokens
    total_gap = float(total["gap_win16_minus_base"])
    assert abs(total_gap) > 1e-4  # the two trainings differ, so the check below is not trivial

    finals = {n: json.loads((diag_runs["runs"] / n / "final.json").read_text())["val_overall"]
              for n in ("base-s0", "win16-s0")}
    assert total_gap == pytest.approx(finals["win16-s0"] - finals["base-s0"], abs=1e-5)

    for fam, n_buckets in (("input", 5), ("target", 5), ("position", 3)):
        fam_rows = [r for r in rows if r["family"] == fam]
        assert len(fam_rows) == n_buckets
        # every token falls in exactly one bucket of each family
        assert sum(int(r["tokens"]) for r in fam_rows) == n_tokens
        assert sum(float(r["token_share"]) for r in fam_rows) == pytest.approx(1.0, abs=1e-9)
        # the per-bucket contributions add up to the total gap
        assert abs(sum(float(r["contribution"]) for r in fam_rows) - total_gap) < 1e-6
    cats = {r["bucket"] for r in rows if r["family"] == "input" and int(r["tokens"]) > 0}
    assert cats == set(D.CATEGORIES)
    assert [r["bucket"] for r in rows if r["family"] == "position"] == ["0-15", "16-63", "64-127"]

    a2 = list(csv.DictReader(open(out / "a2_routing.csv")))
    assert len(a2) == 4 * 2  # runs x layers
    for r in a2:
        assert 0 <= float(r["switch_rate"]) <= 1
        assert 0 <= float(r["router_entropy"]) <= np.log2(4) + 1e-9
        assert 0.25 - 1e-9 <= float(r["top1_prob"]) <= 1
        for k in ("token_determinism", "source_determinism"):
            assert float(r[k]) <= 1 + 1e-9
    summary = (out / "a1_summary.md").read_text()
    assert "## Gap by source x input category" in summary and "Check against final.json" in summary


def test_determinism_and_categories(tmp_path):
    # an expert fully determined by the token -> 1; independent of it -> 0
    assert D.determinism(np.array([[10, 0], [0, 10]])) == pytest.approx(1.0)
    assert D.determinism(np.array([[5, 5], [5, 5]])) == pytest.approx(0.0)
    # frequent = top ids not already space/punct/digit
    np.array([9] * 50 + [0] * 40 + [5] * 30 + [7] * 3, dtype=np.uint16).tofile(tmp_path / "train.bin")
    strings = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i"]
    cats = D.token_categories(strings, str(tmp_path / "train.bin"))
    assert D.CATEGORIES[cats[0]] == "space"  # frequent, but already a category
    assert D.CATEGORIES[cats[9]] == "frequent" and D.CATEGORIES[cats[5]] == "frequent"
    assert D.CATEGORIES[cats[1]] == "other"  # never seen
