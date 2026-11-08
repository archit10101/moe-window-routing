"""Analysis-script tests (PROMPT.md part 6 test_analysis_runs, plus decision-rule checks)."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO, make_synthetic_data

sys.path.insert(0, os.path.join(REPO, "analysis"))
import analyze  # noqa: E402

ANALYZE = os.path.join(REPO, "analysis", "analyze.py")
MAIN_RUNS = ["base-s0", "base-s1", "base-s2", "win16-s0", "win16-s1", "win16-s2", "win1-s0", "win64-s0"]
OUTPUTS = ["sanity.png", "main_comparison.png", "per_source.csv", "per_source.md", "window_sweep.png",
           "loads.png", "routing_heatmap.png", "routing_mi.csv", "summary.md"]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("smoke")
    data = root / "data"
    make_synthetic_data.make(str(data))
    proc = subprocess.run([sys.executable, "train.py", "config/runs/smoke-cpu.py", f"--data_dir={data}",
                           f"--out_root={root / 'runs'}"], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return root / "runs" / "smoke-cpu"


def fake_runs(smoke_run, dest, names):
    for name in names:
        shutil.copytree(smoke_run, dest / name, ignore=shutil.ignore_patterns("ckpt"))
    return dest


def set_final(run_dir, value):
    lines = [json.loads(l) for l in open(run_dir / "log.jsonl")]
    for r in lines:
        if r["type"] == "final":
            r["val_overall"] = value
    (run_dir / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))


def test_analysis_runs(smoke_run, tmp_path):
    runs = fake_runs(smoke_run, tmp_path / "runs", MAIN_RUNS)
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, ANALYZE, "--runs_dir", str(runs), "--out_dir", str(out)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in OUTPUTS:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    summary = (out / "summary.md").read_text()
    assert "**Verdict: " in summary
    # identical copies: s = 0, gap = 0, sanity holds -> INCONCLUSIVE
    assert "**Verdict: INCONCLUSIVE**" in summary and "Verdict: INCONCLUSIVE" in proc.stdout
    assert "low-sample" in (out / "per_source.md").read_text()


def test_missing_runs_give_no_verdict(smoke_run, tmp_path):
    runs = fake_runs(smoke_run, tmp_path / "runs", ["base-s0", "win16-s0"])
    proc = subprocess.run([sys.executable, ANALYZE, "--runs_dir", str(runs), "--out_dir", str(tmp_path / "out")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "No verdict yet" in proc.stdout


def test_decision_rule():
    B = [3.00, 3.02, 3.01]  # s = 0.01 when W has the same spread
    shift = lambda d: [b - d for b in B]  # noqa: E731
    assert analyze.decide(B, shift(0.05), 3.01, 3.00)["verdict"] == "PROMISING"
    assert analyze.decide(B, shift(0.05), 3.01, 3.05)["verdict"] == "INCONCLUSIVE"  # win64 too high
    assert analyze.decide(B, shift(-0.05), 3.01, 3.00)["verdict"] == "NEGATIVE"
    assert analyze.decide(B, shift(0.01), 3.01, 3.00)["verdict"] == "INCONCLUSIVE"
    assert analyze.decide(B, shift(0.05), 3.10, 3.00)["verdict"] == "INVALID: sanity check failed"
    r = analyze.decide(B, shift(0.05), 3.01, 3.00)
    assert r["s"] == pytest.approx(0.01) and r["gap"] == pytest.approx(0.05)


def test_lr_check_tie_and_write(smoke_run, tmp_path):
    names = ["lr-base-1e-3", "lr-base-2e-3", "lr-base-4e-3", "lr-win16-1e-3", "lr-win16-2e-3", "lr-win16-4e-3"]
    runs = fake_runs(smoke_run, tmp_path / "runs", names)
    for name, v in zip(names, [3.30, 3.205, 3.20, 3.40, 3.25, 3.35]):
        set_final(runs / name, v)
    chosen = tmp_path / "chosen_lr.py"
    proc = subprocess.run([sys.executable, ANALYZE, "--runs_dir", str(runs), "--out_dir", str(tmp_path / "o"),
                           "--lr_check", "--write", "--chosen_lr_path", str(chosen)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    ns = {}
    exec(chosen.read_text(), ns)
    assert ns["LR_BASE"] == 2e-3  # 4e-3 is best, 2e-3 is within 0.01 and lower
    assert ns["LR_WIN"] == 2e-3


def test_last_record_per_step_wins(smoke_run, tmp_path):
    run = fake_runs(smoke_run, tmp_path / "runs", ["base-s0"]) / "base-s0"
    with open(run / "log.jsonl", "a") as f:
        f.write(json.dumps({"type": "resume", "step": 20}) + "\n")
        f.write(json.dumps({"type": "eval", "step": 20, "tokens": 20 * 8192, "val": {}, "val_overall": 1.5}) + "\n")
    r = analyze.read_run(str(run))
    assert r["eval"][20]["val_overall"] == 1.5
