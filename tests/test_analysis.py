"""Analysis-script tests (PROMPT.md part 6 test_analysis_runs, plus decision-rule checks)."""
import json
import math
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
    # direct win1-s0 vs base-s0 comparison (identical copies -> zero difference)
    assert "## Direct sanity comparison: win1-s0 vs base-s0" in summary
    assert "| win1-s0 - base-s0 (final val_overall) | +0.00000 |" in summary
    assert "win1-s0 - base-s0 (final val_overall)" in proc.stdout


def test_direct_comparison(smoke_run, tmp_path):
    runs_dir = fake_runs(smoke_run, tmp_path / "runs", MAIN_RUNS)
    base0 = json.loads((runs_dir / "base-s0" / "final.json").read_text())["val_overall"]
    set_final(runs_dir / "win1-s0", base0 + 0.0123)
    cmp = analyze.direct_comparison(analyze.read_runs(str(runs_dir)))
    assert cmp["final_diff"] == pytest.approx(0.0123)
    assert cmp["max_abs_curve_diff"] == pytest.approx(0.0123)  # earlier eval points are identical
    assert cmp["n_points"] >= 3
    assert set(cmp["per_source"]) == {"book", "c4", "github"}
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, ANALYZE, "--runs_dir", str(runs_dir), "--out_dir", str(out)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "| win1-s0 - base-s0 (final val_overall) | +0.01230 |" in (out / "summary.md").read_text()
    # the verdict still follows the part 8 rule against mean(B): s = 0 here, so the sanity check fails
    assert "**Verdict: INVALID: sanity check failed**" in (out / "summary.md").read_text()


def test_missing_runs_give_no_verdict(smoke_run, tmp_path):
    runs = fake_runs(smoke_run, tmp_path / "runs", ["base-s0", "win16-s0"])
    proc = subprocess.run([sys.executable, ANALYZE, "--runs_dir", str(runs), "--out_dir", str(tmp_path / "out")],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2
    assert "No verdict yet" in proc.stdout
    assert "Not available yet: needs finished win1-s0 and base-s0." in proc.stdout


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


def write_partial_log(run_dir, evals):
    """A mid-training log: header plus eval records {step: val_overall}; no final, no done."""
    run_dir.mkdir(parents=True)
    recs = [{"type": "header", "run_name": run_dir.name}]
    recs += [{"type": "eval", "step": s, "tokens": s * 65536, "val": {"c4": v}, "val_overall": v}
             for s, v in sorted(evals.items())]
    (run_dir / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


@pytest.fixture
def partial_runs(tmp_path):
    runs = tmp_path / "runs"
    logs = {
        "base-s0": {0: 10.8, 500: 4.10, 1000: 3.90},
        "base-s1": {0: 10.8, 500: 4.12, 1000: 3.91},
        "base-s2": {0: 10.8, 500: 4.14},                      # slowest: limits the shared step to 500
        "win16-s0": {0: 10.8, 500: 4.00, 1000: 3.80, 1500: 3.70},
        "win16-s1": {0: 10.8, 500: 4.04, 1000: 3.82},
        "win16-s2": {},                                      # started, but no eval logged yet
        "win1-s0": {0: 10.8, 500: 4.11, 1000: 3.90},
        "win64-s0": {0: 10.8, 500: 4.05, 1000: 3.85},
        "lr-base-1e-3": {0: 10.8},                           # not a main run: must not limit the step
    }
    for name, evals in logs.items():
        write_partial_log(runs / name, evals)
    return runs


def test_partial_compare(partial_runs):
    p = analyze.partial_compare(analyze.read_runs(str(partial_runs)))
    assert p["step"] == 500 and p["tokens"] == 500 * 65536
    assert p["limiting"] == ["base-s2"]
    assert p["not_started"] == ["win16-s2"]
    assert "lr-base-1e-3" not in p["values"]
    base, win = p["arms"]["base"], p["arms"]["win16"]
    assert base["runs"] == ["base-s0", "base-s1", "base-s2"] and win["runs"] == ["win16-s0", "win16-s1"]
    assert base["mean"] == pytest.approx(4.12) and base["std"] == pytest.approx(0.02)
    assert (base["min"], base["max"]) == (pytest.approx(4.10), pytest.approx(4.14))
    assert win["mean"] == pytest.approx(4.02) and win["std"] == pytest.approx(0.02 ** 0.5 * 0.2)
    assert p["gap"] == pytest.approx(0.10)
    assert p["s"] == pytest.approx(((0.02 ** 2 + win["std"] ** 2) / 2) ** 0.5)
    assert p["win1_minus_base0"] == pytest.approx(0.01)
    assert p["arms"]["win1"]["n"] == 1 and math.isnan(p["arms"]["win1"]["std"])
    assert p["arms"]["win64"]["mean"] == pytest.approx(4.05)


def test_partial_cli(partial_runs, tmp_path):
    proc = subprocess.run([sys.executable, ANALYZE, "--runs_dir", str(partial_runs), "--partial"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "Partial comparison at step 500 (32.8M tokens)" in out
    assert "Limited by: base-s2" in out and "Not started yet (no eval logged): win16-s2" in out
    assert "base       3    4.12000   0.02000    4.10000    4.14000" in out
    assert "win1       1    4.11000         -" in out
    assert "gap = mean(base) - mean(win16): +0.10000" in out
    assert "win1-s0 - base-s0: +0.01000" in out
    assert not (tmp_path / "out").exists()  # partial mode writes no files

    empty = tmp_path / "empty"
    write_partial_log(empty / "lr-base-1e-3", {0: 10.8})
    proc = subprocess.run([sys.executable, ANALYZE, "--runs_dir", str(empty), "--partial"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "no main run" in proc.stderr


def test_last_record_per_step_wins(smoke_run, tmp_path):
    run = fake_runs(smoke_run, tmp_path / "runs", ["base-s0"]) / "base-s0"
    with open(run / "log.jsonl", "a") as f:
        f.write(json.dumps({"type": "resume", "step": 20}) + "\n")
        f.write(json.dumps({"type": "eval", "step": 20, "tokens": 20 * 8192, "val": {}, "val_overall": 1.5}) + "\n")
    r = analyze.read_run(str(run))
    assert r["eval"][20]["val_overall"] == 1.5
