"""Tests for analysis/followup.py (FOLLOWUP_PROMPT.md part 8, test_followup_runs)."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import REPO, make_synthetic_data

sys.path.insert(0, os.path.join(REPO, "analysis"))
import followup as FU  # noqa: E402

FOLLOWUP = os.path.join(REPO, "analysis", "followup.py")
OUTPUTS = ["followup_summary.md", "followup_runs.csv", "followup_E5_gap.png", "followup_E3_gap.png",
           "followup_E2_gap.png", "followup_E4_gap.png", "router_grad_norm.png"]

# final val_overall for each fake run; main runs give s = 0.01 and original gap +0.05
FINALS = {
    "base-s0": 3.60, "base-s1": 3.61, "base-s2": 3.59, "win16-s0": 3.65, "win16-s1": 3.64, "win16-s2": 3.66,
    "both16-s0": 3.57, "both16-s1": 3.58,                    # -0.03: clearly better
    "win16top1-s0": 3.601, "win16top1-s1": 3.611,            # +0.001: within noise, penalty vanishes
    "win16top2-s0": 3.625, "win16top2-s1": 3.635,            # +0.025: penalty shrinks by 0.025 > 2s
    "base-sh2048-s0": 3.55, "base-sh2048-s1": 3.56,
    "win16-sh2048-s0": 3.56, "win16-sh2048-s1": 3.57,        # gap +0.01: shrinks by 0.04 > 2s
    "base-rlr3-s0": 3.60, "base-rlr3-s1": 3.61,
    "win16-rlr3-s0": 3.65, "win16-rlr3-s1": 3.66,            # gap +0.05: does not shrink
    "base-rg-s0": 3.601,                                     # matches base-s0 within 0.002
    "win16-rg-s0": 3.655,                                    # differs from win16-s0 by 0.005: flagged
}


@pytest.fixture(scope="module")
def smoke_rg(tmp_path_factory):
    root = tmp_path_factory.mktemp("fu")
    data = root / "data"
    make_synthetic_data.make(str(data))
    proc = subprocess.run([sys.executable, "train.py", "config/runs/smoke-cpu.py", f"--data_dir={data}",
                           f"--out_root={root / 'train'}", "--log_router_grad=True"],
                          cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return root / "train" / "smoke-cpu"


def rewrite_log(run_dir, fn):
    recs = [fn(json.loads(l)) for l in open(run_dir / "log.jsonl")]
    (run_dir / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def make_runs(smoke, dest, finals=FINALS):
    for name, value in finals.items():
        shutil.copytree(smoke, dest / name, ignore=shutil.ignore_patterns("ckpt"))

        def set_final(r, value=value):
            if r["type"] == "final":
                r["val_overall"] = value
            return r
        rewrite_log(dest / name, set_final)
    return dest


def run_followup(runs, out):
    proc = subprocess.run([sys.executable, FOLLOWUP, "--runs_dir", str(runs), "--out_dir", str(out)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_followup_runs(smoke_rg, tmp_path):
    runs = make_runs(smoke_rg, tmp_path / "runs")
    out = tmp_path / "out"
    stdout = run_followup(runs, out)
    for name in OUTPUTS:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name

    assert "s = 0.01000, original gap (win16 - base) = +0.05000" in stdout
    # one interpretation line per group
    assert "E5: both16 - base = -0.0300 (clearly better): context helps as an addition." in stdout
    assert "E3: win16top1 - base = +0.0010 (within noise) vs original penalty +0.0500: penalty vanishes." in stdout
    assert "E3: win16top2 - base = +0.0250 (clearly worse) vs original penalty +0.0500: penalty shrinks." in stdout
    assert "E3: penalty shrinks or vanishes with fewer window layers: topic routing only belongs at the top." in stdout
    assert "gap shrinks by more than 2s: the shared expert was the bottleneck" in stdout
    assert "E4: win16-rlr3 - base-rlr3 = +0.0500" in stdout and "does not shrink by more than 2s" in stdout
    assert "not clearly smaller" in stdout  # identical copies: ratio 1
    assert "E4: neither test points to slow router learning" in stdout
    # logging check
    assert "base-rg-s0 vs base-s0: 3.60100 vs 3.60000, difference +0.00100 (within 0.002: ok)" in stdout
    assert "win16-rg-s0 vs win16-s0" in stdout and "FLAG" in stdout

    rows = (out / "followup_runs.csv").read_text().splitlines()
    assert rows[0].startswith("run,final_val_overall,group,comparison")
    assert any(r.startswith("both16-s0,3.570000,E5,base-s0,0,3.600000,-0.030000,-0.030000,clearly better")
               for r in rows)
    assert len(rows) == 1 + len(FINALS)
    summary = (out / "followup_summary.md").read_text()
    assert "| c4 |" in summary and "Mean paired difference: **-0.03000**" in summary


def test_followup_smaller_router_grads_and_missing_runs(smoke_rg, tmp_path):
    finals = {k: v for k, v in FINALS.items() if not k.startswith(("both16", "win16top"))}
    runs = make_runs(smoke_rg, tmp_path / "runs", finals)

    def halve(r):
        if r["type"] == "train" and "router_grad_norm" in r:
            r["router_grad_norm"] = [g * 0.5 for g in r["router_grad_norm"]]
        return r
    rewrite_log(runs / "win16-rg-s0", halve)
    stdout = run_followup(runs, tmp_path / "out")
    assert "L2 0.50" in stdout and "window router gradients clearly smaller" in stdout
    assert "E4: slow router learning explains part of the gap." in stdout
    assert "E5: not available yet" in stdout and "E3: not available yet" in stdout
    for name in OUTPUTS:
        assert (tmp_path / "out" / name).exists(), name


def test_label_boundaries():
    assert FU.label(-0.021, 0.01) == "clearly better"
    assert FU.label(0.021, 0.01) == "clearly worse"
    assert FU.label(0.02, 0.01) == "within noise" and FU.label(-0.02, 0.01) == "within noise"
