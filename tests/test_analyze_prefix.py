"""analyze.py --prefix / --lr_file (docs/M2D2_PROMPT.md part 7) and its unchanged default behavior."""
import json
import os
import subprocess
import sys

import pytest

import make_analyze_fixture as F
from conftest import REPO

sys.path.insert(0, os.path.join(REPO, "analysis"))
import analyze as A  # noqa: E402

M2D2_FINALS = {"base-s0": 3.50, "base-s1": 3.52, "base-s2": 3.51, "win16-s0": 3.49, "win16-s1": 3.50,
               "win16-s2": 3.48, "win1-s0": 3.505, "win64-s0": 3.495}
SLIM120_FINALS = {"base-s0": 3.80, "base-s1": 3.81, "base-s2": 3.82, "win16-s0": 3.90, "win16-s1": 3.91,
                  "win16-s2": 3.89}
M2D2_LR = {"lr-base-2.5e-4": 3.40, "lr-base-5e-4": 3.305, "lr-base-1e-3": 3.30,
           "lr-win16-2.5e-4": 3.45, "lr-win16-5e-4": 3.36, "lr-win16-1e-3": 3.38}


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return F.smoke_run(os.path.realpath(tmp_path_factory.mktemp("az")))


def analyze(*args):
    proc = subprocess.run([sys.executable, F.ANALYZE, *map(str, args)], capture_output=True, text=True, timeout=120)
    return proc


def test_analyze_prefix_and_default_unchanged(smoke, tmp_path):
    # 1. without the new options, every output is byte-identical to the pre-change reference
    ref = json.load(open(F.FIXTURE))
    work = os.path.realpath(tmp_path / "ref")
    os.makedirs(work)
    got = F.capture(work, smoke=smoke)
    for mode in ("main", "no_verdict", "lr_check", "partial"):
        assert got[mode] == ref[mode], f"{mode} output changed (reference: {ref['platform']}, mpl {ref['matplotlib']})"

    # one folder holding the existing runs and both prefixed families
    runs = tmp_path / "runs"
    F.copy_runs(smoke, str(runs), F.MAIN_FINALS)
    F.copy_runs(smoke, str(runs), {f"m2d2-{k}": v for k, v in {**M2D2_FINALS, **M2D2_LR}.items()})
    F.copy_runs(smoke, str(runs), {f"slim120-{k}": v for k, v in SLIM120_FINALS.items()})

    # 2. --prefix m2d2- uses only the m2d2- runs, matched with the prefix stripped
    proc = analyze("--runs_dir", runs, "--out_dir", tmp_path / "m2d2", "--prefix", "m2d2-")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = (tmp_path / "m2d2" / "summary.md").read_text()
    want = A.decide([3.50, 3.52, 3.51], [3.49, 3.50, 3.48], 3.505, 3.495)
    assert f"**Verdict: {want['verdict']}**" in summary
    assert "| mean(B) | 3.51000 |" in summary and "| mean(W) | 3.49000 |" in summary
    assert "Runs with prefix `m2d2-`" in summary and "Skipped criteria" not in summary
    for name in ref["main"]["files"]:  # the same set of output files as the default analysis
        assert (tmp_path / "m2d2" / name).exists(), name

    # 3. slim120- has no win1/win64: the decision uses the available criteria and says which were skipped
    proc = analyze("--runs_dir", runs, "--out_dir", tmp_path / "slim", "--prefix", "slim120-")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = (tmp_path / "slim" / "summary.md").read_text()
    assert "**Verdict: NEGATIVE**" in summary  # gap = 3.81 - 3.90 = -0.09, far beyond 2s
    assert "Skipped criteria" in summary and "no finished win1-s0" in summary and "no finished win64-s0" in summary
    assert "| win1-s0 - mean(B) | skipped (no finished win1-s0) |" in summary
    # without --prefix the decision still uses only the unprefixed runs (the tables list every run, as before)
    proc = analyze("--runs_dir", runs, "--out_dir", tmp_path / "plain")
    assert proc.returncode == 0
    head = got["main"]["stdout"].split("## Direct sanity comparison")[0]
    assert proc.stdout.startswith(head)  # same verdict, mean(B), mean(W), gap, s, win1 and win64 rows

    # 4. --lr_file: the M2D2 LR check writes its own file, labeled as calibration losses
    for name in M2D2_LR:
        path = runs / f"m2d2-{name}" / "log.jsonl"
        recs = [json.loads(l) for l in open(path)]
        recs[0]["config"]["eval_split"] = "calib"
        path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    repo_lr = open(os.path.join(REPO, "config", "chosen_lr.py"), "rb").read()
    lr_file = tmp_path / "chosen_lr_m2d2.py"
    proc = analyze("--runs_dir", runs, "--out_dir", tmp_path / "lr", "--prefix", "m2d2-", "--lr_check", "--write",
                   "--lr_file", lr_file)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "final calib_overall (calibration split)=3.3000" in proc.stdout
    ns = {}
    exec(lr_file.read_text(), ns)
    assert ns["LR_BASE"] == 5e-4  # 1e-3 is best (3.300), 5e-4 within 0.01 and lower
    assert ns["LR_WIN"] == 5e-4
    assert "calibration split" in lr_file.read_text()
    assert open(os.path.join(REPO, "config", "chosen_lr.py"), "rb").read() == repo_lr


def test_decide_available_matches_decide():
    B, W = [3.60, 3.62, 3.61], [3.66, 3.64, 3.65]
    for win1, win64 in ((3.605, 3.67), (3.70, 3.60), (3.61, 3.50)):
        full, avail = A.decide(B, W, win1, win64), A.decide_available(B, W, win1, win64)
        assert avail["skipped"] == [] and {k: avail[k] for k in full} == full
    # without win64, PROMISING needs only the gap; without win1, the sanity check cannot fail
    better = [3.50, 3.51, 3.52]
    assert A.decide_available(B, better, 3.61, None)["verdict"] == "PROMISING"
    assert A.decide(B, better, 3.61, 3.99)["verdict"] == "INCONCLUSIVE"
    assert A.decide_available(B, W, None, 3.67)["verdict"] == "NEGATIVE"
    assert A.decide(B, W, 3.99, 3.67)["verdict"] == "INVALID: sanity check failed"
