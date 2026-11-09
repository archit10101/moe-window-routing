"""Launcher checks that run on the laptop: syntax, and submit.sh against a fake sbatch."""
import glob
import os
import subprocess
import sys

import pytest

from conftest import REPO

SCRIPTS = sorted(glob.glob(os.path.join(REPO, "scripts", "**", "*.sh"), recursive=True)
                 + glob.glob(os.path.join(REPO, "scripts", "**", "*.sbatch"), recursive=True))


def test_scripts_exist():
    names = {os.path.relpath(p, REPO) for p in SCRIPTS}
    for want in ["slurm/run_group.sbatch", "slurm/submit.sh", "slurm/submit_lr_check.sh", "slurm/submit_main.sh",
                 "slurm/submit_extra_seeds.sh", "slurm/submit_smoke.sh", "slurm/benchmark.sh", "slurm/status.sh",
                 "slurm/submit_followups.sh", "slurm/submit_diagnostics.sh",
                 "aws/setup_instance.sh", "aws/launch_run.sh", "aws/launch_instance.sh"]:
        assert f"scripts/{want}" in names, want


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: os.path.relpath(p, REPO))
def test_bash_syntax(path):
    text = open(path).read()
    assert text.startswith("#!/bin/bash")
    assert "set -uo pipefail" in text
    proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.fixture
def fake_sbatch(tmp_path):
    calls = tmp_path / "sbatch_calls.txt"
    fake = tmp_path / "sbatch"
    fake.write_text(f'#!/bin/bash\necho "$@" >> "{calls}"\necho "Submitted batch job 1"\n')
    fake.chmod(0o755)
    env = dict(os.environ, SBATCH=str(fake), PYTHON=sys.executable, OUT_ROOT=str(tmp_path / "runs"),
               PATH=f"{tmp_path}:{os.environ['PATH']}")

    def run(script, *args):
        proc = subprocess.run(["bash", os.path.join(REPO, "scripts", "slurm", script), *args],
                              capture_output=True, text=True, env=env)
        lines = calls.read_text().splitlines() if calls.exists() else []
        return proc, lines

    return run


def test_submit_resources(fake_sbatch):
    proc, calls = fake_sbatch("submit.sh", "g", "b200", "01:00:00", "lr-base-1e-3", "lr-base-2e-3", "lr-base-4e-3")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 1
    c = calls[0]
    for part in ["-p mit_normal_gpu", "--gres=gpu:b200:1", "-c 12", "--mem=144G", "-t 01:00:00", "-J g",
                 "run_group.sbatch lr-base-1e-3 lr-base-2e-3 lr-base-4e-3", "slurm/%x-%j.out"]:
        assert part in c, (part, c)


def test_submit_caps_cpus_and_memory(fake_sbatch):
    runs = ["lr-base-1e-3", "lr-base-2e-3", "lr-win16-1e-3", "lr-win16-2e-3", "lr-win16-4e-3"]
    proc, calls = fake_sbatch("submit.sh", "g", "b200", "01:00:00", *runs)
    assert proc.returncode == 0, proc.stderr
    assert "-c 16" in calls[0] and "--mem=200G" in calls[0]


def test_submit_refuses(fake_sbatch):
    proc, calls = fake_sbatch("submit.sh", "g", "b200", "01:00:00", "no-such-run")
    assert proc.returncode != 0 and "does not exist" in proc.stderr
    proc, calls = fake_sbatch("submit.sh", "g", "b200", "01:00:00", "base-s0")  # LR_BASE is None
    assert proc.returncode != 0 and "still None" in proc.stderr
    assert calls == []


def test_convenience_scripts(fake_sbatch):
    proc, calls = fake_sbatch("submit_lr_check.sh", "b200")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 2 and all("-c 12" in c for c in calls)
    proc, calls = fake_sbatch("submit_lr_check.sh", "l40s")
    assert proc.returncode == 0 and len(calls) == 2 + 6
    proc, calls = fake_sbatch("submit_smoke.sh", "b200")
    assert proc.returncode == 0 and "-t 00:30:00" in calls[-1]
    assert calls[-1].endswith("smoke-gpu-base smoke-gpu-win16")
    proc, _ = fake_sbatch("submit_main.sh", "b200")
    assert proc.returncode != 0 and "still None" in proc.stderr  # until chosen_lr.py is written


def test_status(tmp_path):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "log.jsonl").write_text(
        '{"type": "train", "step": 50, "tokens": 3276800, "ce": 5.5, "tok_per_s": 1000.0}\n'
        '{"type": "eval", "step": 50, "tokens": 3276800, "val": {}, "val_overall": 5.6}\n')
    proc = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "status.py"), str(tmp_path / "runs")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "r1" in proc.stdout and "5.6000" in proc.stdout and "no" in proc.stdout


# ---- follow-up launchers (FOLLOWUP_PROMPT.md part 9) ----

FOLLOWUP_ORDER = ["both16-s0", "both16-s1",
                  "win16top1-s0", "win16top1-s1", "win16top2-s0", "win16top2-s1",
                  "base-sh2048-s0", "win16-sh2048-s0", "base-sh2048-s1", "win16-sh2048-s1",
                  "base-rlr3-s0", "win16-rlr3-s0", "base-rlr3-s1", "win16-rlr3-s1", "base-rg-s0", "win16-rg-s0"]


def submitted_runs(calls):
    return [c.split("run_group.sbatch ")[1] for c in calls]


def test_submit_followups_all(fake_sbatch):
    proc, calls = fake_sbatch("submit_followups.sh", "b200")
    assert proc.returncode == 0, proc.stderr
    assert submitted_runs(calls) == FOLLOWUP_ORDER  # one run per job, priority order E5, E3, E2, E4
    for c, run in zip(calls, FOLLOWUP_ORDER):
        for part in ["-t 01:30:00", "-c 4", "--mem=48G", "--gres=gpu:b200:1", f"-J {run}"]:
            assert part in c, (part, c)


def test_submit_followups_groups(fake_sbatch):
    proc, calls = fake_sbatch("submit_followups.sh", "b200", "E3", "E5")  # still submitted in priority order
    assert proc.returncode == 0, proc.stderr
    assert submitted_runs(calls) == FOLLOWUP_ORDER[:6]
    proc, calls2 = fake_sbatch("submit_followups.sh", "b200", "E9")
    assert proc.returncode == 2 and "unknown group" in proc.stderr
    assert len(calls2) == len(calls)  # nothing more submitted


def test_submit_diagnostics(fake_sbatch):
    proc, calls = fake_sbatch("submit_diagnostics.sh", "b200")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 1
    for part in ["-p mit_normal_gpu", "--gres=gpu:b200:1", "-c 8", "--mem=64G", "-t 01:00:00", "-J diagnostics",
                 "TIKTOKEN_CACHE_DIR=", "submit_diagnostics.sh --in-job"]:
        assert part in calls[0], (part, calls[0])
    assert "TIKTOKEN_CACHE_DIR" in proc.stderr  # warns that the cache is not primed


def test_status_lists_followup_runs(tmp_path):
    for name in FOLLOWUP_ORDER:
        (tmp_path / "runs" / name).mkdir(parents=True)
    proc = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "status.py"), str(tmp_path / "runs")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    listed = [line.split()[0] for line in proc.stdout.splitlines()[1:]]
    assert sorted(listed) == sorted(FOLLOWUP_ORDER)
