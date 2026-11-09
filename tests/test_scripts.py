"""Launcher checks that run on the laptop: syntax, and submit.sh against a fake sbatch."""
import glob
import json
import os
import shutil
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
                 "slurm/submit_m2d2_lr_check.sh", "slurm/submit_slim120.sh", "slurm/submit_locked_test.sh",
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
    fake.write_text(f'#!/bin/bash\necho "$@" >> "{calls}"\necho "Submitted batch job $((100 + $(wc -l < "{calls}")))"\n')
    fake.chmod(0o755)
    env = dict(os.environ, SBATCH=str(fake), PYTHON=sys.executable, OUT_ROOT=str(tmp_path / "runs"),
               PATH=f"{tmp_path}:{os.environ['PATH']}")

    def run(script, *args, env=env, stdin=None):
        proc = subprocess.run(["bash", os.path.join(REPO, "scripts", "slurm", script), *args],
                              capture_output=True, text=True, env=env, input=stdin)
        lines = calls.read_text().splitlines() if calls.exists() else []
        return proc, lines

    run.env = env  # for tests that extend the environment
    run.run = run  # fake_sbatch.run(script, ..., env=..., stdin=...)
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


# ---- M2D2 launchers (M2D2_PROMPT.md part 8) ----

M2D2_LR = ["m2d2-lr-base-2.5e-4", "m2d2-lr-base-5e-4", "m2d2-lr-base-1e-3",
           "m2d2-lr-win16-2.5e-4", "m2d2-lr-win16-5e-4", "m2d2-lr-win16-1e-3"]
M2D2_MAIN_ORDER = ["m2d2-base-s0", "m2d2-win1-s0", "m2d2-win16-s0",
                   "m2d2-base-s1", "m2d2-win16-s1", "m2d2-base-s2", "m2d2-win16-s2",
                   "m2d2-base-s3", "m2d2-win16-s3", "m2d2-base-s4", "m2d2-win16-s4", "m2d2-win64-s0"]


def test_submit_m2d2_lr_check(fake_sbatch):
    proc, calls = fake_sbatch("submit_m2d2_lr_check.sh", "b200")
    assert proc.returncode == 0, proc.stderr
    assert submitted_runs(calls[:6]) == M2D2_LR
    assert all("-t 00:20:00" in c and "--gres=gpu:b200:1" in c for c in calls[:6])
    chain = calls[6]
    for part in ["-p mit_normal ", "--no-requeue", "--dependency=afterany:101:102:103:104:105:106",
                 "submit_m2d2_lr_check.sh --chain b200"]:
        assert part in chain, (part, chain)
    assert len(calls) == 7


def chain_env(tmp_path, fake_sbatch_env):
    """A temporary code copy (so the chain writes its own chosen_lr_m2d2.py) and fake module/conda/python."""
    code = tmp_path / "code"
    for d in ("analysis", "config", "scripts"):
        shutil.copytree(os.path.join(REPO, d), code / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("train.py", "model.py", "configurator.py"):
        shutil.copy(os.path.join(REPO, f), code / f)
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    for name, body in {"module": "exit 0", "conda": "echo /nonexistent", "python": f'exec {sys.executable} "$@"'}.items():
        (fakebin / name).write_text(f"#!/bin/bash\n{body}\n")
        (fakebin / name).chmod(0o755)
    return dict(fake_sbatch_env, CODE_DIR=str(code), PROJECT=str(tmp_path / "project"),
                PATH=f"{fakebin}:{fake_sbatch_env['PATH']}"), code


def write_lr_runs(out_root, finals):
    for name, v in finals.items():
        d = out_root / name
        d.mkdir(parents=True)
        recs = [{"type": "header", "run_name": name, "config": {"eval_split": "calib"}},
                {"type": "final", "step": 458, "tokens": 30_015_488, "val": {"history": v}, "val_overall": v}]
        (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_m2d2_chain_submits_main_runs(fake_sbatch, tmp_path):
    env, code = chain_env(tmp_path, fake_sbatch.env)
    write_lr_runs(tmp_path / "runs", dict(zip(M2D2_LR, [3.40, 3.305, 3.30, 3.45, 3.36, 3.38])))
    repo_lr = open(os.path.join(REPO, "config", "chosen_lr_m2d2.py")).read()
    proc, calls = fake_sbatch.run("submit_m2d2_lr_check.sh", "--chain", "b200", env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "calibration split" in proc.stdout
    assert "LR_BASE = 0.0005" in proc.stdout and "LR_WIN = 0.0005" in proc.stdout  # printed chosen LRs
    assert submitted_runs(calls) == M2D2_MAIN_ORDER
    assert all("-t 00:40:00" in c and "-c 4" in c for c in calls)
    assert "LR_BASE = 0.0005" in (code / "config" / "chosen_lr_m2d2.py").read_text()
    assert open(os.path.join(REPO, "config", "chosen_lr_m2d2.py")).read() == repo_lr  # repo untouched


def test_m2d2_chain_analysis_failure_submits_nothing(fake_sbatch, tmp_path):
    env, _ = chain_env(tmp_path, fake_sbatch.env)
    (tmp_path / "runs").mkdir()  # no LR runs finished
    write_lr_runs(tmp_path / "runs", {"m2d2-base-s0": 3.5})
    proc, calls = fake_sbatch.run("submit_m2d2_lr_check.sh", "--chain", "b200", env=env)
    assert proc.returncode != 0 and "submitting no main runs" in proc.stderr
    assert calls == []


def test_submit_slim120(fake_sbatch):
    proc, calls = fake_sbatch("submit_slim120.sh", "b200")
    assert proc.returncode == 0, proc.stderr
    assert submitted_runs(calls) == ["slim120-base-s0", "slim120-win16-s0", "slim120-base-s1", "slim120-win16-s1",
                                     "slim120-base-s2", "slim120-win16-s2"]
    assert all("-t 00:40:00" in c for c in calls)


def test_submit_locked_test_asks_first(fake_sbatch):
    proc, calls = fake_sbatch.run("submit_locked_test.sh", "b200", stdin="n\n")
    assert proc.returncode != 0 and "the locked test is used once; continue? [y/N]" in proc.stdout + proc.stderr
    assert "not submitted" in proc.stdout and calls == []
    proc, calls = fake_sbatch.run("submit_locked_test.sh", "b200", stdin="")  # no answer = no
    assert proc.returncode != 0 and calls == []
    proc, calls = fake_sbatch.run("submit_locked_test.sh", "b200", stdin="y\n")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 1
    for part in ["--gres=gpu:b200:1", "-J locked-test", "--no-requeue", "submit_locked_test.sh --in-job"]:
        assert part in calls[0], (part, calls[0])


def test_status_lists_m2d2_runs(tmp_path):
    names = M2D2_LR + M2D2_MAIN_ORDER + ["slim120-base-s0", "slim120-win16-s0"]
    for name in names:
        (tmp_path / "runs" / name).mkdir(parents=True)
    proc = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "status.py"), str(tmp_path / "runs")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert sorted(line.split()[0] for line in proc.stdout.splitlines()[1:]) == sorted(names)


def test_submit_locked_test_covers_all_designs(fake_sbatch):
    text = open(os.path.join(REPO, "scripts", "slurm", "submit_locked_test.sh")).read()
    assert "--reference_data_dir" in text and "--data_dir" not in text.split("--in-job", 1)[1]
    proc, calls = fake_sbatch.run("submit_locked_test.sh", "b200", stdin="y\n")
    assert proc.returncode == 0, proc.stderr
    listed = proc.stdout.split("Runs: ", 1)[1].splitlines()[0].split()
    want = ([f"m2d2-{a}-s{s}" for a in ("base", "win16") for s in range(5)]
            + [f"{p}-{a}-s{s}" for p in ("m2d2h4", "m2d2r4") for s in range(4) for a in ("base", "win16")])
    assert sorted(listed) == sorted(want) and len(listed) == 26
