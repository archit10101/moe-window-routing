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
                 "slurm/submit_m2d2_all.sh", "slurm/after_m2d2_all.sh", "slurm/m2d2_common.sh",
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


# ---- submit_m2d2_all.sh / after_m2d2_all.sh (M2D2_UNEVEN_PROMPT.md part 8) ----

ALL_ORDER = (["m2d2-base-s0", "m2d2-win1-s0", "m2d2-win16-s0",
              "m2d2h4-base-s0", "m2d2h4-win16-s0", "m2d2r4-base-s0", "m2d2r4-win16-s0"]
             + [f"{p}-{a}-s{s}" for s in (1, 2, 3) for p in ("m2d2", "m2d2h4", "m2d2r4") for a in ("base", "win16")]
             + ["m2d2-base-s4", "m2d2-win16-s4", "m2d2-win64-s0"])
M2D2_CONFIGS = ["Technology_and_applied_sciences", "Natural_and_physical_sciences", "Mathematics_and_logic",
                "Health_and_fitness", "History_and_events", "Culture_and_the_arts", "Society_and_social_sciences",
                "Philosophy_and_thinking"]


def project_with_raw(tmp_path):
    project = tmp_path / "project"
    for c in M2D2_CONFIGS:
        (project / "raw_m2d2" / c).mkdir(parents=True)
        for f in ("train.txt", "valid.txt", "test.txt"):
            (project / "raw_m2d2" / c / f).write_text("x\n")
    return project


def fake_build(folder, uneven=False, ok=True):
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text("{}")
    lines = ["OK: none of the 10 doc_hashes appears in two splits or two domains." if ok else "something else"]
    if uneven and ok:
        lines.append("OK: all test/val/calib files are byte-identical to the reference build /x (24 files)")
    (folder / "sanity.txt").write_text("\n".join(lines) + "\n")


def job_names(calls):
    return [c.split(" -J ")[1].split()[0] for c in calls]


def test_m2d2_common_configs_match_prepare():
    sys.path.insert(0, os.path.join(REPO, "data", "m2d2"))
    import prepare
    text = open(os.path.join(REPO, "scripts", "slurm", "m2d2_common.sh")).read()
    assert sorted(prepare.DOMAINS.values()) == sorted(M2D2_CONFIGS)
    assert all(c in text for c in M2D2_CONFIGS)
    assert prepare.REFERENCE_OK in text  # the sanity line build_ok looks for


def test_submit_all_dry_run(fake_sbatch, tmp_path):
    project = project_with_raw(tmp_path)
    env = dict(fake_sbatch.env, PROJECT=str(project))
    proc, calls = fake_sbatch.run("submit_m2d2_all.sh", "b200", "--dry_run", env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert calls == []  # nothing submitted
    sb = [l[len("DRY RUN: sbatch "):] for l in proc.stdout.splitlines() if l.startswith("DRY RUN: sbatch ")]
    names = job_names(sb)
    assert names == (["build-m2d2-even", "build-m2d2-hist4x", "build-m2d2-rep4x"] + M2D2_LR + ["m2d2-after"]
                     + ["slim120-base-s0", "slim120-win16-s0", "slim120-base-s1", "slim120-win16-s1",
                        "slim120-base-s2", "slim120-win16-s2"])
    assert "--dependency" not in sb[0] and "build_m2d2.sbatch title" in sb[0]
    for c in sb[1:3] + sb[3:9]:
        assert "--dependency=afterok:900" in c, c
    assert "--train_quota_override history=60000000 --variant hist4x --reference_dir" in sb[1]
    assert "--train_quota_override philosophy=60000000 --fallback_domain natural_sciences --variant rep4x" in sb[2]
    assert "-p mit_normal --no-requeue" in sb[9] and "--dependency=afterany:903:904:905:906:907:908" in sb[9]
    assert sb[9].endswith("after_m2d2_all.sh b200 901 902")
    planned = [l.split()[3] for l in proc.stdout.splitlines() if l.startswith("DRY RUN: submit.sh ")]
    assert planned == ALL_ORDER
    plan = {l.split()[3]: l for l in proc.stdout.splitlines() if l.startswith("DRY RUN: submit.sh ")}
    assert "00:50:00" in plan["m2d2h4-base-s2"] and "--dependency=afterok:901" in plan["m2d2h4-base-s2"]
    assert "--dependency=afterok:902" in plan["m2d2r4-win16-s3"] and "00:40:00" in plan["m2d2-win64-s0"]
    assert "dependency tree:" in proc.stdout and "would submit 28 runs" in proc.stdout


def test_submit_all_real_and_skips_existing_builds(fake_sbatch, tmp_path):
    project = project_with_raw(tmp_path)
    env = dict(fake_sbatch.env, PROJECT=str(project))
    proc, calls = fake_sbatch.run("submit_m2d2_all.sh", "b200", env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(calls) == 16 and job_names(calls)[:3] == ["build-m2d2-even", "build-m2d2-hist4x", "build-m2d2-rep4x"]
    assert all("--dependency=afterok:101" in c for c in calls[1:9])
    assert "--dependency=afterany:104:105:106:107:108:109" in calls[9] and calls[9].endswith("b200 102 103")
    assert all("--dependency" not in c for c in calls[10:])

    # even and hist4x already built and checked: only rep4x is built, with no dependency to wait for
    project2 = project_with_raw(tmp_path / "second")
    fake_build(project2 / "data_m2d2w8")
    fake_build(project2 / "data_m2d2w8_hist4x", uneven=True)
    fake_build(project2 / "data_m2d2w8_rep4x_unused")  # unrelated folder, ignored
    calls_before = len(calls)
    proc, calls = fake_sbatch.run("submit_m2d2_all.sh", "b200", env=dict(env, PROJECT=str(project2)))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    new = calls[calls_before:]
    assert job_names(new)[:7] == ["build-m2d2-rep4x"] + M2D2_LR
    assert all("--dependency" not in c for c in new[:7])
    assert new[7].endswith(f"after_m2d2_all.sh b200 none {100 + calls_before + 1}")
    assert "skipping" in proc.stdout


def test_submit_all_refuses(fake_sbatch, tmp_path):
    env = dict(fake_sbatch.env, PROJECT=str(tmp_path / "empty"))
    proc, calls = fake_sbatch.run("submit_m2d2_all.sh", "b200", env=env)
    assert proc.returncode != 0 and "prepare.py --download --raw_dir" in proc.stderr and calls == []
    project = project_with_raw(tmp_path)
    fake_build(project / "data_m2d2w8", ok=False)
    proc, calls = fake_sbatch.run("submit_m2d2_all.sh", "b200", env=dict(env, PROJECT=str(project)))
    assert proc.returncode != 0 and "move it away" in proc.stderr and calls == []


def after_env(tmp_path, fake_sbatch, squeue_state):
    env, code = chain_env(tmp_path, fake_sbatch.env)
    write_lr_runs(tmp_path / "runs", dict(zip(M2D2_LR, [3.40, 3.305, 3.30, 3.45, 3.36, 3.38])))
    sq = tmp_path / "fakebin" / "squeue"
    sq.write_text(f"#!/bin/bash\necho '{squeue_state}'\n")
    sq.chmod(0o755)
    return env, code


def test_after_m2d2_all_submits_in_order(fake_sbatch, tmp_path):
    env, code = after_env(tmp_path, fake_sbatch, "PENDING")
    fake_build(tmp_path / "project" / "data_m2d2w8_rep4x", uneven=True)  # rep4x already built and checked
    proc, calls = fake_sbatch.run("after_m2d2_all.sh", "b200", "777", "none", env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LR_BASE = 0.0005" in proc.stdout
    assert submitted_runs(calls) == ALL_ORDER
    for c, run in zip(calls, ALL_ORDER):
        assert ("-t 00:50:00" if run.startswith(("m2d2h4", "m2d2r4")) else "-t 00:40:00") in c, c
        assert ("--dependency=afterok:777" in c) == run.startswith("m2d2h4-"), c


def test_after_m2d2_all_failed_build_and_failed_analysis(fake_sbatch, tmp_path):
    env, code = after_env(tmp_path, fake_sbatch, "")  # the hist4x build job is gone and its folder is not built
    fake_build(tmp_path / "project" / "data_m2d2w8_rep4x", uneven=True)
    proc, calls = fake_sbatch.run("after_m2d2_all.sh", "b200", "778", "none", env=env)
    assert proc.returncode != 0 and "hist4x build failed" in proc.stderr
    assert submitted_runs(calls) == [r for r in ALL_ORDER if not r.startswith("m2d2h4-")]

    env2, _ = chain_env(tmp_path / "noanalysis", fake_sbatch.env)
    (tmp_path / "runs").mkdir(exist_ok=True)
    proc, calls2 = fake_sbatch.run("after_m2d2_all.sh", "b200", "none", "none",
                                   env=dict(env2, OUT_ROOT=str(tmp_path / "empty_runs")))
    assert proc.returncode != 0 and "submitting no runs" in proc.stderr and len(calls2) == len(calls)


def test_status_lists_builds(tmp_path):
    project = tmp_path / "project"
    fake_build(project / "data_m2d2w8")
    fake_build(project / "data_m2d2w8_hist4x", uneven=True, ok=False)
    (project / "data_m2d2w8_rep4x").mkdir(parents=True)
    (project / "runs" / "m2d2h4-base-s0").mkdir(parents=True)
    proc = subprocess.run(["bash", os.path.join(REPO, "scripts", "slurm", "status.sh")], capture_output=True,
                          text=True, env=dict(os.environ, PROJECT=str(project), PYTHON=sys.executable,
                                              PATH="/usr/bin:/bin"))
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "m2d2h4-base-s0" in out and "M2D2 data builds:" in out
    assert "data_m2d2w8            checks passed" in out
    assert "data_m2d2w8_hist4x     CHECKS NOT PASSED" in out
    assert "data_m2d2w8_rep4x      no build yet" in out
