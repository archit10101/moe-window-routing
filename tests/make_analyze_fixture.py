#!/usr/bin/env python
"""Capture analysis/analyze.py's default outputs into tests/fixtures/analyze_reference.json.

Captured from the code before the M2D2 options (--prefix, --lr_file) were added (commit c47e9fe).
test_analyze_prefix_and_default_unchanged reruns capture() and requires identical results: every
output file's SHA-256 and the printed text, for main mode, the no-verdict case, the LR check with
--write, and --partial. Do NOT rerun this script on later code.

Usage (repo root): .venv/bin/python tests/make_analyze_fixture.py
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

import matplotlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import make_synthetic_data  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "analyze_reference.json")
ANALYZE = os.path.join(REPO, "analysis", "analyze.py")
MAIN_RUNS = ["base-s0", "base-s1", "base-s2", "win16-s0", "win16-s1", "win16-s2", "win1-s0", "win64-s0"]
LR_RUNS = {"lr-base-1e-3": 3.30, "lr-base-2e-3": 3.205, "lr-base-4e-3": 3.20,
           "lr-win16-1e-3": 3.40, "lr-win16-2e-3": 3.25, "lr-win16-4e-3": 3.35}
# distinct final values so the decision rule has real numbers to work with
MAIN_FINALS = {"base-s0": 3.60, "base-s1": 3.62, "base-s2": 3.61, "win16-s0": 3.66, "win16-s1": 3.64,
               "win16-s2": 3.65, "win1-s0": 3.605, "win64-s0": 3.67}


def smoke_run(workdir):
    data = os.path.join(workdir, "data")
    make_synthetic_data.make(data)
    proc = subprocess.run([sys.executable, "train.py", "config/runs/smoke-cpu.py", f"--data_dir={data}",
                           f"--out_root={os.path.join(workdir, 'train')}"], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return os.path.join(workdir, "train", "smoke-cpu")


def copy_runs(smoke, dest, finals):
    for name, value in finals.items():
        shutil.copytree(smoke, os.path.join(dest, name), ignore=shutil.ignore_patterns("ckpt"))
        path = os.path.join(dest, name, "log.jsonl")
        recs = [json.loads(l) for l in open(path)]
        for r in recs:
            if r["type"] == "final":
                r["val_overall"] = value
        with open(path, "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))


def run(args, workdir):
    proc = subprocess.run([sys.executable, ANALYZE] + args, capture_output=True, text=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout.replace(workdir, "<WORK>")}


def outputs(out_dir):
    return {name: hashlib.sha256(open(os.path.join(out_dir, name), "rb").read()).hexdigest()
            for name in sorted(os.listdir(out_dir))}


def capture(workdir, smoke=None):
    smoke = smoke or smoke_run(workdir)
    res = {"platform": platform.platform(), "matplotlib": matplotlib.__version__}

    main = os.path.join(workdir, "main_runs")
    copy_runs(smoke, main, MAIN_FINALS)
    res["main"] = run(["--runs_dir", main, "--out_dir", os.path.join(workdir, "out_main")], workdir)
    res["main"]["files"] = outputs(os.path.join(workdir, "out_main"))

    few = os.path.join(workdir, "few_runs")
    copy_runs(smoke, few, {k: MAIN_FINALS[k] for k in ("base-s0", "win16-s0")})
    res["no_verdict"] = run(["--runs_dir", few, "--out_dir", os.path.join(workdir, "out_few")], workdir)
    res["no_verdict"]["files"] = outputs(os.path.join(workdir, "out_few"))

    lr = os.path.join(workdir, "lr_runs")
    copy_runs(smoke, lr, LR_RUNS)
    chosen = os.path.join(workdir, "chosen_lr.py")
    res["lr_check"] = run(["--runs_dir", lr, "--out_dir", os.path.join(workdir, "out_lr"), "--lr_check", "--write",
                           "--chosen_lr_path", chosen], workdir)
    res["lr_check"]["chosen_lr"] = "".join(l for l in open(chosen) if not l.startswith("# Written by"))

    res["partial"] = run(["--runs_dir", main, "--partial"], workdir)
    return res


def main():
    if os.path.exists(FIXTURE) and "--force" not in sys.argv:
        sys.exit(f"{FIXTURE} exists; it must stay the pre-M2D2 reference")
    with tempfile.TemporaryDirectory() as d:
        ref = capture(os.path.realpath(d))
    with open(FIXTURE, "w") as f:
        json.dump(ref, f, indent=1)
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
