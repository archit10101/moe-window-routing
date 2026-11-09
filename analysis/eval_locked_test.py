#!/usr/bin/env python
"""Evaluate final checkpoints on the locked test split, once (docs/M2D2_PROMPT.md parts 2 and 7).

    python analysis/eval_locked_test.py --runs_dir <dir> --data_dir <dir> --out_dir <dir>
        --runs m2d2-base-s0,...,m2d2-win16-s4 [--device cuda] [--log_file <path>]
        [--allow_reuse_of_locked_test]

Uses train.py's own evaluation (train.evaluate) and offset scheme (train.load_val_sets) on
test_<source>.bin instead of val_<source>.bin, with every run's final-eval settings. Writes
<run>/test_results.json per run and out_dir/summary.md (+ test_summary.json) with the decision-rule
quantities of analyze.py.

Every invocation is appended to the log (default $PROJECT/analysis/locked_test_log.txt) with a
timestamp, the git commit and the run list. If the log already records a completed evaluation, the
script refuses to run unless --allow_reuse_of_locked_test is given; that use is logged and the
summary is marked "test set reused".
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import nullcontext

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402
import train as T  # noqa: E402

PROJECT = os.environ.get("PROJECT", "/home/architta/orcd/scratch/moe-pilot")
DEFAULT_LOG = os.path.join(PROJECT, "analysis", "locked_test_log.txt")


def git_state():
    commit, dirty = T.git_info()
    return f"{commit or 'unknown'}{' (dirty)' if dirty else ''}"


def log_event(path, event, **fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    parts = [time.strftime("%Y-%m-%d %H:%M:%S"), event] + [f"{k}={v}" for k, v in fields.items()]
    with open(path, "a") as f:
        f.write(" | ".join(parts) + "\n")


def previous_completions(path):
    if not os.path.exists(path):
        return []
    return [line.strip() for line in open(path) if " | COMPLETED | " in line]


def arm_of(name):
    """('base' | 'win16' | 'win1' | 'win64', seed) from a run name such as m2d2-base-s3, else (None, None)."""
    stem = name.rsplit("-s", 1)
    if len(stem) != 2 or not stem[1].isdigit():
        return None, None
    for arm in ("win16", "win64", "win1", "base"):
        if stem[0] == arm or stem[0].endswith("-" + arm):
            return arm, int(stem[1])
    return None, None


@torch.no_grad()
def evaluate_run(run_dir, data_dir, device):
    ckpt = torch.load(os.path.join(run_dir, "ckpt", "latest.pt"), map_location="cpu", weights_only=False)
    cfg = T.with_defaults(ckpt["config"])
    cfg.update(data_dir=data_dir, device=device)
    model = T.build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    step = ckpt["step"]
    del ckpt
    is_cuda = device.startswith("cuda")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if is_cuda and cfg["dtype"] == "bfloat16" else nullcontext()
    test_sets = T.load_val_sets(data_dir, cfg, split="test", allow_locked_test=True)
    test, overall, _ = T.evaluate(model, test_sets, cfg["final_eval_batches"], cfg, device, ctx)
    del model
    if is_cuda:
        torch.cuda.empty_cache()
    return dict(test=test, test_overall=overall, step=step, tokens=step * cfg["batch_size"] * cfg["block_size"],
                final_eval_batches=cfg["final_eval_batches"], eval_batch_size=cfg["eval_batch_size"])


def summarize(results, reused):
    by_arm = {}
    for name, r in results.items():
        arm, seed = arm_of(name)
        if arm:
            by_arm.setdefault(arm, {})[seed] = r
    B = [by_arm["base"][s]["test_overall"] for s in sorted(by_arm.get("base", {}))]
    W = [by_arm["win16"][s]["test_overall"] for s in sorted(by_arm.get("win16", {}))]
    win1 = by_arm.get("win1", {}).get(0, {}).get("test_overall")
    win64 = by_arm.get("win64", {}).get(0, {}).get("test_overall")
    res = A.decide_available(B, W, win1, win64) if len(B) >= 2 and len(W) >= 2 else None

    sources = sorted(set.intersection(*[set(r["test"]) for r in results.values()]))
    L = ["# Locked test split: results", ""]
    if reused:
        L += ["**TEST SET REUSED**: this evaluation ran with --allow_reuse_of_locked_test after the locked test "
              "split had already been evaluated. Treat these numbers as no longer held out.", ""]
    L += [f"Runs: {', '.join(results)}. Same evaluation code and offset scheme as the final eval, on test_*.bin.", ""]
    if res is None:
        L += ["No decision quantities: need at least 2 base-s* and 2 win16-s* runs.", ""]
    else:
        L += [f"**Verdict (test split): {res['verdict']}**", "",
              "Decision rule as in analyze.py: s = sqrt((var(B) + var(W)) / 2), ddof = 1; gap = mean(B) - mean(W) "
              "(positive = window arm better).", "",
              "| quantity | value |", "|---|---|", f"| mean(B) | {res['mean_B']:.5f} |",
              f"| mean(W) | {res['mean_W']:.5f} |", f"| gap = mean(B) - mean(W) | {res['gap']:+.5f} |",
              f"| s | {res['s']:.5f} |", f"| 2s | {2 * res['s']:.5f} |"]
        if res["skipped"]:
            L += ["", "Skipped criteria: " + "; ".join(res["skipped"]) + "."]
        L.append("")
    L += ["## test_overall per run", "", "| run | test_overall | step |", "|---|---|---|"]
    L += [f"| {n} | {r['test_overall']:.5f} | {r['step']} |" for n, r in results.items()]
    if "base" in by_arm and "win16" in by_arm:
        L += ["", "## Per source (mean over seeds; win16 - base)", "", "| source | base | win16 | win16 - base |",
              "|---|---|---|---|"]
        for src in sources:
            b = sum(r["test"][src] for r in by_arm["base"].values()) / len(by_arm["base"])
            w = sum(r["test"][src] for r in by_arm["win16"].values()) / len(by_arm["win16"])
            L.append(f"| {src} | {b:.5f} | {w:.5f} | {w - b:+.5f} |")
    return "\n".join(L) + "\n", res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run names")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_file", default=DEFAULT_LOG)
    ap.add_argument("--allow_reuse_of_locked_test", action="store_true",
                    help="evaluate even though the log records a completed locked-test evaluation")
    args = ap.parse_args()
    names = [r.strip() for r in args.runs.split(",") if r.strip()]
    git = git_state()
    info = dict(git=git, host=socket.gethostname(), runs=",".join(names), data_dir=args.data_dir)

    done = previous_completions(args.log_file)
    if done and not args.allow_reuse_of_locked_test:
        log_event(args.log_file, "REFUSED", reason="a completed evaluation is already on record", **info)
        sys.exit("eval_locked_test.py: the locked test split was already evaluated:\n  " + "\n  ".join(done)
                 + "\nRefusing to evaluate it again. Pass --allow_reuse_of_locked_test only if you accept that "
                   "the test split is no longer held out (the reuse is logged and marked in summary.md).")
    missing = [n for n in names if not os.path.exists(os.path.join(args.runs_dir, n, "ckpt", "latest.pt"))]
    if missing:
        log_event(args.log_file, "REFUSED", reason=f"missing checkpoints {missing}", **info)
        sys.exit(f"eval_locked_test.py: no final checkpoint for {missing} in {args.runs_dir}")
    for n in names:
        if not os.path.exists(os.path.join(args.runs_dir, n, "done")):
            print(f"WARNING: {n} has no done file; evaluating its latest checkpoint anyway")

    if args.allow_reuse_of_locked_test:
        log_event(args.log_file, "REUSE", reason="--allow_reuse_of_locked_test", previous=len(done), **info)
    log_event(args.log_file, "START", **info)

    results = {}
    for n in names:
        print(f"evaluating {n} on the locked test split ({args.device})", flush=True)
        r = evaluate_run(os.path.join(args.runs_dir, n), args.data_dir, args.device)
        r.update(run_name=n, split="test", evaluated_at=time.strftime("%Y-%m-%d %H:%M:%S"), git=git,
                 test_set_reused=args.allow_reuse_of_locked_test)
        with open(os.path.join(args.runs_dir, n, "test_results.json"), "w") as f:
            json.dump(r, f, indent=1)
        results[n] = r
        print(f"  {n}: test_overall {r['test_overall']:.5f}", flush=True)

    text, res = summarize(results, args.allow_reuse_of_locked_test)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(text)
    with open(os.path.join(args.out_dir, "test_summary.json"), "w") as f:
        json.dump(dict(runs=names, results={n: dict(test=r["test"], test_overall=r["test_overall"])
                                            for n, r in results.items()},
                       decision=res, test_set_reused=args.allow_reuse_of_locked_test, git=git), f, indent=1)
    log_event(args.log_file, "COMPLETED", verdict=(res or {}).get("verdict"),
              gap=None if res is None else f"{res['gap']:+.5f}", reused=args.allow_reuse_of_locked_test, **info)
    print(text)


if __name__ == "__main__":
    main()
