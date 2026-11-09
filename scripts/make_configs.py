#!/usr/bin/env python
"""Generate config/runs/<name>.py for every run in the pilot (docs/PROMPT.md part 5).

Each generated file execs config/pilot_base.py and overrides only what differs.
Main runs read their LR from config/chosen_lr.py and stop with a clear message
while it is still None. smoke-cpu.py is written by hand and is not touched.

Usage (from the repository root):
    python scripts/make_configs.py [--out config/runs]
"""
import argparse
import os

M = 1_000_000
BASE = [0] * 8
M2D2_BASE = "config/pilot_m2d2.py"
M2D2_LR_FILE = "config/chosen_lr_m2d2.py"


def win(n):
    return [0, 0, 0, 0, n, n, n, n]


def runs():
    """name -> dict(window_layers, seed, lr, total_tokens, extra). lr is a float or 'LR_BASE'/'LR_WIN'."""
    table = {}
    for lr_name, lr in (("1e-3", 1e-3), ("2e-3", 2e-3), ("4e-3", 4e-3)):
        table[f"lr-base-{lr_name}"] = dict(window_layers=BASE, seed=0, lr=lr, total_tokens=100 * M)
        table[f"lr-win16-{lr_name}"] = dict(window_layers=win(16), seed=0, lr=lr, total_tokens=100 * M)
    for s in range(5):  # seeds 3 and 4 are the optional extra seeds
        table[f"base-s{s}"] = dict(window_layers=BASE, seed=s, lr="LR_BASE", total_tokens=500 * M)
        table[f"win16-s{s}"] = dict(window_layers=win(16), seed=s, lr="LR_WIN", total_tokens=500 * M)
    # sanity check: n=1 is per-token routing, so it uses the baseline's LR and seed and should
    # track base-s0 almost exactly (a deliberate change from the PROMPT.md table, which says LR_WIN)
    table["win1-s0"] = dict(window_layers=win(1), seed=0, lr="LR_BASE", total_tokens=500 * M)
    table["win64-s0"] = dict(window_layers=win(64), seed=0, lr="LR_WIN", total_tokens=500 * M)
    smoke_extra = dict(log_interval=10, eval_interval=100)
    table["smoke-gpu-base"] = dict(window_layers=BASE, seed=0, lr=2e-3, total_tokens=200 * 65536, extra=smoke_extra)
    table["smoke-gpu-win16"] = dict(window_layers=win(16), seed=0, lr=2e-3, total_tokens=200 * 65536,
                                    extra=smoke_extra)

    # follow-up runs (docs/FOLLOWUP_PROMPT.md part 6): the LR used by the main runs (5e-4), written
    # literally rather than read from chosen_lr.py; 500M tokens; router gradients always logged
    def followup(windows, seed, **extra):
        return dict(window_layers=windows, seed=seed, lr=5e-4, total_tokens=500 * M,
                    extra={**extra, "log_router_grad": True})

    for s in (0, 1):
        table[f"both16-s{s}"] = followup(win(16), s, router_input="both")                    # E5
        table[f"win16top1-s{s}"] = followup([0, 0, 0, 0, 0, 0, 0, 16], s)                     # E3
        table[f"win16top2-s{s}"] = followup([0, 0, 0, 0, 0, 0, 16, 16], s)                    # E3
        table[f"base-sh2048-s{s}"] = followup(BASE, s, shared_hidden=2048)                    # E2
        table[f"win16-sh2048-s{s}"] = followup(win(16), s, shared_hidden=2048)                # E2
        table[f"base-rlr3-s{s}"] = followup(BASE, s, router_lr_mult=3.0)                      # E4
        table[f"win16-rlr3-s{s}"] = followup(win(16), s, router_lr_mult=3.0)                  # E4
    table["base-rg-s0"] = followup(BASE, 0)                                                   # E4
    table["win16-rg-s0"] = followup(win(16), 0)                                               # E4

    # M2D2-Wiki-8 replication (docs/M2D2_PROMPT.md part 6): config/pilot_m2d2.py sets the data folder,
    # the 120M budget and eval_interval 100; main runs read only config/chosen_lr_m2d2.py
    def m2d2(windows, seed, lr, tokens=120 * M, eval_split="val"):
        return dict(window_layers=windows, seed=seed, lr=lr, total_tokens=tokens, base_config=M2D2_BASE,
                    lr_file=M2D2_LR_FILE if isinstance(lr, str) else None, extra={"eval_split": eval_split})

    for lr_name, lr in (("2.5e-4", 2.5e-4), ("5e-4", 5e-4), ("1e-3", 1e-3)):
        table[f"m2d2-lr-base-{lr_name}"] = m2d2(BASE, 0, lr, tokens=30 * M, eval_split="calib")
        table[f"m2d2-lr-win16-{lr_name}"] = m2d2(win(16), 0, lr, tokens=30 * M, eval_split="calib")
    for s in range(5):
        table[f"m2d2-base-s{s}"] = m2d2(BASE, s, "LR_BASE")
        table[f"m2d2-win16-s{s}"] = m2d2(win(16), s, "LR_WIN")
    table["m2d2-win1-s0"] = m2d2(win(1), 0, "LR_BASE")
    table["m2d2-win64-s0"] = m2d2(win(64), 0, "LR_WIN")

    # uneven designs (docs/M2D2_UNEVEN_PROMPT.md part 5): 165M tokens on builds where one domain has 4x the
    # training data; same recipe and LRs (chosen_lr_m2d2.py) as the even main runs
    for prefix, base_config in (("m2d2h4", "config/pilot_m2d2_hist4x.py"), ("m2d2r4", "config/pilot_m2d2_rep4x.py")):
        for s in range(4):
            for arm, windows, lr in (("base", BASE, "LR_BASE"), ("win16", win(16), "LR_WIN")):
                table[f"{prefix}-{arm}-s{s}"] = dict(m2d2(windows, s, lr, tokens=165 * M), base_config=base_config)

    # SlimPajama control at the same 120M budget, with the SlimPajama LR (5e-4, both arms)
    for s in range(3):
        for arm, windows in (("base", BASE), ("win16", win(16))):
            table[f"slim120-{arm}-s{s}"] = dict(window_layers=windows, seed=s, lr=5e-4, total_tokens=120 * M,
                                                extra={"eval_interval": 100, "eval_split": "val"})
    return table


def render(name, spec):
    base = spec.get("base_config", "config/pilot_base.py")
    lines = [f"# {name}: generated by scripts/make_configs.py; edit the table there, not this file.",
             f"exec(open('{base}').read())"]
    lr = spec["lr"]
    if isinstance(lr, str) and spec.get("lr_file") == M2D2_LR_FILE:
        lines += [
            f"exec(open('{M2D2_LR_FILE}').read())",
            "if LR_BASE is None or LR_WIN is None:",
            f"    raise SystemExit('{name}: LR_BASE / LR_WIN in {M2D2_LR_FILE} are not both set. Finish the M2D2 LR '",
            "                     'check, then run: python analysis/analyze.py --runs_dir <runs> --out_dir <dir> '",
            f"                     '--prefix m2d2- --lr_check --write --lr_file {M2D2_LR_FILE}')",
        ]
    elif isinstance(lr, str):
        lines += [
            "exec(open('config/chosen_lr.py').read())",
            f"if {lr} is None:",
            f"    raise SystemExit('{name}: {lr} in config/chosen_lr.py is still None. Finish the LR check, '",
            "                     'then run: python analysis/analyze.py --runs_dir <runs> --out_dir <dir> --lr_check --write')",
        ]
    lines += [
        "",
        f"run_name = {name!r}",
        f"seed = {spec['seed']}",
        f"window_layers = {spec['window_layers']!r}",
        f"lr = {lr if isinstance(lr, str) else repr(lr)}",
        f"total_tokens = {spec['total_tokens']:_}",
    ]
    for k, v in spec.get("extra", {}).items():
        lines.append(f"{k} = {v!r}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("config", "runs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, spec in runs().items():
        with open(os.path.join(args.out, f"{name}.py"), "w") as f:
            f.write(render(name, spec))
    print(f"wrote {len(runs())} configs to {args.out}")


if __name__ == "__main__":
    main()
