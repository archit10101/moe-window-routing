#!/usr/bin/env python
"""Compare the window-routing penalty across datasets (docs/M2D2_PROMPT.md part 7).

    python analysis/compare_datasets.py --runs_dir <dir> --out_dir <dir> [--with_test]

Three groups, all computed from the runs' logs (nothing hard-coded):
  SlimPajama-500M   the existing pilot runs (base-s*, win16-s*, win1-s0, win64-s0)
  SlimPajama-120M   slim120-* (same 120M budget and schedule length as M2D2)
  M2D2-Wiki-8       m2d2-*

gap = mean(win16) - mean(base) in final val_overall: the window arm's penalty (positive = worse).
Note the sign is the opposite of analyze.py's gap (mean(B) - mean(W)). Verdicts use analyze.py's
decision rule, skipping criteria whose runs a group lacks.

Main comparison: delta = gap_M2D2 - gap_Slim120, with a rough scale sqrt(s_M^2/n_M + s_S^2/n_S)
(n = seeds per arm); "clearly" means beyond 2 x scale.

--with_test also reads the M2D2 runs' test_results.json (written once by eval_locked_test.py).
Without it, nothing here reads test results.
"""
import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

GROUPS = [("SlimPajama-500M", ""), ("SlimPajama-120M", "slim120-"), ("M2D2-Wiki-8", "m2d2-")]
SAMPLING_NOTE = ("Training samples random 1,024-token windows with replacement. M2D2-Wiki-8 trains 120M tokens over "
                 "~120M training tokens: ~63% of positions seen at least once (1 - e^-1), ~26% two or more times; both "
                 "M2D2 arms share this regime. SlimPajama-120M draws 120M tokens from 600M (~18% coverage, almost no "
                 "repeats); the SlimPajama-500M pilot covered ~57% (500M over 600M).")


def nanfmt(x, spec):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def group_runs(all_runs, prefix):
    return A.select_prefix(all_runs, prefix) if prefix else all_runs


def header_lr(run):
    return ((run.get("header") or {}).get("config") or {}).get("lr")


def arm_finals(runs, arm, value=A.final_val):
    return {int(r["name"].rsplit("-s", 1)[1]): value(r) for r in A.seed_runs(runs, arm) if value(r) is not None}


def spread(B, W):
    if len(B) < 2 or len(W) < 2:
        return float("nan")
    return math.sqrt((np.var(list(B), ddof=1) + np.var(list(W), ddof=1)) / 2)


def stats(runs, value=A.final_val, single=None):
    """Group quantities from per-run values (value(run) -> float or None)."""
    single = single or (lambda name: value(runs[name]) if name in runs else None)
    B, W = arm_finals(runs, "base", value), arm_finals(runs, "win16", value)
    out = dict(n_base=len(B), n_win=len(W), available=bool(B and W))
    if not out["available"]:
        return out
    mb, mw = float(np.mean(list(B.values()))), float(np.mean(list(W.values())))
    common = sorted(set(B) & set(W))
    win1, win64, base0 = single("win1-s0"), single("win64-s0"), single("base-s0")
    verdict, skipped = "n/a (needs 2 seeds per arm)", []
    if len(B) >= 2 and len(W) >= 2:
        d = A.decide_available(list(B.values()), list(W.values()), win1, win64)
        verdict, skipped = d["verdict"], d["skipped"]
    gap = mw - mb
    return dict(out, mean_base=mb, mean_win=mw, gap=gap, s=spread(B.values(), W.values()),
                ppl_pct=100 * (math.exp(gap) - 1), paired_wins=sum(W[s] < B[s] for s in common), paired=len(common),
                sanity=None if win1 is None or base0 is None else win1 - base0, verdict=verdict, skipped=skipped,
                lr_base=sorted({header_lr(r) for r in A.seed_runs(runs, "base")} - {None}),
                lr_win=sorted({header_lr(r) for r in A.seed_runs(runs, "win16")} - {None}))


def test_stats(runs, runs_dir):
    def load(run):
        path = os.path.join(runs_dir, run.get("full_name", run["name"]), "test_results.json")
        return json.load(open(path)) if os.path.exists(path) else None
    tests = {n: load(r) for n, r in runs.items()}
    value = lambda r: (tests.get(r["name"]) or {}).get("test_overall")  # noqa: E731
    st = stats(runs, value=value)
    st["reused"] = any((t or {}).get("test_set_reused") for t in tests.values())
    st["per_source"] = per_source(runs, lambda r: (tests.get(r["name"]) or {}).get("test"))
    return st


def per_source(runs, values=lambda r: r["final"]["val"] if r["final"] else None):
    """source -> (mean base, mean win16, gap, seed spread)."""
    rows = {}
    base = [v for v in (values(r) for r in A.seed_runs(runs, "base")) if v]
    win = [v for v in (values(r) for r in A.seed_runs(runs, "win16")) if v]
    if not base or not win:
        return rows
    for src in sorted(set.intersection(*[set(v) for v in base + win])):
        b, w = [v[src] for v in base], [v[src] for v in win]
        rows[src] = (float(np.mean(b)), float(np.mean(w)), float(np.mean(w) - np.mean(b)), spread(b, w))
    return rows


def gap_curve(runs):
    bands = [A.mean_band([r for r in A.seed_runs(runs, arm) if r["eval"]]) for arm in ("base", "win16")]
    if any(b is None for b in bands):
        return None
    (tb, mb, _, _), (tw, mw, _, _) = bands
    db, dw = dict(zip(tb.tolist(), mb.tolist())), dict(zip(tw.tolist(), mw.tolist()))
    common = sorted(set(db) & set(dw))
    return np.array(common, dtype=float), np.array([dw[t] - db[t] for t in common])


def mi_by_layer(runs, arm):
    """Mean source-expert mutual information (bits) per layer over the arm's runs with routing counts."""
    rows = []
    for r in A.seed_runs(runs, arm):
        if r["routing"]:
            rows.append([A.mutual_information(A.routing_matrix(r, li)[1])
                         for li in range(len(r["routing"]["counts"]))])
    return np.mean(rows, axis=0) if rows else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--with_test", action="store_true", help="add the M2D2 locked-test gap (test_results.json)")
    args = ap.parse_args()
    all_runs = A.read_runs(args.runs_dir)
    if not all_runs:
        sys.exit(f"no runs with log.jsonl in {args.runs_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    groups = {label: group_runs(all_runs, prefix) for label, prefix in GROUPS}
    st = {label: stats(runs) for label, runs in groups.items()}
    tst = test_stats(groups["M2D2-Wiki-8"], args.runs_dir) if args.with_test else None

    # ---- compare_summary.md ----
    L = ["# Window routing across datasets", "",
         "gap = mean(win16) - mean(base) in final val_overall: the window arm's penalty (positive = window worse). "
         "Perplexity % = e^gap - 1. s = sqrt((var(base) + var(win16)) / 2), ddof = 1. Verdicts: analyze.py's "
         "decision rule (skipping criteria a group has no runs for).", "",
         "| dataset | mean base | mean win16 | gap | s | perplexity % | seeds (base/win16) | paired wins (win16 < base) "
         "| sanity win1-s0 - base-s0 | LR base / win16 | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(label, g):
        if not g["available"]:
            return f"| {label} | not available | | | | | {g['n_base']}/{g['n_win']} | | | | |"
        lrs = f"{', '.join(f'{x:g}' for x in g['lr_base']) or '-'} / {', '.join(f'{x:g}' for x in g['lr_win']) or '-'}"
        return (f"| {label} | {g['mean_base']:.5f} | {g['mean_win']:.5f} | {g['gap']:+.5f} | {nanfmt(g['s'], '.5f')} | "
                f"{g['ppl_pct']:+.2f}% | {g['n_base']}/{g['n_win']} | {g['paired_wins']} of {g['paired']} | "
                f"{nanfmt(g['sanity'], '+.5f')} | {lrs} | {g['verdict']} |")

    for label, _ in GROUPS:
        L.append(row(label, st[label]))
    if tst is not None:
        L.append(row("M2D2-Wiki-8 (locked test)", tst))
    skipped = [f"{label}: {'; '.join(g['skipped'])}" for label, g in st.items() if g.get("skipped")]
    if skipped:
        L += ["", "Skipped decision criteria: " + " | ".join(skipped) + "."]
    if tst is not None:
        L += ["", "M2D2 validation vs locked test: gap "
              + (f"{st['M2D2-Wiki-8']['gap']:+.5f} ({st['M2D2-Wiki-8']['verdict']}) vs {tst['gap']:+.5f} ({tst['verdict']})"
                 if st["M2D2-Wiki-8"]["available"] and tst["available"] else "not available")]
        if tst.get("reused"):
            L.append("**TEST SET REUSED**: at least one test_results.json comes from a reuse of the locked test split.")

    m, s120 = st["M2D2-Wiki-8"], st["SlimPajama-120M"]
    L += ["", "## Main comparison: M2D2-Wiki-8 vs SlimPajama at the same 120M budget", ""]
    if m["available"] and s120["available"] and not (math.isnan(m["s"]) or math.isnan(s120["s"])):
        delta = m["gap"] - s120["gap"]
        n_m, n_s = min(m["n_base"], m["n_win"]), min(s120["n_base"], s120["n_win"])
        scale = math.sqrt(m["s"] ** 2 / n_m + s120["s"] ** 2 / n_s)
        if delta < -2 * scale:
            reading = "penalty clearly smaller on M2D2-Wiki-8"
        elif delta > 2 * scale:
            reading = "penalty clearly larger on M2D2-Wiki-8"
        else:
            reading = "penalty not distinguishable between the datasets"
        L += [f"Δ = gap_M2D2 - gap_Slim120 = {m['gap']:+.5f} - ({s120['gap']:+.5f}) = **{delta:+.5f}**; "
              f"rough scale sqrt(s_M²/n_M + s_S²/n_S) = {scale:.5f} (n = {n_m}, {n_s} seeds per arm); "
              f"2 × scale = {2 * scale:.5f}.", "", f"**{reading}.**"]
        print(f"delta = {delta:+.5f}, scale = {scale:.5f}: {reading}")
    else:
        L.append("Not available yet: needs at least 2 finished seeds per arm in both m2d2- and slim120- runs.")
        print("delta: not available yet")
    L += ["", "## Sampling regime", "", SAMPLING_NOTE, ""]
    text = "\n".join(L)
    with open(os.path.join(args.out_dir, "compare_summary.md"), "w") as f:
        f.write(text)

    # ---- compare_per_domain.md ----
    P = ["# Per-domain / per-source gaps (win16 - base, final val_overall per source; mean over seeds)", "",
         "## M2D2-Wiki-8 domains", "", "| domain | base | win16 | gap | seed spread |", "|---|---|---|---|---|"]
    m2 = per_source(groups["M2D2-Wiki-8"])
    P += [f"| {src} | {b:.5f} | {w:.5f} | {g:+.5f} | {nanfmt(s, '.5f')} |" for src, (b, w, g, s) in m2.items()] \
        or ["| not available | | | | |"]
    if tst is not None and tst.get("per_source"):
        P += ["", "## M2D2-Wiki-8 domains, locked test split", "", "| domain | base | win16 | gap | seed spread |",
              "|---|---|---|---|---|"]
        P += [f"| {src} | {b:.5f} | {w:.5f} | {g:+.5f} | {nanfmt(s, '.5f')} |"
              for src, (b, w, g, s) in tst["per_source"].items()]
    for label in ("SlimPajama-120M", "SlimPajama-500M"):
        rows = per_source(groups[label])
        P += ["", f"## {label} sources (reference)", "", "| source | base | win16 | gap | seed spread |",
              "|---|---|---|---|---|"]
        P += [f"| {src} | {b:.5f} | {w:.5f} | {g:+.5f} | {nanfmt(s, '.5f')} |" for src, (b, w, g, s) in rows.items()] \
            or ["| not available | | | | |"]
    with open(os.path.join(args.out_dir, "compare_per_domain.md"), "w") as f:
        f.write("\n".join(P) + "\n")

    # ---- compare_gap_curves.png ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in (("M2D2-Wiki-8", "tab:green"), ("SlimPajama-120M", "tab:purple")):
        c = gap_curve(groups[label])
        if c is not None and len(c[0]):
            ax.plot(c[0] / 1e6, c[1], "o-", ms=3, color=color, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("gap: mean(win16) - mean(base) val_overall")
    ax.set_title("Window-routing penalty over training (same 120M budget)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "compare_gap_curves.png"), dpi=130)
    plt.close(fig)

    # ---- compare_mi.png ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (label, _) in zip(axes, GROUPS):
        for arm, color in (("base", "tab:blue"), ("win16", "tab:orange")):
            mi = mi_by_layer(groups[label], arm)
            if mi is not None:
                ax.plot(np.arange(1, len(mi) + 1), mi, "o-", color=color, label=arm)
        ax.set_title(f"{label}: source-expert MI")
        ax.set_xlabel("layer")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    axes[0].set_ylabel("mutual information (bits), mean over seeds")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "compare_mi.png"), dpi=130)
    plt.close(fig)

    print(text)


if __name__ == "__main__":
    main()
