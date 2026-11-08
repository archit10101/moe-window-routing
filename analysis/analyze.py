#!/usr/bin/env python
"""Analysis for the MoE window-routing pilot (docs/PROMPT.md part 8).

    python analysis/analyze.py --runs_dir <dir> --out_dir <dir> [--lr_check [--write]]

LR-check mode picks each arm's LR from the lr-* runs (and with --write updates
config/chosen_lr.py). Main mode writes the plots, tables and summary.md and
applies the decision rule. Uses only numpy, matplotlib, json, csv and the
standard library. Runs are identified by their folder names.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR_TIE = 0.01            # an LR within this much of the best counts as tied; pick the lowest
HEATMAP_LAYER = 7        # 1-indexed layer for routing_heatmap.png
LOW_SAMPLE = {"book": "low-sample (~225k validation tokens)"}


# -----------------------------------------------------------------------------
# reading runs

def read_run(run_dir):
    """Parse log.jsonl, keeping the last record for each step (after a resume)."""
    run = dict(name=os.path.basename(run_dir), header=None, train={}, eval={}, final=None, routing=None)
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue  # partially written last line
            kind = r.get("type")
            if kind == "header":
                run["header"] = r
            elif kind == "train":
                run["train"][r["step"]] = r
            elif kind == "eval":
                run["eval"][r["step"]] = r
            elif kind == "final":
                run["final"] = r
    rc = os.path.join(run_dir, "routing_counts.json")
    if os.path.exists(rc):
        with open(rc) as f:
            run["routing"] = json.load(f)
    return run


def read_runs(runs_dir):
    runs = {}
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        if os.path.isfile(os.path.join(d, "log.jsonl")):
            runs[name] = read_run(d)
    return runs


def final_val(run):
    return None if run is None or run["final"] is None else run["final"]["val_overall"]


def curve(run):
    """(tokens, val_overall) from eval records, ending with the final evaluation if present."""
    pts = {r["tokens"]: r["val_overall"] for _, r in sorted(run["eval"].items())}
    if run["final"] is not None:
        pts[run["final"]["tokens"]] = run["final"]["val_overall"]
    toks = sorted(pts)
    return np.array(toks, dtype=float), np.array([pts[t] for t in toks])


def seed_runs(runs, prefix):
    """Runs named <prefix>-s<N>, sorted by seed."""
    out = []
    for name, run in runs.items():
        if name.startswith(prefix + "-s") and name[len(prefix) + 2:].isdigit():
            out.append((int(name[len(prefix) + 2:]), run))
    return [r for _, r in sorted(out, key=lambda x: x[0])]


# -----------------------------------------------------------------------------
# LR check

def lr_check(runs, write, chosen_path):
    chosen = {}
    lines = []
    for arm, prefix, var in (("base", "lr-base-", "LR_BASE"), ("win16", "lr-win16-", "LR_WIN")):
        results = []
        for name, run in runs.items():
            if not name.startswith(prefix):
                continue
            lr = float(name[len(prefix):])
            v = final_val(run)
            results.append((lr, v))
            lines.append(f"{name:<16} lr={lr:<8g} final val_overall="
                         + (f"{v:.4f}" if v is not None else "NOT FINISHED"))
        done = [(lr, v) for lr, v in results if v is not None]
        if not done:
            sys.exit(f"lr check: no finished {prefix}* runs in the runs folder")
        if len(done) < len(results):
            lines.append(f"WARNING: {len(results) - len(done)} {arm} LR run(s) unfinished; choosing among finished ones")
        best = min(v for _, v in done)
        tied = [lr for lr, v in done if v - best <= LR_TIE]
        chosen[var] = min(tied)
        best_lr = min(lr for lr, v in done if v == best)
        lines.append(f"{arm}: best lr {best_lr:g} ({best:.4f}); within {LR_TIE} of best: "
                     f"{sorted(tied)}; chosen {var} = {chosen[var]:g}")
    print("\n".join(lines))
    if write:
        with open(chosen_path, "w") as f:
            f.write("# Peak learning rates chosen by the LR check.\n")
            f.write(f"# Written by analysis/analyze.py --lr_check --write on {time.strftime('%Y-%m-%d %H:%M')}.\n")
            for line in lines:
                f.write(f"# {line}\n")
            f.write(f"LR_BASE = {chosen['LR_BASE']!r}\nLR_WIN = {chosen['LR_WIN']!r}\n")
        print(f"wrote {chosen_path}")
    return chosen


# -----------------------------------------------------------------------------
# decision rule

def decide(B, W, win1, win64):
    """PROMPT.md part 8, exactly. B, W: final val_overall per seed (at least 2 each)."""
    B, W = np.asarray(B, float), np.asarray(W, float)
    s = math.sqrt((np.var(B, ddof=1) + np.var(W, ddof=1)) / 2)
    mean_b, mean_w = float(B.mean()), float(W.mean())
    gap = mean_b - mean_w  # positive: the window arm is better
    sanity = abs(win1 - mean_b) <= 2 * s
    if not sanity:
        verdict = "INVALID: sanity check failed"
    elif gap > 2 * s and win64 <= mean_b + 2 * s:
        verdict = "PROMISING"
    elif gap < -2 * s:
        verdict = "NEGATIVE"
    else:
        verdict = "INCONCLUSIVE"
    return dict(verdict=verdict, s=s, gap=gap, mean_B=mean_b, mean_W=mean_w, sanity_ok=sanity,
                win1_minus_meanB=win1 - mean_b, win64_minus_meanB=win64 - mean_b)


# -----------------------------------------------------------------------------
# outputs

def plot_sanity(runs, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in seed_runs(runs, "base"):
        t, v = curve(run)
        ax.plot(t / 1e6, v, lw=0.8, color="tab:blue", alpha=0.8, label=run["name"])
    if "win1-s0" in runs:
        t, v = curve(runs["win1-s0"])
        ax.plot(t / 1e6, v, lw=2.5, color="tab:red", label="win1-s0")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("val_overall (CE)")
    ax.set_title("Sanity: window n=1 vs baseline seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def mean_band(arm_runs):
    curves = [curve(r) for r in arm_runs]
    common = sorted(set.intersection(*[set(t.tolist()) for t, _ in curves])) if curves else []
    if not common:
        return None
    vals = np.array([[dict(zip(t.tolist(), v))[c] for c in common] for t, v in curves])
    return np.array(common), vals.mean(0), vals.min(0), vals.max(0)


def plot_main(runs, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for prefix, color in (("base", "tab:blue"), ("win16", "tab:orange")):
        arm = seed_runs(runs, prefix)
        band = mean_band(arm)
        if band is None:
            continue
        t, mean, lo, hi = band
        ax.plot(t / 1e6, mean, color=color, lw=1.8, label=f"{prefix} (mean of {len(arm)} seeds)")
        ax.fill_between(t / 1e6, lo, hi, color=color, alpha=0.25, lw=0)
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("val_overall (CE)")
    ax.set_title("Baseline vs window-16 routing (band = min-max over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_per_source(runs, csv_path, md_path):
    finished = {n: r for n, r in runs.items() if r["final"] is not None and not n.startswith(("lr-", "smoke"))}
    names = sorted(finished)
    base = [r for r in seed_runs(finished, "base")]
    win = [r for r in seed_runs(finished, "win16")]
    sources = sorted(set().union(*[set(r["final"]["val"]) for r in finished.values()])) if finished else []
    header = ["source"] + names + ["base_mean", "win16_mean", "win16_minus_base", "note"]
    rows = []
    for src in sources + ["val_overall"]:
        def get(r):
            return r["final"]["val_overall"] if src == "val_overall" else r["final"]["val"].get(src)
        vals = [get(finished[n]) for n in names]
        bm = np.mean([get(r) for r in base]) if base else None
        wm = np.mean([get(r) for r in win]) if win else None
        diff = wm - bm if bm is not None and wm is not None else None
        rows.append([src] + vals + [bm, wm, diff, LOW_SAMPLE.get(src, "")])

    def fmt(x):
        return "" if x is None else (f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x))

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([fmt(x) for x in row])
    with open(md_path, "w") as f:
        f.write("Final cross-entropy per validation source. win16_minus_base < 0 means the window arm is better.\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(x) for x in row) + " |\n")
    return rows, header


def plot_window_sweep(runs, res, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs, ys = [], []
    for n, val in ((1, final_val(runs.get("win1-s0"))), (16, res["mean_W"] if res else None),
                   (64, final_val(runs.get("win64-s0")))):
        if val is not None:
            xs.append(n)
            ys.append(val)
    ax.plot(xs, ys, "o-", color="tab:orange", label="window arm (n=16: mean over seeds)")
    if res is not None:
        ax.axhline(res["mean_B"], color="tab:blue", label="baseline mean")
        ax.axhspan(res["mean_B"] - 2 * res["s"], res["mean_B"] + 2 * res["s"], color="tab:blue", alpha=0.15,
                   label="baseline mean ± 2s")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 16, 64])
    ax.set_xticklabels(["1", "16", "64"])
    ax.set_xlabel("router window n (layers 5-8)")
    ax.set_ylabel("final val_overall (CE)")
    ax.set_title("Window sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_loads(runs, path):
    names = [n for n in ("base-s0", "win16-s0") if n in runs]
    fig, axes = plt.subplots(1, max(1, len(names)), figsize=(6.5 * max(1, len(names)), 4.5), squeeze=False)
    for ax, name in zip(axes[0], names):
        recs = [r for _, r in sorted(runs[name]["train"].items())]
        if not recs:
            continue
        t = np.array([r["tokens"] for r in recs]) / 1e6
        mx = np.array([r["load_max"] for r in recs])
        mn = np.array([r["load_min"] for r in recs])
        colors = plt.cm.viridis(np.linspace(0, 1, mx.shape[1]))
        for layer in range(mx.shape[1]):
            ax.plot(t, mx[:, layer], color=colors[layer], lw=1, label=f"layer {layer + 1}")
            ax.plot(t, mn[:, layer], color=colors[layer], lw=1, ls="--")
        ax.set_title(f"{name}: max (solid) and min (dashed) expert load")
        ax.set_xlabel("training tokens (M)")
        ax.set_ylabel("fraction of routed slots")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def routing_matrix(run, layer_idx):
    counts = run["routing"]["counts"][layer_idx]
    sources = sorted(counts)
    return sources, np.array([counts[s] for s in sources], dtype=float)


def mutual_information(mat):
    """MI (bits) between source and expert from a source x expert count matrix."""
    p = mat / mat.sum()
    ps, pe = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (ps @ pe)[nz])).sum())


def plot_heatmap_and_mi(runs, png_path, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "layer", "window", "mi_bits"])
        for name, run in runs.items():
            if run["routing"] is None:
                continue
            for li, window in enumerate(run["routing"]["window_layers"]):
                _, mat = routing_matrix(run, li)
                w.writerow([name, li + 1, window, f"{mutual_information(mat):.5f}"])

    names = [n for n in ("base-s0", "win16-s0") if n in runs and runs[n]["routing"] is not None]
    fig, axes = plt.subplots(1, max(1, len(names)), figsize=(6 * max(1, len(names)), 4.5), squeeze=False)
    for ax, name in zip(axes[0], names):
        n_layers = len(runs[name]["routing"]["counts"])
        li = min(HEATMAP_LAYER, n_layers) - 1
        sources, mat = routing_matrix(runs[name], li)
        share = mat / mat.sum(1, keepdims=True)
        im = ax.imshow(share, cmap="viridis", vmin=0, aspect="auto")
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xlabel("expert")
        note = "" if li == HEATMAP_LAYER - 1 else f" (model has {n_layers} layers)"
        ax.set_title(f"{name}: layer {li + 1} routing share{note}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def main_analysis(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = [r for r in seed_runs(runs, "base") if r["final"] is not None]
    win = [r for r in seed_runs(runs, "win16") if r["final"] is not None]
    win1, win64 = final_val(runs.get("win1-s0")), final_val(runs.get("win64-s0"))
    missing = []
    if len(base) < 2:
        missing.append(f"at least 2 finished base-s* runs (have {len(base)})")
    if len(win) < 2:
        missing.append(f"at least 2 finished win16-s* runs (have {len(win)})")
    if win1 is None:
        missing.append("finished win1-s0")
    if win64 is None:
        missing.append("finished win64-s0")

    res = None
    if not missing:
        res = decide([final_val(r) for r in base], [final_val(r) for r in win], win1, win64)

    plot_sanity(runs, os.path.join(out_dir, "sanity.png"))
    plot_main(runs, os.path.join(out_dir, "main_comparison.png"))
    rows, header = write_per_source(runs, os.path.join(out_dir, "per_source.csv"),
                                    os.path.join(out_dir, "per_source.md"))
    plot_window_sweep(runs, res, os.path.join(out_dir, "window_sweep.png"))
    plot_loads(runs, os.path.join(out_dir, "loads.png"))
    plot_heatmap_and_mi(runs, os.path.join(out_dir, "routing_heatmap.png"), os.path.join(out_dir, "routing_mi.csv"))

    lines = ["# MoE window-routing pilot: summary", ""]
    if res is None:
        lines += ["**No verdict yet.** Missing: " + "; ".join(missing) + ".", ""]
    else:
        lines += [f"**Verdict: {res['verdict']}**", "",
                  "Decision rule (PROMPT.md part 8): s = sqrt((var(B) + var(W)) / 2) with ddof = 1; "
                  "gap = mean(B) - mean(W) (positive = window arm better). Sanity: |win1-s0 - mean(B)| <= 2s. "
                  "PROMISING: gap > 2s and win64-s0 <= mean(B) + 2s. NEGATIVE: gap < -2s. Otherwise INCONCLUSIVE.", "",
                  "| quantity | value |", "|---|---|",
                  f"| mean(B) | {res['mean_B']:.5f} |", f"| mean(W) | {res['mean_W']:.5f} |",
                  f"| gap = mean(B) - mean(W) | {res['gap']:+.5f} |", f"| s | {res['s']:.5f} |",
                  f"| 2s | {2 * res['s']:.5f} |",
                  f"| win1-s0 - mean(B) | {res['win1_minus_meanB']:+.5f} (sanity {'ok' if res['sanity_ok'] else 'FAILED'}) |",
                  f"| win64-s0 - mean(B) | {res['win64_minus_meanB']:+.5f} |", ""]
    lines += ["## Final val_overall per run", "", "| run | final val_overall | tokens |", "|---|---|---|"]
    for name, run in runs.items():
        if run["final"] is not None:
            lines.append(f"| {name} | {run['final']['val_overall']:.5f} | {run['final']['tokens']:,} |")
    lines += ["", f"B = {[round(final_val(r), 5) for r in base]} ({[r['name'] for r in base]})",
              f"W = {[round(final_val(r), 5) for r in win]} ({[r['name'] for r in win]})",
              f"win1-s0 = {win1}", f"win64-s0 = {win64}", "",
              "Per-source table: per_source.md. Routing mutual information: routing_mi.csv.",
              "The book source has only ~225k validation tokens, so its row is low-sample.", ""]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lr_check", action="store_true", help="pick each arm's LR from the lr-* runs")
    ap.add_argument("--write", action="store_true", help="with --lr_check: write config/chosen_lr.py")
    ap.add_argument("--chosen_lr_path", default=os.path.join(REPO, "config", "chosen_lr.py"))
    args = ap.parse_args()
    if args.write and not args.lr_check:
        ap.error("--write only applies with --lr_check")
    runs = read_runs(args.runs_dir)
    if not runs:
        sys.exit(f"no runs with log.jsonl in {args.runs_dir}")
    if args.lr_check:
        lr_check(runs, args.write, args.chosen_lr_path)
        return
    res = main_analysis(runs, args.out_dir)
    if res is None:
        sys.exit(2)


if __name__ == "__main__":
    main()
