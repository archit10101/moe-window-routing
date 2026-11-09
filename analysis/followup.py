#!/usr/bin/env python
"""Follow-up analysis (docs/FOLLOWUP_PROMPT.md part 7).

    python analysis/followup.py --runs_dir <dir> --out_dir <dir>

Each follow-up variant is paired with the run of the same seed it is compared against.
Differences are variant - comparison in final val_overall (positive = variant worse), the
convention of FOLLOWUP_PROMPT.md. The noise scale s and the original gap are computed from
the main experiment's finished base-s* and win16-s* runs:
    s = sqrt((var(base) + var(win16)) / 2), ddof = 1;  original gap = mean(win16) - mean(base).
A mean paired difference below -2s is "clearly better", above +2s "clearly worse", otherwise
"within noise". Missing runs are reported as not available.
"""
import argparse
import csv
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

# group -> (question, {variant prefix: comparison prefix}); runs are <prefix>-s<seed>
GROUPS = {
    "E5": ("Does adding context to token identity help?", {"both16": "base"}),
    "E3": ("Does the penalty shrink with fewer window layers?", {"win16top1": "base", "win16top2": "base"}),
    "E2": ("Was the shared expert the bottleneck?", {"win16-sh2048": "base-sh2048"}),
    "E4": ("Does slow router learning explain the gap?", {"win16-rlr3": "base-rlr3"}),
}
RG_PAIRS = (("base-rg-s0", "base-s0"), ("win16-rg-s0", "win16-s0"))
RG_TOL = 0.002           # logging router gradients must not change training by more than this
GRAD_RATIO_SMALLER = 0.8  # window router grads "clearly smaller": median ratio below this in every window layer
DEFAULT_WINDOW_LAYERS = [4, 5, 6, 7]  # 0-indexed layers 5-8, used if a run's header lacks window_layers


def label(d, s):
    if d < -2 * s:
        return "clearly better"
    if d > 2 * s:
        return "clearly worse"
    return "within noise"


def noise_scale(runs):
    B = [A.final_val(r) for r in A.seed_runs(runs, "base") if r["final"] is not None]
    W = [A.final_val(r) for r in A.seed_runs(runs, "win16") if r["final"] is not None]
    if len(B) < 2 or len(W) < 2:
        sys.exit(f"followup: need at least 2 finished base-s* and win16-s* main runs (have {len(B)} and {len(W)})")
    s = math.sqrt((np.var(B, ddof=1) + np.var(W, ddof=1)) / 2)
    return s, float(np.mean(W) - np.mean(B)), B, W


def pair_variant(runs, variant, comparison):
    """Paired results for every seed where both runs are finished."""
    pairs = []
    for run in A.seed_runs(runs, variant):
        seed = run["name"][len(variant) + 2:]
        other = runs.get(f"{comparison}-s{seed}")
        if run["final"] is None or other is None or other["final"] is None:
            continue
        pairs.append(dict(seed=int(seed), run=run, other=other, diff=A.final_val(run) - A.final_val(other)))
    if not pairs:
        return None
    sources = sorted(set.intersection(*[set(p["run"]["final"]["val"]) & set(p["other"]["final"]["val"])
                                        for p in pairs]))
    per_source = {src: float(np.mean([p["run"]["final"]["val"][src] - p["other"]["final"]["val"][src]
                                      for p in pairs])) for src in sources}
    curves = []
    for p in pairs:
        (tv, vv), (tc, vc) = A.curve(p["run"]), A.curve(p["other"])
        cv, cc = dict(zip(tv.tolist(), vv.tolist())), dict(zip(tc.tolist(), vc.tolist()))
        curves.append({t: cv[t] - cc[t] for t in set(cv) & set(cc)})
    common = sorted(set.intersection(*[set(c) for c in curves]))
    gap_curve = (np.array(common, dtype=float), np.array([np.mean([c[t] for c in curves]) for t in common]))
    return dict(variant=variant, comparison=comparison, pairs=pairs, mean=float(np.mean([p["diff"] for p in pairs])),
                per_source=per_source, curve=gap_curve)


def original_gap_curve(runs):
    """mean(win16) - mean(base) over the main seeds, per eval step shared by all of them."""
    bands = [A.mean_band([r for r in A.seed_runs(runs, arm) if r["final"] is not None]) for arm in ("base", "win16")]
    if any(b is None for b in bands):
        return None
    (tb, mb, _, _), (tw, mw, _, _) = bands
    db, dw = dict(zip(tb.tolist(), mb.tolist())), dict(zip(tw.tolist(), mw.tolist()))
    common = sorted(set(db) & set(dw))
    return np.array(common, dtype=float), np.array([dw[t] - db[t] for t in common])


def router_grads(run):
    """(tokens, [steps x layers] router grad norms) from a run's train records, or None."""
    recs = [r for _, r in sorted(run["train"].items()) if "router_grad_norm" in r]
    if not recs:
        return None
    return np.array([r["tokens"] for r in recs], dtype=float), np.array([r["router_grad_norm"] for r in recs])


def grad_ratios(runs):
    """Per layer: median win16-rg / median base-rg router grad norm over the second half of training."""
    b, w = runs.get("base-rg-s0"), runs.get("win16-rg-s0")
    gb, gw = (router_grads(b) if b else None), (router_grads(w) if w else None)
    if gb is None or gw is None:
        return None
    half = lambda g: g[1][g[0] >= g[0].max() / 2]  # noqa: E731
    ratios = np.median(half(gw), axis=0) / np.median(half(gb), axis=0)
    windows = ((w.get("header") or {}).get("config") or {}).get("window_layers")
    win_layers = [i for i, n in enumerate(windows) if n > 0] if windows else DEFAULT_WINDOW_LAYERS
    smaller = bool(all(ratios[i] < GRAD_RATIO_SMALLER for i in win_layers))
    return dict(ratios=ratios.tolist(), window_layers=win_layers, clearly_smaller=smaller)


def interpret(group, results, s, G, grads):
    """Interpretation lines for one group (FOLLOWUP_PROMPT.md part 7 table)."""
    avail = {k: v for k, v in results.items() if v is not None}
    if not avail:
        return [f"{group}: not available yet (no finished variant/comparison pairs)."]
    if group == "E5":
        r = avail["both16"]
        reading = {"clearly better": "context helps as an addition",
                   "within noise": "context adds nothing at this scale",
                   "clearly worse": "even added context hurts"}[label(r["mean"], s)]
        return [f"E5: both16 - base = {r['mean']:+.4f} ({label(r['mean'], s)}): {reading}."]
    if group == "E3":
        lines, shrank = [], []
        for v, r in avail.items():
            if label(r["mean"], s) == "within noise":
                verdict = "vanishes"
            elif G - r["mean"] > 2 * s:
                verdict = "shrinks"
            else:
                verdict = "does not shrink"
            shrank.append(verdict != "does not shrink")
            lines.append(f"E3: {v} - base = {r['mean']:+.4f} ({label(r['mean'], s)}) vs original penalty {G:+.4f}: "
                         f"penalty {verdict}.")
        if all(shrank) and len(avail) == 2:
            lines.append("E3: penalty shrinks or vanishes with fewer window layers: topic routing only belongs at the top.")
        else:
            lines.append("E3: penalty does not shrink with fewer window layers in every variant: "
                         "restricting the window to the top layers does not remove the cost.")
        return lines
    if group == "E2":
        r = avail["win16-sh2048"]
        if G - r["mean"] > 2 * s:
            reading = "gap shrinks by more than 2s: the shared expert was the bottleneck"
        else:
            reading = "gap does not shrink by more than 2s: the shared expert was not the bottleneck"
        return [f"E2: win16-sh2048 - base-sh2048 = {r['mean']:+.4f} vs original gap {G:+.4f} "
                f"(shrink {G - r['mean']:+.4f}, 2s = {2 * s:.4f}): {reading}."]
    if group == "E4":
        r = avail["win16-rlr3"]
        shrinks = G - r["mean"] > 2 * s
        lines = [f"E4: win16-rlr3 - base-rlr3 = {r['mean']:+.4f} vs original gap {G:+.4f} "
                 f"(shrink {G - r['mean']:+.4f}, 2s = {2 * s:.4f}): gap {'shrinks' if shrinks else 'does not shrink'} "
                 "by more than 2s with a faster router."]
        if grads is None:
            lines.append("E4: router grad norms of base-rg-s0 / win16-rg-s0 not available.")
            smaller = False
        else:
            smaller = grads["clearly_smaller"]
            ratios = ", ".join(f"L{i + 1} {x:.2f}" for i, x in enumerate(grads["ratios"]))
            lines.append(f"E4: router grad norm ratio win16-rg / base-rg (median, second half of training): {ratios}; "
                         f"window router gradients {'clearly smaller' if smaller else 'not clearly smaller'} "
                         f"(< {GRAD_RATIO_SMALLER} in every window layer).")
        if shrinks or smaller:
            lines.append("E4: slow router learning explains part of the gap.")
        else:
            lines.append("E4: neither test points to slow router learning as an explanation of the gap.")
        return lines
    raise ValueError(group)


def plot_gap(group, results, orig, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_curve = False
    for v, r in results.items():
        if r is None or not len(r["curve"][0]):
            continue
        t, d = r["curve"]
        ax.plot(t / 1e6, d, "o-", ms=3, label=f"{v} - {r['comparison']} (mean of {len(r['pairs'])} seeds)")
        any_curve = True
    if orig is not None and group in ("E2", "E4", "E3"):
        ax.plot(orig[0] / 1e6, orig[1], color="gray", ls="--", lw=1, label="original gap: win16 - base")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("val_overall difference (positive = variant worse)")
    ax.set_title(f"{group}: {GROUPS[group][0]}" + ("" if any_curve else " (no runs yet)"))
    if any_curve or orig is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"followup_{group}_gap.png"), dpi=130)
    plt.close(fig)


def plot_router_grads(runs, out_dir):
    series = {n: router_grads(runs[n]) for n in ("base-rg-s0", "win16-rg-s0") if n in runs}
    series = {n: g for n, g in series.items() if g is not None}
    n_layer = max((g[1].shape[1] for g in series.values()), default=8)
    cols = 4
    fig, axes = plt.subplots(math.ceil(n_layer / cols), cols, figsize=(4 * cols, 3.2 * math.ceil(n_layer / cols)),
                             squeeze=False)
    for li in range(n_layer):
        ax = axes[li // cols][li % cols]
        for n, color in (("base-rg-s0", "tab:blue"), ("win16-rg-s0", "tab:orange")):
            if n in series:
                ax.plot(series[n][0] / 1e6, series[n][1][:, li], color=color, lw=1, label=n)
        ax.set_title(f"layer {li + 1}")
        ax.set_yscale("log")
        ax.set_xlabel("tokens (M)")
    for li in range(n_layer, axes.size):
        axes[li // cols][li % cols].axis("off")
    if series:
        axes[0][0].legend(fontsize=8)
    else:
        fig.suptitle("router_grad_norm not logged yet (base-rg-s0 / win16-rg-s0)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "router_grad_norm.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    runs = A.read_runs(args.runs_dir)
    if not runs:
        sys.exit(f"no runs with log.jsonl in {args.runs_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    s, G, B, W = noise_scale(runs)
    orig = original_gap_curve(runs)
    grads = grad_ratios(runs)

    L = ["# Follow-up experiments: summary", "",
         "Differences are variant - comparison in final val_overall, paired by seed (positive = variant worse).",
         f"Noise scale from the main runs: s = {s:.5f} (2s = {2 * s:.5f}); original gap mean(win16) - mean(base) = "
         f"{G:+.5f} ({len(B)} base and {len(W)} win16 seeds).", ""]
    csv_rows, interp = [], []
    for group, (question, variants) in GROUPS.items():
        results = {v: pair_variant(runs, v, c) for v, c in variants.items()}
        L += [f"## {group}: {question}", ""]
        for v, c in variants.items():
            r = results[v]
            if r is None:
                L += [f"{v} vs {c}: not available yet.", ""]
                continue
            L += [f"### {v} vs {c}", "", "| seed | variant | final | comparison | final | difference |",
                  "|---|---|---|---|---|---|"]
            for p in r["pairs"]:
                L.append(f"| {p['seed']} | {p['run']['name']} | {A.final_val(p['run']):.5f} | {p['other']['name']} | "
                         f"{A.final_val(p['other']):.5f} | {p['diff']:+.5f} |")
                csv_rows.append([group, p["run"]["name"], p["other"]["name"], p["seed"], A.final_val(p["run"]),
                                 A.final_val(p["other"]), p["diff"], r["mean"], label(r["mean"], s)])
            L += ["", f"Mean paired difference: **{r['mean']:+.5f}** ({label(r['mean'], s)}; 2s = {2 * s:.5f})", "",
                  "| source | mean paired difference |", "|---|---|"]
            L += [f"| {src} | {d:+.5f} |" for src, d in r["per_source"].items()]
            L.append("")
        lines = interpret(group, results, s, G, grads)
        interp += lines
        L += ["**Interpretation:**", ""] + [f"- {x}" for x in lines] + [""]
        plot_gap(group, results, orig, args.out_dir)

    L += ["## Router-gradient logging check", ""]
    for rg, ref in RG_PAIRS:
        a, b = A.final_val(runs.get(rg)), A.final_val(runs.get(ref))
        if a is None or b is None:
            line = f"{rg} vs {ref}: not available yet."
        else:
            ok = abs(a - b) <= RG_TOL
            line = (f"{rg} vs {ref}: {a:.5f} vs {b:.5f}, difference {a - b:+.5f} "
                    + ("(within 0.002: ok)" if ok else "**FLAG: differs by more than 0.002; logging may have changed training**"))
        L.append(f"- {line}")
        interp.append(line)
    if grads is not None:
        L += ["", "Router grad norm ratio win16-rg / base-rg per layer (median over the second half of training): "
              + ", ".join(f"L{i + 1} {x:.3f}" for i, x in enumerate(grads["ratios"]))]

    with open(os.path.join(args.out_dir, "followup_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    # one row per finished run; the pairing columns are filled for follow-up variants
    paired = {r[1]: r for r in csv_rows}
    with open(os.path.join(args.out_dir, "followup_runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "final_val_overall", "group", "comparison", "seed", "comparison_final", "difference",
                    "group_mean_difference", "label"])
        for name, run in runs.items():
            if run["final"] is None:
                continue
            if name in paired:
                g, _, comp, seed, fv, cf, d, gm, lab = paired[name]
                w.writerow([name, f"{fv:.6f}", g, comp, seed, f"{cf:.6f}", f"{d:+.6f}", f"{gm:+.6f}", lab])
            else:
                w.writerow([name, f"{A.final_val(run):.6f}", "", "", "", "", "", "", ""])
    plot_router_grads(runs, args.out_dir)
    print(f"s = {s:.5f}, original gap (win16 - base) = {G:+.5f}")
    print("\n".join(interp))
    print(f"wrote followup_summary.md, followup_runs.csv, followup_<group>_gap.png, router_grad_norm.png to {args.out_dir}")


if __name__ == "__main__":
    main()
