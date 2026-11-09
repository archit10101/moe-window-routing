#!/usr/bin/env python
"""Uneven-domain design analysis (docs/M2D2_UNEVEN_PROMPT.md part 6).

    python analysis/uneven.py --runs_dir <dir> --out_dir <dir> [--with_test] [--data_root <dir>]

Designs: even (m2d2-*), hist4x (m2d2h4-*) and rep4x (m2d2r4-*). Hypothesis: a domain's window-routing
penalty shrinks when that domain gets 4x the training data. All numbers come from the run logs, each
uneven design's meta.json (the boosted domain is read from it, never assumed) and routing_counts.json.

Gap: g[X][s][d] = loss(win16-s, d) - loss(base-s, d) on final-eval per-domain validation loss, paired by
seed (base and win16 of a seed share their initialization). Positive = window worse.

  1. swap contrast (primary): c[X][s] = g[X][s][h] - g[X][s][r]; B = mean c[hist4x] - mean c[rep4x];
     SE = sqrt(var(c[hist4x])/n_h4 + var(c[rep4x])/n_r4). B < -2SE supports the hypothesis,
     B > +2SE contradicts it, otherwise not distinguishable.
  2. difference-in-differences vs even, per uneven design U with boosted b:
     k[X][s] = g[X][s][b] - mean_{d != b} g[X][s][d]; DiD = mean k[U] - mean k[even] (same SE and rule),
     the ratio version, and the placebo rank of b among all domains' changes.
  3. crossover: mean g[U][b] +/- 2 SE.
  4. absolute losses on b, even -> U, per arm.
  5. expert usage: effective number of experts and source-expert mutual information per layer.

meta.json is looked up at each run's configured data_dir; with --data_root it is <data_root>/<folder name>.
--with_test repeats everything on the locked-test results (<run>/test_results.json).
"""
import argparse
import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

DESIGNS = [("even", "m2d2-"), ("hist4x", "m2d2h4-"), ("rep4x", "m2d2r4-")]
UNEVEN = ("hist4x", "rep4x")


# -----------------------------------------------------------------------------
# data

def expected_seeds(prefix):
    """Seeds the design is configured for (from config/runs), so missing runs can be reported."""
    paths = glob.glob(os.path.join(REPO, "config", "runs", f"{prefix}base-s*.py"))
    return sorted(int(os.path.basename(p)[len(prefix) + 6:-3]) for p in paths)


def val_losses(run):
    return run["final"]["val"] if run["final"] else None


def test_losses_reader(runs_dir):
    def read(run):
        path = os.path.join(runs_dir, run.get("full_name", run["name"]), "test_results.json")
        return json.load(open(path))["test"] if os.path.exists(path) else None
    return read


def val_routing(run):
    return run["routing"]["counts"] if run["routing"] else None


def test_routing_reader(runs_dir):
    def read(run):
        path = os.path.join(runs_dir, run.get("full_name", run["name"]), "test_results.json")
        return json.load(open(path)).get("routing_counts") if os.path.exists(path) else None
    return read


def design_data(all_runs, prefix, losses, routing):
    runs = A.select_prefix(all_runs, prefix)
    arm = {a: {int(r["name"].rsplit("-s", 1)[1]): r for r in A.seed_runs(runs, a)} for a in ("base", "win16")}
    seeds = sorted(s for s in set(arm["base"]) & set(arm["win16"])
                   if losses(arm["base"][s]) and losses(arm["win16"][s]))
    base = {s: losses(arm["base"][s]) for s in seeds}
    win = {s: losses(arm["win16"][s]) for s in seeds}
    domains = sorted(set.intersection(*[set(base[s]) & set(win[s]) for s in seeds])) if seeds else []
    gap = {s: {d: win[s][d] - base[s][d] for d in domains} for s in seeds}
    header = next((r["header"] for a in arm.values() for r in a.values() if r.get("header")), None) or {}
    cfg = header.get("config") or {}
    rout = {a: [x for x in (routing(arm[a][s]) for s in seeds) if x] for a in ("base", "win16")}
    return dict(prefix=prefix, seeds=seeds, expected=expected_seeds(prefix), domains=domains, base=base, win=win,
                gap=gap, data_dir=cfg.get("data_dir"), total_tokens=cfg.get("total_tokens"), routing=rout)


def read_meta(design, data_root):
    d = design["data_dir"]
    if d is None:
        return None
    path = os.path.join(data_root, os.path.basename(os.path.normpath(d)), "meta.json") if data_root else \
        os.path.join(d, "meta.json")
    return json.load(open(path)) if os.path.exists(path) else None


# -----------------------------------------------------------------------------
# statistics

def mean_se(values):
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), se=float("nan"))
    sd = float(v.std(ddof=1)) if len(v) >= 2 else float("nan")
    return dict(n=len(v), mean=float(v.mean()), sd=sd, se=sd / math.sqrt(len(v)) if len(v) >= 2 else float("nan"))


def two_sample(a, b):
    """mean(a) - mean(b) with SE sqrt(var(a)/n_a + var(b)/n_b), sample variances."""
    ma, mb = mean_se(a), mean_se(b)
    diff = ma["mean"] - mb["mean"]
    se = math.sqrt(ma["sd"] ** 2 / ma["n"] + mb["sd"] ** 2 / mb["n"]) if ma["n"] >= 2 and mb["n"] >= 2 else float("nan")
    return diff, se


def verdict(x, se):
    if math.isnan(se):
        return "not computable (needs at least 2 seeds per design)"
    if x < -2 * se:
        return "supports the hypothesis"
    if x > 2 * se:
        return "contradicts it"
    return "not distinguishable"


def mean_gap(design, d):
    return mean_se(design["gap"][s][d] for s in design["seeds"])


def effective_experts(counts):
    p = np.asarray(counts, dtype=float)
    p = p / p.sum() if p.sum() else p
    p = p[p > 0]
    return float(math.exp(-(p * np.log(p)).sum())) if len(p) else float("nan")


def expert_usage(design):
    """Per arm and layer: mean over seeds of MI (bits) and, per domain, the effective number of experts."""
    out = {}
    for arm, rows in design["routing"].items():
        if not rows:
            continue
        n_layer = len(rows[0])
        layers = []
        for li in range(n_layer):
            doms = sorted(set.intersection(*[set(r[li]) for r in rows]))
            mats = [np.array([r[li][d] for d in doms], dtype=float) for r in rows]
            layers.append(dict(
                mutual_information_bits=float(np.mean([A.mutual_information(m) for m in mats])),
                effective_experts={d: float(np.mean([effective_experts(r[li][d]) for r in rows])) for d in doms}))
        out[arm] = layers
    return out


def analyze_designs(designs, metas):
    res = dict(seeds={k: dict(complete=v["seeds"], expected=v["expected"]) for k, v in designs.items()})
    res["partial"] = any(len(v["seeds"]) < len(v["expected"]) for v in designs.values())
    boosted = {u: (metas.get(u) or {}).get("boosted_domain") for u in UNEVEN}
    res["boosted"] = boosted
    res["meta"] = {u: {k: (metas.get(u) or {}).get(k) for k in ("variant", "requested_boosted_domain", "fallback_used",
                                                              "train_tokens_by_source")} for u in UNEVEN}
    even = designs["even"]
    res["even_mean_gap"] = float(np.mean([np.mean(list(even["gap"][s].values())) for s in even["seeds"]])) \
        if even["seeds"] else float("nan")

    # 1. swap contrast
    h, r = boosted["hist4x"], boosted["rep4x"]
    if h and r and all(designs[u]["seeds"] for u in UNEVEN):
        c = {u: {s: designs[u]["gap"][s][h] - designs[u]["gap"][s][r] for s in designs[u]["seeds"]} for u in UNEVEN}
        B, se = two_sample(c["hist4x"].values(), c["rep4x"].values())
        res["swap"] = dict(h=h, r=r, c={u: {str(s): v for s, v in c[u].items()} for u in UNEVEN},
                           mean_c={u: mean_se(c[u].values())["mean"] for u in UNEVEN}, B=B, SE=se, MDE=2 * se,
                           B_over_even_mean_gap=B / res["even_mean_gap"] if res["even_mean_gap"] else float("nan"),
                           verdict=verdict(B, se))
    else:
        res["swap"] = None

    # 2-4. per uneven design
    res["designs"] = {}
    for u in UNEVEN:
        U, b = designs[u], boosted[u]
        if not (b and U["seeds"] and even["seeds"]):
            res["designs"][u] = None
            continue
        others = [d for d in U["domains"] if d != b]
        k = {X: [designs[X]["gap"][s][b] - np.mean([designs[X]["gap"][s][d] for d in others])
                 for s in designs[X]["seeds"]] for X in ("even", u)}
        did, se = two_sample(k[u], k["even"])
        delta = {d: mean_gap(U, d)["mean"] - mean_gap(even, d)["mean"] for d in U["domains"]}
        ranked = sorted(U["domains"], key=lambda d: (delta[d], d))  # most negative change = largest shrinkage
        ratio = {d: mean_gap(U, d)["mean"] / mean_gap(even, d)["mean"] if mean_gap(even, d)["mean"] else float("nan")
                 for d in U["domains"]}
        cross = mean_gap(U, b)
        if math.isnan(cross["se"]):
            cross_text = "not computable (needs at least 2 seeds)"
        elif cross["mean"] + 2 * cross["se"] < 0:
            cross_text = f"yes: win16 beats the baseline on {b} (gap below 0 by more than 2 SE)"
        elif cross["mean"] - 2 * cross["se"] > 0:
            cross_text = f"no: the baseline is still better on {b} (gap above 0 by more than 2 SE)"
        else:
            cross_text = f"no clear difference between win16 and the baseline on {b} (within 2 SE of 0)"
        absolute = {}
        for arm, key in (("base", "base"), ("win16", "win")):
            e = mean_se(even[key][s][b] for s in even["seeds"])["mean"]
            v = mean_se(U[key][s][b] for s in U["seeds"])["mean"]
            absolute[arm] = dict(even=e, uneven=v, improvement=e - v)
        res["designs"][u] = dict(
            boosted=b, k={X: [float(x) for x in k[X]] for X in k}, DiD=did, SE=se, verdict=verdict(did, se),
            budgets_tokens=dict(even=even["total_tokens"], uneven=U["total_tokens"]),
            ratio_boosted=ratio[b], ratio_others={d: ratio[d] for d in others}, delta=delta,
            placebo_rank=ranked.index(b) + 1, n_domains=len(ranked),
            crossover=dict(mean=cross["mean"], se=cross["se"], low=cross["mean"] - 2 * cross["se"],
                           high=cross["mean"] + 2 * cross["se"], text=cross_text),
            absolute=absolute, absolute_extra_win16_gain=absolute["win16"]["improvement"] - absolute["base"]["improvement"])

    # per-domain table and expert usage
    res["per_domain"] = {X: {d: dict(gap=mean_gap(D, d), base=mean_se(D["base"][s][d] for s in D["seeds"])["mean"],
                                     win16=mean_se(D["win"][s][d] for s in D["seeds"])["mean"])
                             for d in D["domains"]} for X, D in designs.items()}
    res["expert_usage"] = {X: expert_usage(D) for X, D in designs.items()}
    return res


# -----------------------------------------------------------------------------
# outputs

def f(x, spec="+.5f"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def summary_lines(res, label):
    L = []
    seeds = ", ".join(f"{X} {len(v['complete'])} of {len(v['expected'])} seeds" for X, v in res["seeds"].items())
    if res["partial"]:
        L += [f"**PARTIAL** ({label}): not every configured run has finished: {seeds}.", ""]
    else:
        L += [f"Seeds ({label}): {seeds}.", ""]
    sw = res["swap"]
    L.append(f"- **Primary (swap contrast): {sw['verdict']}** (B = {f(sw['B'])}, 2·SE = {f(sw['MDE'], '.5f')})"
             if sw else "- **Primary (swap contrast): not available** (needs both uneven designs and their meta.json)")
    for u in UNEVEN:
        D = res["designs"][u]
        if D is None:
            L.append(f"- {u}: not available")
            continue
        L.append(f"- **{u} (boosted {D['boosted']}), difference-in-differences vs even: {D['verdict']}** "
                 f"(DiD = {f(D['DiD'])}, 2·SE = {f(2 * D['SE'], '.5f')})")
        L.append(f"- {u} crossover: {D['crossover']['text']}")
    L += ["", "The hypothesis predicts negative values: a domain's window penalty (gap = win16 - base, positive = "
               "window worse) shrinks when that domain gets 4x the data.", ""]
    if sw:
        L += ["## 1. Swap contrast (primary)", "",
              f"h = {sw['h']} (hist4x boosted), r = {sw['r']} (rep4x boosted); c[X][s] = g[X][s][h] - g[X][s][r].", "",
              "| quantity | value |", "|---|---|",
              f"| mean c[hist4x] | {f(sw['mean_c']['hist4x'])} |", f"| mean c[rep4x] | {f(sw['mean_c']['rep4x'])} |",
              f"| B = mean c[hist4x] - mean c[rep4x] | {f(sw['B'])} |", f"| SE | {f(sw['SE'], '.5f')} |",
              f"| minimum detectable effect (2·SE) | {f(sw['MDE'], '.5f')} |",
              f"| B / even design's mean per-domain gap ({f(res['even_mean_gap'])}) | {f(sw['B_over_even_mean_gap'], '+.3f')} |",
              f"| verdict | {sw['verdict']} |", "",
              "Per-seed c: " + "; ".join(f"{u}: " + ", ".join(f"s{s} {f(v)}" for s, v in sw["c"][u].items())
                                       for u in UNEVEN), ""]
    for u in UNEVEN:
        D = res["designs"][u]
        if D is None:
            continue
        b = D["boosted"]
        L += [f"## 2. {u}: difference-in-differences vs even (boosted {b})", "",
              f"k[X][s] = g[X][s][{b}] - mean of the other domains' gaps. DiD = mean k[{u}] - mean k[even] = "
              f"{f(D['DiD'])}, SE {f(D['SE'], '.5f')}: {D['verdict']}.", "",
              ("Note: the budgets differ ("
               + (f"{D['budgets_tokens']['uneven']:,} vs {D['budgets_tokens']['even']:,} tokens"
                  if D["budgets_tokens"]["uneven"] and D["budgets_tokens"]["even"] else "uneven vs even")
               + "). Subtracting the other domains' mean removes effects common to all domains, but not effects that "
                 "scale with each domain's gap size."), "",
              f"Ratio mean g[{u}][d] / mean g[even][d]: {b} {f(D['ratio_boosted'], '.3f')} (boosted) vs "
              + ", ".join(f"{d} {f(v, '.3f')}" for d, v in D["ratio_others"].items()) + ".", "",
              f"Placebo rank of {b}: {D['placebo_rank']} of {D['n_domains']} (1 = largest shrinkage).", "",
              "| domain | Δ = mean g[U] - mean g[even] |", "|---|---|"]
        L += [f"| {d}{' (boosted)' if d == b else ''} | {f(v)} |" for d, v in sorted(D["delta"].items(), key=lambda x: x[1])]
        c = D["crossover"]
        L += ["", f"## 3. {u}: crossover on {b}", "",
              f"mean g[{u}][{b}] = {f(c['mean'])} ± {f(2 * c['se'], '.5f')} (2 SE): [{f(c['low'])}, {f(c['high'])}]. "
              f"{c['text']}.", "",
              f"## 4. {u}: absolute loss on {b}, even -> {u} (mean over seeds)", "",
              "| arm | even | uneven | improvement |", "|---|---|---|---|"]
        L += [f"| {arm} | {f(a['even'], '.5f')} | {f(a['uneven'], '.5f')} | {f(a['improvement'])} |"
              for arm, a in D["absolute"].items()]
        L += ["", f"The extra data improved win16 by {f(D['absolute_extra_win16_gain'])} more than the baseline on {b}.", ""]
    return L


def per_domain_lines(res, label):
    L = [f"## {label}", "", "| domain | design | mean gap | seed spread | base loss | win16 loss |",
         "|---|---|---|---|---|---|"]
    domains = sorted(set().union(*[set(v) for v in res["per_domain"].values()]))
    for d in domains:
        for X in res["per_domain"]:
            row = res["per_domain"][X].get(d)
            if row:
                tag = " (boosted)" if res["boosted"].get(X) == d else ""
                L.append(f"| {d} | {X}{tag} | {f(row['gap']['mean'])} | {f(row['gap']['sd'], '.5f')} | "
                         f"{f(row['base'], '.5f')} | {f(row['win16'], '.5f')} |")
    return L + [""]


def plots(res, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, u in zip(axes, UNEVEN):
        D = res["designs"][u]
        if D:
            items = sorted(D["delta"].items(), key=lambda x: x[1])
            ax.bar([d for d, _ in items], [v for _, v in items],
                   color=["tab:red" if d == D["boosted"] else "tab:gray" for d, _ in items])
            ax.tick_params(axis="x", rotation=45)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{u}: Δ_d = mean gap({u}) - mean gap(even)" + (f"\nboosted domain ({D['boosted']}) in red" if D else ""),
                     fontsize=10)
    axes[0].set_ylabel("change in win16 - base gap (negative = shrinks)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "uneven_delta.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sw = res["swap"]
    if sw:
        for i, u in enumerate(UNEVEN):
            vals = list(sw["c"][u].values())
            ax.scatter([i] * len(vals), vals, color="tab:blue")
            ax.hlines(np.mean(vals), i - 0.2, i + 0.2, color="black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"hist4x (n={len(sw['c']['hist4x'])})", f"rep4x (n={len(sw['c']['rep4x'])})"])
        ax.set_title(f"c = g[{sw['h']}] - g[{sw['r']}] per seed; B = {sw['B']:+.4f} ({sw['verdict']})", fontsize=9)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_ylabel("c (per seed)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "uneven_swap.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"even": "tab:gray", "hist4x": "tab:red", "rep4x": "tab:green"}
    for X, usage in res["expert_usage"].items():
        for arm, ls in (("base", "--"), ("win16", "-")):
            layers = usage.get(arm)
            if not layers:
                continue
            x = np.arange(1, len(layers) + 1)
            axes[0].plot(x, [l["mutual_information_bits"] for l in layers], ls, color=colors[X], label=f"{X} {arm}")
            axes[1].plot(x, [np.mean(list(l["effective_experts"].values())) for l in layers], ls, color=colors[X],
                         label=f"{X} {arm}")
    axes[0].set_title("source-expert mutual information (bits)")
    axes[1].set_title("effective number of experts per domain (mean over domains)")
    for ax in axes:
        ax.set_xlabel("layer")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "uneven_expert_usage.png"), dpi=130)
    plt.close(fig)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def run(runs_dir, data_root, losses, routing):
    all_runs = A.read_runs(runs_dir)
    designs = {X: design_data(all_runs, p, losses, routing) for X, p in DESIGNS}
    metas = {u: read_meta(designs[u], data_root) for u in UNEVEN}
    missing = [u for u in UNEVEN if designs[u]["seeds"] and metas[u] is None]
    if missing:
        sys.exit(f"uneven.py: no meta.json for {missing} (looked in the runs' data_dir"
                 + (f" under {data_root}" if data_root else "") + "); pass --data_root <folder with the data_m2d2w8_* builds>")
    return analyze_designs(designs, metas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--with_test", action="store_true", help="repeat everything on the locked-test results")
    ap.add_argument("--data_root", help="folder holding the data_m2d2w8_* builds, if not at the runs' data_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results = {"validation": run(args.runs_dir, args.data_root, val_losses, val_routing)}
    if args.with_test:
        results["locked_test"] = run(args.runs_dir, args.data_root, test_losses_reader(args.runs_dir),
                                     test_routing_reader(args.runs_dir))

    L = ["# Uneven-domain designs: does more topic data shrink the window penalty?", ""]
    P = ["# Per-domain gaps (win16 - base), all designs", ""]
    for label, res in results.items():
        L += [f"# {label.replace('_', ' ')}", ""] + summary_lines(res, label.replace("_", " "))
        P += per_domain_lines(res, label.replace("_", " "))
    with open(os.path.join(args.out_dir, "uneven_summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(args.out_dir, "uneven_per_domain.md"), "w") as fh:
        fh.write("\n".join(P) + "\n")
    plots(results["validation"], args.out_dir)
    with open(os.path.join(args.out_dir, "uneven_results.json"), "w") as fh:
        json.dump(jsonable(results), fh, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
