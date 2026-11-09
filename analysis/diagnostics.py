#!/usr/bin/env python
"""Diagnostics on finished checkpoints (docs/FOLLOWUP_PROMPT.md part 4).

    python analysis/diagnostics.py --runs_dir <dir> --data_dir <dir> --out_dir <dir>
        [--runs base-s0,...] [--device cuda] [--decoded_vocab vocab.json]

Each run's final checkpoint is evaluated in eval mode under no_grad on the same fixed
final-eval offsets train.py uses (train.load_val_sets), one source at a time, accumulating
sums and counts only.

A1 (E1, where the loss goes): per-token cross-entropy bucketed by input-token category,
target-token category and position. Tokens are weighted like val_overall (each source by its
train share), so bucket CEs average to val_overall and the per-bucket contributions
(gap x token share) sum exactly to the val_overall gap. Gap = win16 - base (positive = the
window arm is worse).

A2 (E4, routing dynamics): per run and layer, switch rate, router entropy, top-1 probability
and token/source determinism of the top-1 expert (pooled eval tokens, unweighted), plus the
global grad norm over training from log.jsonl.

Token categories need the GPT-2 vocabulary (tiktoken). Without network access, set
TIKTOKEN_CACHE_DIR to a primed cache, or pass --decoded_vocab: a JSON list with the decoded
string of every token id.
"""
import argparse
import csv
import gc
import json
import math
import os
import string
import sys
import unicodedata
from contextlib import nullcontext

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402
import train as T  # noqa: E402

CATEGORIES = ["space", "punct", "digit", "frequent", "other"]
DEFAULT_RUNS = [f"base-s{s}" for s in range(5)] + [f"win16-s{s}" for s in range(5)]
ARMS = ("base", "win16")
N_FREQUENT = 100
FREQUENT_PREFIX = 50_000_000  # tokens of train.bin used to find the frequent ids
MIN_TOKEN_COUNT = 20          # token ids seen fewer times are left out of token_determinism
A2_METRICS = ["switch_rate", "router_entropy", "top1_prob", "token_determinism", "source_determinism"]


# -----------------------------------------------------------------------------
# token categories

def _is_punct(ch):
    return ch in string.punctuation or unicodedata.category(ch).startswith("P")


def classify_string(s):
    """space / punct / digit / other for one decoded token ('frequent' is assigned later)."""
    if s and not s.strip():
        return "space"  # whitespace only, including newlines
    t = s.strip()
    if t and all(_is_punct(ch) for ch in t):
        return "punct"
    if t and t.isdigit():
        return "digit"
    return "other"


def decoded_vocab(vocab_size, path=None):
    """Decoded string of every token id (ids beyond the tokenizer's vocabulary decode to '')."""
    if path:
        with open(path) as f:
            strings = json.load(f)
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        strings = [enc.decode([i]) for i in range(enc.n_vocab)]
    return (list(strings) + [""] * vocab_size)[:vocab_size]


def token_categories(strings, train_path):
    """Category index (into CATEGORIES) for every token id."""
    cats = np.array([CATEGORIES.index(classify_string(s)) for s in strings], dtype=np.int64)
    data = np.memmap(train_path, dtype=np.uint16, mode="r")
    counts = np.bincount(np.asarray(data[:FREQUENT_PREFIX]), minlength=len(strings))[:len(strings)]
    top = np.argsort(-counts, kind="stable")[:N_FREQUENT]
    top = top[counts[top] > 0]
    other, frequent = CATEGORIES.index("other"), CATEGORIES.index("frequent")
    cats[top[cats[top] == other]] = frequent  # only ids not already in space/punct/digit
    return cats


def position_buckets(block_size):
    edges = [0, 16, 64, block_size]
    labels = ["0-15", "16-63", f"64-{block_size - 1}"]
    idx = np.zeros(block_size, dtype=np.int64)
    for b in range(3):
        idx[edges[b]:edges[b + 1]] = b
    return labels, idx


# -----------------------------------------------------------------------------
# evaluation of one run

def load_run(run_dir, data_dir, device):
    ckpt = torch.load(os.path.join(run_dir, "ckpt", "latest.pt"), map_location="cpu", weights_only=False)
    cfg = T.with_defaults(ckpt["config"])
    cfg.update(data_dir=data_dir, device=device)
    model = T.build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    del ckpt
    return model, cfg


@torch.no_grad()
def evaluate_run(model, cfg, val_sets, cats, pos_idx, device):
    """Sums and counts over every final-eval token, one source at a time."""
    Tn, ebs, E, L = cfg["block_size"], cfg["eval_batch_size"], cfg["n_exp"], cfg["n_layer"]
    V = cfg["vocab_size"]
    n_src, n_cat, n_pos = len(val_sets), len(CATEGORIES), int(pos_idx.max()) + 1
    is_cuda = device.startswith("cuda")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if is_cuda and cfg["dtype"] == "bfloat16" else nullcontext()
    cat_t = torch.as_tensor(cats, device=device)
    pos_t = torch.as_tensor(pos_idx, device=device)
    z = dict(dtype=torch.float64, device=device)
    acc = {
        "ce": {f: torch.zeros(n_src, n, **z) for f, n in (("input", n_cat), ("target", n_cat), ("position", n_pos))},
        "n": {f: torch.zeros(n_src, n, **z) for f, n in (("input", n_cat), ("target", n_cat), ("position", n_pos))},
        "switch": torch.zeros(L, **z), "pairs": torch.zeros(L, **z), "entropy": torch.zeros(L, **z),
        "top1": torch.zeros(L, **z), "tokens": torch.zeros(L, **z),
        "tok_exp": torch.zeros(L, V * E, dtype=torch.long, device=device),
        "src_exp": torch.zeros(L, n_src, E, dtype=torch.long, device=device),
    }
    for si, (src, (data, offsets, _)) in enumerate(val_sets.items()):
        for j in range(0, len(offsets), ebs):
            x, y = T.make_xy(data, offsets[j:j + ebs], Tn, device)
            B = x.size(0)
            with ctx:
                logits, _ = model(x)
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="none").double()
            del logits
            pos = pos_t.expand(B, Tn).reshape(-1)
            for fam, idx in (("input", cat_t[x].view(-1)), ("target", cat_t[y].view(-1)), ("position", pos)):
                n_b = acc["ce"][fam].size(1)
                acc["ce"][fam][si] += torch.bincount(idx, weights=ce, minlength=n_b)
                acc["n"][fam][si] += torch.bincount(idx, minlength=n_b).double()
            for li, m in enumerate(model.moe_layers()):
                p = F.softmax(m.logits.view(B, Tn, E), dim=-1)
                top_p, top_i = p.max(-1)
                acc["switch"][li] += (top_i[:, 1:] != top_i[:, :-1]).sum()
                acc["pairs"][li] += B * (Tn - 1)
                acc["entropy"][li] += (torch.special.entr(p).sum(-1) / math.log(2)).sum()
                acc["top1"][li] += top_p.sum()
                acc["tokens"][li] += B * Tn
                acc["tok_exp"][li] += torch.bincount((x * E + top_i).view(-1), minlength=V * E)
                acc["src_exp"][li, si] += torch.bincount(top_i.view(-1), minlength=E)
    out = {k: ({f: t.cpu().numpy() for f, t in v.items()} if isinstance(v, dict) else v.cpu().numpy())
           for k, v in acc.items()}
    out["tok_exp"] = out["tok_exp"].reshape(L, V, E)
    return out


# -----------------------------------------------------------------------------
# A1 aggregation

def a1_bucket_ces(acc, shares):
    """Per family: token share and val_overall-weighted CE per bucket, plus the overall CE."""
    n_src_tokens = acc["n"]["input"].sum(1)       # every token falls in exactly one input bucket
    w = shares / n_src_tokens                      # weight per token of each source
    out = {}
    for fam in ("input", "target", "position"):
        W = (w[:, None] * acc["n"][fam]).sum(0)
        S = (w[:, None] * acc["ce"][fam]).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[fam] = dict(share=W / shares.sum(), ce=np.where(W > 0, S / np.where(W > 0, W, 1), np.nan),
                            tokens=acc["n"][fam].sum(0))
    overall = (w[:, None] * acc["ce"]["input"]).sum() / shares.sum()
    return out, float(overall)


def arm_stats(values):
    v = np.asarray(values, dtype=float)
    std = v.std(ddof=1) if len(v) >= 2 else np.nan
    return float(v.mean()), float(std)


def entropy_bits(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def determinism(joint):
    """1 - H(E | X) / H(E) from a [values x experts] count matrix."""
    total = joint.sum()
    if total == 0:
        return float("nan")
    h_e = entropy_bits(joint.sum(0))
    if h_e == 0:
        return float("nan")
    rows = joint.sum(1)
    h_cond = sum(rows[i] / total * entropy_bits(joint[i]) for i in np.nonzero(rows)[0])
    return 1.0 - h_cond / h_e


# -----------------------------------------------------------------------------
# outputs

def fmt(x, spec=".5f"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def write_a1(out_dir, results, labels, sources, final_vals):
    names = list(results)
    arm_runs = {arm: [n for n in names if A.is_seed_run(n, arm)] for arm in ARMS}
    fam_labels = {"input": CATEGORIES, "target": CATEGORIES, "position": labels}
    fam_title = {"input": "input token category (token at t)", "target": "target token category (token at t+1)",
                 "position": "position in the sequence"}
    per_run = {n: results[n]["a1"] for n in names}
    overall = {n: results[n]["overall"] for n in names}
    ob, ow = arm_stats([overall[n] for n in arm_runs["base"]]), arm_stats([overall[n] for n in arm_runs["win16"]])
    total_gap = ow[0] - ob[0]

    rows, md_tables, sums = [], {}, {}
    for fam in ("input", "target", "position"):
        share = per_run[names[0]][fam]["share"]
        tokens = per_run[names[0]][fam]["tokens"]
        table, contrib_sum = [], 0.0
        for b, label in enumerate(fam_labels[fam]):
            ces = {n: per_run[n][fam]["ce"][b] for n in names}
            bm, bs = arm_stats([ces[n] for n in arm_runs["base"]])
            wm, ws = arm_stats([ces[n] for n in arm_runs["win16"]])
            gap = wm - bm
            contrib = share[b] * gap if share[b] > 0 else 0.0
            contrib_sum += contrib
            rows.append([fam, label, int(tokens[b]), share[b]] + [ces[n] for n in names]
                        + [bm, bs, wm, ws, gap, contrib])
            table.append((label, share[b], bm, bs, wm, ws, gap, contrib))
        md_tables[fam] = table
        sums[fam] = contrib_sum
    rows.append(["total", "all", int(per_run[names[0]]["input"]["tokens"].sum()), 1.0]
                + [overall[n] for n in names] + [ob[0], ob[1], ow[0], ow[1], total_gap, total_gap])

    header = (["family", "bucket", "tokens", "token_share"] + [f"ce_{n}" for n in names]
              + ["base_mean", "base_std", "win16_mean", "win16_std", "gap_win16_minus_base", "contribution"])
    with open(os.path.join(out_dir, "a1_buckets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([x if isinstance(x, (str, int)) else f"{x:.17g}" for x in r])  # full precision

    # source x input-category gap table (CE within each source, unweighted)
    src_gap = []
    for si, src in enumerate(sources):
        row = []
        for c in range(len(CATEGORIES)):
            def ce(n):
                acc = results[n]["acc"]
                cnt = acc["n"]["input"][si, c]
                return acc["ce"]["input"][si, c] / cnt if cnt > 0 else np.nan
            row.append(arm_stats([ce(n) for n in arm_runs["win16"]])[0] - arm_stats([ce(n) for n in arm_runs["base"]])[0])
        src_gap.append(row)

    L = ["# A1: where the loss goes (E1)", "",
         f"Runs: base = {arm_runs['base']}, win16 = {arm_runs['win16']}. Gap = win16 - base (positive: window worse).",
         "Tokens are weighted like val_overall (each source by its train share), so the bucket CEs average to "
         "val_overall and the contributions (gap x token share) sum to the val_overall gap.", "",
         f"**Total gap (val_overall, these checkpoints and tokens): {total_gap:+.5f}**", "",
         "## Check against final.json", "", "| run | diagnostics val_overall | final.json val_overall | diff |",
         "|---|---|---|---|"]
    for n in names:
        fv = final_vals.get(n)
        L.append(f"| {n} | {overall[n]:.5f} | {fmt(fv)} | {fmt(overall[n] - fv if fv is not None else None, '+.2e')} |")
    for fam in ("input", "target", "position"):
        L += ["", f"## By {fam_title[fam]}", "",
              "| bucket | token share | base CE (mean ± std) | win16 CE (mean ± std) | gap | contribution | % of gap |",
              "|---|---|---|---|---|---|---|"]
        for label, sh, bm, bs, wm, ws, gap, contrib in md_tables[fam]:
            pct = 100 * contrib / total_gap if total_gap else float("nan")
            L.append(f"| {label} | {sh:.4f} | {fmt(bm)} ± {fmt(bs)} | {fmt(wm)} ± {fmt(ws)} | {fmt(gap, '+.5f')} | "
                     f"{contrib:+.5f} | {fmt(pct, '.1f')} |")
        L.append(f"| **sum** | 1.0000 | | | | {sums[fam]:+.5f} | |")
    L += ["", "## Gap by source x input category (CE within each source; win16 - base)", "",
          "| source | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for src, row in zip(sources, src_gap):
        L.append(f"| {src} | " + " | ".join(fmt(g, "+.4f") for g in row) + " |")
    with open(os.path.join(out_dir, "a1_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, fam in zip(axes, ("input", "target")):
        gaps = [t[6] for t in md_tables[fam]]
        errs = [math.sqrt((t[3] ** 2 + t[5] ** 2) / 2) if not (math.isnan(t[3]) or math.isnan(t[5])) else 0.0
                for t in md_tables[fam]]
        ax.bar(CATEGORIES, gaps, yerr=errs, capsize=4, color="tab:orange")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axhline(total_gap, color="tab:red", lw=1, ls="--", label=f"overall gap {total_gap:+.4f}")
        ax.set_title(f"gap (win16 - base) by {fam} token category")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("CE gap (error bar: pooled seed std)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "a1_gap_by_category.png"), dpi=130)
    plt.close(fig)
    return dict(total_gap=total_gap, contribution_sums=sums)


def write_a2(out_dir, runs_dir, results):
    names = list(results)
    arm_runs = {arm: [n for n in names if A.is_seed_run(n, arm)] for arm in ARMS}
    rows = []
    metrics = {}
    for n in names:
        acc, windows = results[n]["acc"], results[n]["windows"]
        metrics[n] = []
        for li in range(len(windows)):
            tok = acc["tok_exp"][li]
            tok = tok[tok.sum(1) >= MIN_TOKEN_COUNT]
            m = dict(switch_rate=acc["switch"][li] / acc["pairs"][li],
                     router_entropy=acc["entropy"][li] / acc["tokens"][li],
                     top1_prob=acc["top1"][li] / acc["tokens"][li],
                     token_determinism=determinism(tok),
                     source_determinism=determinism(acc["src_exp"][li]))
            metrics[n].append(m)
            rows.append([n, li + 1, windows[li]] + [m[k] for k in A2_METRICS])
    with open(os.path.join(out_dir, "a2_routing.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "layer", "window"] + A2_METRICS)
        for r in rows:
            w.writerow(r[:3] + [f"{x:.6g}" for x in r[3:]])

    n_layer = len(next(iter(metrics.values())))
    L = ["# A2: routing dynamics (E4)", "",
         f"Runs: base = {arm_runs['base']}, win16 = {arm_runs['win16']}. Pooled final-eval tokens (unweighted). "
         f"token_determinism uses token ids seen at least {MIN_TOKEN_COUNT} times. Values: mean ± std over seeds.", ""]
    for k in A2_METRICS:
        L += [f"## {k}", "", "| layer | base | win16 | win16 - base |", "|---|---|---|---|"]
        for li in range(n_layer):
            b = arm_stats([metrics[n][li][k] for n in arm_runs["base"]])
            wv = arm_stats([metrics[n][li][k] for n in arm_runs["win16"]])
            L.append(f"| {li + 1} | {fmt(b[0], '.4f')} ± {fmt(b[1], '.4f')} | {fmt(wv[0], '.4f')} ± {fmt(wv[1], '.4f')} | "
                     f"{fmt(wv[0] - b[0], '+.4f')} |")
        L.append("")
    with open(os.path.join(out_dir, "a2_summary.md"), "w") as f:
        f.write("\n".join(L))

    fig, axes = plt.subplots(1, len(A2_METRICS), figsize=(4 * len(A2_METRICS), 3.8))
    layers = np.arange(1, n_layer + 1)
    for ax, k in zip(axes, A2_METRICS):
        for arm, color in (("base", "tab:blue"), ("win16", "tab:orange")):
            vals = np.array([[metrics[n][li][k] for li in range(n_layer)] for n in arm_runs[arm]])
            ax.plot(layers, vals.mean(0), "o-", color=color, label=arm)
            ax.fill_between(layers, vals.min(0), vals.max(0), color=color, alpha=0.2, lw=0)
        ax.set_title(k)
        ax.set_xlabel("layer")
        ax.set_xticks(layers)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "a2_by_layer.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, color in (("base", "tab:blue"), ("win16", "tab:orange")):
        curves = []
        for n in arm_runs[arm]:
            recs = A.read_run(os.path.join(runs_dir, n))["train"]
            curves.append({r["tokens"]: r["grad_norm"] for r in recs.values()})
        common = sorted(set.intersection(*[set(c) for c in curves])) if curves else []
        if not common:
            continue
        v = np.array([[c[t] for t in common] for c in curves])
        t = np.array(common) / 1e6
        ax.plot(t, v.mean(0), color=color, lw=1.2, label=f"{arm} (mean of {len(curves)})")
        ax.fill_between(t, v.min(0), v.max(0), color=color, alpha=0.25, lw=0)
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("global grad norm (before clipping)")
    ax.set_yscale("log")
    ax.set_title("Global grad norm over training (band = min-max over seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "a2_grad_norm.png"), dpi=130)
    plt.close(fig)
    return metrics


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="comma-separated run names")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--decoded_vocab", help="JSON list of decoded token strings (skips tiktoken)")
    args = ap.parse_args()

    names = []
    for n in [r.strip() for r in args.runs.split(",") if r.strip()]:
        if os.path.exists(os.path.join(args.runs_dir, n, "ckpt", "latest.pt")):
            names.append(n)
        else:
            print(f"WARNING: {n}: no checkpoint in {args.runs_dir}/{n}/ckpt; skipping")
    for arm in ARMS:
        if not any(A.is_seed_run(n, arm) for n in names):
            sys.exit(f"diagnostics: need at least one {arm}-s* run with a checkpoint (have {names})")
    os.makedirs(args.out_dir, exist_ok=True)

    results, val_sets, cats, ref_cfg = {}, None, None, None
    for n in names:
        print(f"evaluating {n} on {args.device}", flush=True)
        model, cfg = load_run(os.path.join(args.runs_dir, n), args.data_dir, args.device)
        eval_keys = ("block_size", "final_eval_batches", "eval_batch_size", "vocab_size")
        if ref_cfg is None:
            ref_cfg = cfg
            val_sets = T.load_val_sets(args.data_dir, cfg)
            strings = decoded_vocab(cfg["vocab_size"], args.decoded_vocab)
            cats = token_categories(strings, os.path.join(args.data_dir, "train.bin"))
            labels, pos_idx = position_buckets(cfg["block_size"])
        elif any(cfg[k] != ref_cfg[k] for k in eval_keys):
            sys.exit(f"diagnostics: {n} differs from {names[0]} in {eval_keys}; runs must share eval settings")
        acc = evaluate_run(model, cfg, val_sets, cats, pos_idx, args.device)
        shares = np.array([s for _, _, s in val_sets.values()], dtype=float)
        a1, overall = a1_bucket_ces(acc, shares)
        results[n] = dict(acc=acc, a1=a1, overall=overall, windows=list(cfg["window_layers"]))
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    final_vals = {n: A.final_val(A.read_run(os.path.join(args.runs_dir, n))) for n in names}
    a1 = write_a1(args.out_dir, results, labels, list(val_sets), final_vals)
    write_a2(args.out_dir, args.runs_dir, results)
    print(f"A1 total gap (win16 - base): {a1['total_gap']:+.5f}; contribution sums: "
          + ", ".join(f"{k} {v:+.5f}" for k, v in a1["contribution_sums"].items()))
    print(f"wrote a1_buckets.csv, a1_summary.md, a1_gap_by_category.png, a2_routing.csv, a2_summary.md, "
          f"a2_by_layer.png, a2_grad_norm.png to {args.out_dir}")


if __name__ == "__main__":
    main()
