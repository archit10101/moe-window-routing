#!/usr/bin/env python
"""Print the latest step, tokens, train CE, tok/s, val_overall and done flag for every run folder.

Usage: python scripts/status.py <out_root>    (stdlib only)
"""
import json
import os
import sys


def summarize(run_dir):
    row = dict(step="-", tokens="-", ce="-", tok_s="-", val="-")
    path = os.path.join(run_dir, "log.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a partially written last line
                if r.get("type") == "train":
                    row.update(step=r["step"], tokens=f"{r['tokens'] / 1e6:.1f}M", ce=f"{r['ce']:.4f}",
                               tok_s=f"{r['tok_per_s']:,.0f}")
                elif r.get("type") in ("eval", "final"):
                    row["val"] = f"{r['val_overall']:.4f}"
                    if r["type"] == "final":
                        row.update(step=r["step"], tokens=f"{r['tokens'] / 1e6:.1f}M")
    row["done"] = "yes" if os.path.exists(os.path.join(run_dir, "done")) else "no"
    return row


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: status.py <out_root>")
    root = sys.argv[1]
    if not os.path.isdir(root):
        sys.exit(f"{root} does not exist yet")
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d)) and d != "slurm")
    fmt = "{:<18} {:>6} {:>9} {:>8} {:>10} {:>11} {:>5}"
    print(fmt.format("run", "step", "tokens", "train_ce", "tok/s", "val_overall", "done"))
    for name in names:
        r = summarize(os.path.join(root, name))
        print(fmt.format(name, r["step"], r["tokens"], r["ce"], r["tok_s"], r["val"], r["done"]))


if __name__ == "__main__":
    main()
