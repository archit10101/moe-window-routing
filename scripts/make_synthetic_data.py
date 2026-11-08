#!/usr/bin/env python
"""Write a small synthetic data folder in the same format as data/slimpajama/prepare.py.

Outputs in --out:
  train.bin          random uint16 tokens below --vocab (default 1000)
  val_<source>.bin   one file per source (three sources by default)
  meta.json          train_tokens_total, train_tokens_by_source, train_share_by_source,
                     val_tokens_by_source, eot_token, dtype

Used by the smoke-cpu config and by the tests. The data has no document
structure; eot_token is recorded only so meta.json has the real schema.

Usage: python scripts/make_synthetic_data.py --out /tmp/synth
"""
import argparse
import json
import os

import numpy as np

# source -> (share of training tokens, validation tokens). "book" is deliberately
# small, like val_book.bin, so evaluation windows overlap there.
DEFAULT_SOURCES = {
    "c4": (0.6, 40_000),
    "github": (0.3, 20_000),
    "book": (0.1, 1_000),
}


def make(out, train_tokens=200_000, sources=None, vocab=1000, seed=0):
    """Write the synthetic data folder and return the meta dict."""
    sources = DEFAULT_SOURCES if sources is None else sources
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)

    rng.integers(0, vocab, size=train_tokens, dtype=np.uint16).tofile(os.path.join(out, "train.bin"))
    val_counts = {}
    for name in sorted(sources):
        n_val = sources[name][1]
        rng.integers(0, vocab, size=n_val, dtype=np.uint16).tofile(os.path.join(out, f"val_{name}.bin"))
        val_counts[name] = n_val

    total_share = sum(s for s, _ in sources.values())
    by_source = {name: int(round(train_tokens * sources[name][0] / total_share)) for name in sorted(sources)}
    meta = {
        "dataset": "synthetic",
        "tokenizer": "none (random ids)",
        "eot_token": 50256,
        "dtype": "uint16",
        "train_tokens_total": train_tokens,
        "train_tokens_by_source": by_source,
        "train_share_by_source": {name: round(c / train_tokens, 4) for name, c in by_source.items()},
        "val_tokens_by_source": val_counts,
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_tokens", type=int, default=200_000)
    ap.add_argument("--vocab", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = make(args.out, train_tokens=args.train_tokens, vocab=args.vocab, seed=args.seed)
    print(f"wrote synthetic data to {args.out}: {meta['train_tokens_total']:,} train tokens, "
          f"val sources {meta['val_tokens_by_source']}")


if __name__ == "__main__":
    main()
