#!/usr/bin/env python
"""Tokenize SlimPajama-6B parquet shards into flat uint16 token files.

Outputs in --out_dir:
  train.bin           training tokens (GPT-2 BPE, end-of-text token after each document)
  val_<source>.bin    validation tokens, one file per source
  meta.json           token counts per source, settings, dataset revision
  sanity.txt          a decoded sample from every output file
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import tiktoken

SOURCE_NAMES = {
    "RedPajamaCommonCrawl": "commoncrawl",
    "RedPajamaC4": "c4",
    "RedPajamaGithub": "github",
    "RedPajamaBook": "book",
    "RedPajamaArXiv": "arxiv",
    "RedPajamaWikipedia": "wikipedia",
    "RedPajamaStackExchange": "stackexchange",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def short_name(meta):
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            meta = {}
    raw = (meta or {}).get("redpajama_set_name") or "unknown"
    return SOURCE_NAMES.get(raw, raw.lower().replace("redpajama", "") or "unknown")


def iter_docs(path, batch_size):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text", "meta"]):
        texts = batch.column("text").to_pylist()
        metas = batch.column("meta").to_pylist()
        yield [t or "" for t in texts], [short_name(m) for m in metas]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="folder that contains data/*.parquet")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_tokens", type=int, default=600_000_000)
    ap.add_argument("--min_val_tokens", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    assert enc.n_vocab <= 65536, "token ids must fit in uint16"

    def encode(texts):
        ids = enc.encode_ordinary_batch(texts, num_threads=args.threads)
        return [x + [eot] for x in ids]

    data_dir = os.path.join(args.raw_dir, "data")
    train_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(data_dir, "validation-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(data_dir, "test-*.parquet")))
    if not train_files or not val_files:
        sys.exit(f"missing parquet files in {data_dir}")
    log(f"{len(train_files)} train files, {len(val_files)} validation, {len(test_files)} test; "
        f"{args.threads} threads")

    # ---- training tokens ----
    train_counts = defaultdict(int)
    total = 0
    used_files = []
    t0 = time.time()
    with open(os.path.join(args.out_dir, "train.bin"), "wb") as f:
        for path in train_files:
            if total >= args.train_tokens:
                break
            used_files.append(os.path.basename(path))
            log(f"train: {os.path.basename(path)}")
            for texts, srcs in iter_docs(path, args.batch_size):
                arrs = []
                for toks, src in zip(encode(texts), srcs):
                    room = args.train_tokens - total
                    if room <= 0:
                        break
                    if len(toks) > room:
                        toks = toks[:room]
                    arrs.append(np.asarray(toks, dtype=np.uint16))
                    train_counts[src] += len(toks)
                    total += len(toks)
                if arrs:
                    np.concatenate(arrs).tofile(f)
                if total >= args.train_tokens:
                    break
            rate = total / max(time.time() - t0, 1e-9) / 1e6
            log(f"  {total / 1e6:.1f}M tokens so far ({rate:.2f}M tokens/s)")
    if total < args.train_tokens:
        log(f"WARNING: only {total:,} training tokens; download more shards and rerun")

    # ---- validation tokens, one file per source ----
    val_arrs = defaultdict(list)
    val_counts = defaultdict(int)
    for path in val_files:
        log(f"validation: {os.path.basename(path)}")
        for texts, srcs in iter_docs(path, args.batch_size):
            for toks, src in zip(encode(texts), srcs):
                val_arrs[src].append(np.asarray(toks, dtype=np.uint16))
                val_counts[src] += len(toks)

    short = {s for s in set(val_counts) | set(train_counts) if val_counts[s] < args.min_val_tokens}
    topped_up = sorted(short)
    if short and test_files:
        log(f"topping up from the test split: {topped_up}")
        for path in test_files:
            for texts, srcs in iter_docs(path, args.batch_size):
                keep = [(t, s) for t, s in zip(texts, srcs)
                        if s in short and val_counts[s] < args.min_val_tokens]
                if not keep:
                    continue
                for toks, (_, src) in zip(encode([t for t, _ in keep]), keep):
                    val_arrs[src].append(np.asarray(toks, dtype=np.uint16))
                    val_counts[src] += len(toks)
    for src, arrs in val_arrs.items():
        np.concatenate(arrs).tofile(os.path.join(args.out_dir, f"val_{src}.bin"))

    # ---- sanity checks ----
    rng = np.random.default_rng(0)
    lines = []
    names = ["train"] + [f"val_{s}" for s in sorted(val_arrs)]
    for name in names:
        path = os.path.join(args.out_dir, f"{name}.bin")
        m = np.memmap(path, dtype=np.uint16, mode="r")
        assert os.path.getsize(path) == 2 * len(m)
        max_id = int(m.max()) if len(m) else -1
        assert max_id < enc.n_vocab, f"{name}: token id {max_id} out of range"
        start = int(rng.integers(0, max(1, len(m) - 200)))
        sample = enc.decode(m[start:start + 200].tolist())
        lines.append(f"===== {name}: {len(m):,} tokens, max id {max_id} =====\n{sample}\n")
    with open(os.path.join(args.out_dir, "sanity.txt"), "w") as f:
        f.write("\n".join(lines))

    # ---- metadata ----
    rev_path = os.path.join(args.raw_dir, "REVISION.txt")
    revision = open(rev_path).read().strip() if os.path.exists(rev_path) else None
    meta = {
        "dataset": "DKYoon/SlimPajama-6B",
        "dataset_revision": revision,
        "tokenizer": "gpt2 (tiktoken)",
        "eot_token": eot,
        "dtype": "uint16",
        "train_files_used": used_files,
        "train_tokens_total": total,
        "train_tokens_by_source": dict(sorted(train_counts.items())),
        "train_share_by_source": {s: round(c / max(total, 1), 4) for s, c in sorted(train_counts.items())},
        "val_tokens_by_source": dict(sorted(val_counts.items())),
        "val_topped_up_from_test": topped_up if test_files else [],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log(f"done in {(time.time() - t0) / 60:.1f} min")
    print(f"{'source':<15}{'train tokens':>15}{'share':>8}{'val tokens':>14}")
    for s in sorted(set(train_counts) | set(val_counts)):
        print(f"{s:<15}{train_counts[s]:>15,}{train_counts[s] / max(total, 1):>8.1%}{val_counts[s]:>14,}")


if __name__ == "__main__":
    main()
