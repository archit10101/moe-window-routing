#!/usr/bin/env python
"""Prepare M2D2-Wiki-8: eight top-level Wikipedia categories of M2D2 (docs/M2D2_PROMPT.md part 4).

Output format matches data/slimpajama/prepare.py: GPT-2 BPE (tiktoken), end-of-text token after
each document, flat uint16 .bin files, meta.json and sanity.txt. Three modes:

  --download --raw_dir <dir>     (login node) download the 8 archives + Philosophy, extract them,
                                 record the dataset commit in raw_dir/REVISION.txt
  --inspect  --raw_dir <dir>     (login node) line statistics per config, a --doc_mode
                                 recommendation, and the Philosophy vs Philosophy_and_thinking overlap
  --build    --raw_dir <dir> --out_dir <dir> --doc_mode {line,blank,heading,block}
             [--block_lines K] [--threads N]
                                 (CPU job) pool each domain's train/valid/test, form documents,
                                 deduplicate, order by a salted content hash, and fill the splits
                                 test (locked) -> val -> calib -> train with whole documents

Documents are hashed on their normalized text (NFC, whitespace runs collapsed, stripped) and
tokenized from their original text (lines joined with "\\n", stripped). When several documents in
a domain share a normalized hash, the copy whose original text has the lowest SHA-256 is kept.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np

REPO_ID = "machelreid/m2d2"
# domain (source) name -> top-level M2D2 config; each archive already contains its subcategories
DOMAINS = {
    "technology": "Technology_and_applied_sciences",
    "natural_sciences": "Natural_and_physical_sciences",
    "mathematics": "Mathematics_and_logic",
    "health": "Health_and_fitness",
    "history": "History_and_events",
    "culture": "Culture_and_the_arts",
    "society": "Society_and_social_sciences",
    "philosophy": "Philosophy_and_thinking",
}
EXTRA_CONFIG = "Philosophy"  # downloaded and inspected only (overlap report), never built
FILES = ("train.txt", "valid.txt", "test.txt")
SPLITS = ("test", "val", "calib", "train")  # fill order
DEFAULT_QUOTAS = {"test": 1_500_000, "val": 1_500_000, "calib": 1_000_000, "train": 15_000_000}
MIN_TOKENS, MAX_TOKENS = 64, 50_000  # including the end-of-text token
SPLIT_SALT = "m2d2w8-v1:"
TRAIN_SHUFFLE_SEED = 12345
DOC_MODES = ("line", "blank", "heading", "block")
TOKENIZE_CHUNK = 256
SAMPLING_NOTE = ("train.py samples random 1,024-token windows with replacement. A budget of ~120M tokens over "
                 "~120M training tokens covers about 63% of positions at least once (1 - e^-1), and ~26% two or "
                 "more times; both M2D2 arms share this regime. For comparison, the SlimPajama pilot covered ~57% "
                 "(500M over 600M), and the slim120 control covers ~18% (120M over 600M) with almost no repeats.")


class BuildError(RuntimeError):
    pass


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# -----------------------------------------------------------------------------
# documents

_TOP_HEADING = re.compile(r"=\s+[^=\s].*?\s+=")  # "= Title =" (wikitext top level), not "= = Sub = ="
_ANY_HEADING = re.compile(r"=+\s+.*?\s+=+")


def is_top_heading(line):
    return bool(_TOP_HEADING.fullmatch(line))


def is_any_heading(line):
    return bool(_ANY_HEADING.fullmatch(line))


def documents_from_lines(lines, mode, block_lines=40):
    """Group the lines of one file into documents (original text: stripped lines joined by "\\n", stripped).

    line:    every non-empty line is a document
    blank:   documents are separated by empty lines
    heading: a document starts at every top-level heading line ("= Title =")
    block:   consecutive groups of block_lines lines (article boundaries not guaranteed)
    """
    if mode not in DOC_MODES:
        raise ValueError(f"unknown doc_mode {mode!r}")
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        buf.clear()
        return text

    for raw in lines:
        line = raw.strip()
        if mode == "line":
            if line:
                yield line
        elif mode == "blank":
            if line:
                buf.append(line)
            elif buf:
                yield flush()
        elif mode == "heading":
            if is_top_heading(line) and any(buf):
                text = flush()
                if text:
                    yield text
            buf.append(line)
        else:  # block
            buf.append(line)
            if len(buf) == block_lines:
                text = flush()
                if text:
                    yield text
    if buf:
        text = flush()
        if text:
            yield text


def iter_config_documents(raw_dir, config, mode, block_lines):
    """Documents of one config, pooling train, valid and test (a document never spans two files)."""
    for name in FILES:
        path = os.path.join(raw_dir, config, name)
        if not os.path.exists(path):
            raise BuildError(f"{path} not found (run --download first)")
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from documents_from_lines(f, mode, block_lines)


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def doc_digest(text):
    """(normalized-text hash, original-text hash), as raw SHA-256 digests."""
    return hashlib.sha256(normalize(text).encode("utf-8")).digest(), hashlib.sha256(text.encode("utf-8")).digest()


def order_key(doc_hash_hex):
    """Content-only order within a domain: first 8 bytes of SHA-256(salt + doc_hash) as an integer."""
    return int.from_bytes(hashlib.sha256((SPLIT_SALT + doc_hash_hex).encode("utf-8")).digest()[:8], "big")


# -----------------------------------------------------------------------------
# download

def extract_archive(tar_path, raw_dir):
    """Extract one M2D2 archive into raw_dir (refusing paths that escape it)."""
    with tarfile.open(tar_path, "r:gz") as tf:
        root = os.path.realpath(raw_dir)
        for m in tf.getmembers():
            target = os.path.realpath(os.path.join(raw_dir, m.name))
            if not (target == root or target.startswith(root + os.sep)) or m.issym() or m.islnk():
                raise BuildError(f"{tar_path}: unsafe member {m.name!r}")
        if hasattr(tarfile, "data_filter"):
            tf.extractall(raw_dir, filter="data")
        else:
            tf.extractall(raw_dir)


def download(raw_dir):
    from huggingface_hub import HfApi, hf_hub_download  # only needed here

    os.makedirs(raw_dir, exist_ok=True)
    revision = HfApi().dataset_info(REPO_ID).sha
    log(f"{REPO_ID} at commit {revision}")
    configs = list(DOMAINS.values()) + [EXTRA_CONFIG]
    for config in configs:
        if all(os.path.exists(os.path.join(raw_dir, config, f)) for f in FILES):
            log(f"{config}: already extracted, skipping")
            continue
        log(f"{config}: downloading")
        path = hf_hub_download(REPO_ID, f"data/{config}.tar.gz", repo_type="dataset", revision=revision,
                               local_dir=os.path.join(raw_dir, "archives"))
        log(f"{config}: extracting {path}")
        extract_archive(path, raw_dir)
        missing = [f for f in FILES if not os.path.exists(os.path.join(raw_dir, config, f))]
        if missing:
            raise BuildError(f"{config}: archive did not contain {missing} under {config}/")
    with open(os.path.join(raw_dir, "REVISION.txt"), "w") as f:
        f.write(revision + "\n")
    log(f"done; revision recorded in {raw_dir}/REVISION.txt")


# -----------------------------------------------------------------------------
# inspect

def file_stats(path):
    lengths, empty, top, anyh, short_long, total = [], 0, 0, 0, 0, 0
    prev = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            total += 1
            if not line:
                empty += 1
                prev = None
                continue
            lengths.append(len(line))
            top += is_top_heading(line)
            anyh += is_any_heading(line)
            if prev is not None and len(prev) < 60 and not prev.endswith((".", "!", "?", ":", ";", ",")) \
                    and len(line) > 200:
                short_long += 1
            prev = line
    lengths = np.array(lengths) if lengths else np.zeros(1)
    nonempty = max(1, total - empty)
    return dict(lines=total, empty_frac=empty / max(1, total), top_heading_frac=top / nonempty,
                any_heading_frac=anyh / nonempty, short_then_long_frac=short_long / nonempty,
                pct=dict(zip((5, 50, 95, 99), np.percentile(lengths, [5, 50, 95, 99]).tolist())))


def recommend(stats):
    """doc_mode recommendation from the pooled statistics of one config, with the reason."""
    if stats["top_heading_frac"] >= 0.001:
        return "heading", f"{stats['top_heading_frac']:.2%} of lines are top-level '= Title =' headings"
    if stats["empty_frac"] >= 0.02:
        return "blank", f"{stats['empty_frac']:.1%} of lines are empty (separators)"
    if stats["pct"][50] >= 500 and stats["any_heading_frac"] < 0.005 and stats["short_then_long_frac"] < 0.01:
        return "line", (f"median line {stats['pct'][50]:.0f} chars, no blank lines or heading patterns: "
                        "lines look like whole articles")
    return "block", ("no reliable article boundaries (short lines without separators or headings); "
                     "WARNING: block mode does not guarantee whole-article integrity")


def pooled(stats_list):
    lines = sum(s["lines"] for s in stats_list)
    w = [s["lines"] / max(1, lines) for s in stats_list]
    out = dict(lines=lines, pct={p: sum(wi * s["pct"][p] for wi, s in zip(w, stats_list)) for p in (5, 50, 95, 99)})
    for k in ("empty_frac", "top_heading_frac", "any_heading_frac", "short_then_long_frac"):
        out[k] = sum(wi * s[k] for wi, s in zip(w, stats_list))
    return out


def doc_hashes(raw_dir, config, mode, block_lines):
    return {doc_digest(t)[0] for t in iter_config_documents(raw_dir, config, mode, block_lines)}


def inspect(raw_dir, overlap_mode=None, block_lines=40):
    recs = {}
    for config in list(DOMAINS.values()) + [EXTRA_CONFIG]:
        if not os.path.isdir(os.path.join(raw_dir, config)):
            print(f"\n=== {config}: not found in {raw_dir}")
            continue
        print(f"\n=== {config}")
        per_file = []
        for name in FILES:
            path = os.path.join(raw_dir, config, name)
            s = file_stats(path)
            per_file.append(s)
            p = s["pct"]
            print(f"  {name:<10} lines {s['lines']:>10,} | length p5/p50/p95/p99 {p[5]:.0f}/{p[50]:.0f}/{p[95]:.0f}/"
                  f"{p[99]:.0f} chars | empty {s['empty_frac']:.2%} | headings: top-level {s['top_heading_frac']:.2%}, "
                  f"any {s['any_heading_frac']:.2%}, short-then-long {s['short_then_long_frac']:.2%}")
        print("  first 8 lines of train.txt:")
        with open(os.path.join(raw_dir, config, "train.txt"), encoding="utf-8", errors="replace") as f:
            for i, line in zip(range(8), f):
                print(f"    {i + 1}: {line.rstrip(chr(10))[:200]!r}")
        mode, why = recommend(pooled(per_file))
        recs[config] = mode
        print(f"  suggests --doc_mode {mode}: {why}")

    built = [recs[c] for c in DOMAINS.values() if c in recs]
    if built:
        counts = Counter(built)
        choice = counts.most_common(1)[0][0]
        print(f"\nRECOMMENDATION: --doc_mode {choice}  (per config: {dict(counts)})")
        if len(counts) > 1:
            print("WARNING: the eight configs disagree; look at the per-config statistics and samples above.")
        if choice == "block":
            print("WARNING: block mode groups --block_lines consecutive lines per unit; whole-article integrity "
                  "is not guaranteed, so near-duplicate text can straddle splits.")
    else:
        choice = None

    mode = overlap_mode or choice or "line"
    a, b = EXTRA_CONFIG, DOMAINS["philosophy"]
    if os.path.isdir(os.path.join(raw_dir, a)) and os.path.isdir(os.path.join(raw_dir, b)):
        ha, hb = doc_hashes(raw_dir, a, mode, block_lines), doc_hashes(raw_dir, b, mode, block_lines)
        both = len(ha & hb)
        print(f"\nOVERLAP ({mode} mode): {both:,} of {len(ha):,} normalized {a} documents "
              f"({both / max(1, len(ha)):.1%}) also appear in {b} ({len(hb):,} documents).")
    else:
        print(f"\nOVERLAP: {a} or {b} not downloaded; skipped.")
    return recs


# -----------------------------------------------------------------------------
# build

def scan_domain(raw_dir, config, mode, block_lines):
    """Pass 1: per normalized hash, the lowest original-text hash, how many copies share it, and the total."""
    best = {}  # norm digest -> [min original digest, copies with that original, total copies]
    n_docs = 0
    for text in iter_config_documents(raw_dir, config, mode, block_lines):
        h, o = doc_digest(text)
        n_docs += 1
        rec = best.get(h)
        if rec is None:
            best[h] = [o, 1, 1]
        else:
            rec[2] += 1
            if o < rec[0]:
                rec[0], rec[1] = o, 1
            elif o == rec[0]:
                rec[1] += 1
    exact = sum(r[1] - 1 for r in best.values())
    ws = sum(r[2] - r[1] for r in best.values())
    return {h: r[0] for h, r in best.items()}, dict(documents=n_docs, unique=len(best),
                                                     exact_duplicates_dropped=exact,
                                                     whitespace_only_duplicates_dropped=ws)


def collect_texts(raw_dir, config, mode, block_lines, keep):
    """Pass 2: the chosen original text for every kept normalized hash."""
    texts = {}
    for text in iter_config_documents(raw_dir, config, mode, block_lines):
        h, o = doc_digest(text)
        if h in keep and h not in texts and keep[h] == o:
            texts[h] = text
    return texts


def fill_domain(domain, texts, enc, quotas, threads):
    """Walk documents in content order and fill test -> val -> calib -> train with whole documents."""
    order = sorted(((order_key(h.hex()), h.hex(), h) for h in texts))
    eot = enc.eot_token
    assigned = {s: [] for s in SPLITS}
    totals = dict.fromkeys(SPLITS, 0)
    skipped = dict(short=0, long=0)
    si = 0
    for start in range(0, len(order), TOKENIZE_CHUNK):
        chunk = order[start:start + TOKENIZE_CHUNK]
        toks = enc.encode_ordinary_batch([texts[h] for _, _, h in chunk], num_threads=threads)
        for (_, hx, _), t in zip(chunk, toks):
            n = len(t) + 1
            if n < MIN_TOKENS:
                skipped["short"] += 1
                continue
            if n > MAX_TOKENS:
                skipped["long"] += 1
                continue
            split = SPLITS[si]
            assigned[split].append((hx, np.asarray(t + [eot], dtype=np.uint16)))
            totals[split] += n
            if totals[split] >= quotas[split]:
                si += 1
                if si == len(SPLITS):
                    return assigned, totals, skipped
    raise BuildError(f"domain {domain} ran out of documents while filling the {SPLITS[si]} split: "
                     f"{totals[SPLITS[si]]:,} of {quotas[SPLITS[si]]:,} tokens "
                     f"(totals so far {totals}, {len(order):,} documents after dedup)")


def build(raw_dir, out_dir, doc_mode, block_lines=40, threads=1, quotas=None, enc=None, overwrite=False):
    """Build M2D2-Wiki-8 into out_dir and return meta. enc defaults to tiktoken's GPT-2 encoding."""
    if doc_mode not in DOC_MODES:
        raise BuildError(f"unknown doc_mode {doc_mode!r}")
    quotas = {**DEFAULT_QUOTAS, **(quotas or {})}
    if enc is None:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    if enc.n_vocab > 65536:
        raise BuildError("token ids must fit in uint16")
    if os.path.exists(os.path.join(out_dir, "meta.json")) and not overwrite:
        raise BuildError(f"{out_dir} already holds a build (meta.json); pass --overwrite to replace it")
    os.makedirs(out_dir, exist_ok=True)
    if doc_mode == "block":
        log(f"WARNING: block mode ({block_lines} lines per unit) does not guarantee whole-article integrity")

    # pass 1: hashes and within-domain duplicates
    keep, dedup = {}, {}
    for domain, config in sorted(DOMAINS.items()):
        log(f"{domain}: scanning {config}")
        keep[domain], dedup[domain] = scan_domain(raw_dir, config, doc_mode, block_lines)

    # cross-domain duplicates are ambiguous: drop them everywhere
    owners = defaultdict(list)
    for domain in sorted(keep):
        for h in keep[domain]:
            owners[h].append(domain)
    pairs = Counter()
    ambiguous = set()
    for h, ds in owners.items():
        if len(ds) > 1:
            ambiguous.add(h)
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    pairs[f"{ds[i]}|{ds[j]}"] += 1
    del owners
    for domain in keep:
        before = len(keep[domain])
        keep[domain] = {h: o for h, o in keep[domain].items() if h not in ambiguous}
        dedup[domain]["cross_domain_dropped"] = before - len(keep[domain])
        dedup[domain]["kept"] = len(keep[domain])

    # pass 2: order by content and fill the splits
    per_domain, train_docs, manifest = {}, [], []
    for domain, config in sorted(DOMAINS.items()):
        texts = collect_texts(raw_dir, config, doc_mode, block_lines, keep[domain])
        assigned, totals, skipped = fill_domain(domain, texts, enc, quotas, threads)
        del texts
        per_domain[domain] = dict(totals=totals, docs={s: len(assigned[s]) for s in SPLITS}, skipped=skipped)
        for split in SPLITS:
            for hx, arr in assigned[split]:
                manifest.append(dict(doc_hash=hx, domain=domain, split=split, n_tokens=int(len(arr))))
            if split == "train":
                train_docs += [arr for _, arr in assigned[split]]
            else:
                np.concatenate([a for _, a in assigned[split]]).tofile(os.path.join(out_dir, f"{split}_{domain}.bin"))
        log(f"{domain}: " + ", ".join(f"{s} {totals[s]:,} tokens / {len(assigned[s]):,} docs" for s in SPLITS)
            + f"; skipped {skipped['short']:,} short, {skipped['long']:,} long")

    perm = np.random.default_rng(TRAIN_SHUFFLE_SEED).permutation(len(train_docs))
    with open(os.path.join(out_dir, "train.bin"), "wb") as f:
        for i in perm:
            train_docs[i].tofile(f)
    del train_docs
    with open(os.path.join(out_dir, "splits_manifest.jsonl"), "w") as f:
        for rec in manifest:
            f.write(json.dumps(rec) + "\n")

    rev_path = os.path.join(raw_dir, "REVISION.txt")
    revision = open(rev_path).read().strip() if os.path.exists(rev_path) else None
    train_total = sum(per_domain[d]["totals"]["train"] for d in per_domain)
    tok = lambda split: {d: per_domain[d]["totals"][split] for d in sorted(per_domain)}  # noqa: E731
    meta = {
        "dataset": "m2d2-wiki-8",
        "source_dataset": REPO_ID,
        "dataset_revision": revision,
        "archives": [f"data/{c}.tar.gz" for _, c in sorted(DOMAINS.items())],
        "domains": dict(sorted(DOMAINS.items())),
        "tokenizer": getattr(enc, "name", "gpt2 (tiktoken)"),
        "eot_token": enc.eot_token,
        "dtype": "uint16",
        "doc_mode": doc_mode,
        "block_lines": block_lines,
        "split_salt": SPLIT_SALT,
        "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
        "quotas_per_domain": quotas,
        "doc_length_filter_tokens": [MIN_TOKENS, MAX_TOKENS],
        "train_tokens_total": train_total,
        "train_tokens_by_source": tok("train"),
        "train_share_by_source": {d: round(c / max(train_total, 1), 4) for d, c in tok("train").items()},
        "val_tokens_by_source": tok("val"),
        "calib_tokens_by_source": tok("calib"),
        "test_tokens_by_source": tok("test"),
        "docs_by_split_and_source": {s: {d: per_domain[d]["docs"][s] for d in sorted(per_domain)} for s in SPLITS},
        "dedup": {d: dedup[d] for d in sorted(dedup)},
        "whitespace_only_duplicates_dropped": sum(dedup[d]["whitespace_only_duplicates_dropped"] for d in dedup),
        "cross_domain_duplicates_by_pair": dict(sorted(pairs.items())),
        "skipped_by_length": {d: per_domain[d]["skipped"] for d in sorted(per_domain)},
        "sampling_note": SAMPLING_NOTE,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    write_sanity(out_dir, manifest, per_domain, enc)
    log(f"done: {train_total:,} training tokens in {out_dir}")
    return meta


def write_sanity(out_dir, manifest, per_domain, enc):
    lines = []
    rng = np.random.default_rng(0)
    names = ["train"] + [f"{s}_{d}" for s in ("test", "val", "calib") for d in sorted(per_domain)]
    for name in names:
        path = os.path.join(out_dir, f"{name}.bin")
        m = np.memmap(path, dtype=np.uint16, mode="r")
        if os.path.getsize(path) != 2 * len(m):
            raise BuildError(f"{path}: size is not a whole number of uint16 tokens")
        max_id = int(m.max())
        if max_id >= enc.n_vocab:
            raise BuildError(f"{name}: token id {max_id} out of range")
        start = int(rng.integers(0, max(1, len(m) - 200)))
        sample = enc.decode(m[start:start + 200].tolist())
        lines.append(f"===== {name}: {os.path.getsize(path):,} bytes, {len(m):,} tokens, max id {max_id} =====\n"
                     f"{sample}\n")
    hashes = [r["doc_hash"] for r in manifest]
    if len(hashes) != len(set(hashes)):
        raise BuildError("a doc_hash appears in two splits or two domains")
    lines.append(f"OK: none of the {len(hashes):,} doc_hashes appears in two splits or two domains.\n")
    lines.append(f"{'domain':<18}" + "".join(f"{s:>14}" for s in SPLITS))
    for d in sorted(per_domain):
        lines.append(f"{d:<18}" + "".join(f"{per_domain[d]['totals'][s]:>14,}" for s in SPLITS))
    with open(os.path.join(out_dir, "sanity.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------

def parse_quotas(text):
    out = {}
    for part in filter(None, (text or "").split(",")):
        k, v = part.split("=")
        if k not in SPLITS:
            raise SystemExit(f"--quotas: unknown split {k!r}")
        out[k] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--build", action="store_true")
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir")
    ap.add_argument("--doc_mode", choices=DOC_MODES)
    ap.add_argument("--block_lines", type=int, default=40)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--quotas", help="override per-domain token quotas, e.g. test=1000,train=5000 (testing only)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing build in --out_dir")
    args = ap.parse_args()
    try:
        if args.download:
            download(args.raw_dir)
        elif args.inspect:
            inspect(args.raw_dir, overlap_mode=args.doc_mode, block_lines=args.block_lines)
        else:
            if not args.out_dir or not args.doc_mode:
                ap.error("--build needs --out_dir and --doc_mode")
            build(args.raw_dir, args.out_dir, args.doc_mode, args.block_lines, args.threads,
                  parse_quotas(args.quotas), overwrite=args.overwrite)
    except BuildError as e:
        sys.exit(f"prepare.py: {e}")


if __name__ == "__main__":
    main()
