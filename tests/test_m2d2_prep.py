"""Tests for data/m2d2/prepare.py on small fake archives (docs/M2D2_PROMPT.md part 9)."""
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from conftest import REPO

sys.path.insert(0, os.path.join(REPO, "data", "m2d2"))
import prepare as P  # noqa: E402

SMALL_QUOTAS = {"test": 1500, "val": 1500, "calib": 1000, "train": 4000}


class ByteEncoder:
    """Stand-in for tiktoken's GPT-2 encoding (no network in tests): one token per UTF-8 byte."""
    name = "bytes (test)"
    eot_token = 256
    n_vocab = 257

    def encode_ordinary_batch(self, texts, num_threads=1):
        return [list(t.encode("utf-8")) for t in texts]

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


ENC = ByteEncoder()


def fake_docs(domain, n, seed=0, min_words=20, max_words=45):
    rng = random.Random(f"{domain}-{seed}")
    words = [f"{domain[:4]}{i}" for i in range(50)] + ["the", "of", "and", "a", "in"]
    return [f"Doc {domain} {k}: " + " ".join(rng.choice(words) for _ in range(rng.randint(min_words, max_words))) + "."
            for k in range(n)]


def write_archive(tmp, config, files):
    """A .tar.gz with the M2D2 layout <config>/{train,valid,test}.txt."""
    path = os.path.join(tmp, f"{config}.tar.gz")
    with tarfile.open(path, "w:gz") as tf:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(f"{config}/{name}")
            info.size = len(data)
            info.mtime = 0
            tf.addfile(info, io.BytesIO(data))
    return path


def make_raw(tmp, docs_by_domain=None, n_docs=80, sep="\n", extra_lines=None):
    """Fake archives for the 8 domains (+ Philosophy), extracted with prepare.extract_archive. Line mode layout:
    one document per line, split over train/valid/test."""
    arch, raw = os.path.join(tmp, "archives"), os.path.join(tmp, "raw")
    os.makedirs(arch, exist_ok=True)
    os.makedirs(raw, exist_ok=True)
    docs_by_domain = docs_by_domain or {}
    for domain, config in list(P.DOMAINS.items()) + [("philosophy_extra", P.EXTRA_CONFIG)]:
        docs = docs_by_domain.get(domain) or fake_docs(domain, n_docs)
        docs = docs + (extra_lines or {}).get(domain, [])
        third = len(docs) // 3
        parts = [docs[:third], docs[third:2 * third], docs[2 * third:]]
        files = {name: sep.join(p) + "\n" for name, p in zip(P.FILES, parts)}
        P.extract_archive(write_archive(arch, config, files), raw)
    return raw


def build(raw, out, mode="line", **kw):
    kw.setdefault("quotas", SMALL_QUOTAS)
    return P.build(raw, out, mode, enc=ENC, **kw)


def read_manifest(out):
    return [json.loads(l) for l in open(os.path.join(out, "splits_manifest.jsonl"))]


def split_docs(path, eot=256):
    """Documents (as token arrays, each ending with eot) in a split file."""
    toks = np.fromfile(path, dtype=np.uint16)
    ends = np.nonzero(toks == eot)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    assert len(ends) == 0 or ends[-1] == len(toks) - 1
    return [toks[s:e + 1] for s, e in zip(starts, ends)]


def norm_hash(text):
    return hashlib.sha256(P.normalize(text).encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------

def test_m2d2_doc_modes(tmp_path):
    lines = ["= Alpha =", "", "Alpha body one.", "Alpha body two.", "", "= = Section = =", "More alpha.",
             "= Beta =", "Beta body.", "", ""]
    assert list(P.documents_from_lines(["  first article  ", "", "second article"], "line")) == \
        ["first article", "second article"]
    assert list(P.documents_from_lines(lines, "blank")) == [
        "= Alpha =", "Alpha body one.\nAlpha body two.", "= = Section = =\nMore alpha.\n= Beta =\nBeta body."]
    assert list(P.documents_from_lines(lines, "heading")) == [
        "= Alpha =\n\nAlpha body one.\nAlpha body two.\n\n= = Section = =\nMore alpha.", "= Beta =\nBeta body."]
    assert list(P.documents_from_lines([f"l{i}" for i in range(7)], "block", block_lines=3)) == [
        "l0\nl1\nl2", "l3\nl4\nl5", "l6"]
    assert P.is_top_heading("= Alpha =") and not P.is_top_heading("= = Section = =")
    assert P.is_any_heading("= = Section = =") and not P.is_any_heading("a = b = c")

    # through an archive: a document never spans train/valid/test
    raw = tmp_path / "raw"
    P.extract_archive(write_archive(str(tmp_path), "Cfg", {"train.txt": "a1\na2\n\nb1\n", "valid.txt": "c1\n",
                                                           "test.txt": "\nd1\nd2"}), str(raw))
    assert list(P.iter_config_documents(str(raw), "Cfg", "blank", 40)) == ["a1\na2", "b1", "c1", "d1\nd2"]

    # unsafe archive members are refused
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tf:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(P.BuildError, match="unsafe"):
        P.extract_archive(str(bad), str(raw))


def test_m2d2_inspect_recommends(tmp_path, capsys):
    long_docs = {d: [doc * 4 for doc in fake_docs(d, 30)]  # long lines, no headings
                 for d in list(P.DOMAINS) + ["philosophy_extra"]}
    raw = make_raw(str(tmp_path), long_docs)
    recs = P.inspect(raw)
    out = capsys.readouterr().out
    assert set(recs.values()) == {"line"} and "RECOMMENDATION: --doc_mode line" in out
    assert "first 8 lines of train.txt" in out and "p5/p50/p95/p99" in out
    assert "OVERLAP (line mode): 0 of" in out  # Philosophy docs differ from Philosophy_and_thinking here
    blank = {"empty_frac": 0.1, "top_heading_frac": 0, "any_heading_frac": 0, "short_then_long_frac": 0,
             "pct": {5: 10, 50: 80, 95: 300, 99: 400}}
    assert P.recommend(blank)[0] == "blank"
    assert P.recommend({**blank, "empty_frac": 0, "top_heading_frac": 0.01})[0] == "heading"
    assert P.recommend({**blank, "empty_frac": 0})[0] == "block"


def test_m2d2_deterministic(tmp_path):
    raw = make_raw(str(tmp_path))
    a, b = tmp_path / "a", tmp_path / "b"
    build(raw, str(a))
    build(raw, str(b))
    names = sorted(os.listdir(a))
    assert names == sorted(os.listdir(b)) and "train.bin" in names and "splits_manifest.jsonl" in names
    for name in names:
        if name == "meta.json":
            ma, mb = json.load(open(a / name)), json.load(open(b / name))
            ma.pop("created"), mb.pop("created")
            assert ma == mb
        else:
            assert (a / name).read_bytes() == (b / name).read_bytes(), name

    # shuffled line order inside the input files: the same split assignment per doc_hash
    rng = random.Random(1)
    shuffled = {}
    for d in P.DOMAINS:
        docs = fake_docs(d, 80)
        rng.shuffle(docs)
        shuffled[d] = docs
    raw2 = make_raw(str(tmp_path / "shuffled"), shuffled)
    c = tmp_path / "c"
    build(raw2, str(c))
    assign = lambda out: {r["doc_hash"]: (r["domain"], r["split"]) for r in read_manifest(out)}  # noqa: E731
    assert assign(a) == assign(c)
    assert (a / "train.bin").read_bytes() == (c / "train.bin").read_bytes()


def test_m2d2_no_cross_split_or_domain(tmp_path):
    shared = "This exact article text appears in both the history and the culture archives, so it is ambiguous."
    exact_dup = "A history article that appears twice, exactly the same both times, inside one domain only here."
    ws_variant = "A  history article that appears twice,\tonce with extra whitespace, inside one domain only here."
    ws_original = "A history article that appears twice, once with extra whitespace, inside one domain only here."
    extra = {"history": [shared, exact_dup, exact_dup, ws_variant, ws_original], "culture": [shared]}
    raw = make_raw(str(tmp_path), extra_lines=extra)
    out = tmp_path / "out"
    meta = build(raw, str(out))
    man = read_manifest(out)
    hashes = [r["doc_hash"] for r in man]
    assert len(hashes) == len(set(hashes))  # no doc in two splits or two domains
    assert norm_hash(shared) not in hashes
    assert meta["cross_domain_duplicates_by_pair"] == {"culture|history": 1}
    assert meta["dedup"]["history"]["cross_domain_dropped"] == 1
    assert meta["dedup"]["culture"]["cross_domain_dropped"] == 1
    assert meta["dedup"]["history"]["exact_duplicates_dropped"] == 1
    assert meta["dedup"]["history"]["whitespace_only_duplicates_dropped"] == 1
    assert meta["whitespace_only_duplicates_dropped"] == 1

    # the whitespace variants share one normalized hash; the kept copy is the lowest original SHA-256
    keep, _ = P.scan_domain(raw, P.DOMAINS["history"], "line", 40)
    h = hashlib.sha256(P.normalize(ws_original).encode()).digest()
    assert keep[h] == min(hashlib.sha256(t.encode()).digest() for t in (ws_variant, ws_original))
    assert "sanity.txt" in os.listdir(out)
    assert "OK: none of the" in (out / "sanity.txt").read_text()


def test_m2d2_whole_documents(tmp_path):
    raw = make_raw(str(tmp_path))
    out = tmp_path / "out"
    build(raw, str(out))
    man = read_manifest(out)
    by_split = {}
    for r in man:
        by_split.setdefault((r["domain"], r["split"]), []).append(r)
    texts = {norm_hash(t): t for d in P.DOMAINS for t in fake_docs(d, 80)}
    for (domain, split), recs in by_split.items():
        if split == "train":
            continue
        docs = split_docs(out / f"{split}_{domain}.bin")
        # the file is exactly the manifest's documents, whole and in order
        assert [norm_hash(ENC.decode(d.tolist())) for d in docs] == [r["doc_hash"] for r in recs]
        assert [len(d) for d in docs] == [r["n_tokens"] for r in recs]
        for d, r in zip(docs, recs):
            assert d[:-1].tolist() == list(texts[r["doc_hash"]].encode("utf-8"))  # never truncated
    train = split_docs(out / "train.bin")
    train_hashes = sorted(norm_hash(ENC.decode(d.tolist())) for d in train)
    assert train_hashes == sorted(r["doc_hash"] for r in man if r["split"] == "train")


def walked_first(template, good_keys):
    """A text built from template whose content order puts it before every good document."""
    i = 0
    while P.order_key(norm_hash(template.format(i))) >= min(good_keys):
        i += 1
    return template.format(i)


def test_m2d2_quotas(tmp_path):
    raw = make_raw(str(tmp_path))
    out = tmp_path / "out"
    meta = build(raw, str(out))
    man = read_manifest(out)
    for domain in P.DOMAINS:
        for split, quota in SMALL_QUOTAS.items():
            recs = [r for r in man if r["domain"] == domain and r["split"] == split]
            total = sum(r["n_tokens"] for r in recs)
            assert total >= quota, (domain, split)
            assert total - quota < recs[-1]["n_tokens"], (domain, split)  # overshoot under one document
            assert meta[f"{split}_tokens_by_source"][domain] == total
    assert set(meta["skipped_by_length"]) == set(P.DOMAINS)
    assert sum(meta["train_share_by_source"].values()) == pytest.approx(1.0, abs=1e-3)
    assert meta["dataset"] == "m2d2-wiki-8" and meta["split_salt"] == "m2d2w8-v1:"
    assert "63%" in meta["sampling_note"]

    # length filters: a stub (< 64 tokens) and a huge document (> 50,000) placed first in content order
    good = fake_docs("health", 80)
    keys = [P.order_key(norm_hash(t)) for t in good]
    short, long_ = walked_first("stub {}", keys), walked_first("y" * 60_000 + " {}", keys)
    texts = {bytes.fromhex(norm_hash(t)): t for t in good + [short, long_]}
    assigned, totals, skipped = P.fill_domain("health", texts, ENC, SMALL_QUOTAS, 1)
    assert skipped == {"short": 1, "long": 1}
    kept = {h for split in assigned.values() for h, _ in split}
    assert norm_hash(short) not in kept and norm_hash(long_) not in kept
    # and the build records them per domain
    raw2 = make_raw(str(tmp_path / "filters"), docs_by_domain={"health": good + [short, long_]})
    meta2 = build(raw2, str(tmp_path / "out2"))
    assert meta2["skipped_by_length"]["health"] == {"short": 1, "long": 1}

    # a domain that runs short stops the build with an error naming it
    raw3 = make_raw(str(tmp_path / "short"), docs_by_domain={"mathematics": fake_docs("mathematics", 20)})
    with pytest.raises(P.BuildError, match="domain mathematics ran out of documents"):
        build(raw3, str(tmp_path / "out3"))

    # an existing build is never overwritten by accident
    with pytest.raises(P.BuildError, match="already holds a build"):
        build(raw, str(out))


def test_m2d2_meta_compatible(tmp_path):
    import train as T
    from test_train import tiny_cfg, train_records

    raw = make_raw(str(tmp_path))
    data = tmp_path / "data"
    build(raw, str(data))
    meta = json.load(open(data / "meta.json"))
    for key in ("train_tokens_total", "train_share_by_source", "val_tokens_by_source", "eot_token", "dtype",
                "created", "calib_tokens_by_source", "test_tokens_by_source"):
        assert key in meta, key
    finals = {}
    for split in ("val", "calib"):
        name = f"m2d2-tiny-{split}"
        cfg = tiny_cfg(str(data), tmp_path / "runs", name, steps=5, eval_split=split)
        assert T.train(cfg) == 0
        log = [json.loads(l) for l in open(tmp_path / "runs" / name / "log.jsonl")]
        finals[split] = [r for r in log if r["type"] == "final"][-1]
        assert set(finals[split]["val"]) == set(P.DOMAINS)
        assert log[0]["config"]["eval_split"] == split
        assert len(train_records(tmp_path / "runs" / name)) == 5
    # same model and training, different evaluation files
    assert finals["val"]["val_overall"] != finals["calib"]["val_overall"]


def test_build_sbatch_syntax():
    path = os.path.join(REPO, "data", "m2d2", "build_m2d2.sbatch")
    text = open(path).read()
    assert text.startswith("#!/bin/bash") and "set -uo pipefail" in text
    for opt in ("#SBATCH -p mit_normal", "#SBATCH -c 16", "#SBATCH --mem=64G", "#SBATCH -t 02:00:00"):
        assert opt in text
    assert subprocess.run(["bash", "-n", path], capture_output=True).returncode == 0
