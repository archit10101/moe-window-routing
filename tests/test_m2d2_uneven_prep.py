"""Uneven-domain builds and title mode (docs/M2D2_UNEVEN_PROMPT.md parts 3, 4 and 9)."""
import json
import os

import pytest

import make_build_fixture as BF
import test_m2d2_prep as TP
from test_m2d2_prep import P, build, fake_docs, make_raw, read_manifest

EVAL_PREFIXES = ("test_", "val_", "calib_")


def eval_files(out):
    return sorted(n for n in os.listdir(out) if n.startswith(EVAL_PREFIXES))


def train_docs_by_domain(out):
    docs = {}
    for r in read_manifest(out):
        if r["split"] == "train":
            docs.setdefault(r["domain"], []).append(r["doc_hash"])
    return docs


def title_layout(domain, n):
    """Articles as M2D2 stores them: blank line, title, blank line, then (multi-line) body."""
    return [f"Title {domain} {k}\n\n{body}\n{body[::-1]}" for k, body in enumerate(fake_docs(domain, n))]


# -----------------------------------------------------------------------------

def test_m2d2_title_mode(tmp_path, capsys):
    lines = ["", "Alpha", "", "Alpha body one.", "Alpha body two.", "", "Beta", "", "Beta body.", "more beta", ""]
    assert list(P.documents_from_lines(lines, "title")) == [
        "Alpha\n\nAlpha body one.\nAlpha body two.", "Beta\n\nBeta body.\nmore beta"]
    # a title on the very first line (the start of the file counts as a blank line)
    assert list(P.documents_from_lines(["First", "", "body a", "body b"], "title")) == ["First\n\nbody a\nbody b"]
    # a file with no blank lines is one document
    assert list(P.documents_from_lines(["one", "two", "three"], "title")) == ["one\ntwo\nthree"]
    # as specified, any non-blank line between blank lines starts a document (one-line paragraphs too)
    assert list(P.documents_from_lines(["T", "", "body", "body", "", "lone line", "", "x", "y"], "title")) == [
        "T\n\nbody\nbody", "lone line\n\nx\ny"]

    # through archives, with inspect: title layout is recognized and its documents are reported
    raw = make_raw(str(tmp_path), docs_by_domain={d: title_layout(d, 40) for d in list(P.DOMAINS) + ["philosophy_extra"]},
                   sep="\n\n")
    docs = list(P.iter_config_documents(raw, P.DOMAINS["history"], "title", 40))
    assert len(docs) == 40 and all(d.startswith("Title history ") for d in docs)
    recs = P.inspect(raw)
    out = capsys.readouterr().out
    assert set(recs.values()) == {"title"} and "RECOMMENDATION: --doc_mode title" in out
    assert "title mode: " in out and "documents; estimated tokens (chars / 4) p5/p50/p95" in out


def test_m2d2_override_default_unchanged(tmp_path):
    """With no new options, the build output is byte-identical to the pre-change reference."""
    ref = json.load(open(BF.FIXTURE))
    assert BF.capture(str(tmp_path)) == ref


@pytest.fixture
def even_and_hist(tmp_path):
    raw = make_raw(str(tmp_path))
    even, hist = tmp_path / "even", tmp_path / "hist"
    build(raw, str(even))
    meta = build(raw, str(hist), train_quota_override=("history", 8000), variant="hist4x", reference_dir=str(even))
    return raw, even, hist, meta


def test_m2d2_override_eval_identical(even_and_hist):
    raw, even, hist, meta = even_and_hist
    assert eval_files(even) == eval_files(hist) and len(eval_files(hist)) == 24
    for name in eval_files(even):
        assert (even / name).read_bytes() == (hist / name).read_bytes(), name
    # meta.json records the hashes, and sanity.txt's check against the even build passed
    import hashlib
    assert meta["eval_files_sha256"] == {n: hashlib.sha256((even / n).read_bytes()).hexdigest() for n in eval_files(even)}
    assert P.REFERENCE_OK in (hist / "sanity.txt").read_text()
    assert meta["variant"] == "hist4x" and meta["boosted_domain"] == "history"
    assert meta["requested_boosted_domain"] == "history" and meta["fallback_used"] is False
    assert meta["train_quota_by_source"]["history"] == 8000
    assert all(v == TP.SMALL_QUOTAS["train"] for d, v in meta["train_quota_by_source"].items() if d != "history")
    # the even build has none of the new keys
    even_meta = json.load(open(even / "meta.json"))
    assert not {"variant", "boosted_domain", "eval_files_sha256"} & set(even_meta)


def test_m2d2_override_train_prefix(even_and_hist):
    raw, even, hist, meta = even_and_hist
    te, th = train_docs_by_domain(even), train_docs_by_domain(hist)
    for domain in P.DOMAINS:
        if domain == "history":
            # the even build's train documents, in the same hash order, followed by more
            assert th[domain][:len(te[domain])] == te[domain] and len(th[domain]) > len(te[domain])
            assert meta["train_tokens_by_source"]["history"] >= 8000
        else:
            assert th[domain] == te[domain], domain
    # train.bin holds exactly the manifest's train documents (shuffled with the same seed)
    train = TP.split_docs(hist / "train.bin")
    assert sorted(TP.norm_hash(TP.ENC.decode(d.tolist())) for d in train) == sorted(sum(th.values(), []))


def test_m2d2_override_runs_out(tmp_path):
    # natural_sciences has plenty of documents; history (80 documents) cannot supply 30,000 train tokens
    raw = make_raw(str(tmp_path), docs_by_domain={"natural_sciences": fake_docs("natural_sciences", 400)})
    with pytest.raises(P.RunOutError, match=r"domain history ran out .* history can supply only [\d,]+ train tokens"):
        build(raw, str(tmp_path / "nofallback"), train_quota_override=("history", 30_000))
    assert not (tmp_path / "nofallback" / "meta.json").exists()

    even = tmp_path / "even"
    build(raw, str(even))
    meta = build(raw, str(tmp_path / "fb"), train_quota_override=("history", 30_000),
                 fallback_domain="natural_sciences", variant="rep4x", reference_dir=str(even))
    assert meta["fallback_used"] is True and meta["requested_boosted_domain"] == "history"
    assert meta["boosted_domain"] == "natural_sciences"
    assert meta["train_quota_by_source"]["natural_sciences"] == 30_000
    assert meta["train_quota_by_source"]["history"] == TP.SMALL_QUOTAS["train"]
    assert meta["train_tokens_by_source"]["natural_sciences"] >= 30_000
    assert meta["train_tokens_by_source"]["history"] == json.load(open(even / "meta.json"))["train_tokens_by_source"]["history"]
    assert P.REFERENCE_OK in (tmp_path / "fb" / "sanity.txt").read_text()
    # identical to building with the override on the fallback domain directly
    direct = build(raw, str(tmp_path / "direct"), train_quota_override=("natural_sciences", 30_000))
    for name in ["train.bin", "splits_manifest.jsonl"] + eval_files(tmp_path / "direct"):
        assert (tmp_path / "fb" / name).read_bytes() == (tmp_path / "direct" / name).read_bytes(), name
    assert direct["fallback_used"] is False

    # the fallback running out too is an error naming it
    with pytest.raises(P.RunOutError, match="domain natural_sciences ran out"):
        build(raw, str(tmp_path / "both"), train_quota_override=("history", 10**6), fallback_domain="natural_sciences")


def test_m2d2_override_reference_mismatch_fails(tmp_path):
    raw = make_raw(str(tmp_path / "a"))
    other = make_raw(str(tmp_path / "b"), docs_by_domain={"culture": fake_docs("culture", 80, seed=9)})
    build(other, str(tmp_path / "other_even"))
    with pytest.raises(P.BuildError, match="FAILED: eval files differ from the reference build .*culture"):
        build(raw, str(tmp_path / "u"), train_quota_override=("history", 8000), reference_dir=str(tmp_path / "other_even"))
    assert "FAILED: eval files differ" in (tmp_path / "u" / "sanity.txt").read_text()
    # no reference build: the check is skipped and says so
    build(raw, str(tmp_path / "u2"), train_quota_override=("history", 8000), reference_dir=str(tmp_path / "missing"))
    assert "SKIPPED: no reference build" in (tmp_path / "u2" / "sanity.txt").read_text()
