#!/usr/bin/env python
"""Capture data/m2d2/prepare.py's default build output into tests/fixtures/m2d2_build_reference.json.

Captured from the code before the uneven-design options (--train_quota_override, --fallback_domain,
--variant, --reference_dir, title mode) were added (commit b5d5717). test_m2d2_override_default_unchanged
rebuilds with capture() and requires identical results: the SHA-256 of every output file and meta.json
(without its "created" timestamp), for a line-mode and a blank-mode build of the fake archives.
Do NOT rerun this script on later code.

Usage (repo root): .venv/bin/python tests/make_build_fixture.py
"""
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_m2d2_prep as T  # noqa: E402  (fake archives, byte tokenizer, small quotas)

FIXTURE = os.path.join(HERE, "fixtures", "m2d2_build_reference.json")


def build_outputs(out_dir):
    res = {}
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name)
        if name == "meta.json":
            meta = json.load(open(path))
            meta.pop("created")
            res[name] = meta
        else:
            res[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return res


def capture(workdir):
    out = {}
    raw = T.make_raw(os.path.join(workdir, "line"))
    T.build(raw, os.path.join(workdir, "out_line"), "line")
    out["line"] = build_outputs(os.path.join(workdir, "out_line"))
    raw = T.make_raw(os.path.join(workdir, "blank"), sep="\n\n")  # documents separated by blank lines
    T.build(raw, os.path.join(workdir, "out_blank"), "blank")
    out["blank"] = build_outputs(os.path.join(workdir, "out_blank"))
    return out


def main():
    if os.path.exists(FIXTURE) and "--force" not in sys.argv:
        sys.exit(f"{FIXTURE} exists; it must stay the pre-uneven reference")
    with tempfile.TemporaryDirectory() as d:
        ref = capture(d)
    with open(FIXTURE, "w") as f:
        json.dump(ref, f, indent=1)
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
