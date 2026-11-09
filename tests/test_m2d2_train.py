"""eval_split (docs/M2D2_PROMPT.md part 5): default unchanged, calib allowed, the locked test split refused."""
import json
import subprocess
import sys

import pytest

import make_defaults_fixture as fixture
import train as T
from conftest import REPO
from test_train import tiny_cfg


def test_eval_split_default_and_test_refused(synth_dir, tmp_path):
    assert T.default_config()["eval_split"] == "val"

    # the default reproduces the pre-change reference bit for bit (init, training, final eval, weights)
    ref = json.load(open(fixture.FIXTURE))
    got = fixture.capture(str(tmp_path / "fixture"))
    for arm in fixture.ARMS:
        assert got["train"][arm] == ref["train"][arm], arm
    # and the default reads val_<source>.bin exactly as before
    cfg = tiny_cfg(synth_dir, tmp_path, "default", steps=2)
    sets = T.load_val_sets(synth_dir, cfg)
    assert [d.filename.split("/")[-1] for d, _, _ in sets.values()] == ["val_book.bin", "val_c4.bin",
                                                                        "val_github.bin"]

    # "test" is refused by training (config, CLI) and by load_val_sets
    with pytest.raises(SystemExit, match="locked test split"):
        T.train(tiny_cfg(synth_dir, tmp_path, "t1", steps=2, eval_split="test"))
    with pytest.raises(SystemExit, match="locked test split"):
        T.load_val_sets(synth_dir, cfg, split="test")
    with pytest.raises(SystemExit, match="must be one of"):
        T.train(tiny_cfg(synth_dir, tmp_path, "t2", steps=2, eval_split="train"))
    proc = subprocess.run([sys.executable, "train.py", "config/runs/smoke-cpu.py", f"--data_dir={synth_dir}",
                           f"--out_root={tmp_path / 'cli'}", "--eval_split=test"],
                          cwd=REPO, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "locked test split" in proc.stderr
    assert not (tmp_path / "cli" / "smoke-cpu" / "log.jsonl").exists()

    # only the locked-test path may read it, explicitly
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "meta.json").write_bytes(open(f"{synth_dir}/meta.json", "rb").read())
    (locked / "test_c4.bin").write_bytes(open(f"{synth_dir}/val_c4.bin", "rb").read())
    sets = T.load_val_sets(str(locked), cfg, split="test", allow_locked_test=True)
    assert list(sets) == ["c4"]
