"""Run configs match the PROMPT.md part 5 table (extra checks beyond part 6)."""
import os
import shutil
import sys

import pytest

import train as T
from conftest import REPO

sys.path.insert(0, os.path.join(REPO, "scripts"))
import make_configs  # noqa: E402

M = 1_000_000
B = [0] * 8
W16 = [0, 0, 0, 0, 16, 16, 16, 16]

# name -> (window_layers, seed, lr, total_tokens); lr is a float or "LR_BASE"/"LR_WIN"
EXPECTED = {
    **{f"lr-base-{s}": (B, 0, v, 100 * M) for s, v in (("1e-3", 1e-3), ("2e-3", 2e-3), ("4e-3", 4e-3))},
    **{f"lr-win16-{s}": (W16, 0, v, 100 * M) for s, v in (("1e-3", 1e-3), ("2e-3", 2e-3), ("4e-3", 4e-3))},
    **{f"base-s{s}": (B, s, "LR_BASE", 500 * M) for s in range(5)},
    **{f"win16-s{s}": (W16, s, "LR_WIN", 500 * M) for s in range(5)},
    "win1-s0": ([0, 0, 0, 0, 1, 1, 1, 1], 0, "LR_BASE", 500 * M),  # same LR as base-s0 (changed from spec)
    "win64-s0": ([0, 0, 0, 0, 64, 64, 64, 64], 0, "LR_WIN", 500 * M),
    "smoke-gpu-base": (B, 0, 2e-3, 200 * 65536),
    "smoke-gpu-win16": (W16, 0, 2e-3, 200 * 65536),
}


@pytest.fixture
def config_tree(tmp_path, monkeypatch):
    """A copy of config/ with chosen LRs filled in, used as the working directory."""
    shutil.copytree(os.path.join(REPO, "config"), tmp_path / "config")
    (tmp_path / "config" / "chosen_lr.py").write_text("LR_BASE = 0.0011\nLR_WIN = 0.0022\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_generated_configs_are_current(tmp_path):
    out = tmp_path / "runs"
    for name, spec in make_configs.runs().items():
        out.mkdir(exist_ok=True)
        (out / f"{name}.py").write_text(make_configs.render(name, spec))
        committed = open(os.path.join(REPO, "config", "runs", f"{name}.py")).read()
        assert committed == (out / f"{name}.py").read_text(), f"config/runs/{name}.py is stale"
    assert set(make_configs.runs()) == set(EXPECTED)


def test_run_table(config_tree):
    lrs = {"LR_BASE": 0.0011, "LR_WIN": 0.0022}
    for name, (windows, seed, lr, tokens) in EXPECTED.items():
        cfg = T.load_config([f"config/runs/{name}.py"])
        assert cfg["run_name"] == name
        assert cfg["window_layers"] == windows, name
        assert cfg["seed"] == seed, name
        assert cfg["lr"] == lrs.get(lr, lr), name
        assert cfg["total_tokens"] == tokens, name
        assert cfg["micro_batch_size"] == T.default_config()["micro_batch_size"], name
    _, steps500, _ = T.derived(T.load_config(["config/runs/base-s0.py"]))
    _, steps100, _ = T.derived(T.load_config(["config/runs/lr-base-1e-3.py"]))
    assert (steps500, steps100) == (7630, 1526)


def test_main_runs_stop_while_lr_unset(monkeypatch):
    monkeypatch.chdir(REPO)  # the committed chosen_lr.py has LR_BASE = LR_WIN = None
    for name in ("base-s0", "win16-s0"):
        with pytest.raises(SystemExit, match="still None"):
            T.load_config([f"config/runs/{name}.py"])


def test_cli_override_and_unknown_key(config_tree):
    cfg = T.load_config(["config/runs/smoke-gpu-base.py", "--benchmark_steps=30", "--data_dir=/x"])
    assert cfg["benchmark_steps"] == 30 and cfg["data_dir"] == "/x"
    with pytest.raises(ValueError, match="Unknown config key"):
        T.load_config(["config/runs/smoke-gpu-base.py", "--not_a_key=1"])
