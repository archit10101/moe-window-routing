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

# FOLLOWUP_PROMPT.md part 6: name -> (window_layers, seed, settings that differ from pilot_base.py).
# Every follow-up run: lr 5e-4, 500M tokens, log_router_grad True.
TOP1 = [0, 0, 0, 0, 0, 0, 0, 16]
TOP2 = [0, 0, 0, 0, 0, 0, 16, 16]
FOLLOWUP = {
    **{f"both16-s{s}": (W16, s, {"router_input": "both"}) for s in (0, 1)},
    **{f"win16top1-s{s}": (TOP1, s, {}) for s in (0, 1)},
    **{f"win16top2-s{s}": (TOP2, s, {}) for s in (0, 1)},
    **{f"base-sh2048-s{s}": (B, s, {"shared_hidden": 2048}) for s in (0, 1)},
    **{f"win16-sh2048-s{s}": (W16, s, {"shared_hidden": 2048}) for s in (0, 1)},
    **{f"base-rlr3-s{s}": (B, s, {"router_lr_mult": 3.0}) for s in (0, 1)},
    **{f"win16-rlr3-s{s}": (W16, s, {"router_lr_mult": 3.0}) for s in (0, 1)},
    "base-rg-s0": (B, 0, {}),
    "win16-rg-s0": (W16, 0, {}),
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
    assert set(make_configs.runs()) == set(EXPECTED) | set(FOLLOWUP) | set(M2D2) | set(SLIM120) | set(UNEVEN)


def test_new_configs(config_tree):
    assert len(FOLLOWUP) == 16
    base = T.default_config()
    for name, (windows, seed, extra) in FOLLOWUP.items():
        cfg = T.load_config([f"config/runs/{name}.py"])
        want = dict(base, run_name=name, window_layers=windows, seed=seed, lr=5e-4, total_tokens=500 * M,
                    log_router_grad=True, **extra)
        # everything else (batch, schedule, balancing, eval, model shape) is the main runs' setting
        diff = {k: (cfg[k], want[k]) for k in want if cfg[k] != want[k]}
        assert not diff, f"{name}: {diff}"
        _, steps, _ = T.derived(cfg)
        assert steps == 7630


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


# docs/M2D2_PROMPT.md part 6: name -> (window_layers, seed, lr, eval_split, total_tokens)
M2D2_DATA = "/home/architta/orcd/scratch/moe-pilot/data_m2d2w8"
SLIM_DATA = "/home/architta/orcd/scratch/moe-pilot/data"
W1 = [0, 0, 0, 0, 1, 1, 1, 1]
W64 = [0, 0, 0, 0, 64, 64, 64, 64]
M2D2 = {
    **{f"m2d2-lr-base-{s}": (B, 0, v, "calib", 30 * M) for s, v in (("2.5e-4", 2.5e-4), ("5e-4", 5e-4), ("1e-3", 1e-3))},
    **{f"m2d2-lr-win16-{s}": (W16, 0, v, "calib", 30 * M) for s, v in (("2.5e-4", 2.5e-4), ("5e-4", 5e-4), ("1e-3", 1e-3))},
    **{f"m2d2-base-s{s}": (B, s, "LR_BASE", "val", 120 * M) for s in range(5)},
    **{f"m2d2-win16-s{s}": (W16, s, "LR_WIN", "val", 120 * M) for s in range(5)},
    "m2d2-win1-s0": (W1, 0, "LR_BASE", "val", 120 * M),
    "m2d2-win64-s0": (W64, 0, "LR_WIN", "val", 120 * M),
}
SLIM120 = {**{f"slim120-base-s{s}": (B, s, 5e-4, "val", 120 * M) for s in range(3)},
           **{f"slim120-win16-s{s}": (W16, s, 5e-4, "val", 120 * M) for s in range(3)}}


def test_m2d2_and_slim120_configs(config_tree):
    assert len(M2D2) + len(SLIM120) == 24
    # different numbers in the two LR files: M2D2 configs must read only chosen_lr_m2d2.py
    (config_tree / "config" / "chosen_lr_m2d2.py").write_text("LR_BASE = 0.0007\nLR_WIN = 0.0003\n")
    lrs = {"LR_BASE": 0.0007, "LR_WIN": 0.0003}
    base = T.default_config()
    for table, data_dir in ((M2D2, M2D2_DATA), (SLIM120, SLIM_DATA)):
        for name, (windows, seed, lr, split, tokens) in table.items():
            cfg = T.load_config([f"config/runs/{name}.py"])
            want = dict(base, run_name=name, window_layers=windows, seed=seed, lr=lrs.get(lr, lr), eval_split=split,
                        total_tokens=tokens, data_dir=data_dir, eval_interval=100)
            diff = {k: (cfg[k], want[k]) for k in want if cfg[k] != want[k]}
            assert not diff, f"{name}: {diff}"
            text = (config_tree / "config" / "runs" / f"{name}.py").read_text()
            assert "chosen_lr.py'" not in text, name  # never the SlimPajama LR file
            if name.startswith("m2d2-"):
                assert "config/pilot_m2d2.py" in text
            _, steps, _ = T.derived(cfg)
            assert steps == {120 * M: 1832, 30 * M: 458}[tokens]


def test_m2d2_main_configs_stop_until_both_lrs_set(config_tree):
    lr_file = config_tree / "config" / "chosen_lr_m2d2.py"
    for text in ("LR_BASE = None\nLR_WIN = None\n", "LR_BASE = 0.0005\nLR_WIN = None\n"):
        lr_file.write_text(text)
        for name in ("m2d2-base-s0", "m2d2-win16-s0", "m2d2-win1-s0"):
            with pytest.raises(SystemExit, match="not both set"):
                T.load_config([f"config/runs/{name}.py"])
    # the committed file starts with both unset, and the SlimPajama LR file is untouched by all of this
    import os
    committed = open(os.path.join(REPO, "config", "chosen_lr_m2d2.py")).read()
    assert "LR_BASE = None" in committed and "LR_WIN = None" in committed
    assert (config_tree / "config" / "chosen_lr.py").read_text() == "LR_BASE = 0.0011\nLR_WIN = 0.0022\n"


# docs/M2D2_UNEVEN_PROMPT.md part 5: name -> (data folder, window_layers, seed, LR variable)
UNEVEN_DATA = {"m2d2h4": "/home/architta/orcd/scratch/moe-pilot/data_m2d2w8_hist4x",
               "m2d2r4": "/home/architta/orcd/scratch/moe-pilot/data_m2d2w8_rep4x"}
UNEVEN = {f"{p}-{arm}-s{s}": (UNEVEN_DATA[p], w, s, lr)
          for p in UNEVEN_DATA for s in range(4) for arm, w, lr in (("base", B, "LR_BASE"), ("win16", W16, "LR_WIN"))}


def test_uneven_configs(config_tree):
    assert len(UNEVEN) == 16
    lr_file = config_tree / "config" / "chosen_lr_m2d2.py"
    # refuse to run while either LR is None
    for text in ("LR_BASE = None\nLR_WIN = None\n", "LR_BASE = None\nLR_WIN = 0.0003\n"):
        lr_file.write_text(text)
        for name in ("m2d2h4-base-s0", "m2d2r4-win16-s3"):
            with pytest.raises(SystemExit, match="not both set"):
                T.load_config([f"config/runs/{name}.py"])
    lr_file.write_text("LR_BASE = 0.0007\nLR_WIN = 0.0003\n")
    lrs = {"LR_BASE": 0.0007, "LR_WIN": 0.0003}
    base = T.default_config()
    even = T.load_config(["config/runs/m2d2-base-s0.py"])
    for name, (data_dir, windows, seed, lr) in UNEVEN.items():
        cfg = T.load_config([f"config/runs/{name}.py"])
        want = dict(base, run_name=name, data_dir=data_dir, window_layers=windows, seed=seed, lr=lrs[lr],
                    total_tokens=165 * M, eval_interval=100, eval_split="val")
        diff = {k: (cfg[k], want[k]) for k in want if cfg[k] != want[k]}
        assert not diff, f"{name}: {diff}"
        # everything except the run identity, data folder and budget matches the even main runs
        same = {k for k in cfg if k not in ("run_name", "data_dir", "total_tokens", "seed", "window_layers", "lr")}
        assert {k: cfg[k] for k in same} == {k: even[k] for k in same}, name
        _, steps, _ = T.derived(cfg)
        assert steps == 2518
