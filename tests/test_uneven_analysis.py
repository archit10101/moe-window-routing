"""analysis/uneven.py on fake run logs with known per-domain losses (docs/M2D2_UNEVEN_PROMPT.md parts 6 and 9)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO

sys.path.insert(0, os.path.join(REPO, "analysis"))
import uneven as U  # noqa: E402

UNEVEN_PY = os.path.join(REPO, "analysis", "uneven.py")
DOMAINS = ["culture", "health", "history", "mathematics", "natural_sciences", "philosophy", "society", "technology"]
CLUSTER = "/home/architta/orcd/scratch/moe-pilot"
OUTPUTS = ["uneven_summary.md", "uneven_per_domain.md", "uneven_delta.png", "uneven_swap.png",
           "uneven_expert_usage.png", "uneven_results.json"]

# seed-level deviations (each has mean 0)
E = [0.002, -0.002, 0.0, 0.001, -0.001]     # even, history
EP = [-0.001, 0.001, 0.002, -0.002, 0.0]    # even, philosophy
A_ = [0.004, -0.004, 0.002, -0.002]         # uneven, history
B_ = [0.001, 0.003, -0.001, -0.003]         # uneven, philosophy
BASE_LOSS = {d: 3.0 + 0.1 * i for i, d in enumerate(DOMAINS)}


def gaps(design, s, eff):
    """Known gaps: 0.05 everywhere, seed deviations on history / philosophy, and the boost effect `eff`
    on the boosted domain (history for hist4x, philosophy for rep4x)."""
    g = {d: 0.05 for d in DOMAINS}
    if design == "even":
        g["history"] += E[s]
        g["philosophy"] += EP[s]
    else:
        g["history"] += A_[s] + (eff if design == "hist4x" else 0)
        g["philosophy"] += B_[s] + (eff if design == "rep4x" else 0)
    return g


def write_run(runs, name, data_dir, losses, tokens, test=False):
    d = runs / name
    d.mkdir(parents=True)
    recs = [{"type": "header", "run_name": name, "config": {"data_dir": data_dir, "total_tokens": tokens}},
            {"type": "final", "step": 1, "tokens": tokens, "val": losses, "val_overall": float(np.mean(list(losses.values())))}]
    (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    arm_counts = [5, 5, 0, 0, 0, 0, 0, 0] if "win16" in name else [1] * 8   # 2 vs 8 effective experts
    counts = [{dom: arm_counts for dom in DOMAINS} for _ in range(2)]
    (d / "routing_counts.json").write_text(json.dumps({"window_layers": [0, 16], "counts": counts}))
    if test:
        (d / "test_results.json").write_text(json.dumps(
            {"test": {k: v + 0.1 for k, v in losses.items()}, "routing_counts": counts}))


def make_runs(tmp, eff, drop=(), test=False):
    runs = tmp / "runs"
    for design, prefix, folder, n, tokens in (("even", "m2d2-", "data_m2d2w8", 5, 120_000_000),
                                              ("hist4x", "m2d2h4-", "data_m2d2w8_hist4x", 4, 165_000_000),
                                              ("rep4x", "m2d2r4-", "data_m2d2w8_rep4x", 4, 165_000_000)):
        for s in range(n):
            g = gaps(design, s, eff)
            for arm, losses in (("base", BASE_LOSS), ("win16", {d: BASE_LOSS[d] + g[d] for d in DOMAINS})):
                name = f"{prefix}{arm}-s{s}"
                if name not in drop:
                    write_run(runs, name, f"{CLUSTER}/{folder}", dict(losses), tokens, test)
    root = tmp / "data_root"
    for variant, boosted in (("hist4x", "history"), ("rep4x", "philosophy")):
        (root / f"data_m2d2w8_{variant}").mkdir(parents=True)
        (root / f"data_m2d2w8_{variant}" / "meta.json").write_text(json.dumps(
            {"variant": variant, "boosted_domain": boosted, "requested_boosted_domain": boosted, "fallback_used": False}))
    return runs, root


def analyze(runs, root):
    return U.run(str(runs), str(root), U.val_losses, U.val_routing)


def test_uneven_analysis_math(tmp_path):
    # --- an injected boost effect: each domain's gap is 0.02 smaller when it is boosted ---
    runs, root = make_runs(tmp_path / "boost", eff=-0.02)
    res = analyze(runs, root)
    sw = res["swap"]
    assert (sw["h"], sw["r"]) == ("history", "philosophy")
    a_minus_b = np.array(A_) - np.array(B_)                  # c = eff + (a - b) for hist4x, (a - b) - eff for rep4x
    se = math.sqrt(2 * np.var(a_minus_b, ddof=1) / 4)
    assert sw["B"] == pytest.approx(-0.04) and sw["SE"] == pytest.approx(se)
    assert sw["MDE"] == pytest.approx(2 * se) and sw["verdict"] == "supports the hypothesis"
    even_mean_gap = 0.05  # the even deviations average to 0
    assert sw["B_over_even_mean_gap"] == pytest.approx(-0.04 / even_mean_gap)

    h4 = res["designs"]["hist4x"]
    k_h4 = np.array(A_) - np.array(B_) / 7                   # k = eff + a - b/7 for hist4x
    k_even = np.array(E) - np.array(EP) / 7                  # k = e - ep/7 for even
    assert h4["DiD"] == pytest.approx(-0.02)
    assert h4["SE"] == pytest.approx(math.sqrt(np.var(k_h4, ddof=1) / 4 + np.var(k_even, ddof=1) / 5))
    assert h4["verdict"] == "supports the hypothesis"
    assert h4["placebo_rank"] == 1 and h4["delta"]["history"] == pytest.approx(-0.02)
    assert h4["ratio_boosted"] == pytest.approx(0.03 / 0.05)
    assert h4["crossover"]["mean"] == pytest.approx(0.03)
    assert h4["crossover"]["se"] == pytest.approx(np.std(A_, ddof=1) / 2)
    assert h4["crossover"]["text"].startswith("no: the baseline is still better on history")
    assert h4["absolute"]["base"]["improvement"] == pytest.approx(0.0)
    assert h4["absolute"]["win16"]["improvement"] == pytest.approx(0.02)
    assert h4["budgets_tokens"] == {"even": 120_000_000, "uneven": 165_000_000}
    r4 = res["designs"]["rep4x"]
    assert r4["boosted"] == "philosophy" and r4["DiD"] == pytest.approx(-0.02) and r4["placebo_rank"] == 1
    assert not res["partial"]

    # expert usage: base routes uniformly (8 effective experts), win16 to 2 experts
    eu = res["expert_usage"]["hist4x"]
    assert eu["base"][0]["effective_experts"]["history"] == pytest.approx(8.0)
    assert eu["win16"][1]["effective_experts"]["culture"] == pytest.approx(2.0)
    assert eu["base"][0]["mutual_information_bits"] == pytest.approx(0.0)

    # --- no effect, and the opposite effect ---
    none = analyze(*make_runs(tmp_path / "none", eff=0.0))
    assert none["swap"]["B"] == pytest.approx(0.0, abs=1e-12) and none["swap"]["verdict"] == "not distinguishable"
    assert none["designs"]["hist4x"]["verdict"] == "not distinguishable"
    opposite = analyze(*make_runs(tmp_path / "opp", eff=+0.02))
    assert opposite["swap"]["verdict"] == "contradicts it" and opposite["designs"]["rep4x"]["verdict"] == "contradicts it"
    big = analyze(*make_runs(tmp_path / "big", eff=-0.08))
    assert big["designs"]["hist4x"]["crossover"]["text"].startswith("yes: win16 beats the baseline on history")

    # --- missing seeds: PARTIAL with counts ---
    partial = analyze(*make_runs(tmp_path / "partial", eff=-0.02, drop=("m2d2r4-win16-s3", "m2d2-base-s4")))
    assert partial["partial"]
    assert partial["seeds"]["rep4x"] == {"complete": [0, 1, 2], "expected": [0, 1, 2, 3]}
    assert partial["seeds"]["even"]["complete"] == [0, 1, 2, 3]
    assert len(partial["swap"]["c"]["rep4x"]) == 3


def test_uneven_cli_outputs_and_test_split(tmp_path):
    runs, root = make_runs(tmp_path, eff=-0.02, drop=("m2d2h4-win16-s3",), test=True)
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, UNEVEN_PY, "--runs_dir", str(runs), "--out_dir", str(out),
                           "--data_root", str(root), "--with_test"], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in OUTPUTS:
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    summary = (out / "uneven_summary.md").read_text()
    first = summary.split("## 1.")[0]
    assert "**PARTIAL** (validation)" in first and "hist4x 3 of 4 seeds" in first  # verdict lines come first
    assert "- **Primary (swap contrast): supports the hypothesis**" in first
    assert "- **hist4x (boosted history), difference-in-differences vs even: supports the hypothesis**" in first
    assert "# locked test" in summary and "**PARTIAL** (locked test)" in summary
    results = json.loads((out / "uneven_results.json").read_text())
    # the test losses are the validation losses + 0.1, so every gap and contrast is the same
    assert results["locked_test"]["swap"]["B"] == pytest.approx(results["validation"]["swap"]["B"])
    assert results["locked_test"]["expert_usage"]["rep4x"]["win16"][0]["effective_experts"]["history"] == pytest.approx(2.0)
    per_domain = (out / "uneven_per_domain.md").read_text()
    assert "| history | hist4x (boosted) |" in per_domain and per_domain.count("| culture | ") == 6

    # without --data_root the cluster paths do not exist here: a clear error
    proc = subprocess.run([sys.executable, UNEVEN_PY, "--runs_dir", str(runs), "--out_dir", str(out)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0 and "--data_root" in proc.stderr


def test_effective_experts():
    assert U.effective_experts([1] * 8) == pytest.approx(8.0)
    assert U.effective_experts([3, 3, 0, 0]) == pytest.approx(2.0)
    assert U.effective_experts([7, 0, 0]) == pytest.approx(1.0)
