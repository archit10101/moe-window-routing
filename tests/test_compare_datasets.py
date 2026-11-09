"""analysis/compare_datasets.py (docs/M2D2_PROMPT.md part 7)."""
import json
import math
import os
import subprocess
import sys

import pytest

import make_analyze_fixture as F
from conftest import REPO

COMPARE = os.path.join(REPO, "analysis", "compare_datasets.py")
OUTPUTS = ["compare_summary.md", "compare_per_domain.md", "compare_gap_curves.png", "compare_mi.png"]
SLIM500 = {"base-s0": 3.60, "base-s1": 3.61, "base-s2": 3.62, "win16-s0": 3.65, "win16-s1": 3.64, "win16-s2": 3.66,
           "win1-s0": 3.605, "win64-s0": 3.66}
SLIM120 = {"base-s0": 3.80, "base-s1": 3.81, "base-s2": 3.82, "win16-s0": 3.90, "win16-s1": 3.89, "win16-s2": 3.91}
M2D2 = {"base-s0": 3.50, "base-s1": 3.51, "base-s2": 3.52, "win16-s0": 3.52, "win16-s1": 3.53, "win16-s2": 3.51,
        "win1-s0": 3.505}


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return F.smoke_run(os.path.realpath(tmp_path_factory.mktemp("cmp")))


def compare(runs, out, *extra):
    return subprocess.run([sys.executable, COMPARE, "--runs_dir", str(runs), "--out_dir", str(out), *extra],
                          capture_output=True, text=True, timeout=120)


def test_compare_datasets_runs(smoke, tmp_path):
    runs = tmp_path / "runs"
    F.copy_runs(smoke, str(runs), SLIM500)
    F.copy_runs(smoke, str(runs), {f"slim120-{k}": v for k, v in SLIM120.items()})
    F.copy_runs(smoke, str(runs), {f"m2d2-{k}": v for k, v in M2D2.items()})
    # an unreadable test_results.json proves the default mode never touches test results
    (runs / "m2d2-base-s0" / "test_results.json").write_text("{ not json")

    proc = compare(runs, tmp_path / "out")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in OUTPUTS:
        assert (tmp_path / "out" / name).exists() and (tmp_path / "out" / name).stat().st_size > 0, name
    summary = (tmp_path / "out" / "compare_summary.md").read_text()
    for label in ("SlimPajama-500M", "SlimPajama-120M", "M2D2-Wiki-8"):
        assert f"| {label} | " in summary
    # gaps: M2D2 +0.01, Slim120 +0.09, both s = 0.01 with 3 seeds per arm
    delta, scale = 0.01 - 0.09, math.sqrt(0.01 ** 2 / 3 + 0.01 ** 2 / 3)
    assert f"**{delta:+.5f}**" in summary and f"= {scale:.5f}" in summary
    assert "**penalty clearly smaller on M2D2-Wiki-8.**" in summary
    assert "delta = -0.08000" in proc.stdout
    assert "| SlimPajama-500M | 3.61000 | 3.65000 | +0.04000 | 0.01000 | +4.08% | 3/3 | 0 of 3 | +0.00500 |" in summary
    assert "| M2D2-Wiki-8 | 3.51000 | 3.52000 | +0.01000 |" in summary and "+1.01%" in summary
    assert "SlimPajama-120M: " in summary and "no finished win1-s0" in summary  # skipped criteria named
    assert "63%" in summary and "18%" in summary  # sampling regime note
    assert "locked test" not in summary
    per_domain = (tmp_path / "out" / "compare_per_domain.md").read_text()
    assert "## M2D2-Wiki-8 domains" in per_domain and "## SlimPajama-500M sources (reference)" in per_domain
    assert per_domain.count("| c4 |") == 3

    # --with_test adds the M2D2 locked-test gap and verdict next to the validation ones
    for name, value in {"base-s0": 3.40, "base-s1": 3.41, "base-s2": 3.42, "win16-s0": 3.47, "win16-s1": 3.46,
                        "win16-s2": 3.48}.items():
        (runs / f"m2d2-{name}" / "test_results.json").write_text(json.dumps(
            {"test_overall": value, "test": {"c4": value}, "test_set_reused": False}))
    proc = compare(runs, tmp_path / "out_test", "--with_test")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = (tmp_path / "out_test" / "compare_summary.md").read_text()
    assert "| M2D2-Wiki-8 (locked test) | 3.41000 | 3.47000 | +0.06000 |" in summary
    assert "M2D2 validation vs locked test: gap +0.01000 (" in summary and "vs +0.06000 (NEGATIVE)" in summary
    assert "TEST SET REUSED" not in summary
    assert "locked test split" in (tmp_path / "out_test" / "compare_per_domain.md").read_text()
