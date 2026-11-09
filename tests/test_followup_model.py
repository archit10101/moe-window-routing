"""Tests for the follow-up options (docs/FOLLOWUP_PROMPT.md part 8)."""
import json

import make_defaults_fixture as fixture


def test_defaults_unchanged(tmp_path):
    """Default settings reproduce the pre-follow-up code bit for bit (init, 10-step training, weights)."""
    with open(fixture.FIXTURE) as f:
        ref = json.load(f)
    got = fixture.capture(str(tmp_path))
    where = f"(fixture: torch {ref['torch_version']} on {ref['platform']}; now: torch {got['torch_version']} on {got['platform']})"
    for arm in fixture.ARMS:
        assert got["init"][arm] == ref["init"][arm], f"{arm}: initial weights changed {where}"
        g, r = got["train"][arm], ref["train"][arm]
        assert g["steps"] == r["steps"], f"{arm}: training trajectory changed {where}"
        assert g["final"] == r["final"], f"{arm}: final eval changed {where}"
        assert g["final_weights"] == r["final_weights"], f"{arm}: trained weights changed {where}"
