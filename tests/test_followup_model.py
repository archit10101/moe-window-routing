"""Tests for the follow-up options (docs/FOLLOWUP_PROMPT.md part 8)."""
import json
import os

import pytest
import torch

import make_defaults_fixture as fixture
import train as T
from conftest import BASELINE, TINY, WINDOW
from model import GPT, GPTConfig
from test_train import read_log, tiny_cfg, train_records


def build(windows, router_input="replace", seed=0, **over):
    """Tiny model the way train.build_model makes it; also returns the global RNG state afterwards."""
    torch.manual_seed(seed)
    model = GPT(GPTConfig(**{**TINY, **over}, window_layers=list(windows), router_input=router_input,
                          ctx_seed=seed + T.CTX_SEED_OFFSET))
    return model, torch.get_rng_state()


def router_logits(model, n_seq):
    return [m.logits.view(n_seq, 128, -1) for m in model.moe_layers()]


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


def test_both_router_keeps_base_params():
    base, rng_base = build(WINDOW, "replace")
    both, rng_both = build(WINDOW, "both")
    sb, sc = base.state_dict(), both.state_dict()
    # every baseline parameter, in the same order and with identical values
    assert [k for k in sc if k in sb] == list(sb)
    for k in sb:
        assert torch.equal(sb[k], sc[k]), k
    # plus exactly the context modules in the window layer (layer 2 of [0, 4])
    assert [k for k in sc if k not in sb] == ["transformer.h.1.moe.router_ctx_norm.weight",
                                              "transformer.h.1.moe.router_ctx.weight"]
    assert torch.equal(sc["transformer.h.1.moe.router_ctx_norm.weight"], torch.ones(TINY["n_embd"]))
    w = sc["transformer.h.1.moe.router_ctx.weight"]
    assert w.shape == (TINY["n_exp"], TINY["n_embd"]) and 0.015 < w.std().item() < 0.025
    # the context init uses its own generator: the global RNG ends where the baseline's does
    assert torch.equal(rng_base, rng_both)
    # and the per-token baseline [0, 0] has the same parameters too (construction ignores window)
    s0 = build(BASELINE)[0].state_dict()
    assert all(torch.equal(s0[k], sb[k]) for k in s0)


def test_both_router_no_future_leakage():
    model = build(WINDOW, "both")[0].eval()
    torch.manual_seed(1)
    x = torch.randint(0, 1000, (1, 128))
    x2 = x.clone()
    x2[0, 100] = (x[0, 100] + 1) % 1000
    with torch.no_grad():
        model(x)
        a = router_logits(model, 1)
        model(x2)
        b = router_logits(model, 1)
    some_differ = False
    for la, lb in zip(a, b):
        assert la.dtype == torch.float32
        assert torch.equal(la[:, :100], lb[:, :100])
        some_differ |= not torch.equal(la[:, 100:], lb[:, 100:])
    assert some_differ


def test_both_router_window0_layers_unchanged():
    base = build(BASELINE, "replace")[0].eval()
    both = build(WINDOW, "both")[0].eval()
    l0 = both.moe_layers()[0]
    assert l0.router_ctx is None and l0.router_ctx_norm is None
    assert not any(".h.0.moe.router_ctx" in k for k in both.state_dict())
    torch.manual_seed(2)
    x = torch.randint(0, 1000, (2, 128))
    with torch.no_grad():
        base(x)
        lb = router_logits(base, 2)
        both(x)
        lc = router_logits(both, 2)
    # the window-0 layer routes exactly like the baseline
    assert torch.equal(lb[0], lc[0])
    # with the context path zeroed, the whole "both" model is exactly per-token routing
    with torch.no_grad():
        both.moe_layers()[1].router_ctx.weight.zero_()
        logits_base, _ = base(x)
        lb = router_logits(base, 2)
        logits_both, _ = both(x)
        lc = router_logits(both, 2)
    assert torch.equal(logits_base, logits_both)
    assert all(torch.equal(a, b) for a, b in zip(lb, lc))


def test_shared_hidden_param_counts():
    small = build(BASELINE)[0].param_counts()
    big = build(BASELINE, shared_hidden=2 * TINY["shared_hidden"])[0].param_counts()
    per_layer = 3 * TINY["n_embd"] * TINY["shared_hidden"]  # tiny analogue of 3 * 512 * 1024 = 1,572,864
    for key in ("total", "non_embedding", "active_non_embedding"):
        assert big[key] - small[key] == TINY["n_layer"] * per_layer, key

    # "both" adds n_exp * C + C per window layer (4,096 + 512 at full size)
    both = build(WINDOW, "both")[0].param_counts()
    for key in ("total", "non_embedding", "active_non_embedding"):
        assert both[key] - small[key] == TINY["n_exp"] * TINY["n_embd"] + TINY["n_embd"], key

    # the real config: exact Part 5 deltas
    with torch.device("meta"):
        real = GPT(GPTConfig()).param_counts()
        sh = GPT(GPTConfig(shared_hidden=2048)).param_counts()
        b16 = GPT(GPTConfig(window_layers=[0, 0, 0, 0, 16, 16, 16, 16], router_input="both")).param_counts()
    for key in ("total", "non_embedding", "active_non_embedding"):
        assert sh[key] - real[key] == 8 * 1_572_864, key
        assert b16[key] - real[key] == 4 * (4096 + 512), key


def test_router_lr_mult(synth_dir, tmp_path):
    model = build(WINDOW, "both")[0]
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu", router_lr_mult=3.0)
    router_ids = {id(p) for n, p in model.named_parameters() if n in set(model.router_param_names())}
    assert len(model.router_param_names()) == 2 * TINY["n_layer"] + 2  # norm + weight per layer, + context pair
    # groups partition the parameters; router groups hold exactly the router params, decay split kept
    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters())
    for g in opt.param_groups:
        assert all((id(p) in router_ids) == g["router"] for p in g["params"])
        assert all((p.dim() >= 2) == (g["weight_decay"] > 0) for p in g["params"])

    cfg = tiny_cfg(synth_dir, tmp_path, "lrm", steps=50, router_lr_mult=3.0)
    _, max_steps, _ = T.derived(cfg)
    for step in (0, 1, max_steps // 2, max_steps - 1):
        base_lr = T.lr_at(step, cfg, max_steps)
        T.set_lr(opt, base_lr)
        for g in opt.param_groups:
            assert g["lr"] == pytest.approx(base_lr * (3.0 if g["router"] else 1.0), rel=1e-12)

    # end to end: the trained optimizer's router groups end at 3 x the logged lr
    cfg = tiny_cfg(synth_dir, tmp_path, "lrm", steps=3, router_lr_mult=3.0, router_input="both")
    assert T.train(cfg) == 0
    last_lr = train_records(tmp_path / "lrm")[-1]["lr"]
    ckpt = torch.load(tmp_path / "lrm" / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)
    for g in ckpt["optimizer"]["param_groups"]:
        assert g["lr"] == pytest.approx(last_lr * (3.0 if g["router"] else 1.0), rel=1e-12)


def test_router_grad_logging(synth_dir, tmp_path):
    on = tiny_cfg(synth_dir, tmp_path, "on", steps=6, log_router_grad=True)
    off = tiny_cfg(synth_dir, tmp_path, "off", steps=6, log_router_grad=False)
    assert T.train(on) == 0 and T.train(off) == 0
    rec_on, rec_off = train_records(tmp_path / "on"), train_records(tmp_path / "off")
    assert len(rec_on) == len(rec_off) == 6
    for a, b in zip(rec_on, rec_off):
        g = a["router_grad_norm"]
        assert len(g) == TINY["n_layer"] and all(v >= 0 for v in g) and any(v > 0 for v in g)
        assert "router_grad_norm" not in b
        assert (a["ce"], a["aux"], a["z"], a["grad_norm"]) == (b["ce"], b["aux"], b["z"], b["grad_norm"])
    wa = torch.load(tmp_path / "on" / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)["model"]
    wb = torch.load(tmp_path / "off" / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_old_checkpoint_config_resumes(synth_dir, tmp_path):
    """A checkpoint whose config predates the follow-up keys still resumes (missing keys = defaults)."""
    cfg = tiny_cfg(synth_dir, tmp_path, "old", steps=6)
    assert T.train(cfg, stop_at_step=3) == 99
    path = os.path.join(tmp_path, "old", "ckpt", "latest.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("router_input", "router_lr_mult", "log_router_grad"):
        del ckpt["config"][k]
    torch.save(ckpt, path)
    assert T.train(cfg) == 0
    assert [r["step"] for r in read_log(tmp_path / "old") if r["type"] == "resume"] == [3]
