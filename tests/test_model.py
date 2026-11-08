"""Model tests from PROMPT.md part 6."""
import torch

from conftest import BASELINE, TINY, WINDOW, tiny_model
from model import GPT, GPTConfig, causal_window_mean


def router_logits(model):
    """Per-layer router logits from the last forward, as [B, T, n_exp]."""
    return [m.logits for m in model.moe_layers()]


def test_window_mean_matches_naive():
    torch.manual_seed(0)
    B, T, C = 2, 256, 8
    h = torch.randn(B, T, C)
    assert not torch.allclose(h[0], h[1])  # the two sequences differ
    for n in (1, 3, 128, 200):
        got = causal_window_mean(h, n)
        want = torch.empty_like(h)
        for b in range(B):
            for t in range(T):
                lo = max(0, t - n + 1)
                want[b, t] = h[b, lo:t + 1].mean(0)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), f"mismatch for n={n}"


def test_no_future_leakage():
    model = tiny_model(WINDOW).eval()
    torch.manual_seed(1)
    x = torch.randint(0, 1000, (1, 128))
    x2 = x.clone()
    x2[0, 100] = (x[0, 100] + 1) % 1000
    with torch.no_grad():
        model(x)
        a = [l.view(1, 128, -1) for l in router_logits(model)]
        model(x2)
        b = [l.view(1, 128, -1) for l in router_logits(model)]
    some_differ = False
    for la, lb in zip(a, b):
        assert la.dtype == torch.float32
        assert torch.equal(la[:, :100], lb[:, :100])
        some_differ |= not torch.equal(la[:, 100:], lb[:, 100:])
    assert some_differ


def test_no_batch_leakage():
    model = tiny_model(WINDOW).eval()
    torch.manual_seed(2)
    x = torch.randint(0, 1000, (2, 128))
    x2 = x.clone()
    x2[1] = torch.randint(0, 1000, (128,))
    with torch.no_grad():
        model(x)
        a = [l.view(2, 128, -1) for l in router_logits(model)]
        model(x2)
        b = [l.view(2, 128, -1) for l in router_logits(model)]
    for la, lb in zip(a, b):
        assert torch.equal(la[0], lb[0])
    assert any(not torch.equal(la[1], lb[1]) for la, lb in zip(a, b))


def test_identical_init_across_modes():
    sa = tiny_model(BASELINE, seed=0).state_dict()
    sb = tiny_model(WINDOW, seed=0).state_dict()
    assert list(sa.keys()) == list(sb.keys())
    for k in sa:
        assert sa[k].shape == sb[k].shape, k
        assert torch.equal(sa[k], sb[k]), k


def test_router_learns_from_lm_loss():
    # aux_coef = z_coef = 0: the loss is cross-entropy only
    for windows in (BASELINE, WINDOW):
        model = tiny_model(windows).train()
        torch.manual_seed(3)
        x = torch.randint(0, 1000, (4, 128))
        y = torch.randint(0, 1000, (4, 128))
        _, ce = model(x, y)
        ce.backward()
        for i, m in enumerate(model.moe_layers()):
            g = m.router.weight.grad
            assert g is not None, f"layer {i}: no router gradient"
            assert (g != 0).all(), f"layer {i}: zero entries in the router gradient"


def test_pooled_load_counts():
    batch_size = 16
    micro = batch_size // 4
    model = tiny_model(WINDOW).train()
    torch.manual_seed(4)
    x = torch.randint(0, 1000, (batch_size, 128))
    k = TINY["top_k"]
    pooled = torch.zeros(TINY["n_layer"], TINY["n_exp"])
    slots = 0
    for i in range(4):
        model(x[i * micro:(i + 1) * micro])
        pooled += torch.stack([m.counts for m in model.moe_layers()])
        slots += micro * 128 * k
    assert torch.equal(model.load_fractions(), pooled / slots)
    for m, p in zip(model.moe_layers(), pooled):
        assert torch.equal(m.count_buf, p)
        assert m.slots_buf.item() == slots

    model.reset_load_buffers()
    for m in model.moe_layers():
        assert torch.count_nonzero(m.count_buf) == 0
        assert m.slots_buf.item() == 0

    # eval-mode forwards use this batch's counts and leave the buffers alone
    model.eval()
    with torch.no_grad():
        model(x[:micro])
    for m in model.moe_layers():
        assert torch.count_nonzero(m.count_buf) == 0
        assert m.slots_buf.item() == 0
        assert torch.equal(m.f, m.counts / (micro * 128 * k))


def test_param_counts_real_config():
    with torch.device("meta"):
        model = GPT(GPTConfig())  # the real 8-layer config
    counts = model.param_counts()
    expected = dict(non_embedding=71_348_736, active_non_embedding=33_599_488, total=97_628_672)
    for key, want in expected.items():
        assert abs(counts[key] - want) / want <= 0.01, f"{key}: {counts[key]:,} vs {want:,}"
