"""
GPT with a mixture-of-experts layer in every block, for the window-routing pilot.

The skeleton (embeddings, pre-norm blocks, causal attention, tied head, init)
is nanoGPT's. The MoE layer follows docs/PROMPT.md part 3:
  - one shared SwiGLU expert that every token goes through,
  - n_exp routed SwiGLU experts, top-k routing with renormalized gates,
  - no router noise, no capacity limits, no token dropping,
  - the router input is either the token's own hidden state (window == 0) or
    the causal mean of the last `window` hidden states (window >= 1).
Only the router input differs between the two arms; module construction never
depends on `window`.
"""

import inspect
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """nanoGPT's LayerNorm, without bias."""

    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class RMSNorm(nn.Module):
    """RMSNorm with a learnable weight (initialized to ones)."""

    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def causal_window_mean(h, n):
    """Causal mean of h over positions max(0, t-n+1) .. t, for every t.

    h: [B, T, C]. Computed in fp32 with a cumulative sum along T only, so a
    window never reaches before the start of the sequence and the batch
    dimension is never mixed. Returns a tensor with h's dtype.
    """
    assert n >= 1, "window must be >= 1"
    B, T, C = h.shape
    S = torch.cumsum(h.float(), dim=1)
    # S shifted right by n positions along T, zeros filled in at the start
    if n < T:
        S_prev = torch.cat([S.new_zeros(B, n, C), S[:, : T - n]], dim=1)
    else:
        S_prev = torch.zeros_like(S)
    count = torch.arange(1, T + 1, device=h.device, dtype=torch.float32).clamp(max=n)
    return ((S - S_prev) / count[None, :, None]).to(h.dtype)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLU(nn.Module):
    """W_down(silu(W_gate x) * (W_up x)), no biases."""

    def __init__(self, n_embd, hidden):
        super().__init__()
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):

    def __init__(self, config, window):
        super().__init__()
        C = config.n_embd
        self.n_exp = config.n_exp
        self.top_k = config.top_k
        assert 1 <= self.top_k <= self.n_exp
        # plain attribute: it changes the router input, never the parameters
        self.window = int(window)

        self.router_norm = RMSNorm(C, eps=1e-6)
        self.router = nn.Linear(C, self.n_exp, bias=False)
        self.shared = SwiGLU(C, config.shared_hidden)
        self.experts = nn.ModuleList([SwiGLU(C, config.expert_hidden) for _ in range(self.n_exp)])

        # load pooled over the micro-batches of the current optimizer step
        self.register_buffer("count_buf", torch.zeros(self.n_exp), persistent=False)
        self.register_buffer("slots_buf", torch.zeros(()), persistent=False)
        # routing statistics for the analysis tables (only when enabled)
        self.collect_stats = False
        self.register_buffer("stat_counts", torch.zeros(self.n_exp, dtype=torch.long), persistent=False)

        # filled in by every forward
        self.aux = None      # load-balancing loss (scalar, differentiable)
        self.z = None        # router z-loss (scalar, differentiable)
        self.f = None        # load fraction per expert (detached)
        self.counts = None   # expert counts in this forward (detached)
        self.logits = None   # router logits [N, n_exp], fp32 (detached)

    def forward(self, h):
        B, T, C = h.shape
        N = B * T
        k = self.top_k

        # router input, computed on [B, T, C] before any flattening
        r = h if self.window == 0 else causal_window_mean(h, self.window)

        # router math in fp32 with autocast disabled
        with torch.autocast(device_type=h.device.type, enabled=False):
            rn = self.router_norm(r.float())
            logits = F.linear(rn.float(), self.router.weight.float()).reshape(N, self.n_exp)
            probs = F.softmax(logits, dim=-1)
            top_p, top_i = probs.topk(k, dim=-1)  # [N, k]
            gates = top_p / top_p.sum(-1, keepdim=True)

        # dispatch: every (token, slot) pair goes to its expert; nothing is dropped
        x = h.reshape(N, C)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = torch.where(top_i == e)
            y = expert(x[tok]) * gates[tok, slot].unsqueeze(-1)
            out.index_add_(0, tok, y.to(out.dtype))
        out = out + self.shared(x)

        # balancing losses
        with torch.no_grad():
            counts = torch.bincount(top_i.flatten(), minlength=self.n_exp).float()
            if self.training:
                self.count_buf += counts
                self.slots_buf += N * k
                f = self.count_buf / self.slots_buf
            else:
                f = counts / (N * k)
            if self.collect_stats:
                self.stat_counts += counts.long()
        P = probs.mean(0)
        self.aux = self.n_exp * torch.sum(f * P)
        self.z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        self.f = f.detach()
        self.counts = counts.detach()
        self.logits = logits.detach()

        return out.view(B, T, C)


class Block(nn.Module):

    def __init__(self, config, window):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.moe = MoELayer(config, window)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2's 50257, padded up to a multiple of 64
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    n_exp: int = 8
    top_k: int = 2
    expert_hidden: int = 512
    shared_hidden: int = 1024
    window_layers: list = field(default_factory=lambda: [0] * 8)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert len(config.window_layers) == config.n_layer, "window_layers needs one entry per layer"
        assert all(int(w) >= 0 for w in config.window_layers)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config, w) for w in config.window_layers]),
            ln_f=LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # nanoGPT init; norms are already ones
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * config.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("attn.c_proj.weight") or pn.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=resid_std)

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- MoE helpers ----

    def moe_layers(self):
        return [block.moe for block in self.transformer.h]

    def aux_losses(self):
        """Per-layer aux losses from the last forward (differentiable)."""
        return [m.aux for m in self.moe_layers()]

    def z_losses(self):
        """Per-layer z-losses from the last forward (differentiable)."""
        return [m.z for m in self.moe_layers()]

    def load_fractions(self):
        """Per-layer load fraction f (detached), shape [n_layer, n_exp]."""
        return torch.stack([m.f for m in self.moe_layers()])

    def reset_load_buffers(self):
        """Clear the pooled load counts; call after every optimizer step."""
        for m in self.moe_layers():
            m.count_buf.zero_()
            m.slots_buf.zero_()

    def set_routing_stats(self, enabled):
        """Turn routing statistics on or off. Turning them on clears the counts."""
        for m in self.moe_layers():
            m.collect_stats = bool(enabled)
            if enabled:
                m.stat_counts.zero_()

    def routing_counts(self):
        """Accumulated expert counts per layer, as lists of ints."""
        return [m.stat_counts.tolist() for m in self.moe_layers()]

    # ---- parameter counts ----

    def param_counts(self):
        """Parameter counts (the tied head is counted once).

        non_embedding excludes both the token and position embeddings.
        active_non_embedding counts the shared expert plus top_k routed experts.
        """
        cfg = self.config
        total = sum(p.numel() for p in self.parameters())
        emb = self.transformer.wte.weight.numel() + self.transformer.wpe.weight.numel()
        non_emb = total - emb
        per_expert = sum(p.numel() for p in self.moe_layers()[0].experts[0].parameters())
        active = non_emb - cfg.n_layer * (cfg.n_exp - cfg.top_k) * per_expert
        return dict(total=total, non_embedding=non_emb, active_non_embedding=active)

    # ---- forward ----

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"sequence length {t} > block size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # pure next-token cross-entropy; autocast runs cross_entropy in fp32
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
