"""
Training loop for the MoE window-routing pilot (docs/PROMPT.md part 4).

Usage (from the repository root):
    python train.py config/runs/<name>.py [--key=value ...]

One process, one device (a CUDA GPU, or the CPU for tests). Exit codes:
    0   finished (or already done)
    99  interrupted by SIGUSR1/SIGTERM after saving a checkpoint; resumable
"""

import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from model import GPT, GPTConfig

REPO = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG = os.path.join(REPO, "config", "pilot_base.py")

# Used only when meta.json lacks train_share_by_source (PROMPT.md part 2).
FALLBACK_SHARES = {
    "arxiv": 0.0527, "book": 0.0321, "c4": 0.2546, "commoncrawl": 0.4966,
    "github": 0.0766, "stackexchange": 0.0401, "wikipedia": 0.0472,
}

MODEL_KEYS = ["block_size", "vocab_size", "n_layer", "n_head", "n_embd", "n_exp", "top_k",
              "expert_hidden", "shared_hidden", "window_layers", "router_input"]

# Keys that must match between a checkpoint and the config resuming from it.
RESUME_MUST_MATCH = MODEL_KEYS + ["seed", "lr", "min_lr_ratio", "warmup_frac", "weight_decay",
                                  "beta1", "beta2", "grad_clip", "batch_size", "total_tokens",
                                  "aux_coef", "z_coef", "dtype", "router_lr_mult"]

CTX_SEED_OFFSET = 1_000_003  # the "both" context routers are initialized from seed + this

REQUIRED = ["run_name", "lr", "total_tokens"]

CONFIGURATOR_NAMES = {"sys", "literal_eval", "arg", "config_file", "key", "val", "attempt", "f"}


# -----------------------------------------------------------------------------
# config

def default_config():
    """Every default from config/pilot_base.py, as a dict."""
    ns = {}
    with open(BASE_CONFIG) as f:
        exec(f.read(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_")}


def load_config(argv):
    """Run configurator.py on argv (config files and --key=value) and return the config dict.

    Config files are exec'd relative to the current directory, so run from the repository root.
    """
    if not os.path.isfile(os.path.join("config", "pilot_base.py")):
        sys.exit("train.py: run this from the repository root (config/pilot_base.py not found)")
    base = default_config()
    ns = dict(base)
    old_argv = sys.argv
    sys.argv = ["train.py"] + list(argv)
    try:
        with open(os.path.join(REPO, "configurator.py")) as f:
            exec(f.read(), ns)
    finally:
        sys.argv = old_argv
    unknown = [k for k, v in ns.items()
               if not k.startswith("_") and k not in base and k not in CONFIGURATOR_NAMES
               and not k.isupper() and not callable(v) and type(v).__name__ != "module"]
    if unknown:
        sys.exit(f"train.py: unknown config keys {unknown}")
    return {k: ns[k] for k in base}


def validate_config(cfg):
    for k in REQUIRED:
        if cfg.get(k) is None:
            sys.exit(f"train.py: config key '{k}' is required but not set")
    if len(cfg["window_layers"]) != cfg["n_layer"]:
        sys.exit(f"train.py: window_layers has {len(cfg['window_layers'])} entries, n_layer is {cfg['n_layer']}")
    if cfg["batch_size"] % cfg["micro_batch_size"] != 0:
        sys.exit("train.py: batch_size must be a multiple of micro_batch_size")


def derived(cfg):
    tokens_per_step = cfg["batch_size"] * cfg["block_size"]
    max_steps = math.ceil(cfg["total_tokens"] / tokens_per_step)
    grad_accum = cfg["batch_size"] // cfg["micro_batch_size"]
    return tokens_per_step, max_steps, grad_accum


def pick_device(cfg):
    if cfg["device"] == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg["device"]


# -----------------------------------------------------------------------------
# data

def sample_offsets(seed, step, n, data_len, block_size):
    """Offsets for optimizer step `step`: a pure function of (seed, step)."""
    rng = np.random.default_rng([seed, step])
    return rng.integers(0, data_len - block_size - 1, size=n)


def make_xy(data, offsets, block_size, device):
    x = np.stack([data[o:o + block_size] for o in offsets]).astype(np.int64)
    y = np.stack([data[o + 1:o + 1 + block_size] for o in offsets]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def read_meta(data_dir):
    path = os.path.join(data_dir, "meta.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    else:
        print(f"WARNING: {path} not found; using fallback source shares")
    shares = meta.get("train_share_by_source") or FALLBACK_SHARES
    return meta, shares


def load_val_sets(data_dir, cfg):
    """Per source: (memmap, fixed offsets). Offsets are drawn once with rng([12345, i]),
    i = the source's index in sorted order. Offsets may repeat when a file is small."""
    _, shares = read_meta(data_dir)
    T = cfg["block_size"]
    n = cfg["final_eval_batches"] * cfg["eval_batch_size"]
    val_sets = {}
    for i, src in enumerate(sorted(shares)):
        path = os.path.join(data_dir, f"val_{src}.bin")
        if not os.path.exists(path):
            print(f"WARNING: {path} missing; skipping source {src}")
            continue
        data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(data) < T + 1:
            print(f"WARNING: {path} has {len(data)} tokens (< block_size + 1); skipping source {src}")
            continue
        rng = np.random.default_rng([12345, i])
        offsets = rng.integers(0, len(data) - T, size=n)  # o + T + 1 <= len(data)
        val_sets[src] = (data, offsets, shares[src])
    if not val_sets:
        sys.exit(f"train.py: no usable validation files in {data_dir}")
    return val_sets


@torch.no_grad()
def evaluate(model, val_sets, n_batches, cfg, device, ctx, routing=False):
    """Pure next-token cross-entropy per source (fp32 mean over all target tokens) and
    the share-weighted val_overall. Runs in eval mode, so load buffers are untouched.
    With routing=True, also returns per-layer {source: expert counts}."""
    was_training = model.training
    model.eval()
    T, ebs = cfg["block_size"], cfg["eval_batch_size"]
    val, counts = {}, [dict() for _ in range(cfg["n_layer"])]
    for src, (data, offsets, _) in val_sets.items():
        offs = offsets[: n_batches * ebs]
        if routing:
            model.set_routing_stats(True)
        loss_sum, n_tok = 0.0, 0
        for j in range(0, len(offs), ebs):
            x, y = make_xy(data, offs[j:j + ebs], T, device)
            with ctx:
                logits, _ = model(x)
            loss_sum += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1),
                                        reduction="sum").item()
            n_tok += y.numel()
        val[src] = loss_sum / n_tok
        if routing:
            for layer, c in enumerate(model.routing_counts()):
                counts[layer][src] = c
            model.set_routing_stats(False)
    w = sum(val_sets[s][2] for s in val)
    overall = sum(val_sets[s][2] * val[s] for s in val) / w
    if was_training:
        model.train()
    return val, overall, (counts if routing else None)


# -----------------------------------------------------------------------------
# schedule, logging, checkpoints

def lr_at(step, cfg, max_steps):
    """Linear warmup, then cosine decay to lr * min_lr_ratio at the last step."""
    lr, min_lr = cfg["lr"], cfg["lr"] * cfg["min_lr_ratio"]
    warmup = max(1, int(round(cfg["warmup_frac"] * max_steps)))
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - 1 - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))) * (lr - min_lr)


class JsonlLogger:
    def __init__(self, path):
        self.f = open(path, "a")

    def write(self, kind, **fields):
        self.f.write(json.dumps({"type": kind, **fields}) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def git_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        return commit, bool(status)
    except Exception:
        return None, None


def save_ckpt(path, model, optimizer, step, cfg, wall_time_s):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg,
        "wall_time_s": wall_time_s,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, optimizer, cfg):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    old = with_defaults(ckpt["config"])
    bad = [k for k in RESUME_MUST_MATCH if old.get(k) != cfg.get(k)]
    if bad:
        sys.exit(f"train.py: checkpoint {path} was made with a different config for {bad}; "
                 f"refusing to resume (delete the run folder to start over)")
    if old.get("micro_batch_size") != cfg["micro_batch_size"]:
        print(f"WARNING: micro_batch_size changed {old.get('micro_batch_size')} -> {cfg['micro_batch_size']}")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    rng = ckpt["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return ckpt["step"], ckpt.get("wall_time_s", 0.0)


# -----------------------------------------------------------------------------
# training

def with_defaults(cfg):
    """A config (e.g. from an older checkpoint) with any missing keys filled from pilot_base.py."""
    return {**default_config(), **cfg}


def build_model(cfg, device):
    """The GPT for this config, initialized from torch.manual_seed(seed)."""
    torch.manual_seed(cfg["seed"])
    gcfg = GPTConfig(**{k: cfg[k] for k in MODEL_KEYS}, ctx_seed=cfg["seed"] + CTX_SEED_OFFSET)
    return GPT(gcfg).to(device)


def build(cfg, device):
    model = build_model(cfg, device)
    optimizer = model.configure_optimizers(cfg["weight_decay"], cfg["lr"], (cfg["beta1"], cfg["beta2"]),
                                           "cuda" if device.startswith("cuda") else "cpu",
                                           router_lr_mult=cfg["router_lr_mult"])
    return model, optimizer


def set_lr(optimizer, lr):
    """Schedule lr for every param group, scaled by the group's lr_mult (router groups)."""
    for g in optimizer.param_groups:
        g["lr"] = lr * g.get("lr_mult", 1.0)


def train_step(model, optimizer, data, step, cfg, max_steps, grad_accum, device, ctx, router_grad=False):
    """One optimizer step. Returns tensors (no .item()): ce, aux, z (micro-batch means),
    grad_norm, the per-layer load fractions pooled over the step and, with router_grad=True,
    the per-layer router gradient norms measured before clipping."""
    lr = lr_at(step, cfg, max_steps)
    set_lr(optimizer, lr)
    offsets = sample_offsets(cfg["seed"], step, cfg["batch_size"], len(data), cfg["block_size"])
    x_all, y_all = make_xy(data, offsets, cfg["block_size"], device)
    mb = cfg["micro_batch_size"]
    ce_acc = aux_acc = z_acc = 0.0
    for i in range(grad_accum):
        x, y = x_all[i * mb:(i + 1) * mb], y_all[i * mb:(i + 1) * mb]
        with ctx:
            _, ce = model(x, y)
        aux = torch.stack(model.aux_losses()).sum()
        z = torch.stack(model.z_losses()).sum()
        loss = (ce + cfg["aux_coef"] * aux + cfg["z_coef"] * z) / grad_accum
        loss.backward()
        ce_acc = ce_acc + ce.detach()
        aux_acc = aux_acc + aux.detach()
        z_acc = z_acc + z.detach()
    load = model.load_fractions()  # pooled over this step's micro-batches
    router_grad_norm = model.router_grad_norms() if router_grad else None  # before clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    optimizer.step()
    model.reset_load_buffers()
    optimizer.zero_grad(set_to_none=True)
    return dict(ce=ce_acc / grad_accum, aux=aux_acc / grad_accum, z=z_acc / grad_accum,
                grad_norm=grad_norm, load=load, lr=lr, router_grad_norm=router_grad_norm)


def benchmark(model, optimizer, data, cfg, max_steps, grad_accum, device, ctx):
    n = cfg["benchmark_steps"]
    tokens_per_step = cfg["batch_size"] * cfg["block_size"]
    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    model.train()
    for step in range(n):
        t0 = time.time()
        train_step(model, optimizer, data, step, cfg, max_steps, grad_accum, device, ctx)
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"benchmark step {step + 1}/{n}: {tokens_per_step / times[-1]:,.0f} tok/s", flush=True)
    measured = times[5:] if len(times) > 5 else times
    tok_s = tokens_per_step * len(measured) / sum(measured)
    peak = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else float("nan")
    print(f"BENCHMARK run={cfg['run_name']} steps={n} tok_per_s={tok_s:.0f} peak_mem_gb={peak:.2f} "
          f"micro_batch_size={cfg['micro_batch_size']}", flush=True)
    return 0


def train(cfg, stop_at_step=None):
    """Run (or resume) one training run. Returns the process exit code.

    stop_at_step: for tests only; behaves like a signal arriving during that step.
    """
    validate_config(cfg)
    run_dir = os.path.join(cfg["out_root"], cfg["run_name"])
    bench = cfg["benchmark_steps"] > 0
    if not bench and os.path.exists(os.path.join(run_dir, "done")):
        print(f"{run_dir}/done exists; nothing to do")
        return 0

    torch.set_num_threads(cfg["num_threads"])
    device = pick_device(cfg)
    is_cuda = device.startswith("cuda")
    if is_cuda and cfg["dtype"] == "bfloat16":
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()  # plain fp32 (always on CPU)
    tokens_per_step, max_steps, grad_accum = derived(cfg)

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model, optimizer = build(cfg, device)
    counts = model.param_counts()
    print(f"run {cfg['run_name']}: device {device}, window_layers {cfg['window_layers']}")
    print(f"parameters: total {counts['total']:,} | non-embedding {counts['non_embedding']:,} | "
          f"active non-embedding {counts['active_non_embedding']:,}")
    print(f"max_steps {max_steps}, tokens/step {tokens_per_step:,}, grad_accum {grad_accum}", flush=True)

    data = np.memmap(os.path.join(cfg["data_dir"], "train.bin"), dtype=np.uint16, mode="r")
    if bench:
        return benchmark(model, optimizer, data, cfg, max_steps, grad_accum, device, ctx)

    ckpt_dir = os.path.join(run_dir, "ckpt")
    ckpt_path = os.path.join(ckpt_dir, "latest.pt")
    os.makedirs(ckpt_dir, exist_ok=True)
    log = JsonlLogger(os.path.join(run_dir, "log.jsonl"))
    val_sets = load_val_sets(cfg["data_dir"], cfg)

    step, prior_wall = 0, 0.0
    if cfg["resume"] and os.path.exists(ckpt_path):
        step, prior_wall = load_ckpt(ckpt_path, model, optimizer, cfg)
        print(f"resumed from {ckpt_path} at step {step}", flush=True)
        log.write("resume", step=step, hostname=socket.gethostname(), time=time.time())
    else:
        commit, dirty = git_info()
        log.write("header", run_name=cfg["run_name"], config=cfg, git_commit=commit, git_dirty=dirty,
                  hostname=socket.gethostname(),
                  gpu=torch.cuda.get_device_name(device) if is_cuda else None,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  param_counts=counts, max_steps=max_steps, tokens_per_step=tokens_per_step,
                  start_time=time.strftime("%Y-%m-%d %H:%M:%S"))

    # signals: finish the current optimizer step, checkpoint, exit 99
    stop = {"flag": False}

    def on_signal(signum, frame):
        print(f"received signal {signum}; will checkpoint and exit after this step", flush=True)
        stop["flag"] = True

    old_handlers = {s: signal.signal(s, on_signal) for s in (signal.SIGUSR1, signal.SIGTERM)}

    start = time.time()

    def wall():
        return prior_wall + time.time() - start

    def do_eval(n_batches, routing=False):
        return evaluate(model, val_sets, n_batches, cfg, device, ctx, routing=routing)

    try:
        model.train()
        model.reset_load_buffers()
        if step == 0:
            val, overall, _ = do_eval(cfg["eval_batches"])
            log.write("eval", step=0, tokens=0, val=val, val_overall=overall)
            print(f"step 0: val_overall {overall:.4f}", flush=True)

        last_ckpt = time.time()
        t_log, tokens_since_log, paused = time.time(), 0, 0.0
        while step < max_steps:
            log_now = (step + 1) % cfg["log_interval"] == 0
            out = train_step(model, optimizer, data, step, cfg, max_steps, grad_accum, device, ctx,
                             router_grad=cfg["log_router_grad"] and log_now)
            step += 1
            tokens_since_log += tokens_per_step

            if step % cfg["log_interval"] == 0:
                load = out["load"].float().cpu()
                dt = time.time() - t_log - paused
                rec = dict(step=step, tokens=step * tokens_per_step, ce=out["ce"].item(),
                           aux=out["aux"].item(), z=out["z"].item(), lr=out["lr"],
                           grad_norm=out["grad_norm"].item(), tok_per_s=tokens_since_log / max(dt, 1e-9),
                           load_max=load.max(1).values.tolist(), load_min=load.min(1).values.tolist(),
                           time=wall())
                if cfg["log_router_grad"]:
                    rec["router_grad_norm"] = out["router_grad_norm"].float().cpu().tolist()
                log.write("train", **rec)
                print(f"step {step}/{max_steps}: ce {rec['ce']:.4f} aux {rec['aux']:.4f} z {rec['z']:.4f} "
                      f"lr {rec['lr']:.2e} gnorm {rec['grad_norm']:.3f} {rec['tok_per_s']:,.0f} tok/s",
                      flush=True)
                t_log, tokens_since_log, paused = time.time(), 0, 0.0

            t_pause = time.time()
            if step % cfg["eval_interval"] == 0 and step < max_steps:
                val, overall, _ = do_eval(cfg["eval_batches"])
                log.write("eval", step=step, tokens=step * tokens_per_step, val=val, val_overall=overall)
                print(f"step {step}: val_overall {overall:.4f}", flush=True)

            interrupted = stop["flag"] or (stop_at_step is not None and step >= stop_at_step)
            if interrupted and step < max_steps:
                save_ckpt(ckpt_path, model, optimizer, step, cfg, wall())
                print(f"checkpoint saved at step {step}; exiting with code 99", flush=True)
                return 99
            if time.time() - last_ckpt >= cfg["ckpt_interval_minutes"] * 60:
                save_ckpt(ckpt_path, model, optimizer, step, cfg, wall())
                last_ckpt = time.time()
            paused += time.time() - t_pause

        # end of run
        val, overall, routing = do_eval(cfg["final_eval_batches"], routing=True)
        tokens = step * tokens_per_step
        save_ckpt(ckpt_path, model, optimizer, step, cfg, wall())
        with open(os.path.join(run_dir, "routing_counts.json"), "w") as f:
            json.dump({"window_layers": cfg["window_layers"], "counts": routing}, f, indent=1)
        final = dict(step=step, tokens=tokens, val=val, val_overall=overall, wall_time_s=wall())
        with open(os.path.join(run_dir, "final.json"), "w") as f:
            json.dump({"run_name": cfg["run_name"], **final}, f, indent=1)
        log.write("final", **final)
        open(os.path.join(run_dir, "done"), "w").close()
        print(f"done: val_overall {overall:.4f} after {tokens:,} tokens", flush=True)
        return 0
    finally:
        for s, h in old_handlers.items():
            signal.signal(s, h)
        log.close()


def main():
    cfg = load_config(sys.argv[1:])
    sys.exit(train(cfg))


if __name__ == "__main__":
    main()
