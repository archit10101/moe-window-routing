# Follow-up prompt: why does window routing hurt?

## Part 1: Context and goal

The pilot in this repo (spec: `docs/PROMPT.md`) is finished. Result: routing layers 5–8 on a 16-token causal window mean, instead of each token's own hidden state, made the model **worse**.

| Fact | Value |
| --- | --- |
| Final val_overall, baseline (5 seeds) | mean 3.6022, seeds 3.6030 / 3.6074 / 3.5984 / 3.6018 / 3.6006 |
| Final val_overall, win16 (5 seeds) | mean 3.6468, seeds 3.6567 / 3.6332 / 3.6489 / 3.6398 / 3.6551 |
| Gap (win16 − base) | +0.0446; seed spread s = 0.0075 |
| win1-s0 vs base-s0 (sanity) | +0.0009, so the window code path is correct |
| win64-s0 | 3.6557, about the same as win16-s0 |
| Gap over training | 0.115 at 33M tokens, peaks 0.310 at 66M, 0.085 at 131M, 0.050 at 295M, 0.0446 at 500M (flattening) |
| Gap by source | commoncrawl +0.053, c4 +0.050, book +0.041, wikipedia +0.032, stackexchange +0.021, arxiv +0.016, github +0.011 |
| Source–expert mutual information (bits), baseline layers 1–8 | 0.06, 0.08, 0.05, 0.10, 0.09, 0.10, 0.16, 0.26 |
| Same, win16 | 0.05, 0.05, 0.04, 0.04, 0.41, 0.57, 0.56, 0.60 |
| Global grad norm at end of the 100M LR check, 1e-3 | base 0.64, win16 0.42 |
| LR used for every main run | 5e-4 (both arms) |

We want to understand **why**. There are five candidate explanations, each with a test:

| # | Explanation | Test | Type |
| --- | --- | --- | --- |
| E1 | Experts need token identity: topic routing gives each expert every kind of token, diluting its capacity | Split validation loss by token category and position (A1) | Analysis of existing checkpoints |
| E2 | The shared expert was too small to absorb token-level work | Shared expert hidden 1,024 → 2,048 in both arms | Training |
| E3 | Topic routing is forced into layers that don't want it (baseline only routes by source in layer 8) | Window in layer 8 only, and in layers 7–8 only | Training |
| E4 | The window router learns more slowly (smoothed input, weaker gradient) | Routing-dynamics analysis (A2), router-gradient logging, and a faster router LR | Analysis + training |
| E5 | The router needs both token identity and context | Router sees both the token's own state and the window mean | Training |

Build the code for all five, plus tests, Slurm launchers and one follow-up analysis script. Don't run anything that needs a GPU, the cluster or the network.

## Part 2: Ground rules

1. **The existing runs must stay reproducible.** Every new option defaults to the current behavior, and with defaults the model, initialization, data order and training are bit-for-bit unchanged. Every existing test must still pass.
2. **Don't rerun or modify any existing run folder.** New runs get new names.
3. As before: no router noise, capacity limits or token dropping. Comparisons use pure validation cross-entropy. Never weaken or delete a test. Only torch, numpy, tiktoken, pyarrow, matplotlib, pytest and the standard library.
4. All new training runs use LR 5e-4, 500M tokens, and the same batch, schedule and balancing settings as the main runs.
5. If something here conflicts with the code as it now stands, stop and ask.

## Part 3: Fixed facts

- Project folder on Engaging: `/home/architta/orcd/scratch/moe-pilot` (below, `PROJECT`). Data in `PROJECT/data/`, runs in `PROJECT/runs/<run_name>/`, each with `log.jsonl`, `final.json`, `routing_counts.json`, `done` and the final checkpoint `ckpt/latest.pt`.
- Existing runs to analyze: `base-s0`…`base-s4`, `win16-s0`…`win16-s4`, `win1-s0`, `win64-s0`.
- Hardware: `mit_normal_gpu`, at most 2 GPUs at once per user. One B200 trains one run at about 480k tokens/s (about 20–25 minutes per 500M-token run including evals) using 64 GB. Four runs per B200 don't fit in memory, so always one run per job.
- Environment: `module load miniforge`, `source "$(conda info --base)/etc/profile.d/conda.sh"`, `conda activate moe`.

## Part 4: Analyses of existing checkpoints (E1 and E4)

Write `analysis/diagnostics.py`. Usage: `python analysis/diagnostics.py --runs_dir <dir> --data_dir <dir> --out_dir <dir> [--runs base-s0,...] [--device cuda]`. It loads each run's final checkpoint, runs it in eval mode under no_grad on the same fixed final-eval offsets that `train.py` uses (reuse that code; don't reimplement it), and records per-token results.

**A1: where the loss goes (E1).** For every evaluated token, record the per-token cross-entropy (reduction none, fp32). Bucket tokens three ways:

- **By input token category** (the token at position t, which is what the router sees in the baseline):
  - `space`: decodes to whitespace only (including newlines)
  - `punct`: after stripping whitespace, only punctuation characters
  - `digit`: after stripping whitespace, only digits
  - `frequent`: one of the 100 most frequent token ids in the first 50M tokens of `train.bin`, not already in a category above
  - `other`: everything else
- **By target token category:** the same categories, applied to the token at position t+1.
- **By position in the sequence:** 0–15, 16–63, 64–1023. Window routing is closer to per-token routing near the start, where the window covers fewer tokens.

Report, per bucket:

- the token share
- mean CE per run
- mean CE per arm (base, win16) with seed spread
- the gap (win16 − base)
- the bucket's contribution to the overall gap (gap × token share), so the contributions sum to the total gap

Also give a source × input-category gap table.

Outputs: `a1_buckets.csv`, `a1_summary.md`, and `a1_gap_by_category.png` (bar chart of gap per category with seed spread).

**A2: routing dynamics (E4).** On the same tokens, per run and per layer:

| Metric | Definition |
| --- | --- |
| switch_rate | Fraction of adjacent positions (t, t+1) in the same sequence whose top-1 expert differs |
| router_entropy | Mean entropy (bits) of the full 8-expert softmax |
| top1_prob | Mean top-1 probability |
| token_determinism | 1 − H(top-1 expert given input token id) / H(top-1 expert), using only token ids seen at least 20 times |
| source_determinism | Same with source in place of token id |

Also plot the global `grad_norm` from each run's `log.jsonl` over training: the mean across seeds per arm, with a min–max band.

Outputs: `a2_routing.csv`, `a2_summary.md`, `a2_by_layer.png` (each metric vs layer, base vs win16, mean across seeds), and `a2_grad_norm.png`.

Keep memory modest: process one source at a time, accumulate counts and sums rather than storing every token, and free each model before loading the next.

## Part 5: Code changes for the training experiments

Add these config keys, each defaulting to the current behavior:

| Key | Default | Meaning |
| --- | --- | --- |
| `shared_hidden` | 1024 | Hidden size of the shared expert (E2) |
| `router_input` | `"replace"` | `"replace"` = current behavior. `"both"` = the router sees the token's own state and the window mean (E5) |
| `router_lr_mult` | 1.0 | LR multiplier for all router parameters (router_norm and router weights, plus the context router below) (E4) |
| `log_router_grad` | False | If True, add `router_grad_norm` (a list of 8, one per layer, measured before clipping) to each `train` log line (E4) |

**`router_input = "both"`**, in layers whose window n ≥ 1:

- logits = router(router_norm(h)) + router_ctx(router_ctx_norm(causal_window_mean(h, n)))
- `router_ctx` is `Linear(C, 8, bias=False)` and `router_ctx_norm` is an `RMSNorm(C)`.
- **Create these extra modules after the whole model is built,** initialized from a separate `torch.Generator` seeded with seed + 1,000,003 (std 0.02 for the weight, ones for the norm). Every parameter the baseline has then gets byte-identical values for the same seed, and the only difference is the added context path.
- Everything else (fp32 router math, top-2, gates, balancing) is unchanged.
- Layers with window 0 never get the extra modules.

**`router_lr_mult`:** put router parameters in their own optimizer param groups (keeping the decay / no-decay split), with lr scaled by the multiplier at every step of the schedule.

**Parameter counts:** print them as before. Shared hidden 2,048 adds 1,572,864 parameters per layer to both total and active counts. `"both"` adds 4,096 + 512 per window layer.

## Part 6: New runs

All at LR 5e-4, 500M tokens. Same names in `config/runs/`.

| Group | Run names | window_layers | Other settings | Tests |
| --- | --- | --- | --- | --- |
| E5 | both16-s0, both16-s1 | [0,0,0,0,16,16,16,16] | router_input "both" | Does adding context to token identity help? |
| E3 | win16top1-s0, win16top1-s1 | [0,0,0,0,0,0,0,16] | | Window in layer 8 only |
| E3 | win16top2-s0, win16top2-s1 | [0,0,0,0,0,0,16,16] | | Window in layers 7–8 |
| E2 | base-sh2048-s0, base-sh2048-s1 | [0]*8 | shared_hidden 2048 | Baseline with a big shared expert |
| E2 | win16-sh2048-s0, win16-sh2048-s1 | [0,0,0,0,16,16,16,16] | shared_hidden 2048 | Window with a big shared expert |
| E4 | base-rlr3-s0, base-rlr3-s1 | [0]*8 | router_lr_mult 3.0, log_router_grad True | Faster router, baseline |
| E4 | win16-rlr3-s0, win16-rlr3-s1 | [0,0,0,0,16,16,16,16] | router_lr_mult 3.0, log_router_grad True | Faster router, window |
| E4 | base-rg-s0, win16-rg-s0 | as their arm | log_router_grad True only | Router gradients at the normal LR (should match base-s0 / win16-s0 exactly in loss) |

Set `log_router_grad True` for every new run, since it doesn't affect training. That's 16 runs; at 2 at a time, about 3.5 hours.

**Comparisons.** Each new run is paired with the existing run of the same seed:

- E5 and E3 variants vs `base-s0` and `base-s1`.
- `win16-sh2048` vs `base-sh2048`.
- `win16-rlr3` vs `base-rlr3`.

## Part 7: Follow-up analysis script

Write `analysis/followup.py`. Usage: `python analysis/followup.py --runs_dir <dir> --out_dir <dir>`. Use the main experiment's seed spread, s = 0.0075 (computed from the existing base and win16 runs, not hard-coded), as the noise scale. For each group, report:

- every run's final val_overall
- each variant's mean and its paired difference from the matching run (same seed)
- whether the mean difference is beyond 2s, i.e. "clearly better", "clearly worse" or "within noise"
- the per-source paired difference
- the gap curve over training (variant − its comparison, per eval step, mean over seeds) in one plot per group

Interpretation lines to print, per group:

| Group | Question | Reading |
| --- | --- | --- |
| E5 | both16 vs base | Clearly better: context helps as an addition. Within noise: context adds nothing at this scale. Clearly worse: even added context hurts |
| E3 | win16top1, win16top2 vs base | Penalty shrinks or vanishes with fewer window layers: topic routing only belongs at the top |
| E2 | (win16-sh2048 − base-sh2048) vs the original gap 0.0446 | Gap shrinks by more than 2s: the shared expert was the bottleneck |
| E4 | (win16-rlr3 − base-rlr3) vs 0.0446; router grad norms of base-rg-s0 vs win16-rg-s0 | Gap shrinks with a faster router, or the window router's gradients are clearly smaller: slow router learning explains part of the gap |

Also check that base-rg-s0 and win16-rg-s0 match base-s0 and win16-s0 within 0.002 (logging must not change training), and flag it if not.

Outputs: `followup_summary.md`, `followup_runs.csv`, `followup_<group>_gap.png`, and for E4 `router_grad_norm.png` (per layer, base vs window, over training).

## Part 8: Tests (CPU, pytest; add to the existing suite)

| Test | What must hold |
| --- | --- |
| `test_defaults_unchanged` | With default new keys, a tiny model's state_dict and a 10-step loss trajectory are bit-identical to the current code's, captured as a fixture before your changes |
| `test_both_router_keeps_base_params` | For the same seed, the "both" model has every baseline parameter with identical values, plus exactly the extra context modules in window layers |
| `test_both_router_no_future_leakage` | Same as the existing leakage test, for router_input "both" |
| `test_both_router_window0_layers_unchanged` | Layers with window 0 have no context modules and behave exactly like the baseline |
| `test_shared_hidden_param_counts` | shared_hidden 2048 changes the parameter counts by exactly 1,572,864 per layer (scaled to the tiny config) |
| `test_router_lr_mult` | Router param groups get lr × mult at several schedule points; all others are unchanged |
| `test_router_grad_logging` | With log_router_grad True, train lines contain 8 non-negative router grad norms; with False, the key is absent and training is identical |
| `test_diagnostics_runs` | `diagnostics.py` on smoke-cpu outputs, relabeled as base and win16 runs, writes every A1 and A2 file. Category buckets cover every token exactly once, and the per-bucket gap contributions sum to the total gap within 1e-6 |
| `test_followup_runs` | `followup.py` on fake relabeled runs writes every output and an interpretation line per group |
| `test_new_configs` | Every run in Part 6 has a config whose settings match the table |

The whole suite must still pass in under 2 minutes on CPU.

## Part 9: Launchers

In `scripts/slurm/`:

- **`submit_followups.sh <gpu-type> [group ...]`:** one 1-run job per run via the existing `submit.sh`, time 01:30:00, in this priority order: E5, E3, E2, E4. With group names given (for example `E5 E3`), submit only those.
- **`submit_diagnostics.sh <gpu-type>`:** one 1-GPU job (time 01:00:00, 8 CPUs, 64 GB) that runs `analysis/diagnostics.py` on base-s0…s4 and win16-s0…s4, writing to `PROJECT/analysis/diagnostics/`.
- **`status.sh`:** must list the new runs too.

## Part 10: Deliverables, working style and definition of done

**Deliverables:** the code changes, configs, `analysis/diagnostics.py`, `analysis/followup.py`, the tests, the two launchers, and `README_FOLLOWUP.md` with exact commands in order:

1. Pull on Engaging.
2. Submit the diagnostics job.
3. Submit the follow-up runs, in priority order.
4. Monitor with `status.sh`.
5. Run `followup.py`.
6. What each output file shows.

**Working style:** read `docs/PROMPT.md` and the current code first, then propose a plan and wait for approval. Build in this order, testing after each stage: defaults fixture → code changes → their tests → configs → diagnostics → followup analysis → launchers → README. Commit after each stage; don't push.

**Definition of done:**

- [ ] `pytest -q` passes on CPU in under 2 minutes, with no skips, including every old test.
- [ ] `bash -n` passes for every shell script.
- [ ] Parameter counts printed for both16, shared-2048 and the defaults match Part 5.
- [ ] A final summary of what you built, test results, any deviations from this spec, and anything I need to decide.
