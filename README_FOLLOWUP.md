# Follow-up experiments: runbook

This runbook covers why window routing hurt. The spec is `docs/FOLLOWUP_PROMPT.md`. The work has two parts:

- **Diagnostics** on the existing checkpoints:
  - A1: where the loss goes (tests E1).
  - A2: routing dynamics (tests E4).
- **16 new training runs** testing E2–E5, all at LR 5e-4 for 500M tokens, each paired with an existing or new comparison run of the same seed.

| Group | Runs | Question |
|---|---|---|
| E5 | both16-s0, both16-s1 | Router sees the token's own state **and** the 16-token window mean; does that beat base? |
| E3 | win16top1-s0/s1, win16top2-s0/s1 | Window only in layer 8 / layers 7–8; does the penalty shrink? |
| E2 | base-sh2048-s0/s1, win16-sh2048-s0/s1 | Shared expert 1,024 → 2,048 in both arms; does the gap shrink? |
| E4 | base-rlr3-s0/s1, win16-rlr3-s0/s1, base-rg-s0, win16-rg-s0 | 3× router LR; router gradient norms at the normal LR |

Every new run logs per-layer router gradient norms (`log_router_grad = True`), which doesn't change training. With default settings the code reproduces the pilot bit for bit; `tests/test_followup_model.py::test_defaults_unchanged` checks this.

All commands below run on an Engaging login node, from `$PROJECT/code` with the `moe` environment active.

---

## 1. Pull on Engaging

```bash
ssh <you>@<engaging-login-host>
export PROJECT=/home/architta/orcd/scratch/moe-pilot
cd $PROJECT/code
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moe

git status --short             # expect local edits such as config/chosen_lr.py (LR_BASE/LR_WIN = 0.0005)
git stash && git pull && git stash pop
git log --oneline -1           # the latest follow-up commit
cat config/chosen_lr.py        # still your 5e-4 values (the follow-up configs set lr = 5e-4 themselves)
tail -4 config/pilot_base.py   # router_input / router_lr_mult / log_router_grad, defaults
grep micro_batch_size config/pilot_base.py   # whatever the main runs used; new runs inherit it
ls config/runs/ | grep -E "both16|top1|top2|sh2048|rlr3|-rg-"   # the 16 new configs
```

The pull adds code and new files only. Your existing run folders are never touched, and every follow-up run has a new name. Optionally, check the parameter counts:

```bash
python -c "
import torch; from model import GPT, GPTConfig
for name, kw in [('default', {}), ('both16', dict(window_layers=[0,0,0,0,16,16,16,16], router_input='both')),
                 ('shared 2048', dict(shared_hidden=2048))]:
    with torch.device('meta'): print(name, GPT(GPTConfig(**kw)).param_counts())"
# default 97,628,672 total; both16 +18,432 (4 x 4,608); shared 2048 +12,582,912 (8 x 1,572,864)
```

## 2. Submit the diagnostics job

A1 needs tiktoken's GPT-2 vocabulary, and compute nodes may be offline, so download it once on the login node into a shared cache:

```bash
mkdir -p $PROJECT/tiktoken_cache
TIKTOKEN_CACHE_DIR=$PROJECT/tiktoken_cache python -c "import tiktoken; print(tiktoken.get_encoding('gpt2').n_vocab)"   # 50257

scripts/slurm/submit_diagnostics.sh b200      # 1 GPU, 8 CPUs, 64 GB, 1 h; base-s0..s4 and win16-s0..s4
```

Output goes to `$PROJECT/analysis/diagnostics/`, and the log to `$PROJECT/runs/slurm/diagnostics-<jobid>.out`. Submit it first so it takes one of your two GPUs before the follow-up runs queue up.

## 3. Submit the follow-up runs (priority order)

```bash
scripts/slurm/submit_followups.sh b200        # all 16 jobs, in the order E5, E3, E2, E4
# or only some groups (still submitted in priority order):
scripts/slurm/submit_followups.sh b200 E5 E3
```

Each run is its own job: 1 B200, 4 CPUs, 48 GB host memory, 01:30:00. A 500M-token run takes about 20–25 minutes, so the limit leaves room. At most 2 jobs run at once, so all 16 take about 3.5 hours. If a job is interrupted, it checkpoints and requeues itself (up to 3 times), and finished runs are skipped on resubmission.

## 4. Monitor

```bash
scripts/slurm/status.sh                                   # every run folder (old and new) + squeue
tail -f $PROJECT/runs/both16-s0/stdout.log
python analysis/followup.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/followup   # works mid-way too
```

`followup.py` reports any group whose runs aren't finished yet as "not available yet", so you can run it as results come in.

## 5. Run the follow-up analysis

```bash
python analysis/followup.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/followup
cat $PROJECT/analysis/followup/followup_summary.md
# copy everything to the laptop:
scp -r <you>@<engaging-login-host>:$PROJECT/analysis/followup ./analysis_followup
scp -r <you>@<engaging-login-host>:$PROJECT/analysis/diagnostics ./analysis_diagnostics
```

How `followup.py` decides:

- **Noise scale:** s and the original gap come from the finished main runs, not hard-coded:
  - s = √((var(base) + var(win16)) / 2), with ddof 1;
  - original gap = mean(win16) − mean(base), currently about 0.0075 and +0.0446.
- **Labels:** each variant's mean paired difference (variant − comparison, same seed; positive = worse) is "clearly better" below −2s, "clearly worse" above +2s, and "within noise" otherwise.
- **E3:** for each variant, the penalty "vanishes" if its difference is within noise, and "shrinks" if it's more than 2s below the original gap.
- **E2 / E4:** the gap "shrinks" if (original gap − new gap) > 2s.
- **E4 router gradients:** "clearly smaller" means the window router's median grad norm over the second half of training is below 0.8× the baseline's in every window layer.
- **Logging check:** base-rg-s0 and win16-rg-s0 must be within 0.002 of base-s0 and win16-s0. Anything larger is flagged. GPU runs aren't bit-reproducible, so small differences are expected.

## 6. What each output file shows

### Diagnostics (`$PROJECT/analysis/diagnostics/`)

All A1 gaps are win16 − base (positive = window worse). Tokens are weighted like `val_overall`, so the numbers tie back exactly to the 0.0446 gap.

| File | What it shows |
|---|---|
| `a1_summary.md` | **E1.** A check that the recomputed `val_overall` matches `final.json` per run. Then, per bucket: token share, base and win16 CE (mean ± std over seeds), gap, contribution (gap × share) and % of the total gap. The buckets are input-token category (space / punct / digit / frequent / other), target-token category, and position (0–15, 16–63, 64–1023); each family sums to the total gap. Last, a source × input-category gap table. Under E1, the gap concentrates in token categories where the token itself matters. |
| `a1_buckets.csv` | The same numbers at full precision, with per-run CE columns. |
| `a1_gap_by_category.png` | Gap per input and target category, with error bars = pooled seed std; the dashed line is the overall gap. |
| `a2_summary.md` / `a2_routing.csv` | **E4.** Per layer, base vs win16, and per run: switch rate (top-1 expert changes between adjacent tokens), router entropy (bits), mean top-1 probability, token determinism and source determinism (1 − H(expert given token or source) / H(expert); token ids seen ≥ 20 times). Window layers should show low switch rates and token determinism, and high source determinism. |
| `a2_by_layer.png` | Each A2 metric vs layer, base vs win16 (mean, min–max band). |
| `a2_grad_norm.png` | Global gradient norm over training, mean per arm with a min–max band. |

### Follow-up analysis (`$PROJECT/analysis/followup/`)

| File | What it shows |
|---|---|
| `followup_summary.md` | s and the original gap. Per group and variant: every pair's final values, the paired differences, their mean and label, and per-source paired differences. Then the interpretation lines (E5 reading; E3 penalty vanishes / shrinks / doesn't shrink; E2 and E4 whether the gap shrinks by more than 2s; E4 router-gradient ratios per layer), and the rg logging check. The same interpretation lines are printed to the terminal. |
| `followup_runs.csv` | One row per finished run: final `val_overall`, plus group, comparison run, seed, paired difference, group mean and label for follow-up variants. |
| `followup_E5_gap.png`, `followup_E3_gap.png`, `followup_E2_gap.png`, `followup_E4_gap.png` | Variant − comparison `val_overall` over training (mean over seeds, per eval step). E2, E3 and E4 also show the original win16 − base gap curve (dashed) for reference. |
| `router_grad_norm.png` | Router gradient norm (before clipping) per layer over training, base-rg-s0 vs win16-rg-s0 (normal LR). |

---

## Troubleshooting

- **`git stash pop` reports a conflict:** the cluster copy of a file differs from the new version in overlapping lines. Most likely that's `config/pilot_base.py`; the pull only appended lines at its end. Run `git diff`, keep your values (e.g. `micro_batch_size`), keep the three new lines at the end, then `git add config/pilot_base.py && git stash drop`.
- **The diagnostics job fails in tiktoken (download or SSL error):** the cache wasn't primed. Rerun the priming command in step 2 on the login node, then resubmit.
- **"diagnostics: need at least one base-s* run with a checkpoint":** the runs need `ckpt/latest.pt`. Check `ls $PROJECT/runs/base-s0/ckpt/`.
- **A run says "nothing to do":** it already has `done`. To rerun it, move its folder away. Never do this for the original pilot runs.
- **The rg logging check is flagged (> 0.002):** compare `log.jsonl` train CE of base-rg-s0 and base-s0 step by step. Small GPU nondeterminism is normal; a steady divergence from the first steps isn't, and means something other than logging changed.
- **`pytest` on Engaging fails only in `test_defaults_unchanged`:** that test compares bit for bit against a reference captured on the laptop, so a different CPU or BLAS can differ in the last bits. Run the suite on the laptop.
