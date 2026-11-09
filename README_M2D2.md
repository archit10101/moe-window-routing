# M2D2-Wiki-8 replication: runbook

This runbook replicates the per-token vs window-routing comparison on **M2D2-Wiki-8**: eight top-level Wikipedia categories from M2D2. The spec is `docs/M2D2_PROMPT.md`.

> **Running everything at once?** The section [Uneven-domain designs: everything in one go](#uneven-domain-designs-everything-in-one-go) at the end submits the even design below *and* the two uneven designs (hist4x, rep4x) with one command, `scripts/slurm/submit_m2d2_all.sh`. Steps 1–3 here (pull, download, inspect) still apply. The manual steps 4–9 below remain valid for the even design alone, except that the locked test (step 9) now covers all three designs, so it needs the uneven runs to exist.

- **Why this dataset:** every domain is Wikipedia, so any routing specialization has to come from topic, not from writing style.
- **The comparison:** M2D2 against a SlimPajama control at the same 120M-token budget (`slim120-`), and against the original 500M SlimPajama pilot.
- **Unchanged:** the existing SlimPajama runs, configs, `config/chosen_lr.py` and every script's default behavior.

| Runs | What | Budget | LR |
|---|---|---|---|
| m2d2-lr-{base,win16}-{2.5e-4,5e-4,1e-3} | LR check, scored on the **calibration** split | 30M tokens (458 steps) | as named |
| m2d2-base-s0…s4, m2d2-win16-s0…s4 | Main comparison, 5 seeds per arm | 120M tokens (1,832 steps) | LR_BASE / LR_WIN from `config/chosen_lr_m2d2.py` |
| m2d2-win1-s0 | Sanity check (window 1 = per-token) | 120M | LR_BASE |
| m2d2-win64-s0 | Window sweep | 120M | LR_WIN |
| slim120-{base,win16}-s0…s2 | SlimPajama control at the same budget | 120M | 5e-4 |

All runs use `eval_interval = 100` and otherwise the pilot's recipe. Every run is one job on one B200; a 120M run takes about 5–8 minutes.

**Sampling regime.** `train.py` samples random 1,024-token windows with replacement, so "1 epoch" means a 120M-token budget over ~120M training tokens.

| Runs | Positions seen at least once | Repeats |
|---|---|---|
| M2D2 (both arms) | ~63% (1 − e⁻¹) | ~26% seen two or more times |
| slim120 control (120M drawn from SlimPajama's 600M) | ~18% | almost none |
| Original 500M pilot | ~57% | |

So the slim120 control matches M2D2 in budget and schedule, not in coverage. `meta.json` and `compare_summary.md` repeat this note.

**The locked test split** (`test_<domain>.bin`) is read by nothing except `analysis/eval_locked_test.py`, once, in step 9. `train.py` refuses `eval_split='test'`, and the script refuses a second evaluation unless you pass `--allow_reuse_of_locked_test`, which is logged and marked.

All commands run on an Engaging login node, from `$PROJECT/code`, with the `moe` environment active (except where a job does it for you).

---

## 1. Pull on Engaging

```bash
ssh <you>@<engaging-login-host>
export PROJECT=/home/architta/orcd/scratch/moe-pilot
cd $PROJECT/code
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moe

git stash && git pull && git stash pop    # keeps local edits such as config/chosen_lr.py
git log --oneline -1
tail -3 config/pilot_base.py              # ends with eval_split = 'val'
cat config/chosen_lr_m2d2.py              # LR_BASE = None, LR_WIN = None until the LR check
ls config/runs | grep -cE "^(m2d2|slim120)-"   # 24
```

## 2. Download (login node: it needs the network)

```bash
python -c "import huggingface_hub" 2>/dev/null || pip install huggingface_hub
export TIKTOKEN_CACHE_DIR=$PROJECT/tiktoken_cache          # already primed for the follow-up diagnostics
python -c "import tiktoken; print(tiktoken.get_encoding('gpt2').n_vocab)"   # 50257, from the cache

tmux new -s m2d2   # about 4.1 GB plus Philosophy; keeps going if your connection drops
python data/m2d2/prepare.py --download --raw_dir $PROJECT/raw_m2d2
cat $PROJECT/raw_m2d2/REVISION.txt        # the dataset commit every archive was pinned to
ls $PROJECT/raw_m2d2                      # 9 extracted folders, each with train/valid/test.txt
```

The download fetches the 8 top-level archives plus `Philosophy` (used only for the overlap report), then extracts them. Rerunning skips configs that are already extracted.

## 3. Inspect and choose `--doc_mode`

```bash
mkdir -p $PROJECT/analysis
python data/m2d2/prepare.py --inspect --raw_dir $PROJECT/raw_m2d2 | tee $PROJECT/analysis/m2d2_inspect.txt
```

For each config, inspect shows:

- line counts per file;
- line-length percentiles (5/50/95/99);
- the share of empty lines;
- the share of heading-like lines: top-level `= Title =`, any wikitext heading, and "short line followed by a long one";
- the first 8 lines of `train.txt`.

It then prints `RECOMMENDATION: --doc_mode …` and the overlap between `Philosophy` and `Philosophy_and_thinking`.

How to choose:

- **`line`**: lines are whole articles (long lines, no blank lines or heading patterns). The samples should each read as a complete article.
- **`blank`**: articles are separated by empty lines.
- **`heading`**: an article starts at each top-level `= Title =` line.
- **Check the samples, not just the recommendation.** If the recommendation and the samples disagree, or the eight configs disagree (inspect warns about this), go with what the samples show and note it.
- **`block`**: the fallback when article boundaries can't be recovered. Consecutive groups of `--block_lines` lines (default 40) become one unit.
  - Whole-article integrity isn't guaranteed: parts of one article can land in different splits, which slightly weakens the train/val/test separation.
  - Before accepting `block`, look again at the first 8 lines for any boundary marker the heuristics missed (a title line, a URL or ID line).
  - If you do use it, pick `--block_lines` near the typical article length in lines, so that most units are roughly one article.
  - Record that you used `block` and why. Expect lower reported overlap and dedup counts, because shifted block boundaries hide duplicates.

## 4. Build (CPU job, no network)

```bash
sbatch --output=$PROJECT/runs/slurm/%x-%j.out data/m2d2/build_m2d2.sbatch <doc_mode> [block_lines]
squeue -u $USER
tail -f $PROJECT/runs/slurm/build-m2d2-*.out
```

The job runs on `mit_normal` with 16 CPUs, 64 GB and 2 hours. It writes `$PROJECT/data_m2d2w8/`:

- `train.bin` (~120M tokens, shuffled at document level);
- `{test,val,calib}_<domain>.bin`;
- `splits_manifest.jsonl`, `meta.json`, `sanity.txt`.

It refuses to overwrite an existing build, so move the old folder away first if you need to rebuild. If a domain runs out of documents before its training quota fills, the build stops with an error naming it.

Then check the result:

```bash
cd $PROJECT/data_m2d2w8
python -c "
import json; m = json.load(open('meta.json'))
print('train tokens', m['train_tokens_total'], 'doc_mode', m['doc_mode'], 'revision', m['dataset_revision'])
for k in ('train_share_by_source', 'val_tokens_by_source', 'calib_tokens_by_source', 'test_tokens_by_source'): print(k, m[k])
print('dedup', json.dumps(m['dedup'], indent=1))
print('whitespace-only duplicates dropped', m['whitespace_only_duplicates_dropped'])
print('cross-domain duplicates by pair', m['cross_domain_duplicates_by_pair'])
print('skipped by length', m['skipped_by_length'])"
cat sanity.txt | grep -E "^OK|^=====|^[a-z_]+ +[0-9]" | head -60
wc -l splits_manifest.jsonl
cd $PROJECT/code
```

What to look for:

- `train_tokens_total` about 120M, with each domain's share about 0.125.
- Per domain: test ≥ 1.5M, val ≥ 1.5M, calib ≥ 1.0M and train ≥ 15M tokens, each over by less than one document.
- `sanity.txt` says `OK: none of the … doc_hashes appears in two splits or two domains`.
- The cross-domain pair counts look plausible: related categories overlap most, e.g. `natural_sciences|technology`.

## 5. LR check, chained into the main runs

```bash
scripts/slurm/submit_m2d2_lr_check.sh b200
```

This submits the 6 `m2d2-lr-*` jobs (00:20:00 each), then a dependent CPU job, `m2d2-chain`. When all 6 have ended, that job:

1. runs `analyze.py --prefix m2d2- --lr_check --write --lr_file config/chosen_lr_m2d2.py`. The rule: lowest final calibration loss, and if another LR is within 0.01 of it, the lowest such LR.
2. prints the chosen LRs (also in `$PROJECT/runs/slurm/m2d2-chain-<jobid>.out`).
3. submits the 12 main runs (00:40:00 each) in this order: m2d2-base-s0, m2d2-win1-s0, m2d2-win16-s0, the base/win16 pairs for s1–s4, then m2d2-win64-s0.

If the analysis fails (for example, no LR run of an arm finished), it submits nothing and exits non-zero. Fix the cause, then rerun the analysis and submit by hand:

```bash
python analysis/analyze.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/m2d2_lr_check --prefix m2d2- --lr_check --write --lr_file config/chosen_lr_m2d2.py
bash scripts/slurm/submit_m2d2_lr_check.sh --chain b200   # on the login node; it only analyzes and submits
```

## 6. SlimPajama control

```bash
scripts/slurm/submit_slim120.sh b200      # 6 jobs, 00:40:00 each, base/win16 pairs adjacent
```

These use the SlimPajama data, LR 5e-4 for both arms and the 120M budget. They can be submitted at any time; at most 2 jobs run at once.

## 7. Monitor

```bash
scripts/slurm/status.sh                                         # every run folder + squeue
python analysis/analyze.py --runs_dir $PROJECT/runs --prefix m2d2- --partial      # interim per-arm comparison
python analysis/analyze.py --runs_dir $PROJECT/runs --prefix slim120- --partial
```

## 8. Analysis

```bash
python analysis/analyze.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/m2d2 --prefix m2d2-
python analysis/analyze.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/slim120 --prefix slim120-
python analysis/compare_datasets.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/compare
cat $PROJECT/analysis/compare/compare_summary.md
```

Make every decision and finish the write-up of these results **before** step 9.

## 9. Only then: the locked test

```bash
scripts/slurm/submit_locked_test.sh b200       # asks: the locked test is used once; continue? [y/N]
# when the job is done:
cat $PROJECT/analysis/m2d2_locked_test/summary.md
cat $PROJECT/analysis/locked_test_log.txt
python analysis/compare_datasets.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/compare_with_test --with_test
```

The job evaluates all three designs in one invocation: m2d2-base/win16-s0…s4, m2d2h4-base/win16-s0…s3 and m2d2r4-base/win16-s0…s3. Each run reads `test_<domain>.bin` from its own build, after a SHA-256 check that those files are identical to `data_m2d2w8/`'s, with the same evaluation code and offsets as the final eval. It refuses if any of the 26 runs has no checkpoint. Summaries are per design. It writes `<run>/test_results.json` and appends to the log. A second attempt is refused. `--allow_reuse_of_locked_test` overrides that, but the reuse is logged and every summary is marked "TEST SET REUSED".

## 10. What each output means

| Output | Meaning |
|---|---|
| `data_m2d2w8/meta.json` | Token counts per split and domain, dedup statistics (exact, whitespace-only, cross-domain per pair), length-filter skips, doc_mode, revision, split salt, sampling note |
| `data_m2d2w8/splits_manifest.jsonl` | One line per kept document: doc_hash, domain, split, n_tokens. Audits every assignment |
| `data_m2d2w8/sanity.txt` | File sizes, max token id, a decoded sample per file, the no-overlap confirmation and per-domain totals |
| `analysis/m2d2_lr_check/` + `config/chosen_lr_m2d2.py` | The LR-check table (calibration-split losses) and the chosen LRs |
| `analysis/m2d2/summary.md` (and `slim120/`) | `analyze.py`'s decision rule on that group: verdict, mean(B), mean(W), gap = mean(B) − mean(W) (positive = window better), s, and the win1/win64 rows. With `--prefix`, criteria whose runs don't exist are skipped and listed (slim120 has no win1/win64). Also `per_source.*`, `sanity.png`, `main_comparison.png`, `window_sweep.png`, `loads.png`, `routing_heatmap.png` and `routing_mi.csv`, as in the pilot |
| `analysis/compare/compare_summary.md` | One row per dataset (SlimPajama-500M, SlimPajama-120M, M2D2-Wiki-8): mean base, mean win16, gap = win16 − base (**positive = window worse**; note the sign is the opposite of `analyze.py`'s), s, perplexity % (e^gap − 1), seeds per arm, paired wins (seeds where win16 < base), sanity (win1-s0 − base-s0), the LRs used and the verdict. Then the main comparison Δ = gap_M2D2 − gap_Slim120 with scale √(s_M²/n_M + s_S²/n_S): "penalty clearly smaller on M2D2-Wiki-8" or "clearly larger" (beyond 2 × scale), or "not distinguishable". With `--with_test`, the M2D2 locked-test row and the validation-vs-test gap |
| `analysis/compare/compare_per_domain.md` | M2D2 gap per domain (mean, seed spread), plus SlimPajama per-source gaps at 120M and 500M for reference (and the test split with `--with_test`) |
| `analysis/compare/compare_gap_curves.png` | Gap (win16 − base) over training for M2D2 and Slim120, same budget |
| `analysis/compare/compare_mi.png` | Mean source–expert mutual information per layer, base vs win16, for each dataset. A topic router shows high MI in the window layers |
| `analysis/m2d2_locked_test/summary.md`, `<run>/test_results.json` | The locked-test verdict and gap, with per-domain test gaps |
| `analysis/locked_test_log.txt` | Every attempt: START, COMPLETED, REFUSED or REUSE, with timestamp, git commit, host and runs |

---

## Troubleshooting

- **The download fails with a 401/404:** check the dataset name with `python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('machelreid/m2d2').sha)"`. If the dataset requires accepting a license on the Hugging Face website, accept it, then `huggingface-cli login`.
- **"domain … ran out of documents":** the chosen `--doc_mode` produced too few documents of 64–50,000 tokens. Re-inspect, and in `block` mode try another `--block_lines`.
- **"already holds a build":** move `$PROJECT/data_m2d2w8` away before rebuilding. Never rebuild after any M2D2 run has started.
- **The chain job submitted nothing:** read `$PROJECT/runs/slurm/m2d2-chain-<jobid>.out`; the LR-check analysis failed. Then use the commands in step 5.
- **"LR_BASE / LR_WIN … are not both set":** the M2D2 LR check hasn't written `config/chosen_lr_m2d2.py` yet (step 5).
- **`train.py` says eval_split 'test' is refused:** intended. Only `eval_locked_test.py` reads the test split.
- **Plain `analyze.py` on `$PROJECT/runs` now lists m2d2/slim120 runs** in its per-run tables. That's its usual behavior; the pilot's verdict still uses only the unprefixed base/win16 runs. Use `--prefix` for the new groups.

---

## Uneven-domain designs: everything in one go

The spec is `docs/M2D2_UNEVEN_PROMPT.md`.

**Hypothesis:** topic-routed experts need a lot of data from their topic before they learn useful subject knowledge. At 15M tokens per domain they can't, so window routing loses. If one domain gets 4× the data (60M instead of 15M tokens), the window arm's penalty *on that domain* should shrink more than on the others.

| Design | Data folder | Boosted domain | Runs | Budget |
|---|---|---|---|---|
| even | `data_m2d2w8/` | none (15M train tokens per domain) | m2d2-* (LR check, base/win16 s0–s4, win1, win64) | 120M tokens |
| hist4x | `data_m2d2w8_hist4x/` | history | m2d2h4-base/win16-s0…s3 | 165M tokens (2,518 steps) |
| rep4x | `data_m2d2w8_rep4x/` | philosophy (natural_sciences if philosophy runs out) | m2d2r4-base/win16-s0…s3 | 165M tokens |

The uneven builds use exactly the even build's pipeline:

- Every `test_*`, `val_*` and `calib_*` file is byte-identical to the even build's (checked in each build's `sanity.txt`, and again before the locked test).
- Every non-boosted domain has the even build's training documents.
- The boosted domain has the even build's documents followed by more, in the same order.

The uneven runs use the LRs the even design's LR check picks (`config/chosen_lr_m2d2.py`).

**The primary test** compares hist4x with rep4x. They have the same budget, schedule and LRs, and differ only in which domain was boosted.

### U1. Pull on Engaging

```bash
cd $PROJECT/code
module load miniforge && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate moe
git stash && git pull && git stash pop
ls config/runs | grep -cE "^m2d2[hr]4-"        # 16
```

If the archives aren't downloaded yet, do steps 2 (download) and 3 (inspect) above first. The builds use `--doc_mode title` by default, the blank / title / blank / body layout of M2D2's files. Check that inspect prints `RECOMMENDATION: --doc_mode title` and that the title-mode token lengths look like whole articles (p50 in the hundreds or more). If inspect recommends another mode, add `--doc_mode <mode>` to both commands below. All three builds always use the same mode.

### U2. Dry run first

```bash
scripts/slurm/submit_m2d2_all.sh b200 --dry_run
```

Nothing is submitted, and the job IDs it prints (900, 901, …) are placeholders. Check that:

- **The build lines are right.** A folder that already holds a build that passed its checks says "skipping". Otherwise there is one `build-m2d2-even`, `-hist4x` and `-rep4x` line each, ending in `build_m2d2.sbatch title …`.
- **The uneven builds are configured correctly:**
  - hist4x carries `--train_quota_override history=60000000`;
  - rep4x carries `philosophy=60000000 --fallback_domain natural_sciences`;
  - both carry `--reference_dir …/data_m2d2w8`;
  - both have `--dependency=afterok:<even build>` when the even build is also being submitted.
- **The rest is queued in the right order:**
  - 6 `m2d2-lr-*` jobs (after the even build, if one is submitted);
  - one `m2d2-after` CPU job with `--dependency=afterany:<the 6 LR jobs>` and arguments `b200 <hist4x build id or none> <rep4x build id or none>`;
  - 6 `slim120-*` jobs.
- **The planned main runs are right.** The final block, "the after job will then submit", lists 28 runs in the order m2d2 s0 (base, win1, win16), hist4x s0, rep4x s0, then the three designs' s1, s2 and s3 pairs, then m2d2 s4 and m2d2-win64-s0. Uneven runs show `--dependency=afterok:<their build>`; the time limits are 00:40:00 (even) and 00:50:00 (uneven).

### U3. Submit everything

```bash
scripts/slurm/submit_m2d2_all.sh b200       # add --doc_mode <mode> if inspect recommended another mode
```

It prints every job ID and the dependency tree. What happens next:

1. The builds run on `mit_normal`: 16 CPUs, 64 GB, up to 3 hours each.
2. The LR runs follow the even build.
3. The `m2d2-after` CPU job then:
   - writes `config/chosen_lr_m2d2.py` and prints the chosen LRs (in `$PROJECT/runs/slurm/m2d2-after-<id>.out`);
   - submits the 28 main runs.

   An uneven design whose build failed has its runs skipped. The job then exits non-zero after submitting the rest. Fix the build, rerun `submit_m2d2_all.sh b200` (finished builds are skipped, and finished runs exit immediately), or submit the skipped runs by hand with `scripts/slurm/submit.sh <run> b200 00:50:00 <run>`.

### U4. After the builds: check them

```bash
for v in "" _hist4x _rep4x; do
  d=$PROJECT/data_m2d2w8$v; echo "== $d"
  python -c "
import json; m = json.load(open('$d/meta.json'))
for k in ('doc_mode', 'variant', 'boosted_domain', 'requested_boosted_domain', 'fallback_used', 'train_quota_by_source', 'train_tokens_by_source'): print(k, m.get(k))"
  grep -E "^(OK|FAILED|SKIPPED):" $d/sanity.txt
done
```

What to look for:

- **hist4x:** `boosted_domain` is `history`, `fallback_used` is false, and history has about 60M train tokens against about 15M for every other domain.
- **rep4x:** `boosted_domain` is `philosophy` (or `natural_sciences` with `fallback_used` true; everything downstream reads this from `meta.json`).
- **Uneven builds:** `sanity.txt` has both `OK: none of the …` and `OK: all test/val/calib files are byte-identical to the reference build …`.
- **Everything:** `scripts/slurm/status.sh` lists all three builds as "checks passed".

If a build failed, its dependent jobs sit in the queue with the reason `DependencyNeverSatisfied`. If the even build failed, that includes the LR runs and `m2d2-after`. Remove them with `scancel <ids>`, read `$PROJECT/runs/slurm/build-m2d2-*.out`, fix the cause, move the failed folder away, and rerun `submit_m2d2_all.sh b200`.

### U5. Monitor, and an early read

```bash
scripts/slurm/status.sh
# once s0 and s1 of every design have finished (it says PARTIAL until everything is done):
python analysis/uneven.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/uneven_early
head -20 $PROJECT/analysis/uneven_early/uneven_summary.md
```

### U6. When everything has finished

```bash
python analysis/analyze.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/m2d2 --prefix m2d2-
python analysis/analyze.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/slim120 --prefix slim120-
python analysis/compare_datasets.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/compare
python analysis/uneven.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/uneven
cat $PROJECT/analysis/uneven/uneven_summary.md
```

To analyze copies on the laptop, add `--data_root <folder holding the data_m2d2w8_* builds>` to `uneven.py`. It only needs their `meta.json`.

### U7. Only then: the locked test, once, for all three designs

```bash
scripts/slurm/submit_locked_test.sh b200      # 26 runs; asks: the locked test is used once; continue? [y/N]
# when the job is done:
cat $PROJECT/analysis/m2d2_locked_test/summary.md
python analysis/compare_datasets.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/compare_with_test --with_test
python analysis/uneven.py --runs_dir $PROJECT/runs --out_dir $PROJECT/analysis/uneven_with_test --with_test
```

Before evaluating, the job checks that every run's `test_<domain>.bin` files are byte-identical to `data_m2d2w8/`'s, and refuses if any differs. It logs once, and refuses a second use unless you pass `--allow_reuse_of_locked_test`.

### U8. What each verdict line in `uneven_summary.md` means

Throughout, gap = win16 − base per domain, paired by seed, and **positive means the window arm is worse**. The hypothesis predicts *negative* values below.

- **Primary (swap contrast):**
  - **What it measures:** B = mean over seeds of (gap on history − gap on rep4x's boosted domain) in hist4x, minus the same in rep4x.
  - **"supports the hypothesis"** (B < −2·SE): each of the two domains has a smaller penalty in the design where it got 4× the data. The budget, schedule and LRs are identical, so this is the cleanest test.
  - **"contradicts it"** (B > +2·SE): the penalty is *larger* where the data was added.
  - **"not distinguishable":** within ±2·SE. Read it next to the minimum detectable effect (2·SE) and B as a fraction of the even design's mean gap; together they say how big an effect this many seeds could have detected.
- **Difference-in-differences vs even, per uneven design:**
  - **What it measures:** the boosted domain's gap minus the mean of the other domains' gaps, uneven design minus even design, read with the same rule.
  - **Caveat:** the budgets differ (165M vs 120M tokens). Subtracting the other domains removes shifts common to every domain, but not effects that scale with a domain's gap size.
  - **Supporting numbers:** the ratio table (boosted vs non-boosted domains) and the **placebo rank**. A rank of 1 of 8 means the boosted domain's gap shrank the most of all domains.
- **Crossover:**
  - "yes: win16 beats the baseline on <domain>" means the boosted domain's gap is below 0 by more than 2·SE: with 4× data the window router actually wins there.
  - "no: the baseline is still better" means the gap stays above 0 by more than 2·SE.
  - Otherwise, no clear difference.
- **Absolute losses:** how much each arm's loss on the boosted domain improved from even to uneven, and how much more win16 gained than the baseline.
- **PARTIAL:** some configured runs haven't finished. The line gives the seed counts per design, and the numbers use the finished seeds only.

**Other outputs:**

- `uneven_per_domain.md`: every domain × design, with mean gap, seed spread and base/win16 losses.
- `uneven_delta.png`: each domain's change in gap from even to each uneven design, with the boosted domain in red.
- `uneven_swap.png`: per-seed swap contrasts.
- `uneven_expert_usage.png`: source–expert mutual information and the effective number of experts per layer, base vs win16, per design.
- `uneven_results.json`: every number.

With `--with_test`, everything is repeated on the locked test split, including expert usage from the test-split routing counts.
