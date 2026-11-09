# M2D2-Wiki-8 replication: runbook

This runbook replicates the per-token vs window-routing comparison on **M2D2-Wiki-8**: eight top-level Wikipedia categories from M2D2. The spec is `docs/M2D2_PROMPT.md`.

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

The job evaluates m2d2-base-s0…s4 and m2d2-win16-s0…s4 on `test_<domain>.bin`, with the same evaluation code and offsets as the final eval. It writes `<run>/test_results.json` and appends to the log. A second attempt is refused. `--allow_reuse_of_locked_test` overrides that, but the reuse is logged and every summary is marked "TEST SET REUSED".

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
