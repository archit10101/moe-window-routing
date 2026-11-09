# Follow-up prompt: replicate the comparison on M2D2-Wiki-8

## Part 1: Context and goal

The pilot in this repo (spec: `docs/PROMPT.md`) compared per-token routing (baseline) with routing layers 5–8 on a 16-token causal window mean (win16), on SlimPajama-6B. Result: NEGATIVE. win16 was worse by 0.0446 in validation cross-entropy (seed spread s = 0.0075, 5 seeds per arm, sanity check passed). The penalty was smallest on distinctive sources (github +0.011, arxiv +0.016) and largest on general web text.

**Goal:** run the same comparison on a second dataset, **M2D2-Wiki-8**, then compare against SlimPajama.

M2D2-Wiki-8 is an eight-domain subset of M2D2's Wikipedia portion ([paper](https://arxiv.org/abs/2210.07370), [data](https://huggingface.co/datasets/machelreid/m2d2), license CC-BY-NC-4.0). Every domain is Wikipedia, so writing style and formatting are held constant. A router can't specialize just by recognizing that a document looks like code, a forum or a paper; any specialization has to come from topic. That makes it a cleaner test of whether topic-routed experts help.

## Part 2: Ground rules

1. **Nothing about the existing SlimPajama runs changes.** Existing configs, run folders, `config/chosen_lr.py` and the default behavior of every script stay exactly as they are, and every existing test passes. New runs use the `m2d2-` or `slim120-` name prefixes and new data folders.
2. **Same training recipe as the main pilot** (model, optimizer, schedule shape, balancing, batch 64 × 1,024), except the token budget, which matches this dataset's size (Part 6).
3. No router noise, capacity limits or token dropping. Compare only on pure cross-entropy. Never weaken or delete a test. Only torch, numpy, tiktoken, pyarrow, matplotlib, pytest and the standard library, plus `huggingface_hub` for downloading only.
4. **The locked test split is evaluated exactly once,** after every decision is made (Part 7). Nothing (LR choice, monitoring, analysis code paths) reads `test_*` files before that step.
5. You can't reach the network, the cluster or a GPU from here. Test everything offline with small fake archives.
6. If anything conflicts with the current code, stop and ask.

## Part 3: Fixed facts about M2D2

- **Repository layout:** `data/<Config>.tar.gz`. Each archive extracts to `<Config>/train.txt`, `<Config>/valid.txt` and `<Config>/test.txt`: plain UTF-8 text, one example per line. The repo's loading script (`m2d2.py`) strips each line and yields it as `{"text": line}`. Modern `datasets` won't run loading scripts, so download the archives directly with `huggingface_hub` (`hf_hub_download`, `repo_type="dataset"`).
- **Each top-level category archive already contains its subcategories.** For example, `Culture_and_the_arts.tar.gz` (753 MB) roughly equals the sum of its subcategory archives. Use only the top-level archives:

| Domain name (source) | M2D2 config / archive | Compressed size |
| --- | --- | --- |
| `technology` | `Technology_and_applied_sciences` | 704 MB |
| `natural_sciences` | `Natural_and_physical_sciences` | 463 MB |
| `mathematics` | `Mathematics_and_logic` | 119 MB |
| `health` | `Health_and_fitness` | 280 MB |
| `history` | `History_and_events` | 547 MB |
| `culture` | `Culture_and_the_arts` | 753 MB |
| `society` | `Society_and_social_sciences` | 886 MB |
| `philosophy` | `Philosophy_and_thinking` | 385 MB |

  That's about 4.1 GB compressed in total. **Philosophy note:** M2D2 also has a separate config named `Philosophy` (321 MB). Use `Philosophy_and_thinking`, the top-level Wikipedia category that matches the other seven. Also download `Philosophy` in inspect mode only, and report how many of its normalized documents also appear in `Philosophy_and_thinking`.
- **Unknown:** whether a line is a whole article or a paragraph, and how articles are delimited. This must be measured (Part 4, inspect mode), not assumed.
- **Project folder on Engaging:** `/home/architta/orcd/scratch/moe-pilot` (below, `PROJECT`). Raw archives go in `PROJECT/raw_m2d2/`, prepared data in `PROJECT/data_m2d2w8/`. The GPT-2 tokenizer is cached in `PROJECT/tiktoken_cache`; set `TIKTOKEN_CACHE_DIR`. Compute nodes may not reach the internet, but the login node can.
- **Hardware:** `mit_normal_gpu`, at most 2 GPUs per user, one run per B200 job, about 480k tokens/s. CPU jobs use `mit_normal`. Environment: `module load miniforge` plus conda env `moe`.

## Part 4: Data preparation (`data/m2d2/prepare.py`)

Output format identical to `data/slimpajama/prepare.py`: GPT-2 BPE via tiktoken, end-of-text token 50256 after each document, flat uint16 `.bin` files, `meta.json`, `sanity.txt`. The script has three modes.

**1. `--download --raw_dir <dir>`** (login node): download the 8 archives plus `Philosophy.tar.gz`, then extract them. Record the repo commit hash in `raw_dir/REVISION.txt`.

**2. `--inspect --raw_dir <dir>`** (login node, fast): for each config, print:
- line count per file
- the line-length distribution in characters (percentiles 5, 50, 95, 99)
- the fraction of empty lines
- the fraction of lines that look like headings (for example wikitext `= Title =`, or a short line followed by a long one)
- the first 8 lines of `train.txt`, each truncated to 200 characters

Then print a recommendation for `--doc_mode`:
- `line`: each line is a whole article (long lines, no blank lines, no heading patterns).
- `blank`: articles separated by blank lines.
- `heading`: a new article starts at a top-level heading line.
- `block`: fallback when boundaries can't be recovered. Consecutive groups of K lines (`--block_lines`, default 40) are treated as one unit. Print a clear warning that whole-article integrity isn't guaranteed.

Also print the `Philosophy` vs `Philosophy_and_thinking` overlap.

**3. `--build --raw_dir <dir> --out_dir <dir> --doc_mode <mode> [--block_lines K] [--threads N]`** (CPU job, no network):

1. **Pool the documents.** For each of the 8 domains, read `train.txt`, `valid.txt` and `test.txt` together, ignoring M2D2's own splits, and form documents according to `--doc_mode`.
2. **Normalize and hash.** Normalize each document: Unicode NFC, collapse runs of whitespace, strip. Then doc_hash = SHA-256 of the normalized text.
3. **Deduplicate.**
    - Within a domain: keep one copy per doc_hash.
    - Across domains: a doc_hash appearing in more than one of the 8 domains is ambiguous. Drop it from all of them, and report counts per domain pair.
4. **Order deterministically.** Within each domain, compute u = the first 8 bytes of SHA-256(`"m2d2w8-v1:" + doc_hash`) as an integer, and sort ascending. The order depends only on content, not on file order.
5. **Fill the splits in this order, whole documents only:**

| Split | Token quota per domain | File written |
| --- | --- | --- |
| test (locked) | 1,500,000 | `test_<domain>.bin` |
| validation | 1,500,000 | `val_<domain>.bin` |
| calibration | 1,000,000 | `calib_<domain>.bin` |
| train | 15,000,000 | part of `train.bin` |

   Counts are GPT-2 tokens including end-of-text tokens. Walk the sorted documents and add each whole document to the current split until that split reaches its quota; the last document may overshoot, but never truncate a document. Tokenize only the documents you assign. Stop the domain once train is filled, and stop with a clear error if a domain runs out first.

   Skip documents shorter than 64 tokens and longer than 50,000 tokens, and report how many were skipped. The short ones are mostly stubs; the long-document cap keeps any one article from dominating a split.
6. **Write `train.bin`:** the 8 domains' training documents, shuffled at the document level with a fixed seed (12345). That's about 120M tokens.
7. **Write `splits_manifest.jsonl`:** one line per kept document with doc_hash, domain, split and n_tokens, so every assignment can be audited.
8. **Write `meta.json`.** The same keys as SlimPajama's, so `train.py` works unchanged (`train_tokens_total`, `train_share_by_source`, `val_tokens_by_source`, `eot_token`, dtype, created). Add:
    - `calib_tokens_by_source` and `test_tokens_by_source`
    - doc counts per split and domain
    - the dedup and skip statistics
    - doc_mode and block_lines
    - the archive list, revision, `"dataset": "m2d2-wiki-8"`, and the split-salt string
9. **Sanity checks** in `sanity.txt`:
    - file sizes and maximum token id
    - a decoded sample from every file
    - confirmation that no doc_hash appears in two splits or two domains
    - per-domain token totals per split

Add `data/m2d2/build_m2d2.sbatch`: partition `mit_normal`, 16 CPUs, 64 GB, 2 hours. It runs build mode with `--doc_mode` passed as an argument, so the inspect result decides it.

## Part 5: Small code change: which split to evaluate on

Add config key `eval_split`, default `"val"`. `train.py` loads `<eval_split>_<source>.bin` for its periodic and final evaluations; everything else is unchanged. With the default, behavior is bit-for-bit identical to now. Only `"val"` and `"calib"` are allowed: refuse `"test"` with a clear message (Part 2, rule 4).

## Part 6: Runs

**Token budget:** about 120M training tokens exist, so main runs train for 1 epoch, `total_tokens = 120,000,000` (1,832 steps). LR-check runs use 30M tokens. All runs use `eval_interval = 100` and the same everything else as the main pilot.

- `config/pilot_m2d2.py`: loads `config/pilot_base.py`, then sets `data_dir = PROJECT/data_m2d2w8`, the token budget and the eval interval above.
- `config/chosen_lr_m2d2.py`: `LR_BASE = None`, `LR_WIN = None`. M2D2 main configs read only this file and stop with a clear message while either is None.

| Run names | window_layers | seed | lr | eval_split | Purpose |
| --- | --- | --- | --- | --- | --- |
| m2d2-lr-base-2.5e-4, -5e-4, -1e-3 | [0]*8 | 0 | as named | calib | LR check, baseline |
| m2d2-lr-win16-2.5e-4, -5e-4, -1e-3 | [0,0,0,0,16,16,16,16] | 0 | as named | calib | LR check, window |
| m2d2-base-s0 … s4 | [0]*8 | 0–4 | LR_BASE | val | Main, baseline |
| m2d2-win16-s0 … s4 | [0,0,0,0,16,16,16,16] | 0–4 | LR_WIN | val | Main, window |
| m2d2-win1-s0 | [0,0,0,0,1,1,1,1] | 0 | LR_BASE | val | Sanity check |
| m2d2-win64-s0 | [0,0,0,0,64,64,64,64] | 0 | LR_WIN | val | Window sweep |
| slim120-base-s0 … s2 | [0]*8 | 0–2 | 5e-4 | val | SlimPajama control at the same 120M budget |
| slim120-win16-s0 … s2 | [0,0,0,0,16,16,16,16] | 0–2 | 5e-4 | val | SlimPajama control at the same 120M budget |

The `slim120-` runs use the SlimPajama data and the SlimPajama LR (5e-4 for both arms) with the 120M budget. The cross-dataset comparison then compares like with like: the existing SlimPajama gap was measured at 500M tokens with a different schedule length.

Each 120M run takes about 5–8 minutes on a B200. All 24 runs fit in about 1.5 hours at 2 GPUs.

## Part 7: Analysis

**`analysis/analyze.py`**, two new options (behavior unchanged without them):
- `--prefix <p>`: only runs whose names start with `p`, with the prefix stripped when matching the existing name patterns. It works for `m2d2-` and `slim120-`. If win1-s0 or win64-s0 is missing, the decision rule uses the available criteria and `summary.md` says which ones were skipped.
- `--lr_file <path>`: where `--lr_check --write` writes the chosen LRs (default unchanged: `config/chosen_lr.py`). For M2D2 the LR-check numbers are calibration-split losses; label them that way in the output.

**New `analysis/eval_locked_test.py`:** evaluates the final checkpoints of the given runs on `test_<source>.bin`, using the same evaluation code and offset scheme as the final eval but with the test files. It writes `test_results.json` per run, plus a summary using the same decision-rule quantities as `analyze.py`. Every invocation is logged to `PROJECT/analysis/locked_test_log.txt` with a timestamp, the git hash and the run list, so the single use of the test split is on record.

**New `analysis/compare_datasets.py`:** usage `python analysis/compare_datasets.py --runs_dir <dir> --out_dir <dir> [--with_test]`. Everything is computed from logs; nothing is hard-coded.

1. **`compare_summary.md`:**
    - One row each for SlimPajama-500M (existing runs), SlimPajama-120M (`slim120-`) and M2D2-Wiki-8 (`m2d2-`): mean base, mean win16, gap, s, gap in perplexity % (e^gap − 1), seeds per arm, paired wins, sanity difference (where it exists), chosen LRs, and verdict.
    - With `--with_test`, the M2D2 test-split gap and verdict next to the validation ones.
    - **The main comparison:** Δ = gap_M2D2 − gap_Slim120, with a rough scale of √(s_M2D2²/n_M2D2 + s_Slim120²/n_Slim120). A plain-language line: "penalty clearly smaller on M2D2-Wiki-8", "clearly larger", or "not distinguishable", where clear means beyond 2 × the scale.
2. **`compare_per_domain.md`:** M2D2 per-domain gaps (8 rows, mean and seed spread), and the SlimPajama per-source gaps for reference.
3. **`compare_gap_curves.png`:** gap vs training tokens for M2D2 and Slim120 (same budget), on one plot.
4. **`compare_mi.png`:** mean source–expert mutual information per layer, base vs win16, for M2D2 (over its 8 domains) and SlimPajama.

## Part 8: Launchers (`scripts/slurm/`)

- **`submit_m2d2_lr_check.sh <gpu-type>`:** submits the 6 M2D2 LR runs (1 run per job, time 00:20:00), then a dependent CPU job (`mit_normal`, no requeue, `afterany`) that:
  1. runs `analyze.py --prefix m2d2- --lr_check --write --lr_file config/chosen_lr_m2d2.py`
  2. prints the chosen LRs
  3. submits, as 1-run jobs (time 00:40:00), in this order: m2d2-base-s0, m2d2-win1-s0, m2d2-win16-s0, then the remaining base/win16 pairs s1–s4, then m2d2-win64-s0

  If the analysis fails, it submits nothing and exits non-zero.
- **`submit_slim120.sh <gpu-type>`:** the 6 `slim120-` runs, pairs adjacent (time 00:40:00).
- **`submit_locked_test.sh <gpu-type>`:** one 1-GPU job running `eval_locked_test.py` on m2d2-base-s0…s4 and m2d2-win16-s0…s4. Before submitting, it asks for interactive confirmation ("the locked test is used once; continue? [y/N]").
- **`status.sh`:** also lists the new runs.

## Part 9: Tests (CPU, pytest; add to the existing suite)

Build small fake `.tar.gz` archives with the M2D2 layout (`<Config>/train.txt`, `valid.txt`, `test.txt`) in a temp folder.

| Test | What must hold |
| --- | --- |
| `test_m2d2_doc_modes` | `line`, `blank`, `heading` and `block` each group fake files into the expected documents |
| `test_m2d2_deterministic` | Building twice gives byte-identical outputs and manifest, and shuffling line order within the input files (in line mode) gives the same split assignment per doc_hash |
| `test_m2d2_no_cross_split_or_domain` | No doc_hash appears in two splits or two domains; cross-domain duplicates are dropped and counted; within-domain duplicates are kept once |
| `test_m2d2_whole_documents` | Every document's tokens appear contiguously in exactly one split file, never truncated |
| `test_m2d2_quotas` | Each split reaches its quota, with overshoot under one document; a domain that runs short stops with an error naming it; length filters are applied and counted |
| `test_m2d2_meta_compatible` | `train.py` (tiny config) trains and evaluates on the built fake data with `eval_split` val and calib |
| `test_eval_split_default_and_test_refused` | The default `eval_split` is bit-identical to the current behavior; `"test"` is refused |
| `test_m2d2_and_slim120_configs` | Every config matches the Part 6 table, M2D2 configs use `data_m2d2w8` and `chosen_lr_m2d2.py`, and `slim120-` configs use the SlimPajama data and 5e-4 |
| `test_analyze_prefix_and_default_unchanged` | Without the new options, `analyze.py` output on the existing fake-runs fixture is byte-identical to before; `--prefix` selects correctly; `--lr_file` leaves `config/chosen_lr.py` untouched |
| `test_locked_test_eval_and_log` | `eval_locked_test.py` on fake runs writes results and appends to the log |
| `test_compare_datasets_runs` | On fake relabeled runs for all three groups, every output is written and the Δ line appears |

The whole suite must still pass in under 2 minutes on CPU, with no skips.

## Part 10: Deliverables, working style and definition of done

**Deliverables:** `data/m2d2/prepare.py`, `data/m2d2/build_m2d2.sbatch`, the `eval_split` change, the configs, the `analyze.py` options, `analysis/eval_locked_test.py`, `analysis/compare_datasets.py`, the launchers, the tests, and `README_M2D2.md` with exact commands in order:

1. Pull on Engaging.
2. Download on the login node.
3. Inspect, and how to choose `--doc_mode` from the inspect report (including what to do if it recommends `block`).
4. Submit the build job, then check `meta.json`, `sanity.txt` and the dedup report.
5. `submit_m2d2_lr_check.sh b200`, which chains into the main runs.
6. `submit_slim120.sh b200`.
7. Monitor with `status.sh`.
8. Run `analyze.py --prefix m2d2-` and `--prefix slim120-`, then `compare_datasets.py`.
9. Only then, `submit_locked_test.sh` and `compare_datasets.py --with_test`.
10. What each output means.

**Working style:** read `docs/PROMPT.md`, `data/slimpajama/prepare.py`, `train.py` and the analysis code first. Propose a plan and wait for approval. Build in this order, testing after each stage: prep script (all three modes) → its tests → `eval_split` → configs → analysis changes → locked-test script → compare script → launchers → README. Commit after each stage; don't push.

**Definition of done:**

- [ ] `pytest -q` passes on CPU in under 2 minutes, with no skips, including every old test.
- [ ] `bash -n` passes for every shell script.
- [ ] With defaults, `train.py` and `analyze.py` behave exactly as before.
- [ ] A final summary of what you built, test results, deviations from this spec, and anything I need to decide.
