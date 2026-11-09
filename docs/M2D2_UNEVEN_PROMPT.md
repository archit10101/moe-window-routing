# Follow-up prompt: M2D2-Wiki-8, even design + uneven-domain design

## Part 1: Context and goal

The repo already contains:
- **Main pilot** (spec: `docs/PROMPT.md`). On SlimPajama, routing layers 5–8 on a 16-token causal window mean (win16) was worse than per-token routing by 0.0446 in validation cross-entropy (s = 0.0075, 5 seeds per arm).
- **Follow-up experiments** (spec: `docs/FOLLOWUP_PROMPT.md`, if present).
- **The M2D2-Wiki-8 pipeline**, implemented and tested but **not yet run**. Its spec is `docs/M2D2_PROMPT.md` if present; otherwise the code is the reference: `data/m2d2/prepare.py`, the `m2d2-` / `slim120-` configs, `analysis/eval_locked_test.py`, `analysis/compare_datasets.py`, `README_M2D2.md` and the M2D2 launchers. Call this the **even design**: 8 Wikipedia domains, 15M training tokens each.

This prompt does three things:
1. Runs the even design exactly as implemented. **Don't change its behavior.**
2. Adds an **uneven-domain design** that tests a specific hypothesis.
3. Adds one launcher that runs everything, plus an analysis script for the new design.

**The hypothesis being tested:** topic-routed experts need a lot of data from their topic before they learn useful subject knowledge. At 15M tokens per domain they can't, so window routing loses. If one domain gets much more data, the window arm's disadvantage *on that domain* should shrink more than on domains whose data didn't change.

**Design:**
- Two extra datasets, identical to the even design except that one domain's training quota is 4× larger (60M instead of 15M):
  - `hist4x`: the boosted domain is `history`.
  - `rep4x`: a replicate. The boosted domain is `philosophy`, with `natural_sciences` as a fallback if philosophy runs out of documents.
- Every other domain's training data, and every domain's test, validation and calibration data, is **byte-identical** to the even design.
- The **primary test** compares `hist4x` with `rep4x`. Both have the same total tokens, schedule length and LRs; they differ only in which domain is boosted.

## Part 2: Ground rules

1. **Nothing existing changes behavior.** The SlimPajama runs, the follow-ups and the even M2D2 design (configs, data folders, run names, launchers, default script behavior) stay exactly as they are, and every existing test passes. New things are additions with new names: run prefixes `m2d2h4-` and `m2d2r4-`, data folders `data_m2d2w8_hist4x/` and `data_m2d2w8_rep4x/`.
2. **Same training recipe as the M2D2 main runs**, apart from the token budget (Part 5).
3. **Same restrictions as before:**
   - No router noise, capacity limits or token dropping.
   - Compare only on pure cross-entropy.
   - Never weaken or delete a test.
   - Same allowed libraries as before.
4. **The locked test split is evaluated exactly once,** in a single invocation covering all three designs, after every decision is made (Part 7). Nothing reads `test_*` files before that.
5. You can't reach the network, the cluster or a GPU from here. Test offline with small fake archives, reusing the existing M2D2 test fixtures.
6. If anything here conflicts with the existing code, or the existing implementation deviates from its spec in a way that matters for this prompt, **stop and ask**.

## Part 3: Step 0: check the existing implementation

Before writing anything:
1. Read `data/m2d2/prepare.py`, the M2D2 configs, launchers, analysis scripts and tests. Run `pytest -q`.
2. Report in your plan:
   - Which `--doc_mode` options exist.
   - How documents are pooled, deduplicated, ordered and assigned to splits.
   - Whether `eval_locked_test.py` already refuses a second use.
   - How the final evaluation stores per-source routing counts (whatever `routing_mi.csv` is computed from).
   - Any deviations from `docs/M2D2_PROMPT.md`.
3. **`--doc_mode title`:** if it doesn't exist, add it. M2D2 articles appear as a blank line, a title line, a blank line, then body lines. A new document starts at a non-blank line that has a blank line (or the start of the file) before it and a blank line after it. The title is the document's first line, and the document runs until the next title. Inspect mode should report how many documents `title` mode forms per domain and their token-length percentiles (5, 50, 95), estimated as characters ÷ 4. Use whichever mode the even design was meant to use. If you can't tell which one that is, ask.

## Part 4: Uneven builds (`data/m2d2/prepare.py`)

Add two options to build mode. Without them, output must be byte-identical to now.

- **`--train_quota_override <domain>=<tokens>`:** replaces that domain's train quota.
- **`--fallback_domain <domain>`:** if the override domain runs out of documents before reaching its train quota, rebuild with the override applied to the fallback domain instead. Record this in `meta.json` and print a clear notice. Without a fallback, running out is an error that names the domain and reports how many train tokens it could supply.

**Properties that must hold** (they make the comparison valid):
1. **Same pipeline:** pooling, normalization, hashing, within- and cross-domain deduplication, length filters, ordering and the fill order (test → validation → calibration → train) are exactly the even design's. Only the boosted domain's train quota differs.
2. **Same evaluation files:** every `test_*`, `val_*` and `calib_*` file is byte-identical to the even build's.
3. **Same training data elsewhere:**
   - **Non-boosted domains:** their train documents are exactly the even build's.
   - **Boosted domain:** its train documents are the even build's train documents for that domain, followed by more documents in the same hash order.
4. **Shuffling:** `train.bin` is shuffled at the document level with the same fixed seed as the even design. Training data necessarily differs in order from the even build; that's expected.

**Additions to `meta.json`:**
- `variant` (`"hist4x"` / `"rep4x"`; the even build has none)
- `boosted_domain` (after any fallback), `requested_boosted_domain`, `fallback_used`
- `train_quota_by_source`
- `eval_files_sha256`: the SHA-256 of every test, validation and calibration file

Add a check line to `sanity.txt` comparing those hashes against the even build (`--reference_dir`) when that build exists.

**Builds to run:**

| Variant | Output folder | Options |
| --- | --- | --- |
| even (existing) | `data_m2d2w8/` | as already implemented |
| `hist4x` | `data_m2d2w8_hist4x/` | `--train_quota_override history=60000000` |
| `rep4x` | `data_m2d2w8_rep4x/` | `--train_quota_override philosophy=60000000 --fallback_domain natural_sciences` |

`build_m2d2.sbatch` passes extra arguments through to build mode. Resources: 16 CPUs, 64 GB, 3 hours.

## Part 5: Runs

**Uneven datasets:** each has about 165M training tokens. Set `total_tokens = 165,000,000` (2,518 steps). Use `eval_interval = 100`, `eval_split = "val"`, and the LRs from `config/chosen_lr_m2d2.py`, the ones the even design's LR check picks. Uneven configs stop with a clear message while either LR is None. Everything else matches the even main runs.

| Run names | Data folder | window_layers | Seeds | LR |
| --- | --- | --- | --- | --- |
| even design (existing) | `data_m2d2w8/` | as implemented (6 LR-check runs, `m2d2-base/win16-s0…s4`, `m2d2-win1-s0`, `m2d2-win64-s0`) | as implemented | as implemented |
| `slim120-` (existing) | SlimPajama | as implemented | 0–2 | 5e-4 |
| `m2d2h4-base-s0 … s3` | `data_m2d2w8_hist4x/` | [0]*8 | 0–3 | LR_BASE |
| `m2d2h4-win16-s0 … s3` | `data_m2d2w8_hist4x/` | [0,0,0,0,16,16,16,16] | 0–3 | LR_WIN |
| `m2d2r4-base-s0 … s3` | `data_m2d2w8_rep4x/` | [0]*8 | 0–3 | LR_BASE |
| `m2d2r4-win16-s0 … s3` | `data_m2d2w8_rep4x/` | [0,0,0,0,16,16,16,16] | 0–3 | LR_WIN |

**New configs:** `config/pilot_m2d2_hist4x.py`, `config/pilot_m2d2_rep4x.py` and `config/runs/m2d2h4-*`, `config/runs/m2d2r4-*`. Follow the existing M2D2 config pattern.

**Timing:** each 165M-token run takes about 8–11 minutes on a B200. The 16 uneven runs add about 2.5 GPU-hours. All 40 runs (6 LR + 12 even + 6 slim120 + 16 uneven) take roughly 3–3.5 hours at 2 GPUs, plus queue time.

## Part 6: Analysis (`analysis/uneven.py`, new)

**Usage:** `python analysis/uneven.py --runs_dir <dir> --out_dir <dir> [--with_test]`.
- Everything is computed from logs and `meta.json`. Nothing is hard-coded, including which domain was boosted.
- It works with whatever seeds are complete, and marks the output **PARTIAL** with seed counts when runs are missing.
- With `--with_test`, it repeats every quantity on the locked-test results.

**Definitions:**
- **Gap:** for design X ∈ {even, hist4x, rep4x}, seed s and domain d, `g[X][s][d] = loss(win16-s, d) − loss(base-s, d)`, using final-eval per-domain validation loss. Base and win16 with the same seed share initialization, which is why gaps are paired by seed.
- **Boosted domains:** h = hist4x's boosted domain (history), r = rep4x's boosted domain (read from `meta.json`).

**1. Primary test: swap contrast** (same budget, same schedule):
- Per seed: `c[X][s] = g[X][s][h] − g[X][s][r]`.
- `B = mean_s c[hist4x][s] − mean_s c[rep4x][s]`.
- `SE = sqrt(var(c[hist4x]) / n_h4 + var(c[rep4x]) / n_r4)`, using sample variances.
- **The hypothesis predicts B < 0:** each domain's gap is smaller in the design where that domain got 4× the data.
- **Verdict:**
  - "supports the hypothesis" if B < −2·SE
  - "contradicts it" if B > +2·SE
  - otherwise "not distinguishable"
- Also print the minimum detectable effect (2·SE), and B as a fraction of the even design's mean per-domain gap, so a null result shows how sensitive the test was.

**2. Secondary test: difference-in-differences vs the even design,** for each uneven design U with boosted domain b:
- Per seed: `k[X][s] = g[X][s][b] − mean_{d≠b} g[X][s][d]`.
- `DiD_U = mean k[U] − mean k[even]`, with SE computed the same way.
- Verdict rule as above. Note in the output that the budgets differ (165M vs 120M tokens); subtracting the other domains' mean removes effects common to all domains, but not effects that scale with each domain's gap size.
- Also report the ratio version: `mean g[U][b] / mean g[even][b]` vs the same ratio for each non-boosted domain.
- **Placebo rank:** `Δ_d = mean g[U][d] − mean g[even][d]` for all 8 domains, and the rank of b (1 = largest shrinkage).

**3. Crossover:** the mean `g[U][b]` ± 2·SE across seeds. Say plainly whether win16 beats the baseline on the boosted domain (< 0 beyond 2·SE).

**4. Absolute losses:** for each arm and boosted domain, how much the domain's loss improved from even to U (mean over seeds). This shows whether the extra data helped win16 more than the baseline.

**5. Expert usage:** for layers 1–8, base vs win16, each design and domain:
- **Effective number of experts:** exp(entropy of P(expert | domain)), counting both top-2 slots.
- **Source–expert mutual information**, as in the existing analysis.

Use the routing counts the final evaluation already stores. If per-domain counts aren't available, stop and ask; don't add a new eval pass without approval.

**Outputs:**
- `uneven_summary.md`: verdict lines first, then the tables for items 1–4.
- `uneven_per_domain.md`: all 8 domains × 3 designs, mean gap, seed spread and base/win16 losses.
- `uneven_delta.png`: Δ_d bars per uneven design, boosted domain highlighted.
- `uneven_swap.png`: per-seed c values for hist4x vs rep4x.
- `uneven_expert_usage.png`
- `uneven_results.json`: every number above.

## Part 7: Locked test: one use, all designs

Extend `scripts/slurm/submit_locked_test.sh` and `eval_locked_test.py` so the single locked-test invocation covers `m2d2-base/win16-s0…s4`, `m2d2h4-base/win16-s0…s3` and `m2d2r4-base/win16-s0…s3`.

**Before evaluating anything,** check that each run's `test_<domain>.bin` files have the same SHA-256 as `data_m2d2w8/`'s. If any differs, refuse.

**Keep the existing safeguards:**
- the interactive confirmation
- the log entry
- refusing reuse without `--allow_reuse_of_locked_test`

Then the README order is: analyze all three designs, then run the locked test once, then `compare_datasets.py --with_test` and `uneven.py --with_test`.

## Part 8: Launchers (`scripts/slurm/`)

**`submit_m2d2_all.sh <gpu-type> [--dry_run]`** (new) submits everything in one go:

1. **Builds.**
   - If `PROJECT/raw_m2d2/` lacks the extracted archives, print the download command from `README_M2D2.md` and exit.
   - For each of even, `hist4x` and `rep4x`: skip the build if `<folder>/meta.json` exists and `sanity.txt` reports all checks passed (and, for uneven folders, the eval-file hash check passed); otherwise submit the build job and record its job ID.
   - The uneven builds use `--reference_dir data_m2d2w8`, so if the even build is also being submitted, they depend on it (`afterok`).
2. **LR check:** the 6 even-design LR runs, dependent (`afterok`) on the even build if one was submitted.
3. **`after_m2d2_all.sh`** (new): a dependent CPU job (`mit_normal`, no requeue, `afterany` on the LR runs). It receives the uneven build job IDs as arguments and:
   1. Runs `analyze.py --prefix m2d2- --lr_check --write --lr_file config/chosen_lr_m2d2.py` and prints the chosen LRs. If this fails, it submits nothing and exits non-zero.
   2. Submits 1-run GPU jobs in this order, so every design gets early seeds before any gets late ones (an early read is possible after s0 and s1):
      - `m2d2-base-s0`, `m2d2-win1-s0`, `m2d2-win16-s0`
      - `m2d2h4-base-s0`, `m2d2h4-win16-s0`, `m2d2r4-base-s0`, `m2d2r4-win16-s0`
      - then for s = 1, 2, 3: the m2d2 pair, the m2d2h4 pair, the m2d2r4 pair
      - then the `m2d2-` s4 pair, then `m2d2-win64-s0`
      
      Uneven runs depend (`afterok`) on their build job, if one was submitted. Time limits: 00:40:00 for even runs, 00:50:00 for uneven runs.
   
   Keep the existing `submit_m2d2_lr_check.sh` unchanged.
4. **slim120:** submit via the existing `submit_slim120.sh`.
5. **Summary:** print every job ID, plus the dependency tree.

With **`--dry_run`**, print every `sbatch` command, including dependencies, and submit nothing.

**`status.sh`:** also lists `m2d2h4-` and `m2d2r4-` runs and the build jobs.

## Part 9: Tests (CPU, pytest; add to the existing suite)

| Test | What must hold |
| --- | --- |
| `test_m2d2_title_mode` (if you added it) | Fake files in blank/title/blank/body layout group into the expected documents; a file with no blank lines forms one document |
| `test_m2d2_override_default_unchanged` | With no new options, build output is byte-identical to before |
| `test_m2d2_override_eval_identical` | With an override, every test/val/calib file is byte-identical to the even build's, and the `eval_files_sha256` sanity check passes |
| `test_m2d2_override_train_prefix` | Non-boosted domains have exactly the even build's train documents; the boosted domain's train documents begin with the even build's train documents for that domain, in the same hash order |
| `test_m2d2_override_runs_out` | Running out with no fallback errors, naming the domain and the available tokens; with a fallback, `meta.json` records `fallback_used` and the fallback domain is boosted |
| `test_uneven_configs` | Every new config matches the Part 5 table and refuses to run while LRs are None |
| `test_uneven_analysis_math` | On fake run logs with known per-domain losses, B, DiD, SEs, placebo ranks and crossover match hand-computed values; an injected boost effect gives "supports", none gives "not distinguishable", and an opposite effect gives "contradicts"; missing seeds give PARTIAL |
| `test_locked_test_all_designs` | The extended locked test covers all three designs, refuses a run whose test files differ, still refuses reuse, and logs once |
| `test_submit_all_dry_run` | With a fake `sbatch` on PATH, `--dry_run` and a real run produce the Part 8 order and dependencies; existing builds are skipped |

The whole suite must still pass in under 2 minutes on CPU, with no skips.

## Part 10: Deliverables, working style and definition of done

**Deliverables:**
- the `prepare.py` options (plus `title` mode if needed) and the `build_m2d2.sbatch` pass-through
- the new configs
- `analysis/uneven.py`
- the locked-test extension
- `submit_m2d2_all.sh`, `after_m2d2_all.sh` and the `status.sh` update
- the tests
- a new section in `README_M2D2.md` with exact commands in order:
  1. Pull on Engaging.
  2. `submit_m2d2_all.sh b200 --dry_run`, and what to look for in its output.
  3. `submit_m2d2_all.sh b200`.
  4. After the builds: check each `meta.json` (`boosted_domain`, `fallback_used`, per-domain train tokens) and `sanity.txt`.
  5. Monitor with `status.sh`. When s0 and s1 of every design are done, an early read with `uneven.py` (it will say PARTIAL).
  6. When everything has finished: `analyze.py --prefix m2d2-`, `--prefix slim120-`, `compare_datasets.py`, then `uneven.py`.
  7. Only then, `submit_locked_test.sh`, then `compare_datasets.py --with_test` and `uneven.py --with_test`.
  8. What each verdict line means.

**Working style:**
- Do Step 0 (Part 3) first.
- Propose a plan and wait for approval.
- Build in this order, testing after each stage: prep options → their tests → configs → `uneven.py` → locked-test extension → launchers → README.
- Commit after each stage; don't push.

**Definition of done:**

- [ ] `pytest -q` passes on CPU in under 2 minutes, with no skips, including every old test.
- [ ] `bash -n` passes for every shell script.
- [ ] With defaults, every existing script behaves exactly as before.
- [ ] A final summary of what you built, test results, deviations from this spec, and anything I need to decide.
