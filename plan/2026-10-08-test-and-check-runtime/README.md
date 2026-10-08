# Test and check runtime: faster kernels, tiered checks, quieter output

**Status: implemented (8 Oct 2026)**, see [03](03-implementation.md); approved the same day, with Q1–Q6 answered in [02](02-open-questions.md): W6
is dropped, logs go to `logs/`, and the runner defaults to `-j 2`. The branch
`chore/test-check-runtime` is cut from `feat/shortlist-recall` and merged back into it.
The measurements are in [01](01-measurements.md).

## Objective

Cut the time it takes to verify a change (`uv run pytest` plus the `--check` commands) and
the number of times the same verification is re-run. **No committed record may change.**
Every speed-up below either produces bit-identical numbers or is confined to a mode that
says it reuses cached work.

## Baseline (measured, [01](01-measurements.md))

| What | Now |
|---|---:|
| `uv run pytest` (594 tests) | **290 s**. 90 % of it is one file, `test_llm_recheck.py` |
| all 21 `--check` commands, serial | **5,169 s (86 min)** |
| full-suite runs in past transcripts | 197, of which 151 were piped to `tail`/`grep` and ≤ 144 had no edit since the previous run |

## Workstreams

Ordered by saving per unit of risk. W1–W6 change code. W7–W9 change how things are run.

### Kernels: same numbers, less work (benefit `--check` cold and warm)

| # | Change | Where | Saving (est.) | Equality argument | Risk |
|---|---|---|---|---|---|
| **W1** | numpy confusion-matrix kappa in `cohen_kappa` / `kappa_ci`, with the same ops and order as sklearn's `cohen_kappa_score`. Keep sklearn's `ValueError` when no label of `labels` is in `y1` | `annotation/llm_recheck.py` | **pytest 290 → ~35–45 s** (llm_recheck 260 → ~10–15 s); the same speed-up for `llm_recheck` report regeneration | Prototype matched bit for bit on 80 cases; a permanent test compares with sklearn on random and degenerate inputs | Low |
| **W2** | Module-level memo for `_grams(s)` and `trigram_cosine(a, b)` (pure functions of two strings) | `matching/role_filter.py` | ~50 % of `shortlist_dev` (~340 s) and of `--variants` (~250–300 s); part of `shortlist_final` | Same function, same inputs, cached result | Very low. Memory is bounded by the number of distinct title strings (thousands) |
| **W3** | Per-run memo of `per_jd_auc` keyed by (system, scope, subset, metric), so `paired_delta` reuses a rung's AUCs instead of recomputing them | `evaluation/indomain.py` | ~35 % of indomain v1 (~150 s); part of `ordering`, `stage1` | Reuses the identical computation | Very low |
| **W4** | `CrossEncoder.classify()` reuses the scores `score()` already computed for the same frame (in-process only) | `matching/cross_encoder.py` | ~5 % of indomain v1 (~25 s) and the same in a1_secondary | Same model, same inputs | Very low |
| **W5** | Cache `role_family` features or probabilities per text within a run, so 8 rungs do not re-featurise the same dev CVs | `extraction/role_family.py` | ~18 % of `shortlist_dev` (~120 s) and of `--variants` | Pure function of the text | Low |
| **W6** | Vectorise `pair_cosine_scores`: transform the unique texts once, then a row-wise dot of the L2-normalised rows. **Adopt only if every record reproduces byte for byte**; otherwise drop W6 and record why | `baselines/tfidf.py` | ~18 % of `shortlist_dev`, ~12 % of `retrieval_pilot` (~120 s + ~60 s) | **Not guaranteed**: summation order can differ in the last ulp, and records round to 6 decimals | Medium, gated by the cold `--check` of all records |

Kernel-only estimate (cold, every check rebuilding from scratch): **86 → ~55–60 min.**
The irreducible part is model training (NER 412 s, role family, cross-encoder inference).

### Tiers: what a check has to prove, and when (benefit the everyday loop)

| # | Change | Saving (est.) | Implication |
|---|---|---|---|
| **W7** | **Two tiers per check.** `--check` stays as it is (cold: retrains and re-infers, the reproducibility proof). A new `--check --warm` reuses cached artefacts: the NER weights (the existing `--reuse-weights`), the role-family model (a new read-only "check from cached model"; `--evaluate` today writes the record), and a **persistent cross-encoder score cache** keyed by (weights digest, sha256 of JD text, sha256 of CV text, `max_length`), stored under `data/processed/` (git-ignored, reconstructible). `dev_pools` warm does not force a rebuild | warm total **~20–30 min** serial against 86 (extraction 640 → ~230 s; indomain v1, a1_secondary and retrieval_pilot lose their ~57–63 % cross-encoder share; role_family unmeasured, Q2) | Warm proves "**the code that turns cached model outputs into the record is unchanged**", not "training reproduces". A stale cache could hide a model change, so the key holds the weights digest and the cold tier stays mandatory at defined points (Q1). Needs a decision-log entry (D55) and an `AGENTS.md` change |
| **W8** | **One check runner**: `python -m candidate_screener.checks [--warm\|--cold] [--affected\|--all\|NAME…] [-j 2]`. It runs the registered checks, writes each one's full output to a git-ignored `data/logs/checks/<name>.log`, and prints **one line per check** (`PASS  indomain_v1  41.2 s  log: …`). `--affected` maps `git diff --name-only main…` to checks through a declared table, with a test asserting that every `--check` module is registered | A typical change touches 1–5 checks, not 21. `-j 2` is the RAM ceiling (peak 6.8 GB of 15 GB). The heaviest checks are flagged to run alone | The registry is new code to keep current, which the test enforces. `--affected` is a convenience: any doubt means `--all` |

### Output and agent habits (benefit re-runs, the user's original complaint)

| # | Change | Saving | Implication |
|---|---|---|---|
| **W9a** | pytest: add `--tb=short` to `addopts`. Add scoped `filterwarnings` for the two known benign warnings (sklearn's undefined kappa in `test_llm_recheck`, numpy's All-NaN slice in the two tests that build empty strata), at the test level, **not** globally | The output shrinks to the dot line plus the summary. Real warnings still show | A global filter could hide a new real warning, hence per-test marks |
| **W9b** | Silence the transformers/HF progress bars in `matching/neural.py` before any model loads (`HF_HUB_DISABLE_PROGRESS_BARS`, transformers' `disable_progress_bar()`) | indomain v1 output 25 → 1 line; similar wherever a model loads | None on figures |
| **W9c** | `AGENTS.md` → Testing: **run once, keep the log, read the log.** `uv run pytest > "$LOG" 2>&1; tail -3 "$LOG"`, then `grep` the log, never re-run to see another slice. Use `pytest tests/test_x.py` / `--lf` while iterating, and the full suite once before review. Checks go through the W8 runner. Includes a cost table (suite ~40 s; warm all ~25 min; cold all ~60 min) so an agent knows what a command costs before starting it | Removes most of the ≤ 144 measured re-runs; at the old 290 s each, that is hours per week | A rule only works if it is read. It lives in the Testing section, which every session loads through `CLAUDE.md → AGENTS.md` |

### Pruning (considered, mostly not recommended)

| Candidate | Saving | Recommendation |
|---|---:|---|
| `test_baseline_golden` refit tests: they duplicate `baselines.run --check` | 11 s | **Keep.** They `skipif` without data, cost 11 s, and are the suite's one end-to-end check |
| `test_extraction_vocab::…rebuilds_byte_identically`: duplicates `vocab --check` | 2.4 s | Keep (same reasoning) |
| Low-value `test_llm_recheck` tests (collection is closed, D42) | after W1, ~10 s | **Keep.** Each guards a bug that shipped silently, and W1 removes the cost |
| **Superseded records out of the routine run**: indomain v1 (pre-D53 text), `retrieval` (A1 pools), `retrieval_pilot`, `a1_secondary`. Not deleted: moved to the cold tier only, and run when their code paths change (W8 `--affected` still catches that) | ~1,800 s off the routine **cold** run, ~300 s off warm | **Q3**, the user's call: these are historical records the report still cites |
| `pytest-xdist` | after W1 the suite is ~40 s | **Not recommended.** It adds a dependency and nondeterministic output order for ~25 s |

## Projected outcome

| Loop | Now | After W1–W9 (estimate) |
|---|---:|---:|
| `uv run pytest` | 290 s | **~35–45 s** |
| all checks, cold (reproducibility proof) | 86 min | ~55–60 min (~35–40 min if Q3 = yes) |
| all checks, warm | n/a | **~20–30 min** serial, ~13–18 min at `-j 2` |
| typical change: suite + `--affected --warm` | 5 min + whichever checks were remembered | ~1–8 min |
| re-runs to read another slice of output | ≤ 144 observed | ~0: the log is kept |

The estimates come from profile shares times measured wall time. Each is replaced by a
measured figure in the implementation record.

## Work plan (after approval)

1. **W1** with its sklearn-equality test, which must be shown to fail against a deliberately
   wrong kappa (`AGENTS.md` Testing rule 2). Re-time pytest.
2. **W2–W5**, one commit each. After each, run the cold `--check` of every check that
   imports the touched module. All must PASS byte-for-byte.
3. **W6** last among the kernels. If any record differs, revert and note it here.
4. **W9a/W9b** (output).
5. **W7** warm tier and the cross-encoder score cache, then **W8** runner and registry
   test.
6. **W9c** `AGENTS.md`, D55 in `docs/decisions.md`, `README.md` commands.
7. Review: `uv run pytest`, `checks --cold --all` (the full proof, ~1 h), and
   `checks --warm --all`. Record measured before and after in `03-implementation.md`.

**Acceptance:** every committed record reproduces under `--cold`; `git status` stays clean
after `--all`; the suite passes without `data/` and without the `ml`/`app` groups;
before/after timings are recorded.

## Explicit assumptions

- A1. "Optimisation" must not change any committed figure. Hence the bit-identity bar and
  W6's gate. (The bootstrap-over-pairs in `kappa_ci` is out of scope: changing it would
  change the records.)
- A2. Training reproducibility still has to be proven, just not on every edit. Hence two
  tiers rather than dropping the retrain.
- A3. A runner is code, so it goes in the package (`AGENTS.md`: no scripts under `plan/`).
