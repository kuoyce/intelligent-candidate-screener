# Two files, two draws: separating option D from the instrument recheck

**Date:** 6 Sep 2026
**Status:** approved 6 Sep 2026; implemented — see [01-implementation.md](01-implementation.md)
**Branch:** `feat/recheck-file-split`

## Why

`verify --derived` has failed on two checks since the calibration branch merged
(`plan/2026-09-05-prompt-calibration/02-calibration-run.md` §"Not done"). Neither is a
figure error; both are bookkeeping that has outgrown the shape it was written in. Until
they clear there is no clean state to run a second production recheck from, and
`--dispatch --run 2` is not safe to invoke.

```
[FAIL] resume: 50 already-judged pairs re-dispatched
[FAIL] scope: 559 judged pair(s) are not among the 100 A1 recheck pairs
```

## What is actually wrong

### 1. The `scope` failure — two draws in one file

`llm-recheck.csv` holds 879 rows across four runs:

| run | rows | model | agent | what it is |
|---|---|---|---|---|
| 1 | 100 | `claude-sonnet-5` | `47fc2e48` | the D33 recheck, full draw |
| 2 | **659** | `claude-sonnet-5` | `47fc2e48` | **20 rows of D33's subsample + 639 rows of option D** |
| 3 | 20 | `claude-sonnet-5` | `47fc2e48` | D33's second self-consistency subsample |
| 4 | 100 | `claude-haiku-4-5` | `75e00ea3` | the D34 recheck under the rewritten guide |

`check_llm_recheck`'s scope check asserts every pair in the file is one of the 100 A1
recheck pairs. Option D judges all 659 A1 pairs by design, so 559 are out of scope and the
check is red for a reason that is not a defect in the data.

### 2. Run 2 is a mixed run, and nothing refused it

This is the finding that shapes the fix. `dispatch_all_a1` dedups on `(pair_id, run)`
against the prompt file, exactly as `dispatch` does. When option D was dispatched as run 2,
20 pairs were **already** dispatched under run 2 — D33's `--subsample 20` self-consistency
run — so they were skipped as "already dispatched" and their existing labels became part of
what reads as one 659-pair run. The timestamps separate them cleanly:

```
run 2, pairs in the recheck draw   : 20 judged 07:01:45-07:01:58,  80 judged 14:26+
run 2, pairs outside it            : 559 judged 14:26-23:12
```

Two consequences:

- **A published figure is a blend.** `output/annotation/llm-recheck-report.json` currently
  quotes `run1_vs_run2` = 0.829 over n=100. `plan/2026-08-30-llm-recheck/01-findings.md`
  quotes 0.857 over n=20 for the same comparison. They are different measurements: the
  first silently widened when option D landed on run 2. Self-consistency is a property of
  one judge over *one draw judged twice*; 80 of those 100 pairs were judged once in run 1
  and once in a dispatch that was not the recheck.
- **The dedup key cannot tell the two apart.** `(pair_id, run)` is the only idempotency
  key, and a run number is currently free to hold both draws. `--dispatch --run 2` today
  would silently skip 100 pairs it believes it already sent, because option D sent them.

### 3. The `resume` failure — a stale human queue, not the LLM dispatch

This one is unrelated to option D. `check_judging_queue` asserts the committed
`judging-queue.csv` never re-presents a pair that carries a row in `judgements.csv` — D28's
only invariant for the resume mechanism. The committed queue was built while the A1 recheck
leg was 50 pairs and unlabelled; all 100 A1 pairs now carry a `yc` label, so 50 of the
queue's 250 rows are stale. `queue --build --seed 0` is the fix and is also the documented
resume command.

## Decisions taken (user, 6 Sep 2026)

| # | Decision |
|---|---|
| B1 | Option D's rows move to their own committed file, `docs/data/manifests/llm-recheck-full-a1.csv`. `llm-recheck.csv` stays the instrument-validation file. |
| B2 | **Reporting reads the recheck file only.** `judge_frames` does not open the full-A1 file. Strict separation: no option-D row reaches any kappa. |
| B3 | The draw is **stamped at dispatch**, not inferred at read time. `dispatch` writes `draw="a1_recheck"`, `dispatch_all_a1` writes `draw="a1_full"`; `collect` routes on it. |
| B4 | **A run number belongs to exactly one draw.** `dispatch` refuses a run that already carries `a1_full` prompts and vice versa. The next production recheck is `--run 5`. |
| B5 | The option-D dataset is **data-only for now** — collected, provenance-checked, not analysed. It was judged under the retired pre-D34 instrument `47fc2e48`, so a kappa over it would describe a judge this repository no longer commits. Recorded as **Q34**. |
| B6 | The human queue is rebuilt and the rebuilt manifest committed. |

## Design

### D35 — the draw is a column, and a run holds one of them

The unit of provenance grows by one field. `agent_sha256` says *which instrument*; `run`
says *which sitting*; `draw` says *which population was sampled*. Run 2 proves the third is
not derivable from the first two: one run number, one instrument, two populations, and no
check anywhere could see it.

`draw` is stamped on the prompt record at dispatch and copied onto the collected row. It is
also a **new column in `LLM_COLUMNS`**, appended after `judged_at`, so both committed CSVs
carry it and `check_llm_recheck`'s schema check pins it.

Guard, in `dispatch` and `dispatch_all_a1` both:

```
existing = {r["draw"] for r in read_jsonl(PROMPTS) if r["run"] == run}
if existing - {this_draw}:
    raise MixedDraw(f"run {run} already carries {existing} prompts; a run number holds "
                    f"one draw. Choose an unused run number.")
```

This is the check that would have refused option D's dispatch onto run 2, and it is what
makes `--dispatch --run N` safe to type again.

### Backfilling the four runs already on disk

`llm-recheck-prompts.jsonl` is collected data in a git-ignored directory: it cannot be
regenerated, so it is stamped in place, once, by `backfill_draw()`. The rewrite adds a
field and touches nothing a hash is taken over (`prompt_sha256` covers the prompt text
alone), so provenance is preserved.

The split of run 2 is **derived, not asserted from file order**: D33's subsample is
`subsample_ids(recheck_ids, 20, seed=0)`, which is deterministic and still computable. The
backfill asserts all three of the following before it writes, and refuses if any fails:

1. Runs 1 and 4 are exactly the 100 recheck pairs → `a1_recheck`.
2. Run 3 is exactly `subsample_ids(..., 20, 0)` → `a1_recheck`.
3. Run 2 splits into that same 20 (→ `a1_recheck`) and 639 others (→ `a1_full`), and every
   one of the 639 is a pair in `fit/test.parquet`.

A `draw`-less prompt record reaching `collect()` after this is a hard error naming the
backfill, never a default — a default would refile option D into the recheck file the first
time someone ran `--collect` on a fresh machine.

### What the split does to the published figures

Removing option D from the recheck file **restores** the D33 numbers rather than losing
them. Run 2 in `llm-recheck.csv` becomes the 20-pair subsample it originally was, so all
three self-consistency comparisons survive at their original n:

| | now (blended) | after the split |
|---|---|---|
| `run1_vs_run2` | 0.829, n=100 | recomputed over n=20 — D33 reported 0.857 |
| `run1_vs_run3` | 0.857, n=20 | unchanged |
| `run2_vs_run3` | 1.000, n=20 | unchanged |
| `per_instrument[47fc2e48].rows` | 779 | 120 |

The exact figures are whatever `--report` computes; the table above is the expectation
against which the run is read, not a prediction to be written into the docs. AGENTS.md's
"test-retest reliability is 0.857-1.000 over three runs" is checked against the regenerated
report and restated if it moved.

No kappa involving `a1`, `human` or `llm` changes: every one of them is already restricted
by index intersection to the 100 recheck pairs, and option D contributed no pair outside
them to any of those series.

### The full-A1 file is 639 of 659, and that is stated rather than patched

The 20 pairs D33 subsampled onto run 2 are filed as `a1_recheck`, because that is the
dispatch that sent them. So the option-D dataset holds 639 pairs, not the 659 the plan
`plan/2026-08-30-expand-annotation/README.md` §objective 3 described. The missing 20 do
have a label from the same instrument on the same day — it lives in `llm-recheck.csv`, and
joining the two files to reach 659 is exactly the pooling B2 forbids in reporting.

The alternative — refiling those 20 as `a1_full` so the analytical file is complete — would
delete D33's self-consistency measurement, which is the load-bearing row of
`plan/2026-08-30-llm-recheck/01-findings.md` (§"Why self-consistency is the load-bearing
row"): without it, low agreement with A1 reads as a noisy judge rather than as a judge
measuring something A1 does not contain. A complete 659 costs 20 spawns under a fresh
`a1_full` run number, and under B5 nothing is waiting on it. Recorded with Q34.

### Checks

`check_llm_recheck` gains a second block over the full-A1 file, and its existing scope check
stops seeing option D:

| check | file | assertion |
|---|---|---|
| schema | both | columns == `LLM_COLUMNS` (now including `draw`) |
| draw purity | `llm-recheck.csv` | every row `draw == "a1_recheck"` |
| draw purity | `llm-recheck-full-a1.csv` | every row `draw == "a1_full"` |
| scope | `llm-recheck.csv` | every pair is one of the 100 recheck pairs |
| scope | `llm-recheck-full-a1.csv` | every pair is one of the 659 in `fit/test.parquet` |
| one draw per run | both | no run number appears in both files |
| provenance | both | `prompt_sha256` matches the dispatched prompt |
| scheme | both | labels inside A1's 3 classes |

## Work

| # | Task | Files |
|---|---|---|
| 1 | `draw` in `LLM_COLUMNS`; `FULL_A1_CSV` constant; stamp `draw` in both dispatch paths | `annotation/llm_recheck.py` |
| 2 | `MixedDraw` guard in both dispatch paths | `annotation/llm_recheck.py` |
| 3 | `backfill_draw()` + `--backfill-draw` CLI flag, with the three assertions | `annotation/llm_recheck.py` |
| 4 | `collect()` routes by `draw`, writes both CSVs, refuses a `draw`-less record | `annotation/llm_recheck.py` |
| 5 | `judge_frames` reads `RECHECK_CSV` only (unchanged code, docstring states it is now a *pure* recheck file) | `annotation/llm_recheck.py` |
| 6 | Split the checks; add the full-A1 block | `data/verify_derived.py` |
| 7 | Four regression tests, each shown to fail against the code it guards | `tests/test_llm_recheck.py` |
| 8 | Run `--backfill-draw`, `--collect`, `--report`; commit both CSVs and the report | manifests, `output/annotation/` |
| 9 | Run `queue --build --seed 0`; commit the rebuilt manifest and report | `judging-queue.csv`, `judging-queue-report.json` |
| 10 | Docs: `AGENTS.md`, the A1 card, the data catalog, this plan's implementation record | as listed below |

### The regression tests

Each must be demonstrated red against the code it guards, per `AGENTS.md` §Testing.

1. `test_collect_routes_each_draw_to_its_own_file` — a mixed dispatch (one `a1_recheck`,
   one `a1_full`) produces one row in each file, never two in one. Fails against a
   `collect()` that writes a single CSV.
2. `test_dispatch_refuses_a_run_number_already_used_by_the_other_draw` — both directions.
   Fails against the current `(pair_id, run)`-only dedup, which silently skips.
3. `test_judge_frames_never_reads_the_full_a1_file` — an option-D row for a recheck pair,
   with a label that would move the kappa, changes no figure. Fails against a `judge_frames`
   that unions the two files.
4. `test_collect_refuses_a_prompt_record_with_no_draw` — the backfill is not optional.
   Fails against any default value for `draw`.

### Documentation

- `AGENTS.md` — a paragraph on D35 beside the existing `agent_sha256` paragraph (an
  instrument is not the only thing a figure must not be pooled across); the file names in
  §"Its labels never enter `judgements.csv`"; the test-retest sentence if the regenerated
  report moves it; Q34.
- `docs/data/cards/A1-resume-job-description-fit.md` — the new manifest beside
  `llm-recheck.csv` in the evidence list; the test-retest sentence at line 49.
- `docs/data/data-catalog.md` — a row for `llm-recheck-full-a1.csv`; the label-quality row
  is **not** restated, because B5 leaves option D unanalysed. Any figure that moves is
  marked *Superseded 6 Sep 2026* in place.
- `plan/2026-09-06-recheck-file-split/01-implementation.md` — the record, including any
  deviation and the before/after self-consistency table as actually measured.
- `plan/2026-08-30-expand-annotation/README.md` — a note that objective 3's output landed
  in its own file, and that its item 4 (a full-A1 report section) is deferred under B5.

## Open questions

- **Q34 — the option-D dataset was judged under a retired instrument.** All 659 rows carry
  `agent_sha256 = 47fc2e48`, the pre-D34 guide whose tie-break drove `No Fit` (D34: 82 of
  its own `No Fit`s, only 8 citing a different profession). D34 moved kappa(yc, llm) from
  0.287 to 0.352 and the binary collapse from 0.421 to 0.614, so the two instruments are
  not interchangeable. Re-judging all 659 under the committed judge is a run, not a code
  change: `--dispatch --all-a1 --run 6`, ~659 spawns. Whether that is worth the spend is a
  budget decision under D25's logic, and it is not taken here.
  **Deferred (user, 6 Sep 2026)**: Q34 stays open and unworked *unless* option D is
  revisited, or a decision comes to rest on option D's output. Either trigger re-opens it,
  and the first thing it re-opens is whether a dataset judged under `47fc2e48` may be used
  at all, or must be re-judged under the committed instrument first.
- **Q35 — one published figure was a blend and nobody could see it.** `run1_vs_run2` widened
  from n=20 to n=100 the moment option D landed on run 2, with no exception and a plausible
  number, and it was quoted in the committed report for six days. D35's guard stops a
  *run* from mixing draws. It does not stop a future artefact from mixing something else
  the schema has no column for. No action proposed; recorded because the failure mode has
  now occurred twice (D34's instrument pooling, this).

## Acceptance

- `uv run pytest` green, including the four new tests, each shown red first.
- `uv run python -m candidate_screener.data.verify --derived` green — **both** pre-existing
  failures cleared, no new ones.
- `llm-recheck.csv` is 240 rows (100 + 20 + 20 + 100), every one `a1_recheck`.
- `llm-recheck-full-a1.csv` is 639 rows, every one `a1_full`, every pair in `test.parquet`.
- 240 + 639 = 879, the row count of the file being split. Nothing is created or dropped.
- `judging-queue.csv` re-presents no judged pair.
- No row of either CSV appears in `judgements.csv` (the existing separation check).
