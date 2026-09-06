# Implementation record — D35, two files, two draws

**Date:** 6 Sep 2026 · **Branch:** `feat/recheck-file-split` · **Plan:** [README.md](README.md)

Both `verify --derived` failures are cleared and the acceptance criteria are met. One
finding during implementation changed a derivation the plan had specified; it is §Deviation
V7 below.

## What was built

| # | Change | Where |
|---|---|---|
| 1 | `draw` in `LLM_COLUMNS`; `FULL_A1_CSV`, `RECHECK_DRAW`, `FULL_A1_DRAW`, `DRAW_FILES` | `annotation/llm_recheck.py` |
| 2 | `MixedDraw` + `assert_run_holds_one_draw`, called by both dispatch paths | `annotation/llm_recheck.py` |
| 3 | `backfill_draw()` + `--backfill-draw`, three assertions before it writes | `annotation/llm_recheck.py` |
| 4 | `collect()` routes by `draw`, one CSV per draw, refuses a draw-less record | `annotation/llm_recheck.py` |
| 5 | `LEGACY_MIXED_RUNS = {2}` — the declared, closed exception | `annotation/llm_recheck.py` |
| 6 | Checks run per file: schema, draw purity, provenance, scheme, scope; plus one-draw-per-run | `data/verify_derived.py` |
| 7 | Five regression tests, each shown red against the code it guards | `tests/test_llm_recheck.py` |

`judge_frames` needed no code change — it already read `RECHECK_CSV` alone. Its docstring
now says that is a decision rather than an accident, and a test holds it.

## Deviation V7 — the subsample was drawn from a 50-pair draw, not the current 100

The plan specified run 2's split as `subsample_ids(recheck_ids, 20, seed=0)` over the
recheck draw, "deterministic and still computable". It is computable, but not over today's
draw: **7 of the 20 reproduce, not 20**. The backfill's own assertion caught it on the first
run, against run 3's pair set, and refused to write.

The cause is V6. D33 dispatched `--subsample 20` on 30 Aug while the machine leg was still
**50 pairs** (15/15/20); V6 raised it to 100 later the same day. `subsample_ids` sorts and
permutes the list it is handed, so widening the draw re-draws the subsample. The derivation
now runs over `LEGACY_SUBSAMPLE_STRATA = {"Good Fit": 15, "Potential Fit": 15, "No Fit": 20}`
and reproduces run 3's 20 pairs **exactly**, which is what licenses using it to split run 2.

Recorded in AGENTS.md beside V6, because it generalises: raising a stratum count is
retroactive against anything already seeded off the draw, not only against the labels the
nesting assertion protects.

## Deviation V8 — run 2 appears in both files, and the check says so

The plan's check table has "one draw per run: no run number appears in both files". Run 2
does appear in both — 20 recheck rows and 639 option-D rows — because it *is* the historical
mixture, and no rewrite of collected provenance would be honest. `LEGACY_MIXED_RUNS`
declares it and the check reports the exception in its message rather than passing silently.
The set is closed by construction: `assert_run_holds_one_draw` sees both draws on run 2 and
refuses whichever one asks, so run 2 can never be dispatched to again.

## What the split did to the figures

`--backfill-draw` stamped 879 records: `run1:a1_recheck` 100, `run2:a1_recheck` 20,
`run2:a1_full` 639, `run3:a1_recheck` 20, `run4:a1_recheck` 100. Re-running it stamps 0.
`--collect` then wrote 240 + 639 = 879 rows. Nothing created, nothing dropped.

| | before | after |
|---|---|---|
| `run1_vs_run2` | 0.829, n=100 | **0.857 [0.417, 1.000], n=20** |
| `run1_vs_run3` | 0.857, n=20 | 0.857, n=20 |
| `run2_vs_run3` | 1.000, n=20 | 1.000, n=20 |
| `per_instrument[47fc2e48].rows` | 779 | **140** |
| every kappa over a1 / human / llm | — | unchanged |

`run1_vs_run2` returns to the value `plan/2026-08-30-llm-recheck/01-findings.md` published,
so **AGENTS.md's "test-retest reliability is 0.857-1.000 over three runs" needed no
restatement** — it had been true when written, and had quietly stopped being what the
committed report said. The plan predicted 120 rows for the old instrument; 140 is right
(100 + 20 + 20) and the plan's arithmetic was wrong, not the code.

The report diff is 6 lines. `coverage`, `pairwise_kappa`, `per_class_agreement` and
`per_class_llm_vs_a1` are byte-identical: option D contributed no pair to any series that
`judge_frames` builds, which is the point of B2.

## The human queue

`queue --build --seed 0`: 250 → **200 rows**, 500 already judged and skipped, all of them
`targeted` in-domain. The A1 leg now contributes 0 rows because all 100 of its pairs carry a
`yc` label. `judging-queue-report.json` regenerated. The stale 50 that failed the check are
gone, and D28's invariant — a rebuild never re-presents a labelled pair — holds again.

## Verification

| Gate | Result |
|---|---|
| `uv run pytest` | **193 passed** |
| `uv run python -m candidate_screener.data.verify --derived` | **all derived checks passed** |
| `uv run python -m candidate_screener.baselines.run --check` | **the baseline reproduces** |

Each of the five new tests was run against a mutation of the code it guards and observed to
fail, per AGENTS.md §Testing rule 2:

| Test | Mutation it was shown red against |
|---|---|
| `test_collect_routes_each_draw_to_its_own_file` | `collect()` writes every row to `RECHECK_CSV` |
| `test_collect_refuses_a_prompt_record_with_no_draw` | a missing `draw` defaults to `RECHECK_DRAW` |
| `test_dispatch_refuses_a_run_number_already_used_by_the_other_draw` | `assert_run_holds_one_draw` removed from both paths |
| `test_dispatch_refuses_a_pre_d35_run_rather_than_guessing_its_draw` | same |
| `test_judge_frames_never_reads_the_full_a1_file` | `judge_frames` unions the two CSVs |

## Open questions

- **Q34 — option D was judged under the retired instrument.** Deferred by the user on 6 Sep
  2026: it stays open and unworked unless option D is revisited, or a decision comes to rest
  on its output. Either trigger reopens first whether labels carrying `47fc2e48` may be used
  at all, or must be re-judged under the committed judge (`--dispatch --all-a1 --run 6`).
- **Q35 — a published figure was a blend for six days and nothing could see it.** D35 stops
  a *run* mixing draws. It does not stop a future artefact mixing something the schema has no
  column for; that is now the second occurrence (D34's instrument pooling, this). No action
  proposed.

## What the next recheck run looks like

Runs 1-4 are used and run 2 is closed to further dispatch. The next production recheck is:

```bash
uv run python -m candidate_screener.annotation.llm_recheck --dispatch --run 5
uv run python -m candidate_screener.annotation.llm_recheck --judge --run 5 --concurrency 10
uv run python -m candidate_screener.annotation.llm_recheck --collect --report
```

A run number already holding the other draw now raises `MixedDraw` at dispatch rather than
skipping pairs, so the question the plan opened with — whether option D's 659 pairs must be
excluded from a second recheck run — no longer needs an operator to know the answer.
