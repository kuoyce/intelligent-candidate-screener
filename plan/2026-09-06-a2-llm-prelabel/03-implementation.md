# Implementation Record — Task 0.0, interruption safety (F1–F4)

**Date:** 6 Sep 2026 · **Branch:** `feat/a2-llm-prelabel`

Task 0.0 of [`02-work-plan.md`](02-work-plan.md), the prerequisite to every dispatch in
this plan. Nothing in Phase 0 or 1 has been dispatched yet.

## What changed

`src/candidate_screener/annotation/llm_recheck.py`:

| | |
|---|---|
| `FAILED` | New quarantine file, `data/processed/indomain/llm-judge-failed.jsonl`. Git-ignored like the rest of `data/`. |
| `JudgeUnavailable` | Raised by `judge_one` when the spawn produced no answer. Carries a coarse `kind` used only for the operator summary. |
| `JudgeFailures` | Raised by `judge()` and `assert_run_complete()`. Its message states the recovery procedure. |
| `classify_failure` / `QUOTA_MARKERS` | Labels an exhausted balance or a throttle as `quota` rather than `exitN`. Cosmetic — every kind is handled identically. |
| `judge_one` | Wraps `subprocess.run`; `TimeoutExpired` and `OSError` become `JudgeUnavailable`. **Any non-zero exit** now raises rather than returning a synthetic `raw` string. |
| `judge` | Routes `JudgeUnavailable` and any other exception to `FAILED`, counts by kind, and raises `JudgeFailures` if anything failed. |
| `run_coverage` / `assert_run_complete` | F3's primitive and its refusal, for `a2_report` (Task 0.4) to call. |
| CLI | `--coverage`; `--judge` catches `JudgeFailures`, prints the summary and returns 1. |

## Deviation D-V7 — a non-zero exit is quarantined even when stdout is non-empty

The approved design spec §6 wrote F1's trigger as *"non-zero exit with empty stdout"*,
matching the condition in the code being replaced. **Implemented stricter: any non-zero
exit is an infrastructure failure**, whatever is on stdout.

A healthy judge exits 0. A non-zero exit with partial stdout is most likely a truncated
stream, and `parse_return` is strict enough to reject it — at which point it is filed
`parsed_ok=False` and becomes **indistinguishable from a judge that answered badly**. That
conflation is the thing F1 exists to end, so the narrower trigger would have left a
version of the bug in place. The stricter rule costs nothing: a pair wrongly quarantined
is retried, and a retry is cheap.

The complementary half is asserted as its own test: **a clean exit with junk on stdout is
still data.** F1 must not swallow a misbehaving judge — that is a real finding about the
instrument, and only spawns that never ran are quarantined.

## Deviation D-V8 — an existing test asserted the bug

`test_a_judge_that_crashes_is_recorded_rather_than_dropped` required a crashed judge to
land in `RAW` as `<judge failed: …>` and to be counted in `judged_now`. That is precisely
the behaviour that made a crashed pair un-retryable while the run reported itself
complete. Rewritten as `test_a_judge_that_crashes_is_recorded_but_not_as_a_return`, with
the history in its docstring; the finding is still recorded, in the quarantine file.

`_staged_judge` also gained a `FAILED` monkeypatch. Without it the fixture wrote quarantine
rows into the **real** `data/processed/indomain/`, where `run_coverage` would have counted
them. Caught during this task; the stray file was removed.

## Tests

Six new, in `tests/test_llm_recheck.py`, plus the rewritten one.

Both mutations were run and the tests shown to fail, per the repo's regression rule:

**Mutation 1 — route failures to `RAW` (pre-F1):** 4 tests fail. Notably
`test_no_figure_may_be_quoted_over_a_run_that_is_short` fails as *DID NOT RAISE* — the run
looks complete, which is the bug stated exactly.

**Mutation 2 — return the summary instead of raising (F2 removed):** 3 tests fail.

## Acceptance checks

| Check | Result |
|---|---|
| `uv run pytest` | **199 passed** |
| `data.verify --derived` | **all derived checks passed** |
| `baselines.run --check` | **the baseline reproduces** |
| A1 figures unmoved | `llm-recheck-report.json` re-generated: only `generated_at` differs. Every kappa identical, and the timestamp change was reverted. |

That last row is what F1/F2 touching shared code required — the fixes reach the A1 recheck
path, and the committed D33/D34 figures did not move.

## Not done here

F4 is operator practice (`--limit 100` chunks), already in the work plan's commands.
F3's *refusal* is wired when `a2_report` exists (Task 0.4); its primitive is in place and
tested.

## Open questions

Unchanged: Q35 (second annotator, incoming), Q36 (Phase 0 answers), Q37 (deferred), Q34
(deferred, no run number reserved).
