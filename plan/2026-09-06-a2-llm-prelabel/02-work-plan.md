# Work Plan — A2 LLM pre-labelling

Four phases. **Phase 0 is the whole value of the plan** and needs nothing pending to
complete. Phases 2–3 are gated and may never run.

---

## Phase 0 — Retrospective: judge the 400 already-labelled pairs

**Needs nothing pending.** These pairs have `yc` labels; the judge has never seen them.
This is the n=400 in-domain measurement that does not exist today.

### Task 0.0 — Interruption safety (F1–F4) — **blocks every dispatch**

Design spec §6. A credit exhaustion or timeout mid-run currently records the failure as a
finished pair, so a re-run resumes nothing and the report quotes a kappa over a
latency-ordered subset. Nothing may be dispatched until this is fixed.

| Fix | Change |
|---|---|
| **F1** | `judge_one` distinguishes infra failure (non-zero exit, empty stdout) and timeout from an unparseable judge answer. Infra failures go to `llm-judge-failed.jsonl`, **not** `RAW`, so `judge()`'s resume set retries them. |
| **F2** | `judge()` returns a `failed` count and **raises** if it is non-zero. |
| **F3** | `a2_report` carries `n_dispatched` beside `n_scored` and refuses a kappa while they differ. |
| **F4** | Operator practice: run in `--limit 100` chunks. No code. |

F1 and F2 are edits to shared code (`judge_one`, `judge`) and therefore apply to the A1
recheck path too. That is intended — the bug is not A2-specific. Re-run
`verify --derived` afterwards to confirm the committed A1 figures have not moved.

### Task 0.1 — Add the A2 draws to `llm_recheck.py`

```python
A2_GENERIC_DRAW  = "a2_generic"
A2_TARGETED_DRAW = "a2_targeted"
A2_GENERIC_CSV   = MANIFESTS / "llm-a2-generic.csv"
A2_TARGETED_CSV  = MANIFESTS / "llm-a2-targeted.csv"
DRAW_FILES = {RECHECK_DRAW: RECHECK_CSV, FULL_A1_DRAW: FULL_A1_CSV,
              A2_GENERIC_DRAW: A2_GENERIC_CSV, A2_TARGETED_DRAW: A2_TARGETED_CSV}
```

`collect()` already routes by `draw` through `DRAW_FILES`; adding the entries is the whole
routing change. Verify by reading `collect()` rather than assuming.

### Task 0.2 — `dispatch_a2(stratum, run, seed=0)`

Modelled on `dispatch_all_a1`, sourcing from `queue.indomain_rows()` filtered to the
stratum. Same order of operations, no shortcuts:

`assert_agent_definition_current()` → `assert_run_holds_one_draw` →
build records with `draw` stamped → `guide_example_pairs()` overlap →
`assert_not_anchoring` per pair → `redact.assert_clean` over all prompts →
dedup on `(pair_id, run)` → append to `PROMPTS`.

Records carry `stratum` alongside `draw`, so a collected row can be checked against the
file it landed in rather than trusted.

### Task 0.3 — Dispatch and judge, `a2_generic`, run 5

```bash
uv run python -m candidate_screener.annotation.llm_recheck --dispatch --a2 generic --run 5
uv run python -m candidate_screener.annotation.llm_recheck --judge --run 5 --limit 100   # x4
uv run python -m candidate_screener.annotation.llm_recheck --collect
```

400 spawns, 10 concurrent, isolated subprocesses. Option D was 639 at this shape, so the
scale is known-feasible. Chunked per F4: if credits run out, `--judge --run 5` is re-run
after topping up and resumes exactly the pairs that failed (F1).

### Task 0.4 — `a2_frames` / `a2_report`

Per §2 of the design spec. Per stratum, three-class **and** binary, per-class breakdown,
confusion matrix both directions, CI on every kappa. Writes
`output/annotation/llm-a2-report.json`.

### Task 0.5 — `check_llm_a2` in `verify --derived`

Provenance, never determinism (Q31's rule): `agent_sha256` on every row, `prompt_sha256`
against the dispatched prompt, every judged pair present in `indomain_rows()` for its
stratum, and **no row in the wrong file for its `draw`**.

### Phase 0 output

kappa(yc, llm) on A2/generic, n=400 — the first in-domain agreement figure in the project,
and the answer to **Q36**.

---

## Phase 1 — Prospective: pre-label the 200 pending pairs

Runs **before** the human session on batch 2. That ordering is what makes the result a
genuine held-out prospective measurement: the judge sees no label, and the annotator never
sees the judge's.

### Task 1.1 — Dispatch `a2_targeted`, run 6

```bash
uv run python -m candidate_screener.annotation.llm_recheck --dispatch --a2 targeted --run 6
uv run python -m candidate_screener.annotation.llm_recheck --judge --run 6 --limit 100   # x2
uv run python -m candidate_screener.annotation.llm_recheck --collect
```

### Task 1.2 — Confirm the UI is unaffected

`annotation.ui` serves from `queue.build_queue`, which drops judged pairs using
`judged_pair_ids()` — read from `judgements.csv`, which this plan never writes. The LLM
labels are invisible to the annotator by the existing mechanism. **Add a test** asserting
that a pair present in an A2 LLM file still appears in the human queue, so a later
refactor cannot quietly make the judge's output suppress human work.

### Task 1.3 — Human session runs (external to this plan)

Batch 2, 200 pairs. Unchanged by anything here.

### Task 1.4 — Score prospectively

Re-run `a2_report`. kappa(yc, llm) on A2/targeted, n=200, prospective. Quoted **separately
from generic**, never averaged with it (D29).

---

## Phase 2 — GATED: establish the human ceiling and evaluate *X*

**Gate: Phase 0 and 1 complete.** Blocked on a decision, not on code.

### Task 2.1 — Second annotator, double-labelled subset

**Confirmed incoming, and not before this plan is approved** (Q35, 6 Sep 2026). Phases 0
and 1 therefore run without waiting, and the relative gate is the expected path rather
than the fallback.

D29's session design already calls for two annotators and 30% double-labelling; it has
never been run. The cheapest version that answers the question: **one second annotator,
~100 pairs drawn across both strata**, labelled independently through the existing UI.
~1 person-day, against the ~2.5 team-days D25 caps a session at.

Two constraints on the draw, both from existing decisions:

- It must span **both strata** and be quoted per stratum (D29). A ceiling measured only on
  `generic` does not license a substitution on `targeted`.
- The second annotator must not see the judge's labels, `yc`'s labels, or `selection_reason`.
  The existing UI gives this for free — it serves from `judgements.csv` and this plan never
  writes there — but D29's "both annotators cover every title" rule means the draw is
  stratified by title, not handed out by expertise.

### Task 2.2 — Evaluate the gate

Per design spec §4, per stratum:

- kappa(yc, llm) ≥ 0.75 × kappa(yc, annotator2), **and**
- per-class agreement > 0.40 on all three classes

The absolute fallback survives only as a contingency if Task 2.1 slips; it is no longer
the anticipated route.

### Task 2.3 — Write the finding

`03-findings.md`, in the shape of `plan/2026-08-30-llm-recheck/01-findings.md`: what was
measured, at what n, under which instrument, and what it licenses. **A negative result is
a complete result** — "the judge cannot substitute on A2" closes the question, and is
worth the same write-up.

---

## Phase 3 — GATED: replacement

**Gate: Phase 2 passes on at least one stratum.** Scope per design spec §5, and it needs
its own plan and its own approval — D36 crosses a line D33 drew deliberately. Not
specified further here; specifying an unapproved decision's implementation invites
building it.

---

## Acceptance checks (workflow step 4)

Run at the end of Phase 0 and again at the end of Phase 1:

```bash
uv run pytest
uv run python -m candidate_screener.data.verify --derived
uv run python -m candidate_screener.baselines.run --check
```

New tests required, each shown to fail against the code it guards (the repo's regression
rule):

1. `assert_run_holds_one_draw` refuses an A2 draw on a run holding `a1_recheck`, **and**
   the reverse.
2. `collect()` routes an `a2_generic` row to `llm-a2-generic.csv` and never to
   `llm-recheck.csv`.
3. `judge_frames()` output is byte-identical before and after A2 files exist on disk.
   This is the test that protects the D33/D34 figures.
4. The A2 prompt frame carries no column joinable to a human label.
5. A pair judged by the LLM still appears in the human queue (Task 1.2).
6. `a2_report` raises rather than returns if asked for a figure pooled across strata.
7. **F1** — a spawn that exits non-zero with empty stdout leaves no row in `RAW`, and the
   next `judge()` call has that pair in `todo`. Shown to fail against today's code, where
   the pair is absorbed as "done".
8. **F2** — `judge()` raises when any spawn failed.
9. **F3** — `a2_report` refuses a kappa when `n_scored < n_dispatched`.

## Documentation updates (workflow step 5), same commit

- **AGENTS.md** — a paragraph under the annotation section: the judge is pointed at four
  populations now, `judge_frames` still reads the recheck alone, and A2 figures are quoted
  per stratum. State plainly that no replacement decision has been taken.
- **`docs/data/data-catalog.md`** — the two new manifest files, with row counts.
- **This plan** — `03-findings.md` at the end of Phase 2, and any deviation recorded in
  place.

## Cost

| Phase | Spawns | Model | Human |
|---|---|---|---|
| 0 | 400 | `claude-haiku-4-5` | — |
| 1 | 200 | `claude-haiku-4-5` | — |
| 2 | — | — | ~1 day (2nd annotator, ~100 pairs) |

600 spawns total. Option D's 639 is the precedent for the scale.

## Open questions

| # | Question | Status |
|---|---|---|
| Q35 | kappa(yc, annotator2) on A2? | **Answered 6 Sep 2026 — second annotator confirmed incoming, arriving after approval.** Phases 0–1 do not wait; the relative gate is the expected path. Open until measured. |
| Q36 | Does judge–human agreement differ A1 vs A2? | Confirmed in scope — Phase 0 answers it |
| Q37 | May `yc` and `llm` labels coexist in one evaluation layer? | **Deferred by decision, 6 Sep 2026.** Revisit at Phase 3; not worked before then |
| Q34 | Option D's 639 pre-D34 rows | Untouched, still deferred. **No run number reserved** — AGENTS.md corrected in this commit |
