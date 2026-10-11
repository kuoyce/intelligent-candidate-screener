# Design Specification — A2 LLM pre-labelling

## 1. Two new draws, two new files

D35 says a draw is a population, one run holds one of them, and one file holds one of them.
D29 says `generic` and `targeted` are two populations and their average names nothing.
Composed, those give **two** new draws, not one:

| Draw | Population | File | n |
|---|---|---|---|
| `a2_generic` | batch 1, the unstratified D14 draw | `docs/data/manifests/llm-a2-generic.csv` | 400 |
| `a2_targeted` | batch 2, the keyword-scoped batches pooled | `docs/data/manifests/llm-a2-targeted.csv` | 200 |

A single `a2_indomain` draw would be cheaper by one file and would make the first pooled
figure across the two strata a one-line mistake rather than an impossible one. The whole
point of D29 is that this figure has no referent; the file boundary is what enforces it.

`DRAW_FILES` gains both. `assert_run_holds_one_draw` needs no change — it already refuses
any run number another draw holds, in both directions, for any draw in the map.

### Run numbers

Runs 1–4 are spent (1, 2, 3 recheck under `47fc2e48`; 2 also `a1_full`; 4 recheck under
`75e00ea3`). This plan uses **run 5** for `a2_generic` and **run 6** for `a2_targeted`.

AGENTS.md previously wrote option D's hypothetical re-judge as `--run 6`. That reservation
is removed in the same commit as this plan: a deferred task does not hold a run number
against work that is happening, and Q34 takes the next free number whenever it is
revisited.

## 2. `judge_frames` and `report` are not touched

`judge_frames` reads `RECHECK_CSV` alone, by design and with a docstring saying why. A2
labels must not reach it. They get their own builder:

```python
def a2_frames(stratum: str) -> dict[str, pd.Series]   # human, llm, indexed by pair_id
def a2_report(seed: int = 0) -> dict                  # per stratum, never pooled
```

writing `output/annotation/llm-a2-report.json`, separate from `llm-recheck-report.json`.
Two reports rather than a section appended to one, for the same reason as two files: the
A1 recheck figures are quoted in AGENTS.md and in the D33/D34 findings, and a change to
this plan must not be able to move them.

`a2_report` quotes, **per stratum and never averaged**:

- kappa(yc, llm) three-class, with bootstrap CI, and the binary collapse beside it (D34's
  standing rule — that run is where the two moved apart)
- the per-class breakdown, because A1-vs-llm's entire positive kappa came from one cell and
  a headline number hid it
- the confusion matrix, both directions
- `n`, `agent_sha256`, `model`, `run`

## 3. Blinding and isolation — unchanged, and that is the point

The A2 dispatch is the `dispatch_all_a1` shape: load a frame, render, assert, append. It
reuses without modification:

- `assert_isolated` — sandbox cwd, no `CLAUDE.md`/`AGENTS.md`/`.claude` above it (V4)
- `assert_not_anchoring` — no label text in the prompt
- `redact.assert_clean` — PII sweep over every prompt (rule 4; A2 is Djinni, but the sweep
  is unconditional)
- `guide_example_pairs()` overlap check — empty since D34 removed all nine examples, kept
  because a future guide edit could reintroduce them
- `assert_agent_definition_current()` — refuses to dispatch a judge that is not what the
  guide renders

One thing is **new and load-bearing**: for `a2_generic`, `yc`'s label already exists in
`judgements.csv`. The prompt is rendered from `indomain_rows()`, which carries no label
column for A2 (`a1_label` is `None` throughout), so no label can reach the prompt by
construction. `assert_not_anchoring` is the belt; the frame's shape is the braces. A test
asserts the A2 prompt frame has no column joinable to a human label.

## 4. The threshold *X*, and why it cannot be set yet

### What is known

kappa(yc, llm) on the A1 recheck, committed instrument, n=100: **0.352** three-class,
**0.614** [0.447, 0.760] binary. The judge's own test–retest is 0.857–1.000.

### What is not known, and blocks *X*

**There is no kappa(human, human) in this repository.** D29's session design — two
annotators, both covering every title, 30% double-labelled — was never run. All 500
labels are `yc`'s. So there is no measurement of how much two competent humans disagree on
*this* instrument over *this* corpus, and therefore no scale on which 0.352 is either good
or bad.

This matters more than it sounds. Three-class kappa over an ordinal scale with a genuinely
ambiguous middle class (`Potential Fit`) is routinely low between humans. If two annotators
reach 0.45 here, then a judge at 0.352 is at **78% of the human ceiling** and the
replacement question is live. If two annotators reach 0.80, the judge at 0.352 is not
close and the answer is no. The same number supports opposite decisions, and nothing on
file distinguishes them.

### Proposed gate

*X* is defined **relative to the human ceiling**, not absolutely:

> The judge may substitute for a human annotator on a stratum when
> kappa(yc, llm) ≥ **0.75 × kappa(yc, annotator2)** on that stratum, measured on the same
> pairs, three-class; **and** the per-class agreement is above 0.40 on every one of the
> three classes.

The second clause is not decoration. A1-vs-llm scored 0.90 / 0.27 / **0.067** per class
and still produced a positive pooled kappa; a headline figure that can be carried entirely
by the easy class is not a substitution criterion. Both clauses must hold.

**Q35 is answered as of 6 Sep 2026: a second annotator is confirmed, arriving after this
plan is approved.** So the relative gate is the route this plan expects to take, and
Phases 0–1 run in the meantime rather than waiting on it. The fallback below is retained
as a contingency against slippage, not as the anticipated path.

### If the second annotator slips

Then *X* is an **explicit, flagged assumption** rather than a measurement, and the plan
degrades to:

> Absolute fallback, used only with the limitation stated in every figure it touches:
> three-class ≥ 0.45 and binary ≥ 0.70, per stratum, plus the per-class clause above.

0.45 is chosen as *above the judge's current A1 figure*, not from an adjective table — the
argument being that A2 is the corpus the guide was calibrated to score and a judge that
does no better there than on the corpus it disagrees with has shown nothing. This is a
weaker gate and the plan says so wherever it is used.

## 5. What "replace A1 as the baseline" would mean

Worth stating precisely, because it is a bigger change than it sounds and this plan does
not execute it.

A1 is the evaluation ground truth today: `a1_label` on `fit/test.parquet` drives the pools,
and D26's attainable ceilings (0.7806 strict / 0.7188 graded) are properties of A1's judged
supply. A13 — *the A1 labels are correct* — **fails**: kappa(A1, yc) = 0.034,
kappa(A1, llm) = 0.085. The ceilings are soft and every figure quoted as a fraction of
attainable inherits the doubt. That is the case for replacement.

Replacement means all of:

1. `annotation/layer.py` (specified in the 2026-08-29 wave, **never built**) gains a third
   `label_source`: `{a1, yc, llm}` rather than the spec's `{a1, ours}`.
2. Ceilings recomputed **per stratum** over the A2 label supply. They will not be 0.7806;
   that number is A1's and does not travel.
3. `evaluation/retrieval.py` and `metrics.py` gain an A2 evaluation path, with `n` and
   value taken from the same view (the design spec's `Figure` rule).
4. Three reported views, always side by side: **A1-only** (comparable to published work,
   with A13's failure stated), **A2/yc** (human, n=400 generic), **A2/yc+llm** (the wide
   internal figure, `label_source` per row).
5. A decision record — provisionally **D36** — stating that LLM labels may serve as
   evaluation ground truth. This crosses a line D33 drew deliberately ("its labels never
   enter `judgements.csv`", "a third opinion, not an answer key") and cannot be done by
   implication. The files stay separate either way; what changes is whether the evaluation
   layer is permitted to read them.

**None of steps 1–5 are in this plan.** They are gated on §4, and §4 is gated on a
measurement that does not exist yet.

## 6. Interruption — what happens if credits run out mid-run

Traced through the code on 6 Sep 2026, before dispatching anything. **The current handling
is unsafe at 400 spawns and must be fixed before Phase 0 runs.** No run so far has
exercised it: all 879 raw returns on file are clean, so option D got lucky.

### The failure path as it stands

1. `claude -p` exits non-zero with empty stdout. `judge_one` turns that into a *data row*:
   `raw = "<judge exited 1: Credit balance is too low>"`.
2. That row is written to `llm-recheck-raw.jsonl` and flushed.
3. `judge()` builds its resume set as `done = {(pair_id, run) for r in read_jsonl(RAW)}`.
   **The failed pair is now "done".** Re-running `--judge --run 5` reports
   `remaining: 0` and judges nothing. The run looks complete.
4. Credit failures return in milliseconds, so the entire remaining tail burns through in
   one pass. There is no backoff and no pause — 150 pairs can die between two heartbeats.
5. `collect()` files them `parsed_ok=False`, label `""`.
6. The frame builder filters to `parsed_ok=True`, and `kappa_ci` reports `n` as the number
   of rows it was *handed*. So the report shows a smaller `n` and a plausible kappa, with
   nothing anywhere saying 150 pairs were dispatched and never answered.
7. Worst of it: the survivors are whatever `as_completed` returned before the wall — that
   is completion-latency order, which correlates with document length and plausibly with
   label. **The subset is not random, and the figure over it is not the figure.**

This is the same shape as the three baseline bugs and as D35's `run1_vs_run2`: no
exception, no row count that looks wrong, a plausible number.

### Required fixes, all prerequisites to Phase 0

**F1 — an infrastructure failure is not data.** `judge_one` must separate *the judge
answered something unparseable* (data; belongs in `RAW`) from *the process never ran*
(not data). Non-zero exit with empty stdout, and the `except Exception` timeout path in
`judge()`, are both the latter. Neither may be written to `RAW`. They go to a quarantine
file the resume set does not read, so **re-running `--judge` retries them**. Getting
credits back and re-running is then the whole recovery procedure.

**F2 — `judge()` refuses to look successful.** It returns a `failed` count and raises if
any spawn failed, rather than returning normally. The operator must not reach `--collect`
without knowing.

**F3 — no kappa over incomplete coverage.** `a2_report` carries `n_dispatched` beside
`n_scored` on every figure and **refuses to emit a kappa while they differ**. This is
D26's "every figure travels with its ceiling" applied to coverage: a kappa over 250 of 400
is a different measurement from a kappa over 400, and the report must not be able to print
the first while looking like the second.

**F4 — pace it.** `--limit` already exists. Phase 0 runs in chunks of 100, so an
interruption costs at most one chunk's worth of retries rather than the tail of the run.
No code needed.

### After the fixes

Credit exhaustion becomes: some pairs fail, `judge()` raises and says how many, nothing
enters `RAW`, nothing is collected, no figure is computed. Top up, re-run `--judge --run
5`, and it picks up exactly the pairs that failed. **Interruption costs money and time,
never a silently biased figure.**

## 7. What this plan explicitly does not do

- Does not dispatch, re-judge, or analyse option D. Q34 stays deferred and untouched.
- Does not modify `judgements.csv`, `llm-recheck.csv`, `llm-recheck-full-a1.csv`,
  `judge_frames`, `report`, or `llm-recheck-report.json`.
- Does not build `layer.py` or change any evaluation metric.
- Does not restate D26's ceilings. That change is still outstanding and still separate.
- Does not edit `docs/annotation-guide.md`. A guide edit is a new judge (D34) and would
  invalidate the comparison this plan exists to make.
