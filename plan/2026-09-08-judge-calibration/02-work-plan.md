# Work Plan — judge calibration

Three moves, strict order. Each ends at a committed finding and stops cleanly if the
next is not approved. Every task lists the acceptance check that proves it.

Prerequisites **M1** and **M2** block every dispatch in this plan.

---

## Prerequisites — provenance the dispatch does not currently have

### M1 — model is an axis; `a2_frames` must not blend two

**Blocks:** every dispatch in moves 2 and 3.

`a2_frames` selects `frame.agent_sha256 == current` and takes a per-pair `majority`.
A second model under the same guide has the same `agent_sha256`, `majority` returns
`None` on a two-element tie, and `.dropna()` deletes it — so every pair the two models
disagree on vanishes and the survivors are the pairs they agreed on. Non-random, smaller
n, higher kappa, no exception, no wrong row count.

- Key the live series on `(agent_sha256, model)`; return the rest under `per_model`.
- `a2_report` quotes each `(instrument, model)` separately; no field blends two.
- `self_consistency_kappa` requires both to match — runs 5 and 7 are a model comparison,
  never a reliability figure.

**Regression test** — `test_two_models_of_one_guide_are_never_one_series`: stage a draw
file with run 5 (`haiku`) and run 7 (`sonnet`) disagreeing on half the pairs; assert the
live series has the full n and that a `per_model` key holds the other. **Shown to fail**
against the current selection, which returns half the rows.

### M2 — the recorded model must be the model that ran

`MODEL = "claude-haiku-4-5"` is a module constant written into the dispatch record, and
the subprocess is launched as `claude -p --agent a1-judge` with **no `--model` flag**. The
model column is therefore *asserted*, not enforced: it records what the constant said, not
what answered. Runs 1-6 are affected — their model provenance rests on the constant having
been correct at the time.

- `--model` on `dispatch` and `judge`, defaulting to `MODEL`, threaded into the prompt
  record and into the subprocess argv.
- `judge()` refuses a run whose dispatch record names a model other than the one it was
  asked to run.
- Runs 1-6 are **not** re-stamped. Their provenance stays as recorded, with this
  limitation noted in the plan record and in `AGENTS.md`.

**Regression test** — `test_a_run_is_judged_by_the_model_its_dispatch_names`: assert the
argv carries `--model` and that a mismatch raises. **Shown to fail** against the current
argv, which carries no model at all.

---

## Move 1 — the consensus reference (no spawns, no human hours)

### Task 1.1 — `consensus` in `a2_frames`

`a2_frames` gains a `consensus` series: for the pairs where **every** annotator present
gave the same label, that label; otherwise absent. Two annotators today; the rule is
written for *n* so a third does not change the API.

Beside it, `consensus_coverage`: the number of agreeing pairs over the number of
pairs with two or more annotators. **A consensus figure is never returned without it.**

### Task 1.2 — `a2_report` emits a `consensus` block

Per stratum, beside `vs_llm` and `human_ceiling`:

```
"consensus": {"n_multi_labelled": 126, "n_agreed": 60, "coverage": 0.476,
              "label_distribution": {...},
              "vs_llm": {"three_class": {...}, "binary": {...},
                         "per_class": {...}, "confusion_rows_consensus": {...}},
              "note_selection": "the pairs two annotators agreed on are not a random
                                 subsample; this is an upper estimate"}
```

`note_never_pooled_references` at top level, alongside the existing
`note_never_pooled_annotators`.

**Regression tests**, each shown to fail against a mutation:

1. `test_a_consensus_figure_never_travels_without_its_coverage` — remove `coverage`; the
   report must refuse to build.
2. `test_a_pair_two_annotators_read_differently_is_not_in_the_consensus` — stage a
   disagreement; assert the pair is absent, not resolved by order or by majority-of-two.
3. `test_consensus_and_single_annotator_figures_are_not_the_same_field` — assert no key
   holds both, and that `n` differs and is reported.

### Task 1.3 — the interchangeability gate

`a2_report` emits, per stratum and **on the pairs all three series share**, the four-row
table of design spec §4: for each human, kappa(that human, other human) beside kappa(that
human, llm), three-class and binary, with intervals.

The per-class clause is applied to the human pair as well as the judge, and a class the
humans fail is reported as `unidentified_by_instrument`, not as a judge failure.

**Regression test** — `test_the_gate_compares_on_the_pairs_the_three_series_share`: stage
a judge series covering more pairs than the second annotator; assert the ceiling and the
judge figure carry the same `n`. **Shown to fail** against a version that reads
kappa(yc, llm) from the n=400 block — the mistake that would read 0.380 against a ceiling
measured on a different 126 pairs.

### Task 1.4 — write the finding

`03-consensus-findings.md`. The numbers of README §"the answer this plan proposes", the
§4 table, and an explicit statement of what consensus does and does not license (design
spec §3).

**Move 1 output:** a committed report, a committed finding, **D37** (the reference is an
axis). Cost: no spawns, ~half a day.

---

## Move 2 — the model control, Q32 (gated on M1, M2 and approval)

### Task 2.1 — register the reading in advance

Design spec §5's three outcomes are committed **before** the dispatch, in
`04-run7-registration.md`, with the exact statistic and cut point each names. A result
that can be read either way afterwards measures nothing.

### Task 2.2 — dispatch and judge

```
--dispatch --a2 generic  --run 7 --model claude-sonnet-5
--judge    --run 7
--dispatch --a2 targeted --run 8 --model claude-sonnet-5
--judge    --run 8
```

Two runs because D35 gives a run one draw. `--coverage` before `--collect` on each;
`assert_run_complete` refuses a short run, and F1/F2 quarantine any spawn that never
answered. Run 5 lost 202 of 395 spawns to a usage limit — **budget for a retry pass**,
and do not collect until coverage is 400/400 and 200/200.

### Task 2.3 — collect, verify, report

`--collect`; `check_llm_a2` in `verify --derived` extended to the new runs (scope,
stratum purity, corpus, `agent_sha256` against the committed guide, `model` against the
dispatch record). `a2_report` now quotes `haiku` and `sonnet` separately per stratum.

### Task 2.4 — write the finding

`05-model-control-findings.md`, answering Q32 in the terms registered in 2.1.

**Move 2 output:** Q32 closed; a decision on the committed model. Cost: ~600 spawns plus
retries, ~1 day.

---

## Move 3 — the graded instrument (gated on move 2 returning "no material model effect")

### Task 3.1 — edit the guide

Both parts of design spec §6 in one edit: a positive condition for `Good Fit`, and a 0-100
score emitted beside the class. `--agent` regenerates `.claude/agents/a1-judge.md`;
`verify --derived` fails if the committed judge is not what the guide renders. Guide and
agent in **one commit**, as for D34.

### Task 3.2 — the score is parsed, stored and never silently absent

`collect()` stores `score`; a row that parses a class but no score is `parsed_ok=False`,
not a row with a null score that a later mean would skip.

### Task 3.3 — dispatch runs 9 and 10, judge, collect

Same procedure as 2.2. New `agent_sha256`: **no figure from 9/10 pools with 5-8**, and
`a2_report` must show them as a separate instrument, not as more runs.

### Task 3.4 — fit the cut points, and label them as fitted

Cut points fitted against `consensus`, per stratum, reported with coverage, selection-bias
note, and an explicit `fitted_on` field naming the pairs. **A fitted cut point is not an
out-of-sample result on the pairs it was fitted on**, and the report says so in a field,
not in prose.

**Regression test** — `test_a_fitted_cut_point_names_the_pairs_it_was_fitted_on`.

### Task 3.5 — write the finding

`06-graded-instrument-findings.md`, and re-run the move 1 gate under the new instrument.

**Move 3 output:** a re-calibrated judge, or a recorded failure to re-calibrate. Cost:
~600 spawns, ~2 days.

---

## Acceptance checks (workflow step 4)

Run at the end of every move, and no move is complete without all three green:

```bash
uv run pytest
```

```bash
uv run python -m candidate_screener.data.verify --derived
```

```bash
uv run python -m candidate_screener.baselines.run --check
```

Plus, per move:

- **M1/M2** — both regression tests shown to fail against the code they guard, in the
  plan record, not just asserted.
- **Move 1** — `a2_report` rebuilt; consensus figures reproduce the README table
  (0.389 n=60, 0.401 n=97).
- **Move 2** — `--coverage` reports 400/400 and 200/200 before either `--collect`;
  `llm-judge-failed.jsonl` empty for runs 7 and 8.
- **Move 3** — the committed agent is what the guide renders; every run 9/10 row carries
  the new `agent_sha256` and a parsed score.

## Documentation updates (workflow step 5), same commit as the change

- `AGENTS.md` — D37 (reference axis), D38 (model axis), the rewritten gate, and the M2
  limitation that runs 1-6 have asserted rather than enforced model provenance.
- `docs/data/data-catalog.md` — new runs and files; a superseded figure marked
  *Superseded {date}* in place, never deleted.
- `docs/annotation-guide.md` — move 3 only.
- this plan's findings documents.

## Cost

| Move | Spawns | Human | Reversible? |
|---|---|---|---|
| M1 + M2 | 0 | ~0.5 day | yes |
| 1 — consensus reference | 0 | ~0.5 day | yes |
| 2 — model control | ~600 + retries | ~1 day | yes; runs 7/8 are additive |
| 3 — graded instrument | ~600 | ~2 days | new `agent_sha256`; the old judge stays on file |

Total ~1,200 spawns, ~4 team-days, **no human annotation**. Moves 2 and 3 are separately
approvable and each is worth stopping at.

## Open questions

- **Q32** — model control. Move 2 closes it.
- **Q38** — the `Good`/`Potential` boundary. Move 3 acts on it, gated on move 2.
- **Q39/Q40** — the old gate's two clauses. Design spec §4 replaces the relative clause
  and rescopes the absolute one; both close when move 1 lands.
- **Q41** — no inter-annotator agreement on the shortlist pick. Not fixable without a
  sitting; the ordering rule is recorded for the next campaign.
- **Q42** — `yc`'s `No Fit` shift is confounded with D34. **Unresolvable without human
  time.** Standing limitation.
- **Q43 (new)** — runs 1-6 record a model that was never passed to the subprocess (M2).
  Is the recorded provenance of runs 1-6 trustworthy enough to keep quoting, or must the
  D33/D34 figures carry a provenance caveat? Raised by M2, **not resolved by it**.
