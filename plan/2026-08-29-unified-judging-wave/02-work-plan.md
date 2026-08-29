# Work Plan — Phase 5

> **Revised 29 August 2026 by D25–D27, before approval.** The judging wave is deferred; the
> session is 250 pairs, not the union of every system's top-k. Tasks 5.1, 5.2, 5.3 and 5.5 are
> **done** (`bfafc9c`, `0b1d81e`); 5.4's queue is built and awaits people; 5.6 is now one
> ~2.5 team-day session gated on people rather than on Stage 4. The original sizing is kept
> below, struck where superseded, because the reasoning is the audit trail.

Effort is wall-clock for one person unless marked **team**. Tasks 5.1–5.5 need **no annotator
time** and can run now; 5.6 is the only human-gated step.

## Sequencing

| Order | Task | Depends on | Effort | Gated on |
|---|---|---|---|---|
| 1 | **5.1** judgement layer + overlay | — | ½ day | — |
| 2 | **5.2** record the attainable ceiling | 5.1 | ½ day | ~~Q27~~ **closed by D26** |
| 3 | **5.3** extend the annotation guide | 3.4a | ¼ day | 3.4a exists |
| 4 | **5.4** wave 1 — A1 recheck | 5.1, 5.3 | **½ team-day** | People |
| 5 | **5.5** queue builder + sizing pass | 5.1 | ½ day | **Q24** |
| 6 | **5.6** ~~wave 2~~ **the session** | all above | **~2.5 team-days** | people *(was: D22 + Stage 4 frozen)* |
| 7 | **5.7** re-report | 5.6 | ½ day | — *(was: Q26, dissolved by D26)* |

The plan's shape followed from D22: everything mechanical now, one human step at the end.
**Under D25 the human step moves to the front instead** — it no longer needs any system's
top-k, so nothing is gained by waiting for Stage 4, and the in-domain set is wanted *before*
Stage 2 rather than after Stage 4. That reordering is the largest practical effect of D25.

---

## 5.1 — The judgement layer and the overlay

1. ~~`annotation/layer.py` and the overlay `apply(pools, judgements) -> pools'`.~~
   **Dropped 29 Aug 2026.** The overlay existed to fold a wave's *new* relevance judgements
   into the pools. The A1 recheck compares against labels that already exist and changes no
   relevance, and D25 defers the wave, so nothing consumes an overlay. Writing one now would be
   untested machinery guarding a case that does not occur — YAGNI, with the reason recorded so
   that whoever runs the wave knows it was a deferral and not an oversight.
2. Write the empty `judgements.csv` with its committed header, so the schema exists before any
   label does — a schema settled after labels exist is a migration.
3. Register the six `verify --derived` checks from [`01-design-spec.md` §6](01-design-spec.md#6-determinism-and-commit-policy).

**Acceptance:** overlay applied to an empty layer returns a frame equal to its input; a synthetic
judgement changes exactly one row's `relevance` and nothing else; the A1-downgrade guard raises;
region isolation rejects a planted `train`-region row. All on inline synthetic corpora — the
suite must still run on a fresh clone with no `data/`.

---

## 5.2 — Record the attainable ceiling *(Q18a, step 1 of the Q18 path)*

The Q18 figures are currently **measured but not reproducible from the package** — computed ad
hoc, deliberately not committed under `plan/`. This is the task that fixes that.

1. `metrics.attainable_ceiling(pools, metric, k, definition) -> Figure` — the best value any
   ranker could reach given judged-relevant supply, using the same exclude-zero-relevant rule
   `score` enforces, so its *n* matches.
2. Emit it into `pools-yield.json`, **per view** (A1-only and A1+ours) once the layer is
   non-empty — **under D25 there is only the A1-only view**, since no new A1 relevance
   judgement is collected. D26 settles the reporting: raw *and* ceiling, both.
3. Reproduce the ad-hoc numbers as a regression test: P@5 strict **0.7806**, graded **0.7188**;
   P@10 strict **0.5677**, graded **0.4734**; and the depth-invariance property — identical at
   N20, N100 and Nfull, which is what proves it is a property of the split rather than of pool
   depth.
4. Add the provisional-figure line to every interim Stage 1–3 report (see D22's cost).

**Acceptance:** the four ceilings reproduce to 4 dp; depth-invariance holds across all three
variants; `Figure` still refuses to exist without its *n*; the Q18 doc's *Provenance* section is
updated from "measured-but-unpinned" to a command.

**Q27 closed by D26: both.** `retrieval-metrics.json` carries `value`, `ceiling` and
`pct_of_attainable` on every precision row; `pools-yield.json` carries the ceilings and the
unwinnable-slot arithmetic (34/155 strict at k=5, 90/320 graded).

> **Done 29 Aug 2026 (`bfafc9c`).** All four ceilings reproduce to 4 dp and depth-invariance
> holds across N20/N100/Nfull. `attainable_ceiling` refuses nDCG with the reason — it
> normalises by an ideal from the same pool, so a perfect ranker reaches 1.0 at any density.
> Recall@10's ceiling is **0.983**, which is the measurement D26 headlines it on.

---

## 5.3 — Extend the annotation guide

`docs/annotation-guide.md` is task 3.4a's deliverable and is **not rewritten here**. It gains:

1. An **A1 section** — US resumes at 5,134 median chars against Djinni's 1,525. Worked examples
   from A1, and an explicit note that the corpora differ in length, market and format, so the
   same 3-class scheme is being applied to visibly different documents (**A16**).
2. The instruction that pairs arrive shuffled and context-free, and that **no rank, score or
   system attribution will be shown** — with the reason, so it reads as method rather than
   withholding.
3. Unchanged: the class definitions, the tie-breaking rule, the what-not-to-consider list, and
   the "domain-literate, not professional recruiters" statement for the report.

**Acceptance:** one guide covers both corpora; no A1 example in the guide contains PII.

---

## 5.4 — Wave 1: the A1 recheck *(tests A13)*

Runs alongside task 3.4b's 200-pair wave 1, in the same session, from the same guide.

1. Sample ~50 **already-judged** A1 pairs, stratified across `good`/`potential`/`no`, seed
   recorded. De-identify.
2. Seed them into the wave-1 queue, blind and shuffled among the Djinni pairs.
3. Report the agreement rate and κ against A1's own labels, per class and overall.
4. Write the result into the Q18 analysis as the measured status of **A13**.

**Acceptance:** κ reported with its interpretation band; every disagreement logged by `pair_id`
with both labels retained; annotators demonstrably could not tell recheck pairs from fresh ones
(no column, no ordering, no formatting difference).

**This is the task most likely to change the plan.** If our labels diverge materially from A1's,
the Q18 ceiling is soft rather than hard, and 5.6's design should sample judged pairs too.

---

## 5.5 — Queue builder and the sizing pass

**Zero annotator time. Do this before anyone is scheduled.**

1. `annotation/queue.py`: given per-system score frames, take each system's top-k per query,
   union, deduplicate against `judgements.csv` **and** against A1's existing labels, shuffle at a
   recorded seed, strip the blinded columns, write `judging-queue.csv`.
2. **Measure A17.** Report the union size against the per-system sum, so the overlap assumption is
   a number rather than a hope. Under D23 the per-system upper bound is **500** (100 queries ×
   top-5) and the union is what actually gets judged.
3. Report the same for the A2 side over task 3.4's 40 JDs at top-10 (Q14's wave-2 depth).
4. **Report the 36 currently-unscoreable queries as their own line** in `judging-report.json` —
   queue size and, after 5.6, how many became scoreable. D23 spends ~180 judgements on a return
   that is unknown until judged; that line is what makes the bet auditable either way.

**Acceptance:** no pair in the queue is already judged by anyone; the queue carries no
`surfaced_by`, `best_rank` or `selection_reason`; rebuilding at the same seed and the same scores
is byte-identical; the union-vs-sum ratio is recorded in `judging-report.json`.

> **Done 29 Aug 2026 (`0b1d81e`), at a different size.** Under D25 the queue is the 250-pair
> session, not the top-k union: 200 in-domain pairs plus 50 blind A1 recheck pairs. Steps 2–4
> above sized the deferred wave and are retained for whoever runs it. **Q24 is closed by D27** —
> `evaluation.retrieval` scored the baseline over the pools, and the answer is in
> `output/baselines/retrieval-metrics.json`.

---

## 5.6 — The session *(D25)*

**No longer blocked by Stage 4.** One session, both corpora, one guide, ~250 judgements —
200 in-domain (task 3.4b) and 50 blind A1 recheck. Run it **before Stage 2**, since nothing in
it depends on a system's output any more.

1. Dispatch the shuffled, blinded, de-identified queue.
2. Double-label 30% for κ, matching task 3.4's protocol.
3. Collect `shortlist_pick` per query (Q14's one top-1 pick, ~30 s per query).
4. Adjudicate disagreements in a review session; retain original labels alongside adjudicated
   ones.
5. Freeze. Append to `judgements.csv`; never overwrite a row.

**Acceptance:** every dispatched pair has a label or a recorded reason it has none; κ reported per
corpus (**A16**); zero pairs judged twice by the same annotator; the region-isolation check passes
on the appended layer.

**Effort: ~2.5 team-days for 250 judgements.** Phase 3's implied rate is ~130–175 per team-day
(260 judgements in 1.5–2 team-days), which puts 250 at 1.5–2 team-days. The estimate is widened
to 2.5 because **the 50 A1 pairs will run slower than that rate** — it was derived from
1,525-char Djinni CVs and A1 resumes are 5,134 — and because adjudication is included.

---

## 5.7 — Re-report and supersede

1. Recompute every A1 figure under both relevance definitions. **There is only one view under
   D25** — A1-only — because no new A1 relevance judgement is collected; the recheck re-judges
   pairs that already carry a label. The A1-only / A1+ours dual-reporting rule (D20) stays
   written down for whoever runs the deferred wave.
2. ~~Report the contamination rate.~~ **Not measured, by decision (D26).** What *is* measured
   and recorded is the **exposure**: 83.2% of TF-IDF's top-5 slots are unjudged, falling
   monotonically with system quality (random 94.0%, BM25 86.6%). That bounds the bias and shows
   it does not invert the ranking; the fraction genuinely relevant needs the deferred wave.
3. **The A13 result — the finding to watch.** Agreement and κ against A1's own labels on the
   50 recheck pairs, per class. Every ceiling figure in the Q18 analysis assumes A1's labels are
   correct. If ours diverge materially, the ceiling is *soft* rather than hard, and D26's
   limitation line has to say so rather than presenting 0.7806 as a hard bound.
4. ~~Apply Q26's pre-registered rule.~~ **Q26 is dissolved by D26** — under a standing
   limitation there are no pre-wave figures to retract. The figures are final and carry the
   limitation.
5. Supersede in place, per the repo convention — `docs/data/data-catalog.md`, the A1 card,
   `pools-yield.json`, and every stage report — marking each corrected figure *Superseded
   {date}* rather than deleting it.

**Acceptance:** no reported A1 precision figure exists without a view label; every superseded
figure retains its predecessor; the Q18 analysis is closed with a measurement rather than an
argument.

---

## What this plan does not own

- **Task 3.4b wave 1** (the 200-pair banded-lexical set) — unchanged, still owned by Phase 3.
- **The 675-pair data-science shortlist** — `region=train`, for supervised fine-tuning. It may use
  the same guide and interface, but it is **not** part of the evaluation layer, and D19's
  partition plus check 2 of §6 is what keeps it out.
- **Any modelling.** Stage 4 must be frozen before 5.6 runs; this plan does not produce it.
