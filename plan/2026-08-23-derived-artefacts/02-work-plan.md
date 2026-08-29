# Work Plan — Phase 3

Effort is wall-clock for one person unless marked **team**. Every task lists the acceptance
criterion that `verify --derived` checks, so "done" is machine-checkable rather than asserted.

## Sequencing

| Order | Task | Depends on | Effort | Gated on |
|---|---|---|---|---|
| Start now, in parallel | ~~**3.4a** annotation guide~~ **DONE** 29 Aug 2026 | — | ½ day | — |
| 1 | ~~**3.2** leak-free split~~ **DONE** | — | ½ day | Q13 answered from its own output |
| 2 | ~~**3.3** retrieval pools~~ **DONE** | 3.2 | ½ day | Q8 sign-off |
| 3 | **3.5** vocabulary | — | ½ day | — |
| 4 | **3.1** DataTurks repair | — | ½–1 day | — |
| 5 | **3.4b** sampling **DONE** 29 Aug 2026; the labelling run **not started** | 3.4a | **~2.5 team-days** *(D25: 250 pairs, one session, including the 50 A1 recheck)* | People |

3.2 → 3.3 is the compute critical path to the first reportable number. 3.4 is the only
human-gated step; D14 unblocked it, so its guide is written first and the labelling run starts
as soon as annotators are free.

---

## 3.2 — Leak-free A1 split

1. Mint content-hash IDs for all 8,000 A1 rows; assert 643 unique resumes and 351 unique JDs, and
   assert zero hash collisions.
2. Implement the doubly-disjoint split at seed 0. Emit the yield table for hold-out fractions
   {15, 20, 25, 30}% and reproduce the predecessor plan's numbers — this is a regression check on
   the implementation, not new analysis.
3. **Size the three-way split (D16).** Pool train and test first — 642 + 477 unique resumes
   dedupe to 643 because 476 are the same documents, so the published partition is discarded, not
   re-partitioned. Carve val by a second doubly-disjoint hold-out on the train side and emit:
   test yield, val yield, surviving train pairs, and JDs-with-a-Good-Fit on **each** evaluation
   side. Both evaluation folds need enough JDs with a Good Fit to be scoreable; if val cannot
   reach that, fall back to cross-validation on train and say so.
4. Write `fit-split.csv` and `fit-split-yield.json`; build the parquet splits.
5. Write the "not comparable to published numbers" note into the split manifest header **and**
   `docs/data/cards/A1-resume-job-description-fit.md`, so it travels with the data.

**Acceptance:** zero resume-ID and zero JD-ID overlap across splits; discarded-pair count recorded
and explained; re-running at seed 0 is byte-identical.

**Decision surfaced for sign-off:** the hold-out fraction, once step 3's table shows what a
three-way split costs the train set. Q8 and Q13 are closed by D16.

> **DONE 23 Aug 2026 — see [`05-implementation.md`](05-implementation.md).** Built at a **30%**
> hold-out, seed 0, **no val fold**. The predecessor's yield table did not reproduce and its
> 25% recommendation was superseded on measurement: 25% yields 24 JDs with a Good Fit, 30%
> yields 31. A val fold costs ~1,000–1,400 training pairs for 10–11 scoreable queries, so
> D16's stated fallback — doubly-disjoint CV inside train — is taken. New **Q17**: the
> test-side density falls from 18 to 6 Good Fit per JD, which may reinstate Recall@10; decide
> in 3.3.

---

## 3.3 — Retrieval pools

1. For each JD in the evaluation split, assemble relevant sets under both the strict (`Good`) and
   graded (`Good ∪ Potential`) definitions.
2. Sample distractors to `N` ∈ {20, 100, 477} at seed 0; **N=100 primary**.
3. Write `pools.csv` and the joined parquet.
4. Implement `Recall@50`, `Precision@10`, `nDCG@10` with a bootstrap CI over queries, and a
   reporting helper that refuses to emit a figure without its *n*.
5. Add the "distractors are assumed non-relevant" limitation to the pool manifest header.

**Acceptance:** every query is in the evaluation split; no distractor carries a label against its
own query; pool sizes match `pool_variant`; the metric module rejects Recall@10 with an
explanatory error rather than computing it.

> **DONE 23 Aug 2026 — see [`05-implementation.md`](05-implementation.md) §3A.** 100 queries
> over a 193-resume universe (not 477 — that was the shipped split's count), nested variants
> N20 ⊂ N100 ⊂ Nfull. **Q17 closed and it reversed this section's metric set:** the test-side
> density is 6, not 18, so Recall@10 reaches 0.90 for 93.5% of queries and is reinstated,
> while Recall@50 saturates and is retired. Adopted: **Recall@10, Precision@5, nDCG@10**. The
> guard is general — *any* k deeper than the pool is refused — which is what makes it survive
> Q17 flipping which k was safe. New **Q18**: ~96% of a 100-deep pool is assumed non-relevant,
> so precision is biased downward; decide after the Stage 1 baseline whether a judging wave
> over system top-k is warranted.

---

## 3.4 — In-domain evaluation set (200 pairs)

### 3.4a — Annotation guide *(do this first; it is the real blocker)*

> **DONE 29 Aug 2026 — `docs/annotation-guide.md`.** It carried this label from the day the
> plan was written and stayed unwritten through four commits of evaluation-design work; that
> is the single clearest symptom of the pattern D25 exists to stop. It now also covers the A1
> corpus (**A16**) and the blinding, absorbing task 5.3 of the Phase 5 plan.

Write `docs/annotation-guide.md`: Good / Potential / No Fit definitions matching A1's 3-class
scheme (Q10 — proposed yes), 3 worked examples per class drawn from Djinni, an explicit
tie-breaking rule, and a written instruction on what **not** to consider (company prestige,
English level as a proxy for competence, CV length).

State in the guide, for the report: annotators are business-analytics master's students with
software and data-science backgrounds — **domain-literate, not professional recruiters**. This is
team-adjudicated judgement, not recruiter ground truth.

### 3.4b — Sampling and labelling

1. Filter both Djinni tables to English-language records with non-empty text.
2. Build CV text as `Position + CV + Highlights + Moreinfo + Looking For`, **not** the `CV`
   column alone — see [Q12](#q12--the-length-gap-recommendation). Median rises 751 → 1,525 chars.
3. Sample ~40 JDs **without role-family stratification (D14)**, banded by `Exp Years` so the set
   spans seniorities. Record each pair's `Primary Keyword` regardless — the constraint D14 removes
   is on sampling, not on measurement.
4. Draw ~5 CVs per JD, **banded by a cheap TF-IDF/BM25 score against the JD** so each set spans
   high / mid / low similarity — a random draw returns mostly `No Fit` and makes P@k degenerate.
   Reach 200 pairs and write `indomain-pairs.csv`.
   **Write `indomain-holdout.csv` in the same step (D18)**: the 40 JD IDs plus the *entire banded
   candidate pool* each was drawn from, not only the 5 selected. This region is subtracted from any
   A2 pretraining corpus, and reserving the full pool — not just the 200 — is what keeps wave 2
   valid, since wave 2 judges more CVs for those JDs after pretraining has already run.
5. Collect, alongside the 3-class label, **one top-1 shortlist pick per JD** — see
   [Q14](#q14--pool-depth-and-whether-to-collect-rankings). No full rankings.
6. Assign 60 pairs (30%) to all annotators for double labelling; single-label the remaining 140.
7. Compute Cohen's κ (2 annotators) or Fleiss' κ (3+) on the overlap; hold an adjudication session
   on disagreements; record adjudicated labels with the original ones retained.
8. Freeze the set. It is never used for training or tuning.
9. Report the realised family mix as an observation, and add two limitations to §10: the Djinni
   Ukrainian/EE IT market scope (D11), and the unstratified sample (D14).

**Acceptance:** 200 pairs, 60 double-labelled, κ reported with its interpretation band, every
adjudicated disagreement logged.

> **Sampling DONE 29 Aug 2026 (`0b1d81e`)** — `annotation/sample.py`, steps 1–4 and 9.
> 200 pairs, 40 JDs, 200 CVs, drawn from `region == eval` (D19); `indomain-holdout.csv`
> reserves **7,441 documents** — the whole candidate pool per JD, not the 200 drawn pairs
> (D18). Realised family mix recorded as an observation (D14): Java 20, QA/Sales/JavaScript/
> Node.js/DevOps 15 each, and 14 more families.
>
> **Steps 5–8 — the labelling run — are not started.** Under **D25** they run as one session
> of ~250 judgements together with 50 blind A1 recheck pairs, before Stage 2 rather than after
> Stage 4. `docs/data/manifests/judging-queue.csv` is the key; the redacted dispatch file is
> git-ignored under `data/processed/indomain/`.
>
> One deviation from step 3: **experience bands cover `0-1`, `2-3` and `4-6` only.** The JD
> side ships five `Exp Years` categories that map to three bands and has no `7+` — a property
> of how Djinni collected the field, already recorded in `a2_finetune.JD_EXP_TO_BAND`, not a
> sampling error. The set therefore spans junior to mid-senior, not the full seniority range,
> and that is a limitation for §10.

**Open:** Q12 — whether the 7× CV/resume length gap (751 vs 5,134 median chars) needs a
length-normalised scoring mitigation or is reported as a limitation only. Decide once the first
scores exist; it does not block the set being built.

---

## 3.5 — Skill and occupation vocabulary

1. Extract ESCO skills + `altLabels` (~86,694) and occupations + `altLabels` (30,417) to the
   long-form schemas in the design spec, via `parse_skill_cell` for anything touching D2.
2. Left-join D2 `job_skills` frequencies; add unmatched high-frequency tools as project-local rows
   with `source = data_jobs`. Record the match rate — acquisition measured 33/100 exact.
3. Extract `occupationSkillRelations_en.csv` to `requirement-relations.csv`, preserving the
   `essential` / `optional` distinction verbatim.
4. Write `VERSION` with the ESCO release, D2 snapshot date and row counts. Commit all four files
   per D15.
5. Add the ESCO attribution line required by its terms to `data/vocab/VERSION` and to the report's
   data section.

**Acceptance:** `skills.csv` covers every ESCO skill; the D2 match rate is recorded, not assumed;
no committed file under `data/vocab/` contains source document text; the alias lookup carries the
review gate described in the design spec §3.5.

---

## 3.1 — DataTurks repair and de-identification *(decision D5)*

1. Convert JSON-lines → BIO. Repair or drop the 222 mis-offset spans (6.2%); **log every one**
   with its reason.
2. Drop the `Name` and `Email Address` classes; mask the residual 5 emails and 17 phone strings.
3. Merge the label scheme with SkillSpan's and commit `label-map.yaml`.
4. Set up cross-validation over all 220 docs; report mean ± std. Do **not** build a 200/20 split.
5. Inspect what the `Skills` spans actually contain — acquisition found they are often long
   multi-skill blobs, not single skills. If the blob rate is high, that is a finding for the
   extraction design, not something to silently normalise away.

**Acceptance:** the de-identification sweep in `verify --derived` returns zero PII matches; the
repair log accounts for all 222 offset failures; no 200/20 split exists anywhere in the outputs.

---

## Q14 — Pool depth, and whether to collect rankings

**Recommendation: do not collect rankings. Collect one top-1 shortlist pick per JD, and judge in
two waves.**

### Rankings are not what Precision@k needs

P@k asks how many of the *system's* top-k are relevant. Its ground truth is a relevance judgement
per document, not a human ordering. nDCG@k needs *graded* relevance — which **D17 already
supplies**: `Good` = 2, `Potential` = 1, `No Fit` = 0 is exactly nDCG's gain input. A human
ranking adds nothing either metric consumes, costs materially more time per JD, and has no clean
inter-annotator statistic (κ does not apply; you would be reporting Kendall's τ on 5 items, which
is noisy to the point of uselessness).

**Do add one cheap question per JD:** *"among the candidates you marked Good or Potential, which
one would you shortlist first?"* A single top-1 pick, ~30 seconds per JD, ~20 minutes across the
whole set. It breaks ties inside the `Good` class — with 5 candidates and 3 marked `Good`, nDCG
cannot separate any system that orders them differently — and its agreement is a plain rate, no
statistic needed.

### The real constraint is depth, not ordering

40 JDs × 5 CVs = 200 pairs gives a pool **5 deep**. **P@10 is undefined in-domain**; the design as
written supports P@5 and nDCG@5 only. Three ways to spend the same budget:

| Option | Queries *n* | Deepest metric | Cost |
|---|---|---|---|
| (a) 40 JD × 5 | 40 | P@5, nDCG@5 | CI as good as this budget allows |
| (b) 20 JD × 10 | 20 | P@10, nDCG@10 | Halves *n*; CI widens ~1.4× — at *n*=20 the interval swamps the finding |
| ~~**(c) Two waves**~~ | ~~40, rising~~ | ~~P@5 now, **P@10 later**~~ | **Not taken.** D25 defers wave 2; the in-domain set is 40 × 5 and its metrics are **P@5 / nDCG@5**, i.e. option (a) |

> **Resolved 29 Aug 2026 — option (a), not (c).** Wave 1 below is built and stands unchanged.
> **Wave 2 is deferred by D25**: its design lives in
> [`plan/2026-08-29-unified-judging-wave/`](../2026-08-29-unified-judging-wave/README.md), merged there with the A1-pool
> judging wave Q18 asked for, and it is an optional week-11 batch rather than a scheduled step.
> **In-domain metrics are P@5 and nDCG@5.** P@10 in-domain is not reported, because the pool is
> 5 deep and `metrics.score`'s depth guard refuses it — which is the guard working, not a gap.
> The **top-1 shortlist pick** below is kept: it is 30 seconds per JD and it is the only thing
> that breaks ties inside the `Good` class, which nDCG@5 cannot.

**Option (c).** Wave 1 is 40 × 5 now: it yields κ, the calibration data, and P@5 / nDCG@5. Wave 2
runs after the Stage-1 baseline exists and judges the *union of each system's top-10* for the same
40 JDs — TREC-style pooling, which is what P@10 actually requires. Because system variants overlap
heavily in their top-10, wave 2 typically costs far fewer than another 200 judgements. Judging a
random sample now and computing P@10 on a ranked list whose top-10 is mostly unjudged would bias
precision downward by construction.

### One thing wave 1 must get right

Sampling 5 CVs at random from a JD's role family will return mostly `No Fit`, and P@5 on a pool
with ~0 relevant documents is degenerate. **Band the 5 by a cheap lexical score** (TF-IDF or BM25
against the JD) so each JD's set spans high / mid / low similarity. This is ~20 lines and needs no
model. State the consequence: the pool is then not a uniform sample of the corpus, so in-domain
*absolute* precision is not an unbiased estimate of production precision — it is a comparison
instrument between systems. That is what it is for.

---

## Q12 — The length gap: recommendation

**Recommendation: fix the data first, report the residual as a limitation for ranking metrics, and
recalibrate — not length-normalise — for anything thresholded.**

### Measured 23 Aug 2026

**Length carries no label signal in A1.** Median resume length by class is `Good` 5,135 / `No Fit`
5,134 / `Potential` 5,047 chars — indistinguishable. Spearman(length, label) = **−0.001** overall;
median **−0.013** within JD across 219 scoreable queries; and across 456 resumes appearing in ≥5
pairs, Spearman(length, Good-Fit rate) = **−0.0005**. A1's labels are length-agnostic, so a model
trained on them is not rewarded for using length as evidence.

**Half the gap is an artefact of using one Djinni field.** The plan's 751-char median is the `CV`
column alone. Concatenating `Position + CV + Highlights + Moreinfo + Looking For` gives a median of
**1,525 chars** (mean 1,912; p25 929, p75 2,495) — `Moreinfo` alone is populated on 100% of records
at a 538-char median. The gap against A1 falls from **6.8× to 3.4× at zero cost.**

### What follows

1. **Do the concatenation.** It is free, it is not a modelling choice, and it halves the problem.
   Task 3.4b builds the CV text from all five fields, not from `CV` alone.
2. **Ranking metrics need no mitigation.** Every candidate in an in-domain pool is a Djinni CV, so
   the length shift applies uniformly across the pool and cancels under any monotone scoring
   function. Recall@k, P@k and nDCG@k are unaffected. Report the gap as a limitation for context,
   not as a threat to these numbers.
3. **Thresholded outputs are where it actually bites.** Any absolute cut-off ("score > 0.7 =
   Good Fit") and any 3-class accuracy/F1 carried over from A1 will drift, because score
   *distributions* shift with document length even when *ordering* does not. The fix is
   recalibration on in-domain data — refit the threshold on Djinni — **not** length normalisation,
   which is an arbitrary transform applied on the basis of no measured bias.
4. **Budget for the recalibration.** It consumes in-domain labels, which collide with the freeze
   in §5.2. Reserve a calibration fold (~50 of the 200) or use nested cross-validation, and record
   which pairs were spent on calibration so the reported evaluation excludes them.

**Net: Q12 downgrades from a design risk to a data-preparation step plus one limitation line.**

---

## Documentation to update at phase close

Per the project's documentation rules, and because two of these are already stale:

- `plan/2026-08-19-data-strategy/README.md` — mark Q9 and Q11 closed, pointing at D14 and D15.
- `plan/2026-08-19-data-strategy/04-acquisition-implementation.md` §6 — same, plus Q13 added.
- `AGENTS.md` / `README.md` — the `build` command and the `data/vocab/` committed artefact.
- The predecessor plan's §5.2 isolation note — A2 is now *pretraining-eligible outside the holdout
  region* (D18), which is a narrowing of "frozen", and must be stated wherever A2 results appear.
- `docs/data/cards/A1-*.md` — the re-split and the not-comparable note.
- `docs/data/cards/D1-esco.md`, `D2-data-jobs.md` — pointer to the derived vocabulary.
- A new `05-implementation.md` in this folder recording what was built and every deviation.
