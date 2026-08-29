# Phase 3 — Derived Artefacts

**Date:** 23 August 2026
**Status:** Draft for approval
**Predecessor:** [`plan/2026-08-19-data-strategy/`](../2026-08-19-data-strategy/) — Phases 0–2 and 4 complete, merged to `main` (`d5fb992`)
**Scope:** Build the five derived artefacts that Stage 1–4 modelling consumes. No modelling in this phase.

## Contents

| File | Purpose |
|---|---|
| [`01-design-spec.md`](01-design-spec.md) | What each artefact is: schema, identity scheme, determinism and commit policy |
| [`02-work-plan.md`](02-work-plan.md) | Sequenced tasks with acceptance criteria and effort |

## Why this phase exists

Acquisition proved the sources are usable. It also proved three things that make the derived
artefacts mandatory rather than optional:

- The shipped A1 split leaks **476 of 477 test resumes into train** (99.8%). Nothing can be
  reported on it, so a leak-free split must be built before any baseline is run.
- A1's 8,000 rows are a cross-product of **643 resumes × 351 JDs**. Every confidence interval
  must be computed against the document count, which means the pool manifests have to record
  document identity, not row index.
- R3 (candidate pools) and R4 (in-domain evaluation) have **no source that ships them**. They
  are constructed here or they do not exist.

## Inherited decisions

D1–D13 from the predecessor plan stand unchanged. Two questions carried into this phase are now
closed, and both change the work:

| # | Decision | Closes | Consequence |
|---|---|---|---|
| **D14** | The 200-pair in-domain set is **not stratified by role family**. Sample without family constraint and let annotation decide | **Q9** | Simplifies 3.4 and unblocks it immediately. **Cost, stated for the record:** results cannot be broken down by role family, and the sample will be dominated by the largest families in proportion to their supply. §10 gains a limitation line. See [`02-work-plan.md` §3.4](02-work-plan.md#34--in-domain-evaluation-set-200-pairs) |
| **D15** | **Derived** ESCO extracts may be committed. Raw data may not, and no notebook or script may dump bulk source text — samples only | **Q11** | `data/vocab/*.csv` becomes a committed, versioned artefact. Enforced by an explicit commit policy and an output cap — see [`01-design-spec.md` §5](01-design-spec.md#5-commit-policy-d15) |
| **D16** | **Pool A1's shipped train and test, then split three ways.** The published split is discarded entirely, not re-partitioned | **Q8, Q13** | Confirms the existing design — the yield table was already computed over all 8,000 pooled pairs. Pooling is the only coherent option: 642 train + 477 test unique resumes dedupe to **643**, because 476 are the same documents. Q13 is closed to **three-way**; task 3.2 now sizes the val fold rather than choosing between two-way and three-way |
| **D17** | The in-domain set uses **A1's 3-class scheme** (`Good` / `Potential` / `No Fit`) | **Q10** | Public and in-domain results are interpretable together. Doubles as graded relevance for nDCG: `Good`=2, `Potential`=1, `No`=0 |
| **D18** | A2 may supplement the baseline **for unlabelled domain-adaptive pretraining only**. It never contributes fit labels, and the in-domain evaluation documents are excluded from the pretraining corpus | **Q15** | Forces an **eval holdout region** to be reserved *before* pretraining runs — excluding the 200 selected pairs is not sufficient. See [`01-design-spec.md` §3.4](01-design-spec.md#34-dataprocessedindomain--djinni-evaluation-set-decisions-d3-d4-d11-d12-d14). **Narrowed by D19, 29 Aug 2026** — the "never contributes fit labels" restriction now applies to the evaluation region only |
| **D19** | *New, 29 Aug 2026.* A2 is partitioned at document level — every JD `id` and every CV `id` independently — into a `train` region and an `eval` region **before either is sampled** (`a2_finetune.partition_ids`, seed 0, `eval_fraction=0.25`, matching the D16 precedent). Labelled pairs drawn from `train` may be used for supervised fine-tuning; task 3.4's evaluation set (and any wave-2 expansion, Q14) is drawn only from `eval`. D14 is unaffected — task 3.4's own sampling stays unstratified by role family | **Q16** | Enables a personal, data-science-scoped labelling shortlist (`a2_finetune.build_shortlist`, keywords `Data Science`/`Data Engineer`/`Data Analyst`) without risking future eval-set leakage. **`eval_fraction=0.25` is an assumption, not a specification** — no fraction was requested; it is a CLI flag and safely re-cuttable until the first document is labelled. See [`05-implementation.md` §3B](05-implementation.md) |

## Open questions

| # | Question | Status | Blocks |
|---|---|---|---|
| Q12 | Does the 7× length gap between Djinni CVs (751 chars median) and A1 resumes (5,134) need mitigation? | **Evidence gathered, recommendation pending sign-off** — see [`02-work-plan.md` §Q12](02-work-plan.md#q12--the-length-gap-recommendation) | 3.4 interpretation, §10 |
| **Q14** | *New.* Pool depth. 40 JDs × 5 CVs cannot support **Precision@10** — the in-domain pool is 5 deep. Accept P@5 / nDCG@5 in-domain, or re-shape the budget? | **Recommendation pending sign-off** — see [`02-work-plan.md` §Q14](02-work-plan.md#q14--pool-depth-and-whether-to-collect-rankings) | 3.4 sampling design |

Q8–Q11, Q13 and Q15 are closed by D14–D18 above. **Q16 is closed by D19**, 29 Aug 2026 — partition,
not more annotation alone: A2 is split into a document-disjoint `train`/`eval` region, and labelled
`train`-region pairs may be used for fine-tuning.

**Q17 — new, 23 Aug 2026.** Task 3.2 measured the test-side relevance density at **6 Good Fit
per JD**, not the 18 the corpus-wide figure gave. The "do not report Recall@10" rule (D9) was
derived from 18 and may no longer hold. Re-derive the attainability curve on the actual test
pool **inside task 3.3**, before any result exists — the rule is currently stated in the
catalog, the A1 card and the README. See
[`05-implementation.md` §3.5](05-implementation.md).

## Progress

| Task | Status |
|---|---|
| 3.2 leak-free split | **Done** 23 Aug 2026 — 30% hold-out, seed 0, no val fold. [`05-implementation.md`](05-implementation.md) |
| 3.3 retrieval pools | **Done** 23 Aug 2026 — 100 queries, N20/N100/Nfull, metrics guarded. Q17 closed: Recall@10 reinstated, Recall@50 retired |
| 3.4 preliminary — A2 train/eval partition + data-science shortlist | **Done** 29 Aug 2026 — D19 closes Q16. 352,147 A2 documents partitioned (25% eval / 75% train); a 675-pair data-science shortlist drawn from the train region. The 200-pair unstratified in-domain set (D14) itself is not yet built. [`05-implementation.md` §3B](05-implementation.md) |
| 3.1, the remainder of 3.4, 3.5 | Not started |

**Successor:** [`plan/2026-08-29-baseline-repeatability/`](../2026-08-29-baseline-repeatability/README.md)
— Phase 4 takes the 3.2 split and the 3.3 metric discipline and applies both to the first
scored system. Q18 (pool precision bias) remains open here — now analysed and decomposed in
[`plan/2026-08-29-pool-precision-bias/`](../2026-08-29-pool-precision-bias/README.md), which requests decisions Q26 and Q27; Q24 (scoring
the baseline on the 3.3 pools) is raised there and deliberately deferred back to a phase after
it.

## Critical path

```
3.2 split ──► 3.3 pools ──► Stage 1 baseline
3.1 DataTurks repair ──────► Stage 2 extraction
3.5 vocabulary ────────────► Stages 2–4
3.4 in-domain set (human-gated, 1.5–2 team-days) ──► end-to-end evaluation
```

3.2 → 3.3 is the compute critical path and gates the first reportable number. 3.4 is the only
human-gated step and is now unblocked by D14, so it runs in parallel from day one. 3.1 and 3.5
are independent and can be done in any order.

## Explicit assumptions

| # | Assumption | If wrong |
|---|---|---|
| A5 | Phase 3 is modelling-free. Building the artefacts and validating them is the deliverable; the Stage 1 baseline belongs to the next phase | Baseline work moves onto this branch; nothing here needs undoing |
| A6 | A1 documents carry no stable publisher ID, so identity is minted as a content hash (see [`01-design-spec.md` §2](01-design-spec.md#2-document-identity)) | If a publisher ID surfaces later, manifests are regenerable from the same seed |
| A7 | `data/interim/` and `data/processed/` stay git-ignored; only manifests and `data/vocab/` are committed | Adjust the commit policy in §5; no artefact changes |
