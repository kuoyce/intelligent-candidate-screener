# Derived-artefact manifests

`data/` is git-ignored, so these manifests are how a derived artefact stays reproducible
without committing a corpus. A manifest names documents by **content-addressed ID** and
records the seed; running the builder against the same `data/raw/` reproduces the artefact.

```bash
uv run python -m candidate_screener.data.build  --all --seed 0   # rebuild
uv run python -m candidate_screener.data.verify --derived        # acceptance test
```

Document IDs are `sha256(normalise(text))[:12]` prefixed `r_` / `j_` — A1 ships no publisher
ID, so identity is minted from the text (design spec §2). **No manifest carries a timestamp:**
the git history is the timestamp, so a diff is a real change in the data, never a re-run.

## Task 3.2 — the leak-free A1 split

| File | What it is |
|---|---|
| `fit-split.csv` | `doc_id, doc_type, split` — the authoritative assignment. Every stage and every team member must score the same documents |
| `fit-split-yield.json` | Seed, fractions, per-split counts, and the pooling record (what was dropped and why) |
| `fit-split-survey.json` | What each hold-out fraction buys, across 20 seeds — the evidence for choosing 30% |
| `fit-pair-conflicts.csv` | The 7 duplicate `(resume, jd)` pairs found by pooling, 6 of them labelled two ways, and how each was resolved |
| `fit-c4-overlap.csv` | The 7 A1 resumes whose text also appears in the **rejected** `vm-structured-onet` set, with the split each landed in |

> **Results on this split are NOT comparable to any published number on the shipped
> `cnamuangtoun` train/test partition.** That partition leaks 99.8% of its test resumes into
> train; this one is doubly disjoint — nothing in test shares a resume *or* a JD with train.
> The incomparability is the accepted price of removing the leakage (D8) and must be stated
> wherever Stage 1–4 results appear.

**`doc_type` takes only `train`, `val` and `test`.** `discarded` is a *pair*-level outcome —
a pair whose two documents were assigned to different splits — and is counted in
`fit-split-yield.json`, not in this file. A document is never discarded; it is held out.

**`fit-c4-overlap.csv` is a guard, not a defect.** Those 7 resumes belong in A1. The file
exists so that if the rejected C4 derivatives are ever re-imported as extra training data,
the exclusion can be *applied* rather than remembered — the same people would otherwise be
trained on and scored in test at once.

## Task 3.3 — the retrieval pools

| File | What it is |
|---|---|
| `pools.csv` | `query_jd_id, candidate_resume_id, relevance, source, pool_variant` — the authoritative pools. 100 queries over a 193-resume candidate universe |
| `pools-yield.json` | Seed, per-variant pool sizes, queries carrying a relevant document, and the Recall@k ceiling that settles Q17 |

Variants are **nested — N20 ⊂ N100 ⊂ Nfull** — so a sensitivity run differs from the primary
run by depth alone, never by which distractors were drawn. `N` is a *floor*: a query judged
against more resumes than `N` keeps all of them, because discarding a human judgement to hit a
round number is the wrong trade.

> **Distractors are UNJUDGED and assumed non-relevant.** A1 judges a median of 4 resumes per
> test JD, so a 100-deep pool is ~96% assumption. A relevant resume sitting among the
> distractors is scored as a false positive, which biases precision **downward** by
> construction. These pools compare systems against each other; they do not estimate
> production precision. State this wherever a pool-derived figure appears.

`relevance` and `source` are separate columns on purpose: a judged `No Fit` and an unjudged
distractor both score as non-relevant, but only one of them is evidence.

**Reporting is enforced in code**, in `candidate_screener.evaluation.metrics`: a `Figure`
cannot be constructed without its query count, and a `k` deeper than the shallowest pool raises
rather than silently collapsing to `k = pool depth`. Queries with no relevant document are
excluded, not scored as zero — that would report the pool's label sparsity as a property of
the system.

## Task 3.4 preliminary — A2 train/eval partition and data-science shortlist (D19)

A2 (Djinni) ships its own publisher `id` on both tables, so unlike A1 these manifests use it
directly — `doc_id` is not a content hash here.

| File | What it is |
|---|---|
| `a2-partition.csv` | `doc_id, doc_type {jd,cv}, region {train,eval}` — every A2 JD and CV id, independently assigned. **The** guarantee: task 3.4's 200-pair evaluation set must sample only `region == eval`; any fine-tuning label must come only from `region == train` |
| `a2-partition-report.json` | Seed, `eval_fraction` (0.25 — an assumption, not a specification; re-cuttable via `--eval-fraction` until the first document is labelled), per-region counts |
| `a2-datascience-shortlist.csv` | `pair_id, jd_id, cv_id, primary_keyword, exp_band` — candidate pairs for a data-science-focused labelling pass, drawn only from `region == train`, banded by experience |
| `a2-shortlist-report.json` | Seed, keyword scope, sampling parameters, per-cell summary |

> **Why this exists.** D18 restricted A2 to unlabelled pretraining — "never contributes fit
> labels." Q16 asked what happens once a stage needs more *labelled* A2 data than the 200-pair
> set, and deferred the answer. That point arrived: labelled A2 pairs are wanted for
> fine-tuning as well as evaluation. **D19** answers Q16 by partitioning A2 at document level
> before either side is sampled — the same doubly-disjoint discipline `fit-split.csv` already
> applies to A1 (D8), reused via `fit_split.assign_documents` rather than re-derived. D18 is
> narrowed, not lifted: the "never contributes fit labels" restriction now applies to
> `region == eval` only.

The shortlist filters to `Primary Keyword ∈ {Data Science, Data Engineer, Data Analyst}` —
the only bucket `AI Engineer`/`ML Engineer`-titled rows fold into, verified via `Position`
substring matching. It does **not** stratify or otherwise change task 3.4's own 200-pair
build, which stays unstratified across all 41 role families per D14.
