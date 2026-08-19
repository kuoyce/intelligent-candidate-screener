# Data Acquisition and Validation Action Plan

**Target layout** (all of `data/` is git-ignored; nothing raw is committed):

```
data/
  raw/        as-downloaded, never edited
  interim/    parsed / de-identified / repaired
  processed/  model-ready splits and pools
  vocab/      ESCO + skill frequency extracts
docs/data/    dataset cards written by us (licence, provenance, caveats)
```

## Phase 0 — Environment (30 min)

```bash
cd /home/yuchen/project/intelligent-candidate-screener
uv init --python 3.12
uv add pandas pyarrow datasets huggingface_hub requests
uv add --dev jupyterlab           # for the viewing/EDA step
printf 'data/\n.venv/\n*.parquet\n' >> .gitignore
```

No Hugging Face token is needed — every HF source in the catalog is public and ungated
(verified). A Kaggle token is needed only for step 1.6.

## Phase 1 — Download (≈1 hour, ~400 MB)

Run `uv run python plan/2026-08-19-data-strategy/scripts/fetch_sources.py --all`, or step by step:

| # | Source | Command / action | Size | Blocking? |
|---|---|---|---|---|
| 1.1 | A1 fit dataset | `fetch_sources.py --source fit` | 12 MB | No |
| 1.2 | A2 Djinni JDs | `fetch_sources.py --source djinni-jd` | ~146 MB | No |
| 1.3 | A2 Djinni CVs | `fetch_sources.py --source djinni-cv` | ~250 MB | No |
| 1.4 | B2 SkillSpan | `fetch_sources.py --source skillspan` | small | No |
| 1.5 | B1 DataTurks | `fetch_sources.py --source dataturks` (GitHub raw) | 1.2 MB | No |
| 1.6 | C1 Kaggle PDFs | `kaggle datasets download -d snehaanbhawal/resume-dataset` | ~700 MB | Account available; **not on the critical path** |
| 1.7 | C2 HTML fallback | `fetch_sources.py --source livecareer` | ~40 MB | No |
| 1.8 | C3 ResumeAtlas | `fetch_sources.py --source resume-atlas` | 24 MB | No |
| 1.9 | D2 skill frequencies | `fetch_sources.py --source data-jobs` | ~90 MB | No |
| 1.10 | D1 ESCO | **DONE** — v1.2.1 CSV bundle in place at `data/raw/esco_dataset-v1.2.1-classification/` | ~50 MB | Complete |

Everything except 1.6 and 1.10 runs unattended with no credentials.

**Status 19 Aug 2026:** 1.1, 1.2, 1.3, 1.4, 1.5 and 1.10 are complete (429 MB in `data/raw/`).
Outstanding: 1.6 (PDFs, do before the ingestion component in §8 weeks 3–4), 1.7–1.9 (run on demand).

**ESCO contents — Verified:** `skills_en.csv` 13,960 skills (10,734 skill/competence + 3,221
knowledge) carrying **~86,694 alternative labels** for alias normalisation; `occupations_en.csv`
3,043 occupations; `occupationSkillRelations_en.csv` 126,051 occupation↔skill links for
hard-requirement checks; `digitalSkillsCollection_en.csv` 1,284 digital skills. ESCO's
skill/knowledge division maps directly onto SkillSpan's `tags_skill` / `tags_knowledge` layers —
use one label scheme across both.

## Phase 2 — View and validate (≈2 hours)

```bash
uv run python plan/2026-08-19-data-strategy/scripts/profile_sources.py --all
```

This reproduces every **Verified** figure in `02-data-catalog.md`. Treat it as an acceptance
test on the download — if the numbers drift, a publisher has re-uploaded and the catalog needs
re-verification.

Per-source checks to eyeball in Jupyter afterwards:

| Source | What to look at |
|---|---|
| A1 | Read 5 Good Fit and 5 No Fit pairs end to end. Judge for yourselves whether the labels are defensible — this calibrates expectations for decision D2 |
| A2 Djinni | `Primary Keyword` value counts on **both** tables; confirm the join vocabulary overlaps and pick the tech role families for the in-domain set |
| B1 | Render 3 resumes with spans highlighted; confirm the 6.2% offset breakage and inspect what the `Skills` spans actually contain (they are often long multi-skill blobs, not single skills) |
| B2 | Confirm the BIO tag inventory and decide the `tags_skill` / `tags_knowledge` → project-label mapping |
| C1/C2 | Open 3 PDFs and 3 HTML records; note the failure modes the parser must survive |
| C3 | Confirm the text really is pre-normalised, then set it aside for clustering only |

## Phase 3 — Build the derived artefacts (≈2–3 days)

### 3.1 Repair and de-identify DataTurks *(decision D5)*

1. Convert JSON-lines → BIO. **Drop or repair the 222 mis-offset spans** (6.2%); log every one.
2. Strip the `Name` and `Email Address` entity classes and mask the residual 5 emails / 17 phone
   strings found in the raw text.
3. Merge the label scheme with SkillSpan's, then re-split — **200/20 is unusable**; use
   cross-validation over all 220 docs and report mean ± std, not a single test F1.

### 3.2 Re-split the fit dataset at candidate level *(decision D8)*

The shipped split leaks 99.8% of test resumes into train. **Replace it — do not report on it.**

A resume-disjoint split alone would still leave JDs shared across splits, so build a
**doubly-disjoint** split: hold out a set of resumes *and* a set of JDs, take as test only the
pairs where both sides are held out, take as train only the pairs where neither is. Cross pairs
are discarded — that is the cost of a genuinely leak-free evaluation.

**Verified yield** (seed 0, over all 8,000 pairs / 643 unique resumes / 351 unique JDs):

| Hold-out fraction | Test pairs | Test JDs | Test resumes | Test JDs w/ Good Fit | Train pairs |
|---|---|---|---|---|---|
| 15% | 152 | 37 | 68 | 8 | 5,909 |
| 20% | 403 | 67 | 108 | 25 | 4,887 |
| **25% (recommended)** | **560** | **77** | **135** | **30** | **4,218** |
| 30% | 827 | 96 | 163 | 32 | 3,755 |

**Use 25%.** It yields 30 JDs with a Good Fit — slightly *more* than the shipped test split's 28 —
while being genuinely leak-free on both axes, and retains 4,218 training pairs. Below 20% the
evaluation set collapses; above 30% the training set starts to hurt. Open question Q8 confirms.

Fix the seed, and commit the split manifest (resume id → split, JD id → split) so every stage and
every team member scores identical data.

> **Report explicitly:** results on this split are **not comparable** to any published numbers on
> the shipped `cnamuangtoun` split. That is the accepted price of removing the leakage, and the
> reason should be stated wherever Stage 1–4 results appear.

### 3.3 Construct retrieval pools *(decisions D1 + D9 — this is a deliverable, document it)*

For each JD *q* in the evaluation split:

1. **Relevant set** = its resumes labelled `Good Fit`. Run a second, graded pass with
   `Good ∪ Potential` — they answer different questions, so report both.
2. **Distractors** = resumes sampled from the split's unique-resume pool that carry no label
   against *q*. These are **assumed** non-relevant; state that assumption as a limitation.
3. **Pool size N = 100** (Q3 resolved). Report N = 20 (native density) and N = 477 (maximum) as
   sensitivity runs. §3.4's "500 applications per role" is **not reachable** from this data and
   should be restated in the proposal.
4. **Metrics — Recall@50, Precision@10, nDCG@10.** Do **not** report Recall@10: with a median of
   18 relevant resumes per JD it is capped at 0.56 for the median query and is unreachable above
   0.90 for 57% of queries (see `01-requirements-and-findings.md` §2.7).
5. Report *n* (the number of scored queries, ~30) beside every figure, with a bootstrap confidence
   interval. At this sample size the interval is the finding.

Fix the random seed and commit the pool manifest (query id → candidate ids) so all four stages
score identical pools, as §4.4 and Appendix A.2 require.

### 3.4 Build the in-domain evaluation set from Djinni *(decisions D3, D4, D11, D12)*

**Target: 200 adjudicated pairs.** Annotators are business-analytics master's students with
software and data-science backgrounds — domain-literate, but not professional recruiters, which
must be stated in the report per A.7 (this is *team-adjudicated* judgement, not recruiter ground
truth).

1. Filter both Djinni tables to tech/data/software `Primary Keyword` families. All 41 CV-side
   families occur JD-side; spread the sample across ~6–8 of the largest (`JavaScript`, `Java`,
   `Python`, `DevOps`, `.NET`, `QA Automation`, `Node.js`, `PHP`) — confirm via Q9.
2. Sample ~40 JDs across those families and seniority bands.
3. For each JD, build a candidate pool from CVs sharing the role family, banded by
   `Experience Years`; draw ~5 candidates per JD to reach **200 pairs**.
4. **Write the annotation guide first** — Good / Potential / No Fit definitions, worked examples,
   and a tie-breaking rule (§11). Use the **same 3-class scheme as A1** so the public and in-domain
   evaluations are interpretable together (Q10).
5. **Double-label 60 of the 200 pairs (30%)** across all annotators, report Cohen's κ (2
   annotators) or Fleiss' κ (3+), then adjudicate disagreements in a review session. Single-label
   the remaining 140.
6. Budget: ~200 pairs × ~3 min + 60 overlap + adjudication ≈ **1.5–2 team-days**, matching D4.
7. Add LLM-generated synthetic resumes at controlled fit levels (D3) **only** as extra pool
   distractors, clearly flagged — never mixed into the human-labelled evidence.
8. Freeze the set and isolate it from all training and tuning (§5.2).
9. State the market limitation (D11) alongside the results.

### 3.5 Skill vocabulary

ESCO is already in place. Join `skills_en.csv` preferred labels + the ~86,694 `altLabels`
aliases with the `job_skills` frequency counts from D2 to produce `data/vocab/skills.csv` with a
`tech_frequency` column, versioned per §10. Use `occupationSkillRelations_en.csv` (126,051 links,
**67,600 `essential` + 58,451 `optional`**) to seed the **hard vs preferred requirement**
distinction §4.5 calls for — that mapping already exists in ESCO and does not need to be invented.

## Phase 4 — Document (½ day)

Write a short dataset card per adopted source in `docs/data/` covering source URL, licence,
retrieval date, verified row counts, PII status and known defects. Then apply the §5.1/§5.2/§5.3/§7
amendments listed in `01-requirements-and-findings.md` §5, and record the deviations from the
original plan per the project's documentation rules.

## Critical path

```
Phase 0 ──► 1.1 fit ──► 2 profile ──► 3.2 re-split ──► 3.3 pools ──► Stage 1 baseline (§8 week 5)
                └────► 1.2/1.3 Djinni ──► 3.4 in-domain set + annotation (longest lead, start now)
```

**Start 3.4 immediately.** §11 already flags the in-domain set as the primary evidence for
end-to-end performance, and it is the only step gated on human effort rather than compute.

## Blockers to clear before Phase 3

All original blockers are cleared. Remaining confirmations are non-blocking:

| Item | Open question | Needed by | Status |
|---|---|---|---|
| Kaggle PDFs | Q4 | Phase 1.6 / ingestion component | Account available; pull before §8 week 3 |
| ESCO | Q7 | Phase 3.5 | **Done** — in `data/raw/` |
| Core benchmark licence | Q1 | — | **Accepted** |
| Re-split | Q2 / Q8 | Phase 3.2 | Decided: doubly-disjoint at 25%; confirm yield |
| Pool size and metrics | Q3 | Phase 3.3 | Decided: N=100; Recall@50 / P@10 / nDCG@10 |
| Annotation volume and annotators | Q6 | Phase 3.4 | Decided: 200 pairs, 60 double-labelled |
| Djinni role families | Q9 | Phase 3.4 | Confirm the 6–8 families |
