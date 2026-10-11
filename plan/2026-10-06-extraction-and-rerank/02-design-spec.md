# 02 — Design spec

## 1. Principle: no metadata at inference

At inference a system reads **JD text and CV text only**. It does not read `Primary Keyword`,
`Exp Years` / `Experience Years`, `English Level`, or `Position`. Those fields have three
legitimate uses, all outside inference:

1. **Training labels** for extractors, from the A2 `train` region only (D19).
2. **An oracle for measurement**: scoring an extractor, or showing the upper bound of a rule
   that used the true field.
3. **Building candidate pools** (D31) for the annotation sets that already exist.

`Primary Keyword` is not assumed to appear in the CV and is too coarse to mean a role
(Python covers backend, frontend, data). The target is a role family derived from titles.

## 2. A2 with `Position` removed (the final evaluation text)

The CV's `Position` is the first field of `cv_text`. A system that reads it is reading a
title field. The evaluation text variant is called **`text_v2`**:

- CV text = `CV`, `Highlights`, `Moreinfo`, `Looking For` (no `Position`).
- JD text = `Long Description` (unchanged; the JD `Position` was never in it).
- **Residual leak check (P1).** Report how often the removed `Position` string (lower-cased,
  seniority tokens stripped) still appears verbatim in the remaining CV text and in the JD
  text. The field is removed, **no string scrubbing** is applied, and the residual rate is
  reported next to every `text_v2` figure.
- The current ladder is re-scored on `text_v2` as a new variant (new `text_variant` key in
  the record). The `text_v1` figures stay, for the audit trail, and are never mixed with
  `text_v2`.

**Decide before looking:** `text_v2` is fixed by this spec before any held-out `text_v2`
result is seen. The re-scored existing rungs are `exploratory` (their configurations were
seen on `text_v1`). New rungs built under this plan are `held-out`.

## 3. Entity extraction

### 3.1 Order of work (easiest to extract × most value)

| # | Entity | Method (rules first, targeted) | Free ground truth (A2 train) | Value |
|---|---|---|---|---|
| E1 | **Skills** | Done: vocabulary + rules matcher (D49), tagger for JD free text | SkillSpan F1 | rung 1b: +0.044 over metadata alone |
| E2 | **Job title → role family** | Title spans from text (JD first lines and "looking for / seeking X"; CV headline and work-history titles), normalised through an alias table (ESCO occupation alt labels, `data_jobs` titles, A2-train `Position` strings); plus a text → role-family classifier trained on A2-train `Position` | `Position` (normalised), `Primary Keyword` as a coarse check | **Highest**: replaces the keyword that decided the level |
| E3 | **Experience years** | Regexes: JD "3+ years", "2–5 years", "at least"; CV explicit "N years" and date ranges summed (overlaps merged) | `Exp Years`, `Experience Years` bands | Second half of rung 1a |
| E4 | **Seniority** | Title tokens (junior / middle / senior / lead / head) plus E3 | seniority tokens in `Position` | Tie-break and reranker feature |
| E5 | Location, education, English level | Rules | `English Level` exists in A2 | Low; **only if time remains** |

### 3.2 Role family: the label taxonomy

`Primary Keyword` (45 JD / 41 CV classes) is the free supervision but is too generic in
places. Candidate taxonomy, **to be fixed in P2 before any extractor is scored on A2**:

- **T-A:** `Primary Keyword` as given (45 / 41 classes). Simple, and the oracle already uses it.
- **T-B:** normalised `Position` strings clustered to a role family (strip seniority and
  language tokens, merge by alias table, keep families with at least *m* train JDs and CVs).
  Finer than T-A where T-A is generic.

P2 first **profiles** both on the train region (class sizes, how many distinct T-B families
fall under each T-A class) and reports the choice with its evidence. The decision stays
with the user (05, Q-B1).

### 3.3 Existing data checks (P2, before anything is built)

Already known: A2 `Position` is present on 100% of JDs and CVs (01 §3). To check, with
numbers:

1. Which of ESCO, `data_jobs`, livecareer and resume-atlas has a usable title or role
   field, its fill rate, distinct count and PII exposure. (The catalog says these exist; the
   catalog is not trusted, per `AGENTS.md` data rule 3. Figures are marked **Verified**
   only after `profile`.)
2. Title coverage of the alias table on A2-train `Position` (share of strings that map to
   a family).
3. Whether titles appear in the free text at all: for a sample of A2 JDs and of
   `Position`-removed CVs, the share where a title span can be found. This decides whether
   span extraction or a classifier carries E2.

### 3.4 Rungs built from extraction (all text-only)

| Rung | System | Formula |
|---|---|---|
| X1 | **Extracted metadata rule** | `2 × role_match(extracted) + experience_compatible(extracted)`, the 1a formula **unchanged** |
| X2 | X1 + TF-IDF | same fixed rank-normalised sum as 1a+2 |
| X3 | X1 + skill overlap | as 1a+1b |
| X4 | X1 + SBERT weak | as 1a+3b |
| ref | **oracle** 1a / 1a+2 on the true fields | upper bound only, never a result |

`role_match` is a soft agreement (the product or overlap of the two sides' family
probabilities for the classifier, exact match for rule-extracted spans). The formula is
fixed before scoring.

Registered comparisons (05-registration, written before held-out scoring): X2 − TF-IDF;
X2 − oracle 1a+2 (the cost of dropping the field); X4 − SBERT weak; X2 vs SBERT weak.
Metadata is still its own rung, with within-family AUC beside every headline.

## 4. Two stages

```
pool of candidate CVs ──► Stage 1: retrieve  ──► shortlist K=10 ──► Stage 2: rerank ──► ordered list
        (500 per role)    (text-only)                (Fit calls)     (judged CVs scored)
```

**Interface (fixed first, so the two stages develop independently):**

```python
class Retriever(Protocol):
    def shortlist(self, jd: str, pool: Sequence[str], k: int) -> list[int]: ...
class Reranker(Protocol):
    def score(self, jd: str, cvs: Sequence[str]) -> list[float]: ...   # higher = better, within one JD
```

Why concurrent works: the stages are judged on **different things**. Stage 1 is judged by
Recall of fits over big pools; it only has to put the right CVs in the shortlist. Stage 2
is judged on the **judged CVs per JD** (the 10 labelled), whatever stage 1 does. Stage-1
quality bounds what stage 2 can order, which is why stage 1 gets a recall bar and then stops.

### Stage 1 bars (registered before text-only scores are seen)

- **S2, unchanged, measured text-only:** Recall@50 of final-set fits in 500-candidate pools,
  **≥ 0.80** for the classical retriever (TF-IDF family) and **≥ 0.90** for the advanced
  (neural) retriever. (In the proposal these are "Recall@10 > 80% / > 90%"; the repo's S2
  measures Recall@50 on pilot pools.)
- **Recall@10: reported, no bar** until the number is seen. Setting a bar now, without a
  text-only figure, is the only way to keep it non-post-hoc.
- The retrieval pilot reuses `evaluation/retrieval_pilot.py` as it stands (pool construction
  unchanged). Pools are built with the keyword for **distractor sampling only** (a data
  construction aid), never as a feature.

### Stage 2

- **Scope:** orders only the CVs stage 1 / the first-stage fit call keeps. For evaluation,
  that means **judged CVs only**; an unlabelled shortlisted CV is dropped from the ordering
  metric and counted (the count is reported).
- **Candidates:** (i) SBERT weak (existing); (ii) a **cross-encoder** trained on the
  LLM-scored batch (04); (iii) a **bi-encoder** trained on the same data, so the
  cross-encoder is compared with the architecture it costs more than; (iv) the **LLM score
  itself** as a ranker (scored on `llm_free`, per D43); (v) baselines random, TF-IDF,
  SBERT zero-shot.
- **Training signal:** the 0–100 LLM score on the batch (04). Two forms are fixed in advance
  and chosen on the batch's own validation split: pointwise regression on the score, and
  pairwise preference from within-JD score gaps ≥ δ (δ from a two-point grid on validation).
  The pairwise form is expected to be more robust to the score's scale and drift.
- **No hand-tuning on the final set** (D46): hyperparameters and δ on the batch's validation
  split (JD- **and** CV-disjoint from training).

## 5. Ordering evaluation (secondary figure; classification stays Fit / No Fit)

- **Reference ordering:** human grades `yc` and `van` mapped Good = 2, Potential = 1,
  No Fit = 0. Reported against: `yc` alone (600 pairs), `van` alone (326), and the
  **mean of the human grades** where two exist (326 pairs; `yc` alone elsewhere is a
  separate, labelled row). The LLM's own score is **never** the reference.
- **Metrics:** per-JD **nDCG** (score ties share their mean gain; no tie-break luck) and
  per-JD **pairwise agreement** over pairs with different human grades (a score tie counts
  0.5). JDs with no differing pair are excluded and counted.
- **Reporting:** `generic`, `targeted` and `total` together, with *n* (JDs), and a 95% CI
  bootstrapped over **JDs**. System comparisons use **paired** deltas.
- **Baselines:** random, TF-IDF, SBERT zero-shot, and SBERT weak.
- **Bar:** a system must beat random (lower CI bound above random). **No human-ceiling bar**
  (user decision). Inter-annotator agreement is reported as context (D44).
- **Subsets:** every ordering figure is reported on the **full** set and on the **`llm_free`**
  subset (`final_set.LLM_FREE_TIERS` = `humans_agree`, `single_human`: pairs where no LLM label entered the final label), because the final label uses
  the LLM on 107 ties.
- **Fit-only variant:** the same metrics restricted to the pairs the final set calls Fit,
  since stage 2 only orders Fit CVs.
- **Not used:** `shortlist_pick` top-1 hit rate. It is a secondary, descriptive figure at
  most (59 picks, one annotator each).

## 6. End-to-end and the UI

The pipeline (`pipeline.py`) is rewired so the keyword and experience bands come from the
extractors (E2, E3), not the fields. The recruiter UI shows the extracted title, years and
skills as evidence. An unlabelled shortlisted CV is shown, never scored against labels.
