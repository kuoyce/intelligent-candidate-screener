# Soft matching in the learned score

Status: **complete (exploratory).** Result and decision: [01-implementation-record.md](01-implementation-record.md). Exploratory under D56 (extend its wording on approval).
Follows [coverage-weighting-variants](../2026-10-13-coverage-weighting-variants/01-implementation-record.md).

## Question

Embedding matching of concepts at threshold 0.60 lifts the coverage feature by +0.07 per-JD AUC
(D50 selection), +0.02 to +0.04 (D50 held-out) and +0.10 (A1-LLM) over exact matching. Exact
coverage gains did not carry into the learned score before. **Does the soft-matched coverage let
a learned score beat hand-set X2 (0.850 on A2 final)?**

## What is fixed already

- **Threshold 0.60**, chosen in Part C on the 70 D50 selection JDs (positive against exact on both
  held-out sets); 0.55 and 0.65 are run as sensitivity and are **not** alternatives to choose from.
- **Concept source** R+N (seed rules plus the MiniLM tagger). **Weighting** W0: no variant beat it.
- Hand-set X2 stays `2·role + experience + 0.999·TF-IDF` with no coverage term.

## Selection discipline and a disclosure

A2 final is scored once, last. The decision to pursue soft matching was informed by Part B, which
printed A2 coverage-only AUC at 0.7 / 0.8 / 0.9, so A2 has already been seen for that feature;
only the threshold (0.60) is clean of it. That is stated in every result of this plan.

## Features (soft versions of the coverage block)

For a pair, each JD concept `j` has `s(j) = max cosine to any CV concept` (0 if the CV has none)
and is *matched* when `s(j) >= 0.60`. All other features are unchanged.

| Feature | Soft definition |
|---|---|
| `coverage` | weight-matched share: sum of IDF weight(j) over matched j / sum over all JD concepts (weights as W0) |
| `coverage_unweighted` | matched share, weight 1 |
| `log_matched` | `log(1 + #matched)` |

Feature sets: **base (4, soft)** = base four with `coverage` replaced; **all (11, soft)** = the
eleven with the three rows above replaced. The exact versions already exist as references.

## Procedure

1. Embed the distinct concept phrases of A2 final, A1 train, A1-LLM and D50 (local
   `all-MiniLM-L6-v2`, cached); no new tagger inference.
2. Refit, with soft features, exactly what the earlier Stage 2 did: the notebook 18 Part 2 grid
   (feature set × label source A / B / C) and the notebook 19 curves (10/25/50/70 JDs, 30 held out,
   20 repetitions) for logistic base, logistic all and boosting all, on A1-LLM and D50 labels.
3. Sensitivity: the same final fits at thresholds 0.55 and 0.65 (D50 held-out and A1-LLM only).
4. Score A2 final once per fit; report `generic`, `targeted`, `total`, n and a JD-bootstrap CI.

## Bar (unchanged, registered)

- A learner beats hand-set X2 only if its paired total ΔAUC CI is above 0 **and** `generic` and
  `targeted` both have a positive delta.
- A pass counts only if the same learner meets it on both the A1-LLM and the D50 final fit.
  With 3 learners × 2 label sources, a single nominal pass is not evidence.
- Isolating the match key: for each learner and label source, the paired delta of soft against
  the exact version (same features, same labels), reported with its CI.

## Decision rules (registered now)

- **Soft matching works** if a learner passes the bar. Then the follow-up question (more A1 LLM
  labels) reopens under the earlier rule: the A1-LLM curve must also rise by more than 0.01 from
  25 to 70 JDs.
- **Coverage is not the bottleneck** if no learner passes **and** the soft-vs-exact paired delta
  has no CI above 0 on D50 held-out. Then no further coverage-feature work in this line: the next
  levers are the role model and experience extraction (see the owner discussion of 13 Oct).
- Anything between is reported as inconclusive with the paired CIs.

## Constraints and cost

- Code in the package (`evaluation/`), inline synthetic tests, each regression test shown to
  fail against the code it guards; the default feature path stays byte-identical.
- No labels are read by feature code; the scorer reads A2 labels. Branch unchanged.
- Cost: embeddings a few minutes; refits about 3 min per arm; the three-threshold sensitivity
  is three more final-fit runs. Under an hour in total; I will time one fit first.

## Out of scope

Graded (similarity-valued) matching without a threshold, per-concept learned weights, a new
role model, new experience extraction, a different model class. Each is a separate plan.

## Choices explained, and decisions

1. **"Binary at 0.60" in plain words.** For each concept in the JD we find the most similar
   concept in the CV (cosine similarity between phrase embeddings, 0 to 1). *Binary* means we
   then ask a yes/no question: is that best similarity at least 0.60? If yes the concept counts as
   matched, otherwise it does not, and coverage is the share (weighted or not) of JD concepts that
   matched. The alternative, *graded*, would use the similarity itself (0.72, 0.41, ...) as the
   amount of credit and need no cutoff. We use binary because the threshold search (Part C) was
   run for exactly this form; graded matching is a possible later variant, not rejected.
2. **"K2 containment" in plain words.** A cheaper matching rule tried in Part B: two phrases
   match when the words of one are all inside the other (`react` matches `react developer`;
   `sql` does not match `sql server` because the shorter phrase must have at least four
   characters). Soft matching beat it on the selection set at every threshold from 0.50 to
   0.80, so containment is not used here. It stays in the records as a reference.
3. **Original A1 labels (arms A and B): keep.** Decided by the owner, 13 Oct 2026. The grid
   keeps all three label sources (A original all, B original matched, C LLM matched), as in the
   earlier Stage 2. They are cheap, and the pass rule still counts only the A1-LLM and D50 fits.

Open: none.
