# Error analysis 2: where is the headroom?

Status: **approved by the owner 14 Oct 2026; implementation in progress.** Exploratory (D56). It
continues notebook 20, which found where X2 errs but not how much each cause is worth.

## Question

Four things could be holding the score back: the role signal, missing experience, label noise,
and concept coverage. **How much AUC is each worth, so the next plan goes at the biggest?**
Notebook 20 gave the shape (on A2, 71% of X2's mistakes are between two CVs that are both out of
the JD's role family; "Fit out of role, No Fit in role" is only 9% of mistakes though 94% of those
pairs are misordered; years are known on about 29% of pairs; LLM tie-break labels invert 27% of the
time against 10-14% for human labels). It did not size them.

## Scorers analysed

- hand-set X2 (`2·role + experience + 0.999·TF-IDF`);
- the base-4 logistic (exact coverage), fitted on the D50 selection JDs (70) as in the previous
  plan;
- the soft-coverage version of that logistic **only if** the soft-matching plan has run.

## Where the analysis runs (rule 8)

- **Design and sizing on D50** (A2 train region, LLM labels, 61% Fit, 80% same-keyword): split
  into the same 70 selection / 30 held-out JDs as the previous plan.
- **A2 final is for diagnosis and for confirming one finding**, as notebook 20 did. Nothing is
  fitted on it (D46). A feature idea designed from these errors is confirmed on a set it was not
  designed on, in its own plan.
- The same-keyword rate on D50 (80%) makes a role oracle weak there, so the role headroom is
  also sized on A2 final, labelled as such.

## Steps

1. **Who causes each inversion.** For every (Fit, No Fit) pair in a JD that the score orders the
   wrong way, split the score gap into feature terms (`weight × (x_NoFit - x_Fit)`) and name the
   term that contributes most to the flip (role, experience, TF-IDF, coverage). Report the share
   of inversions per culprit, by stratum, grade of the Fit side, and label tier.
2. **Oracle ablations (headroom).** Replace one input with a better one and re-score. Each
   result is a ceiling for that lever, **diagnostic only, never a system**:
   - *role oracle*: `role_match` replaced by the same-Primary-Keyword flag (metadata, forbidden
     as a system input by D53, allowed as a diagnostic);
   - *label ceiling*: AUC on the `humans_agree` tier only, and on `single_human`, against all;
   - *coverage*: exact versus soft (from the soft-matching plan) as the measured lever;
   - *experience*: not an oracle (the true years are not available); sliced instead (step 3).
3. **Slices.** Per-JD AUC and inversion rate by: experience known / unknown (both sides), zero
   concept overlap, JD concept count (<= 3 or more), CV length tercile, stratum.
4. **Manual review (done by the owner).** After steps 1-3 are read.
   - *Sample:* a mistake is a (Fit, No Fit) pair in one JD that the score orders the wrong way.
     Step 1 puts each mistake into a bucket by the feature term that caused it (role, experience,
     TF-IDF, coverage). Draw up to **20 mistakes per bucket** (fewer if a bucket is smaller),
     seed 0, from the D50 selection JDs and, for the role bucket, also from A2 final (labelled).
     Cases are shown in random order with the bucket hidden.
   - *What the reviewer sees:* the JD, the Fit CV and the No Fit CV, all passed through
     `annotation.redact`. A collapsed panel holds the model details (both scores, feature values)
     and the final-set LLM note (D43: diagnostic reading only). Opening it before answering can
     anchor the answer, so the reviewer answers first when they can.
   - *What the reviewer records, per case, one of:* **A** role mismatch but the Fit CV has
     transferable skills; **B** seniority or years are wrong; **C** the No Fit CV matches on
     words but lacks the core requirement; **D** the label looks wrong; **E** other (with a short
     note). The list is fixed here, before anything is read.
   - *Tool:* a small Streamlit page (`evaluation/error_review_app.py`, thin) over tested logic in
     `evaluation/error_review.py` (sampling, redaction, saving). It shows one case at a time with
     the five choices and a note box, saves every answer at once, can be closed and resumed, and
     shows progress per bucket. The frozen annotation instrument (`annotation/`, D42) is not
     touched or reused.
   - *Storage:* append-only `output/extraction/error-review-reasons.csv` with `pair_id`, bucket,
     reason, note, reviewer, timestamp, sample seed and scorer version (the latest row per case
     wins). No resume or JD text is stored. **These are error-reason tags, not relevance labels**:
     they never enter `judgements.csv`, the final set or any training set (D42, D46). A decision-log
     entry records this on approval.
   - *Time:* about 1-2 minutes per case, roughly 2 hours for 80 cases.

## Measures and the rule for what comes next (registered now)

- Per-JD AUC (generic, targeted, total) with JD-bootstrap CIs; ablation deltas are **paired**
  against the unablated score.
- A lever is **worth its own plan** if its oracle or slice shows a paired total delta of at
  least **0.02** with a CI above 0 (owner-confirmed 14 Oct 2026). If two qualify, the larger goes first. If none does, the
  honest finding is that the remaining error is label noise or irreducible with these inputs.
- The manual review informs what to build; it does not decide whether a lever qualifies.

## Constraints and cost

- Code in `evaluation/error_analysis.py` (extending `a1_weights.pair_inversions`); a thin
  notebook reports it. Tests on inline synthetic data, each shown to fail against the code it
  guards.
- Minutes of compute (features cached from the earlier runs). The manual review is the long part.
- No new labels, no model dispatch (D42).

## Decisions (owner, 14 Oct 2026) and remaining questions

1. **Bar of 0.02:** confirmed.
2. **Manual review:** done by the owner, in the Streamlit tool above. 20 per bucket is the
   default; say if you want 10.
3. **Order:** this plan runs first, once finalised; steps 1-3 (minutes) before the review.

Open: none that block approval. Assumed, please confirm: reason tags live in a separate file
and are not labels (decision-log entry added on approval).

## Results of steps 1-3 (14 Oct 2026; record `output/extraction/error-analysis-2.json`)

Code: `evaluation/error_analysis.py` (6 tests, mutation-checked). Reported in
[notebook 21](../../notebooks/21-error-findings-and-manual-review.ipynb); this section is the summary.

- **Culprit term (arithmetic attribution, not cause).** A2, hand-set X2: TF-IDF 56%, role 33%,
  experience 12% of mistakes. Base-4 logistic: TF-IDF 48%, role 35%, coverage 12%, experience 5%.
- **Role oracle** (role_match replaced by the same-keyword flag): hand-set -0.002 [-0.042, 0.031],
  logistic +0.012 [-0.026, 0.042]. **Neither reaches the 0.02 bar**, so a perfect role signal is
  not shown to be the headroom. (An earlier suggestion that the role model was the top lever rested
  on misreading notebook 20: 94% is the inversion rate in a 16-pair group, not a share of mistakes.)
- **Label tiers (A2, different pair subsets):** AUC 0.92 where two humans agree (219 pairs), 0.88
  with one human (274), **0.72 where an LLM broke a human tie (107)**. The largest gap found.
- **Slices (consistent across A2 and both D50 sets):** pairs with no concept overlap and pairs with
  short CVs are misordered more; long CVs least. Experience-known pairs are not consistently better.
- **By the registered rule no lever qualifies from steps 1-3** (no oracle or slice delta of >= 0.02
  with a CI above 0). The manual review (step 4) decides what to build next.
