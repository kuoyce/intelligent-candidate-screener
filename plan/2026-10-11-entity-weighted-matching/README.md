# Entity distribution EDA and a bounded matching experiment

Status: approved by the user; implementation in progress. User requested distribution plots and a new model
on 11 October 2026. This plan supersedes the earlier one-pair EDA proposal; use
all documents of an explicitly sized campaign rather than one illustrative pair.

## Evidence and constraints

- GLiNER extracts KNOWLEDGE, SKILL, TITLE, DEGREE and EXPERIENCE_YEARS. The frozen
  five-label experiment covers 100 CVs and 100 JDs, not the entire A2 corpus.
- Incoming X2 is TF-IDF plus `2 * role_match + experience_compatible`: soft role
  agreement is from a trained role-family classifier, and experience is from
  deterministic text rules. GLiNER spans currently do not enter this score.
- The existing skill-overlap rung already deduplicates concepts and divides by
  the JD skill count. The new hypothesis is about discriminatory weighting and
  evidence context, not fixing a raw-count bug in that rung.
- X2 uses JD Long Description. Adding the given JD title is an independent
  experimental change. CV Position and experience metadata remain excluded.
- The incoming stage-2 record reports no cross-encoder improvement and adopts
  SBERT weak ordering. Do not retrain another neural grid by default.
- Current benchmark classification is Fit/No Fit. Manual Good/Potential/No Fit
  grades may support descriptive EDA and graded ordering, but cannot be used to
  tune a new model on the frozen final set.

## EDA delivery

Add a thin `15-entity-distribution-and-matching.ipynb`, with reusable analysis and
plotting in the extraction package. Save standalone PNGs and aggregate tables.

1. Per-document heatmaps (rows: every sampled document; columns: all five labels),
   anonymous indices, with zero-mention documents retained.
2. Per-label histograms/box plots separately for CVs and JDs; total counts, median,
   coverage and sample sizes. These are descriptive prediction counts, not recall.
3. Repeated mention counts versus unique concepts, common versus rare concepts,
   and duration availability. Keep document length beside count distributions.
4. If independently labelled pairs overlap the extraction campaign, describe
   shared/required/unmatched evidence by human grade, with population and
   provenance stated. Do not manufacture a JD/CV pairing or extrapolate a sample
   to all resumes. Grade plots remain exploratory and do not select weights.
5. Historical artifacts can be inspected by verifying their saved hashes,
   manifests, original text spans and coverage, with producing code digest shown.
   Do not present them as replay under current code or silently bypass checks.

A read-only preview of the frozen 200-document campaign is prepared in `/tmp` for
review; it does not infer new entities or report model improvement.

## Proposed model and formula

Preserve the exact X2 as a control. Fit IDF from unique CV documents in the A2
training region only. Let J be unique affirmative JD SKILL/KNOWLEDGE concepts and
C be unique affirmative CV concepts, canonicalized through explicit vocabulary
aliases; unmatched phrases use exact normalized-string identity, not guessed IDs.
Negated/learning mentions do not prove possession. Missing JD skills carry an
availability flag; missing extraction is not a No Fit judgement.

    w(s) = min(5, 1 + log((N_train + 1) / (df_train(s) + 1)))
    coverage_idf = sum(w(s) for s in J intersect C) / sum(w(s) for s in J)

Empty J gives coverage 0 and an availability flag. Repetition cannot raise the
score, and unrelated CV skills do not add credit. Common skills have less weight;
the fixed cap limits the influence of rare extraction errors. Unknown canonical
phrases have a conservative weight of 1 until training support is available.

    score = 2 * role_match + experience_compatible
            + 0.999 * ((1 - alpha) * tfidf_scaled + alpha * coverage_idf)

`tfidf_scaled` retains X2's per-JD scaling. Register alpha in {0, .25, .5, .75, 1}
before development scoring; alpha 0 must reproduce the matching X2 text variant.
Weights remain bounded; more mentions cannot arbitrarily increase a CV's score.

Keep numeric duration semantics separate from span detection. GLiNER duration
spans require tested parsing of exact/minimum/range expressions and scope before
numeric compatibility. Never add overlapping durations, infer total from a
skill-specific claim, or equate no returned span with zero experience. Start with
the existing tested experience rule as E; a later ablation may replace or enrich
it only where GLiNER evidence has unambiguous scope. DEGREE and TITLE are evidence
features first, not universal hard gates or automatic proficiency signals.

## Registered comparisons and metrics

Use identical candidate pools for all systems. Separate changes:

- Original X2: original long-text representation, no new entities.
- X2 title control: given JD title + JD long text, same formula.
- Title control plus deduplicated unweighted JD concept coverage.
- Title control plus IDF-weighted coverage from vocabulary rules.
- Same weighting from GLiNER SKILL/KNOWLEDGE evidence (isolates extractor).
- Optional duration ablation after scope/normalization tests and coverage audit.

Stage 1 headline: Recall@50. Stage 2 headline: Recall@10 after reordering the same
stage-1 top 50. Show end-to-end Recall@10 as well as conditional shortlist
ordering; a reranker cannot recover a CV omitted by stage 1. Report generic,
targeted and total together, per-JD n and paired bootstrap CIs, with pool
judged-supply/unjudged-distractor limitations. Recall here is judged Fit recall,
not correctness of three-class labels.

Select alpha and representation on train-region development data only, using
existing 250/150-JD dev splits for retrieval. Their same-family proxy relevance
does not establish fit improvement: report the proxy honestly and use permitted
D50 validation ordering as a second signal when available. Freeze configuration
and adoption rule in a registration before any new final-set score. No changing
weights after viewing held-out labels. An improvement is a measured paired delta,
not a promise; retain the baseline if the candidate does not improve it.

## Compute and implementation sequence

1. Complete the approved active merge repair and retain its validation record.
2. After this plan's approval, create a new branch and commit the registration.
3. Deliver sample EDA first; do not extract the full corpus implicitly.
4. Implement tested feature extraction/scoring and a cached experiment runner;
   each model input and cache includes text, extractor and vocabulary provenance.
5. Time a ten-document inference pilot, extrapolate unique development documents,
   and report cost before a larger run. Existing 200-document inference took about
   174 seconds; 14,170 dev CVs alone could take several CPU hours. This is an
   estimate, not an authorized full-corpus campaign.
6. Run a bounded development pilot, with a 30-minute extraction budget. Partial
   coverage cannot be reported as a full comparison. A full development/final
   GLiNER campaign requires an agreed compute budget based on that measurement.
7. Delegate implementation and independent review; prove meaningful regression
   invariants fail old/broken scoring (repeat inflation, unrelated-skill credit,
   leakage of validation frequencies, zero-entity exclusion, alpha-zero drift).
8. Run targeted tests during work, the full suite once before review, and applicable
   cold checks through the registry. Execute notebooks and preserve output/PII caps.

## Open questions / approval choices

- Approval resolves the initial population to existing 100 CVs + 100 JDs, then
  a timed pilot rather than immediate corpus-wide extraction.
- Approval resolves the new model to the bounded ranking blend and registered
  ablations above; it does not guarantee improved held-out recall.
- Three-class classifier: no independent train-region human three-class targets
  identified. Proposed retrieval benchmark first; preserve human grades for
  descriptive evidence. A trained three-class classifier needs an agreed labelled
  training source/protocol; no new annotation wave is authorized by this plan.
- Earlier unspecified "other entities" is resolved provisionally by using the
  five already requested types; no sixth label is introduced.
- Larger inference and new final-set score are gated by measured budget and a
  committed registration, respectively.
