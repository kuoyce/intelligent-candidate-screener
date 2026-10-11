# Registration before development scoring

Approved experiment: an entity-weighted extension of X2. No new development or
final-set results have been computed for this experiment when this file is written.

## Fixed settings

- Text v2; compare original long-text X2 with given JD title + long-text controls.
- Five entity labels, pinned local GLiNER checkpoint, threshold 0.5 and 32-word
  overlap; unknown duration is not zero; experience term remains the tested rule.
- Unique, affirmative SKILL/KNOWLEDGE concepts; explicit vocabulary aliases and
  exact normalized unknown phrases. No metadata in model inputs except JD title.
- CV training-only document frequencies, IDF `min(5, 1+log((N+1)/(df+1)))`;
  unknown-concept weight 1, never learned from development/validation/final data.
- Alpha grid ordered `[0, .25, .5, .75, 1]`; alpha-zero is the exact matching X2
  control. Earlier alpha wins a tie rounded to six decimals.
- Original long-text X2, title X2, title plus unweighted vocabulary coverage,
  title plus IDF vocabulary coverage, and title plus IDF GLiNER coverage are
  separate controls/variants. No neural reranker training.

## Pilot population and selection

Use existing train-region development splits where locally reconstructible,
preserving 500-candidate pools. For the bounded GLiNER pilot, select the first
two dev-train and first two dev-select JDs in sorted ID order, before scores are
seen. The pilot is operational and cannot establish full-dev superiority. IDF
fitting uses separate training documents, not the dev-select concept distribution.

If the inherited role-model or dev-pool caches are unavailable, report that
prerequisite explicitly. Do not replace the trained role model with metadata,
invent human labels, or claim a synthetic probe is a retrieval experiment.
A train-only ten-document runtime pilot can still proceed independently.

GLiNER extraction has a 30-minute budget; incomplete documents/pools cannot be
scored as a completed comparison. Cache predictions by immutable document text,
model, labels, threshold, context/chunk settings and extractor code digest.

Rank selection: highest dev-select Recall@50; ties favor the earlier alpha and
then the unchanged control. Report Recall@10 and paired per-JD deltas too.
Only propose adoption when a paired JD-bootstrap interval for the primary delta
has lower bound above zero and Recall@10 does not fall. A four-query pilot cannot
authorize adoption; a full registered development comparison must precede it.
If D50 validation scores/texts are available, report their ordering separately;
they are not human three-class targets and decide no final-set thresholds.

Stage-2 figures reorder the identical baseline top 50 and keep the denominator
over the full 500-CV pool. A candidate absent from that top 50 cannot be recovered.
Final-set scoring requires a separate frozen adoption registration and agreed
inference budget; it is not part of this bounded pilot.
