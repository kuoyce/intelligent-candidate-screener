# Design Specification — Unified Judging Wave

## 1. Target layout

```
src/candidate_screener/
  annotation/
    queue.py          builds the judging queue; the selection strategies live here
    layer.py          loads our judgements, applies them as an overlay
  evaluation/
    metrics.py        unchanged API; gains `attainable_ceiling` (task 5.2)
docs/data/manifests/
  judgements.csv          COMMITTED — our labels. The whole layer, both corpora
  judging-queue.csv       COMMITTED — what was dispatched, and why
  judging-report.json     COMMITTED — counts, κ, contamination rate, ceiling shift
docs/annotation-guide.md  COMMITTED — extended, not replaced (task 3.4a owns it)
```

`judgements.csv` is committed for the same reason `data/processed/indomain/labels.csv` is
(design spec §5, D15): **labels are our own work and carry no source text.**

## 2. The judgement layer

One table, both corpora, both objectives. A pair appears **at most once**.

| Column | Values | Note |
|---|---|---|
| `pair_id` | `sha256(corpus\|query_id\|doc_id)[:12]` | Content-addressed, per design spec §2 |
| `corpus` | `a1`, `a2` | |
| `query_id`, `doc_id` | `j_…` / `r_…` (A1) · publisher `id` (A2) | Identity scheme is already per-corpus; §2 of the Phase 3 spec |
| `region` | `train`, `eval`, `-` | A2 only. **D19's guarantee**, enforced here not by convention |
| `selection_reason` | `banded_lexical` · `pool_topk` · `double_label` · `a1_recheck` | See §4 |
| `surfaced_by` | `stage1,stage3` (comma-set) or empty | Which systems put it in top-k. **Never shown to the annotator** |
| `best_rank` | int or empty | Best rank across systems. Likewise hidden |
| `annotator` | initials | |
| `label` | `good`, `potential`, `no` | A1's 3-class scheme (D17) |
| `shortlist_pick` | bool | Q14's one top-1 pick per query |
| `notes` | free text | |

### Why this cannot be a column in `pools.csv` (D20)

`verify --derived` acceptance check 6 asserts that `build --seed 0` reproduces every manifest
**byte-for-byte**. `pools.csv` is a pure function of (`data/raw/`, seed). A human judgement is
not a function of a seed. Merging the two would have forced the choice between deleting the
determinism check and never re-running the builder. D20 avoids the choice; it is not a
presentational preference.

## 3. The overlay, and dual reporting

`layer.apply(pools, judgements) -> pools'` returns a **new** frame with `relevance` updated
where we judged, plus a `label_source ∈ {a1, ours}` column. `metrics.score` is untouched — it
reads `pool.relevance` and does not care who wrote it.

Every A1 figure is therefore computed twice, from the same code path:

| View | Pools passed | What it is for |
|---|---|---|
| **A1-only** | `pools` | Comparable to published work on this benchmark. The A1 card's "not comparable" note already limits this; D20 keeps it from getting worse |
| **A1+ours** | `layer.apply(pools, judgements)` | The honest internal figure. Higher precision, higher *n*, higher ceiling |

**Both are reported, always, side by side.** A single number labelled neither is the failure
mode D20 exists to prevent.

Three consequences to carry into §10 of the report:

1. Our annotators are *domain-literate business-analytics students, not professional recruiters*
   (3.4a states this for the report). **A1+ours mixes two label provenances**, and the `label_source`
   column is what makes that auditable rather than merely disclosed.
2. The **ceiling moves**. Task 5.2's attainable ceiling must be computed per view; the A1+ours
   ceiling is strictly higher, because every new `good`/`potential` adds judged-relevant supply.
   This is why Q27 is harder under D20 than it was when it was written.
3. ***n* moves too, under D23.** A1-only stays at *n* = 64 graded / 31 strict; A1+ours may be
   higher, because a query with no judged relevant document becomes scoreable the moment one of
   our judgements marks something relevant. **A value and an *n* must come from the same view** —
   `Figure` enforces that an *n* exists, not that it is the right one.

## 4. Selection strategies

`selection_reason` is the only field distinguishing the two objectives. It is metadata for us,
never an input to the judgement.

| Reason | Corpus | When | Selected by |
|---|---|---|---|
| `banded_lexical` | a2 | Wave 1 (task 3.4b, **unchanged**) | TF-IDF/BM25 bands, high/mid/low |
| `a1_recheck` | a1 | Wave 1 | Random sample of pairs **A1 already labelled** |
| `pool_topk` | a1, a2 | Wave 2 | Union of every stage's top-k, deduplicated. **All 100 A1 queries (D23)**, not only the 64 scoreable |
| `double_label` | both | Both | 30% overlap for κ |

### Wave 1 stays a fair sample — deliberately

The Stage 1 baseline exists (`6de2d3c`), so wave 1 *could* be pooled. It must not be.

A pool drawn from one system is that system's top-k under another name. It would hand Stage 1 a
fully-judged top-k while Stages 2–4 surface unjudged documents that score as false positives —
which is Q18's bias relocated onto the newer models, not removed. Pooling is only unbiased over a
**union of diverse runs**, which is why D22 puts it at the end.

### `a1_recheck` — the cheap test of A13

The Q18 analysis rests on assumption **A13**: *the A1 labels are correct.* Nothing has tested it.

Seed ~50 already-judged A1 pairs into the wave-1 queue, blind and indistinguishable from the
rest, stratified across `good` / `potential` / `no`. Comparing our labels against A1's yields a
plain agreement rate and a κ. If they agree, A13 is measured rather than assumed and the whole
Q18 chain firms up. If they do not, that is a larger finding than Q18 — and far better learned in
wave 1 than after four stages have been reported.

## 5. Blinding and presentation

1. `surfaced_by` and `best_rank` are stripped from the annotator's view. Rank order is a strong
   prior; showing it measures the annotator's trust in the system.
2. The queue is **shuffled** before dispatch, with a recorded seed. Never grouped by query,
   system, or `selection_reason`.
3. `a1_recheck` pairs are indistinguishable from fresh ones.
4. **PII (AGENTS.md rule 4).** A1 resumes are real documents — 44 of them overlap livecareer. The
   A1 queue is de-identified before it reaches an annotator, by the same sweep task 3.1 applies to
   DataTurks. The A2 queue does not need this. `verify --derived` gets the regex sweep over the
   dispatched queue, not just over `data/interim/`.

## 6. Determinism and commit policy

Queue construction is deterministic given (system scores, seed, k). Judgements are not
reproducible by construction and are therefore **committed as data**, not regenerated — the one
place in this project where a manifest is an input rather than an output.

`verify --derived` gains:

1. **Uniqueness** — no `(corpus, query_id, doc_id)` judged twice; `pair_id` collision-free.
2. **Region isolation (D19)** — every `corpus=a2, selection_reason∈{banded_lexical, pool_topk}`
   row has `region=eval`. A `train`-region pair in the evaluation layer is a hard failure.
3. **Identity** — every `query_id`/`doc_id` resolves in the corpus it claims.
4. **Overlay soundness** — `layer.apply` never *downgrades* an A1 label. If we judged a pair A1
   already judged, it lands in the `a1_recheck` report as a disagreement; it does not silently
   overwrite the benchmark.
5. **Blinding** — the dispatched queue file contains no `surfaced_by`, `best_rank` or
   `selection_reason` column.
6. **De-identification** — regex sweep for emails and phone numbers over the dispatched A1 queue
   returns zero.
