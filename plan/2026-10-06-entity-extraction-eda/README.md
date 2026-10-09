# Generated entity extraction EDA

6 October 2026. Authorized by the user's instruction to plan and execute EDA.
Extend the current transformer branch; leave all previously staged changes intact.
No model inference, dependency installation, training, raw-data mutation or new
annotation is needed. The prior commit attempt was interrupted; do not retry it
as part of EDA or combine its staged changes into a new commit.

## Deliverables and design

Create notebook `14-entity-extraction-eda.ipynb`, reusable analysis/plotting helpers
in `candidate_screener.extraction.eda`, standalone PNG charts and aggregate tables
under `output/extraction/eda/`, plus an implementation record and README link.
Use the completed GLiNER run named by `output/extraction/transformer-metrics.json`.
Validate its artifact hashes, sample identity, completion and exact original spans.
Reconstruct rules on the identical sampled training IDs, rather than comparing
GLiNER's 100+100 documents against notebook 11's 1,000+1,000 totals. State clearly
that the rule comparison is regenerated from unchanged rules, not another neural run.

Notebook sections:
1. Input/run inventory and schema, totals and coverage by CV/JD.
2. Side-by-side entity counts and per-label document coverage for both methods.
3. Entity counts per document, including zero-entity documents; label co-occurrence.
4. Most frequent real mention strings by label/source, with case-folded grouping
   documented as descriptive normalization, not canonical linking.
5. GLiNER score distributions by label/type and threshold sensitivity at 0.5–0.9.
   Higher thresholds describe retained counts, not estimated precision.
6. Exact-span/label overlap and disagreement; surface examples and bounded source
   context illustrate shared, rule-only and neural-only predictions. Treat these
   as inspection examples, never gold judgements or accuracy estimates.
7. Field/context distributions and limitations; summarize measured findings.

Include real extracted entity text as explicitly requested. Do not suppress neural
surfaces merely because they are real. Keep previews at five records, short bounded
context and 2,000 text characters; apply contact redaction inside the display path
as a low-cost safeguard, without altering original evidence offsets. Avoid complete
resume dumps. User permission covers real entity exploration, not unrelated data
publication. Charts/tables should remain useful rather than being truncated.
Use existing Matplotlib/Pandas/NumPy; notebook calls helpers and defines no model.
Persist real contextual example tables only under ignored local data; committed
chart/table artifacts show aggregates and a bounded entity vocabulary, no full texts.

## Validation and workflow

Delegate helpers and notebook development; independently review equal-ID comparison,
zero-document handling, overlap denominators and score interpretation. Tests use
synthetic inputs and require no weights/data. Show regression invariants rejecting
mixed populations and including zero-entity documents. Execute notebook and inspect
saved outputs/charts. Run full uv pytest and existing check commands; document prior
baseline/queue failures rather than fixing them in this task. Leave the prior staged
work untouched. No extra permission is needed for this user-authorized EDA scope.

## Open questions

No high-impact question blocks descriptive EDA on generated outputs. No gold labels
exist for these documents; extraction accuracy and downstream ranking usefulness
remain unmeasured. Future supervised evaluation remains outside this EDA scope.
