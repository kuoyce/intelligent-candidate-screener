# B1 · DataTurks — *Entity Recognition in Resumes*

| | |
|---|---|
| Catalog entry | [B1](../data-catalog.md) — Tier 2, resume-native NER seed (decision **D5**) |
| Source | https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy (GitHub mirror, no Kaggle account needed) |
| Retrieved | 19 Aug 2026 |
| On disk | `data/raw/dataturks/` — 1.2 MB, `traindata.json` (200 docs) + `testdata.json` (20 docs) |
| Licence | **None declared** on the repository |
| PII | **DIRECT** — see below |
| Verified rows | 220 documents, 3,556 spans, median 2,931 chars/doc |

## Schema

JSON-lines, spaCy-era: `{content, annotation: [{label, points: [{start, end, text}]}]}`.
`end` is **inclusive**.

## Verified span inventory (19 Aug 2026)

| Entity | Spans | | Entity | Spans |
|---|---|---|---|---|
| Companies worked at | 729 | | Graduation Year | 254 |
| Designation | 521 | | **Email Address** | **252** |
| **Skills** | **472** | | **Name** | **224** |
| Location | 430 | | Years of Experience | 44 |
| College Name | 330 | | UNKNOWN | 2 |
| Degree | 298 | | | |

## Known defects

1. **Direct PII is the label set.** `Name` (224) and `Email Address` (252) are annotated
   entity classes; 5 distinct emails and 17 phone-like strings also remain in the raw text.
2. **222 spans (6.2%) have broken offsets** — the stored `text` does not match the slice its
   offsets point at (inclusive/exclusive `end` plus stray whitespace). Converting these to
   BIO without repair puts tags on the wrong tokens and produces silently wrong NER scores.
3. **`Skills` spans are multi-skill blobs**, frequently a whole skills section rather than
   one skill. They need segmentation against the ESCO vocabulary before skill-level scoring.
4. **20 test documents** cannot carry the §7 extraction-F1 target.

## How it must be used

- De-identify first (action plan §3.1): strip the `Name` / `Email Address` classes and mask
  residual emails and phone numbers. **Never commit un-redacted content** — notebook 04
  redacts at display time for exactly this reason.
- Repair or drop every broken span, logging each one.
- Evaluate with **cross-validation over all 220 documents**, reporting mean ± std. Do not
  report a single F1 on the 20-document shipped split.
- Supplement with B2 SkillSpan, which carries ~20× the skill supervision.

Explored in [`notebooks/04-extraction-sources.ipynb`](../../../notebooks/04-extraction-sources.ipynb).
