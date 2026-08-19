# C3 · `ahmedheakl/resume-atlas`

| | |
|---|---|
| Catalog entry | [C3](../data-catalog.md) — Tier 2, clustering EDA only |
| Source | https://huggingface.co/datasets/ahmedheakl/resume-atlas |
| Retrieved | 19 Aug 2026 |
| On disk | `data/raw/resume-atlas/` — 23.6 MB, 1 file |
| Licence | **MIT** |
| PII | Pre-normalised text; no direct identifiers observed |
| Provenance | Heakl et al., *ResumeAtlas* (arXiv 2406.18125) |
| Verified rows | 13,389 across 43 categories |

## Schema

`Category`, `Text`.

## Critical caveat — verified, not assumed

`Text` is **already lowercased, punctuation-stripped and stopword-removed**. A 200-document
sample contains **0 uppercase characters and 0 sentence punctuation marks**, e.g.

```
education omba executive leadership university texas 20162018 bachelor science accounting
richland college 20052008 training certifications certified management accountant cma ...
```

There are therefore no character offsets, no sentence boundaries and no displayable text.

## How it must be used

**§4.7 clustering EDA only.** It cannot support NER, evidence spans, or anything the
recruiter UI renders. §5.1 of the proposal should say so explicitly.

Explored in [`notebooks/05-raw-formats.ipynb`](../../../notebooks/05-raw-formats.ipynb).
