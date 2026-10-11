"""Skill extraction: the vocabulary, the rules matcher and the learned tagger (P5).

- `vocab`: builds `data/vocab/skills.csv` (ESCO ICT scope + `data_jobs` + the 8 baseline
  regex families), deterministic and digested.
- `rules`: the longest-match phrase matcher over that vocabulary (fixed interface).
- `ner`: a MiniLM token classifier fine-tuned on SkillSpan train.
- `normalise`: alias → canonical label, string matching against SBERT.
- `metrics`: span-level strict / partial P/R/F1 with a bootstrap CI.
- `evaluate`: the committed record `output/extraction/extraction-metrics.json`.
- `title_profile` (plan 2026-10-06 P2) and `role_family` (P3, E2): job title -> role
  family from text only; record `output/extraction/role-family-metrics.json`.
- `transformer`: local five-label GLiNER inspection, including EXPERIENCE_YEARS.
  Package exports below retain the independent field-relative seed-rule workflow
  and its exact label/layer scorer alongside the skill-extraction APIs.

The design is `plan/2026-09-26-top-down-reset/02-design-spec.md` §6; the record of what
was built is `08-extraction.md` in the same directory.
"""

from .rules import extract_document
from .schema import EXTRACTOR_VERSION, validate_entity
from .vocabulary import VOCABULARY_VERSION, vocabulary_digest
from .metrics import score_exact_spans

__all__ = ["extract_document", "validate_entity", "EXTRACTOR_VERSION",
           "VOCABULARY_VERSION", "vocabulary_digest", "score_exact_spans"]
