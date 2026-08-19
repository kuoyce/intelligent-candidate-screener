# D1 · ESCO v1.2.1 classification

| | |
|---|---|
| Catalog entry | [D1](../data-catalog.md) — Tier 1, skill ontology |
| Source | https://esco.ec.europa.eu/en/use-esco/download (registration required; CSV, English) |
| Retrieved | 19 Aug 2026 |
| On disk | `data/raw/esco_dataset-v1.2.1-classification/` — 40.3 MB (verified files: `skills_en.csv`, `occupations_en.csv`, `occupationSkillRelations_en.csv`) |
| Licence | European Commission ESCO terms of use — **redistribution of derived extracts is open question Q11** |
| PII | None |
| Verified rows | 13,960 skills · 3,043 occupations · 126,051 occupation↔skill links |

## Verified figures (19 Aug 2026)

- `skillType`: **10,734 skill/competence + 3,221 knowledge** — maps one-to-one onto
  SkillSpan's `tags_skill` / `tags_knowledge` layers.
- **~86,694 alternative labels** (median 6 per skill; only 18 skills have none), giving
  99,624 distinct surface forms with preferred labels included. This is the alias inventory
  §4.5 normalisation needs.
- `relationType`: **67,600 `essential` + 58,451 `optional`** — the **hard vs preferred
  requirement** distinction §4.5 calls for already exists here and does not need inventing.

## Measured limitation — tech-tool coverage

Against the 100 most frequent skills in [D2](D2-data-jobs.md):

| Match | Covered | Weighted by posting volume |
|---|---|---|
| Exact label or alias | **33 / 100** | 47% |
| Relaxed (every word known to ESCO) | 55 / 100 | 66% |

Absent even under relaxed matching: `aws`, `azure`, `tableau`, `snowflake`, `databricks`,
`gcp`, `kafka`, `docker`, `kubernetes`, `mongodb`, `pytorch`, `terraform`, `scikit-learn`,
`jira`, `github` and ~30 more. ESCO carries abstract competences; the misses are concrete
tools and products — exactly what a technical recruiter screens on.

**Consequence:** ESCO alone is insufficient for this domain. `data/vocab/skills.csv`
(action plan §3.5) must be ESCO preferred labels + aliases left-joined with D2 frequencies,
with unmatched high-frequency tool names added as project-local entries carrying a
`source = data_jobs` provenance flag.

Explored in [`notebooks/06-skill-vocabulary.ipynb`](../../../notebooks/06-skill-vocabulary.ipynb).
