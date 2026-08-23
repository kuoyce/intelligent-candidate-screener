# B2 · `jjzha/skillspan`

| | |
|---|---|
| Catalog entry | [B2](../data-catalog.md) — Tier 1, primary skill-span supervision |
| Source | https://huggingface.co/datasets/jjzha/skillspan |
| Retrieved | 19 Aug 2026 |
| On disk | `data/raw/skillspan/` — 0.7 MB, 3 files |
| Licence | **CC-BY-4.0** |
| PII | Pre-anonymized — entities replaced with placeholders such as `<ORGANIZATION>` |
| Provenance | Zhang et al., *SkillSpan: Hard and Soft Skill Extraction from English Job Postings*, NAACL 2022 |
| Verified rows | 11,543 sentences (4,800 train / 3,174 validation / 3,569 test) |

## Schema

`idx` · `tokens` (list) · `tags_skill` (BIO) · `tags_knowledge` (BIO) · `source`

Two **independent BIO layers over the same tokens**: soft/behavioural skills and
hard knowledge/tools.

## Verified figures (19 Aug 2026)

| Split | Sentences | Tokens | Skill spans | Knowledge spans |
|---|---|---|---|---|
| train | 4,800 | 93,453 | 2,221 | 2,969 |
| validation | 3,174 | 40,057 | 1,070 | 1,093 |
| test | 3,569 | 42,786 | 1,090 | 1,174 |
| **total** | **11,543** | **176,296** | **4,381** | **5,236** |

9,617 spans in total — against 472 `Skills` spans in B1, i.e. **~20× the skill supervision**.

> The catalog previously carried the card claim ">12.5K annotated spans" and an estimated
> "~26×". The measured figures above supersede both.

## Known limitations

- **Domain shift.** Annotated on **job postings, not resumes**. Resume-native evaluation
  must still come from B1 (de-identified), with the domain gap stated.
- The skill/knowledge division has to be mapped onto the project's label scheme.

## Label alignment (decided in notebook 04)

| Project label | SkillSpan layer | ESCO `skillType` | DataTurks |
|---|---|---|---|
| `SKILL` | `tags_skill` | `skill/competence` (10,734) | `Skills` after segmentation |
| `KNOWLEDGE` / tool | `tags_knowledge` | `knowledge` (3,221) | `Skills` after segmentation |
| `TITLE`, `ORG`, `DEGREE`, `EDU`, `YEARS` | — | — | native classes |

One scheme covers all three sources.

Explored in [`notebooks/04-extraction-sources.ipynb`](../../../notebooks/04-extraction-sources.ipynb).
