# Data Catalog

**Compiled:** 19 August 2026 · **Verified** = downloaded and profiled directly; **Card claim** = taken from the publisher's description, unconfirmed.

**Acquisition status: complete.** All eleven adopted sources are on disk (676 MB) and pass
`verify --all`. Per-source dataset cards are in [`cards/`](cards/); per-file sizes and
SHA-256 digests in [`acquisition-manifest.json`](acquisition-manifest.json); the measured
figures behind every **Verified** claim below in
[`profile-metrics.json`](profile-metrics.json). Reproduce with:

```bash
uv run python -m candidate_screener.data.fetch   --all
uv run python -m candidate_screener.data.verify  --all
uv run python -m candidate_screener.data.profile --all
```

Four figures below were **superseded during acquisition** — B2's span volume, C1's
contamination count, D1's tech coverage and D2's skill vocabulary. Each is marked
*Superseded 19 Aug 2026* in place.

Legend — **Tier 1** adopt, **Tier 2** adopt with stated caveats, **Tier 3** evaluated and rejected (recorded so the decision is traceable).

---

## A. Candidate–JD matching (the core progression, R2 + R3)

### A1 · `cnamuangtoun/resume-job-description-fit` — **Tier 1, core benchmark**

| | |
|---|---|
| URL | https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit |
| Size | **Verified** 6,241 train / 1,759 test = 8,000 rows |
| Effective size | **Verified** 642 + 477 unique resumes; 280 + 71 unique JDs |
| Schema | `resume_text` (str), `job_description_text` (str), `label` ∈ {No Fit, Potential Fit, Good Fit} |
| Labels | **Verified** 50.4 / 24.9 / 24.7 % train; 48.7 / 25.2 / 26.0 % test |
| Licence | **None declared** — see open question Q1 |
| PII | Resumes are livecareer-style; names largely stripped, but employers/schools remain |
| Popularity | 798 downloads/30d, 80 likes, last modified 2024-07-25 |
| Role | Common dataset for Stages 1–4 (§5.3.2), and the substrate for synthetic retrieval pools |
| Label quality | *Added 30 Aug 2026 (D33).* **The labels do not reproduce — A13 fails.** On a 50-pair recheck, two judges blind to A1's answer both land at chance against it: kappa(A1, human) = 0.010 (n=41), kappa(A1, `llm:claude-sonnet-5`) = **0.029** (n=50), **0.000** under binary collapse — while agreeing with each other at kappa = 0.291 (0.424 collapsed). The failure is in A1's positives: 14 of its 15 `Good Fit` labels read as `No Fit` to both. **D26's 0.7806 / 0.7188 ceilings are computed off these labels and are therefore soft**; they are not restated here. See the [card](cards/A1-resume-job-description-fit.md) defect 5 and `plan/2026-08-30-llm-recheck/01-findings.md` |
| Caveats | 99.8% resume leakage across the shipped split; 6 conflicting-label pairs; general-industry, not IT-specific; **the labels do not reproduce (above)** |

**Pool feasibility (Verified)** — this determines whether decision D1 is executable:

| | Train | Test |
|---|---|---|
| Resumes per JD — mean / median / max | 22.3 / 14 / 111 | 24.8 / 20 / 89 |
| JDs with ≥5 labelled resumes | 256 | 69 |
| JDs with ≥1 Good Fit | 115 | 28 |
| JDs with ≥2 Good Fit | 86 | 25 |

**Relevance density (Verified)** — Good Fit resumes per test JD: median **18**, Q1 3, Q3 24, max 48.

Two consequences. Pools are natively only ~14–25 candidates, so they **must** be padded with
distractors — pool N = 100 is the adopted setting. And because the median JD has 18 relevant
resumes, **Recall@10 is capped at 0.56 for the median query** on this, the *shipped*, split.

> **Superseded 23 Aug 2026 by task 3.3 (Q17).** The density above is the shipped split's. The
> leak-free split holds resumes out, so each query keeps only its held-out judgements and the
> test-side median falls from 18 to **6 Good Fit per JD**. Measured on the built pools,
> **Recall@10 reaches 0.90 for 93.5% of queries and 1.0 for 87%** — so the ban on Recall@10 is
> lifted and it becomes the primary recall metric. In the same move **Recall@50 is retired**:
> at 6 relevant in a 100-deep pool every query's ceiling is 1.0 and the metric saturates, and
> over the N20 variant it is not even defined. The candidate universe is **193** test resumes,
> not 477. See [`docs/data/manifests/pools-yield.json`](manifests/pools-yield.json) and
> [`plan/2026-08-23-derived-artefacts/05-implementation.md`](../../plan/2026-08-23-derived-artefacts/05-implementation.md).

### A2 · `lang-uk/recruitment-dataset-*` (Djinni) — **Tier 1, in-domain evaluation (R4)**

| | |
|---|---|
| URLs | [job descriptions (en)](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) · [candidate profiles (en)](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english) |
| Size | **Verified** 141,897 JDs · 210,250 candidate CVs (English splits; Ukrainian splits also exist) |
| JD schema | `Position`, `Long Description`, `Company Name`, `Exp Years`, `Primary Keyword`, `English Level`, `Published`, `id` |
| CV schema | `Position`, `Moreinfo`, `Looking For`, `Highlights`, `Primary Keyword`, `English Level`, `Experience Years`, `CV`, `id` |
| Licence | **MIT** (Verified via HF API) |
| PII | **Pre-anonymized by the publishers** — the stated purpose of the release |
| Provenance | Djinni IT hiring platform, 2020–2023. Drushchak & Romanyshyn (2024), *Introducing the Djinni Recruitment Dataset*, UNLP @ LREC-COLING 2024 |
| Role | In-domain end-to-end evaluation set; replaces the "200+ self-collected JDs" plan |
| Caveats | **No fit labels** and no interaction/match table (Verified — only four datasets exist in the org). Ukrainian/EE IT market — see Q5 |
| Join key | **Verified** `Primary Keyword` has 45 JD-side and 41 CV-side values, **41 of 41 shared**. Largest shared families are `JavaScript, Java, DevOps, .NET, QA Automation, Node.js, PHP, Python, Project Manager` — squarely the project's target domain |
| Length | **Verified** median **JD 1,629 chars, CV 751 chars** — the CVs are ~7× shorter than A1's resumes (5,134), a transfer risk for anything tuned on A1. Add to §10 |

**Why this solves R4.** It is IT-domain, both-sided, licence-clean and needs no scraping or
consent process. `Primary Keyword` (role family) is present on *both* CVs and JDs, and with
`Experience Years` and `English Level` gives a structured basis for constructing in-domain
candidate pools; the team then adjudicates fit labels on a sample (decision D4) rather than
inventing the pairs from nothing.

### A3 · `netsol/resume-score-details` — **Tier 2, optional comparator**

| | |
|---|---|
| URL | https://huggingface.co/datasets/netsol/resume-score-details |
| Size | **Card claim** 1,031 resume–JD samples, 648 matched |
| Provenance | **Generated and scored by GPT-4o** |
| Role | Only as an LLM-labelled comparator for §5.3.2's optional Stage-5. Cannot serve as ground truth |
| Caveats | Synthetic labels; using it to validate an LLM comparator is circular |

### A4 · `0xnbk/resume-ats-score-v1-en` — **Tier 3, reject**

| | |
|---|---|
| URL | https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en |
| Size | **Verified** 5,099 train / 1,275 validation = 6,374 |
| Schema | `text` (resume+JD concatenated), `ats_score` (18.3–90.7), `original_label` |
| Licence | Apache-2.0 |
| **Reject because** | **Verified** 97.2% of rows are `cnamuangtoun` resume+JD concatenated — not an independent signal. The `[SEP]` separator its card documents is **absent from all 6,374 rows**, so the resume/JD boundary is unrecoverable from the file. Its score is itself an embedding-similarity output, making retrieval "calibration" circular |
| If retained anyway | The boundary is recoverable by prefix-matching against A1 (`profile --recover-ats-boundary`). The circularity objection is unaffected |

---

## B. Information extraction / NER (R1)

### B1 · DataTurks *Entity Recognition in Resumes* — **Tier 2, resume-native seed (decision D5)**

| | |
|---|---|
| Source | GitHub mirror: https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy (`traindata.json`, `testdata.json`) — **no Kaggle account required** |
| Also at | https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner |
| Size | **Verified** 220 docs (200 train / 20 test), 3,556 spans, median 2,931 chars/doc |
| Entities | **Verified** Companies 729 · Designation 521 · **Skills 472** · Location 430 · College 330 · Degree 298 · Grad Year 254 · Email 252 · Name 224 · Years-of-Experience 44 |
| Licence | **None declared** on the repo (459 stars, last push 2019) |
| PII | **Direct** — `Name` and `Email Address` are annotated entity classes; 5 distinct emails and 17 phone-like strings remain in the raw text |
| Format | JSON-lines, spaCy-era `{content, annotation:[{label, points:[{start,end,text}]}]}` |
| Caveats | **6.2% of spans (222) have offsets that do not match their stored text** — inclusive/exclusive `end` bug. Must be repaired or dropped when converting to BIO. 20 test docs is too few for the §7 F1 target |
| Role | Resume-native gold seed; de-identify before use; supplement with B2 |

### B2 · `jjzha/skillspan` — **Tier 1, primary skill-span supervision**

| | |
|---|---|
| URL | https://huggingface.co/datasets/jjzha/skillspan |
| Size | **Card claim** 14.5K sentences, >12.5K annotated spans; **Verified** train/validation/test splits present, ungated |
| Schema | **Verified** `idx`, `tokens`, `tags_skill`, `tags_knowledge`, `source` — BIO tags, two span layers |
| Licence | **CC-BY-4.0** (Verified) |
| PII | Pre-anonymized — entities replaced with placeholders such as `<ORGANIZATION>` |
| Provenance | Zhang et al., *SkillSpan: Hard and Soft Skill Extraction from English Job Postings* (NAACL 2022) |
| Role | The volume and quality that DataTurks lacks — **~20× more skill supervision** (measured), peer-reviewed guidelines, clean licence |
| **Verified size** | *Superseded 19 Aug 2026:* **11,543 sentences** (4,800 / 3,174 / 3,569), 176,296 tokens, **4,381 skill spans + 5,236 knowledge spans = 9,617 total**. The card's ">12.5K spans" and the earlier "~26×" estimate are replaced by these counts — see [card](cards/B2-skillspan.md) |
| Caveats | Annotated on **job postings, not resumes** — a documented domain shift. Distinguishes *skill* from *knowledge* spans, which must be mapped onto the project's label scheme |

### B3 · `jjzha/green` — **Tier 2, supplementary**

CC-BY-4.0 (Verified), same annotation family and schema as B2, 1K–10K size band. Adds coverage;
adopt only if B2 proves insufficient.

### B4 · `jjzha/kompetencer` — **Tier 3, reject**

CC-BY-4.0, but **Danish** (Verified `language:da`). Out of scope per §5.4 (English only).

---

## C. Resume corpora — raw formats and clustering (R6, R7)

### C1 · Kaggle `snehaanbhawal/resume-dataset` — **Tier 1, the PDF/HTML corpus (decision D6)**

| | |
|---|---|
| URL | https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset |
| Size | **Verified** 2,484 resumes across 24 categories; 120 `INFORMATION-TECHNOLOGY`, 118 `ENGINEERING`, 120 `BUSINESS-DEVELOPMENT` |
| Schema | `ID`, `Resume_str`, `Resume_html`, `Category`; **PDFs on disk** in per-category folders, filename = `ID` |
| Provenance | Scraped from livecareer.com. **Verified — it is NOT A1's upstream source:** 0 exact matches against A1's 643 resumes, and only **40 (6.2%)** share substantial text. Same site and formatting, largely different documents |
| Contamination | *Superseded 19 Aug 2026:* the overlap criterion is now defined and scripted. **0 exact matches**; **44 resumes at ≥0.70 8-gram containment** (6.8% of A1's 643 uniques), 2 at ≥0.90. The exclusion list is written to `data/interim/c1_a1_contamination.csv` by `profile --check livecareer` and **must be excluded** from any NER training set or distractor pool built off this corpus |
| Licence | Kaggle-hosted, publisher terms; verify on the page before redistributing |
| Status | **ACQUIRED 19 Aug 2026** — `data/raw/snehaanbhawal-resume-dataset/`, **Verified 2,484 PDFs** + `Resume.csv` (median HTML 15,025 chars, text 5,886 chars). WSL `Zone.Identifier` markers removed |
| Role | The only identified source giving **real PDFs**, satisfying §5.2's parsing/robustness claims |
| Caveats | Requires a Kaggle account + API token (Q4) |

### C2 · `opensporks/resumes` — **Tier 1, HTML fallback for C1**

| | |
|---|---|
| URL | https://huggingface.co/datasets/opensporks/resumes |
| Size | **Verified** 2,484 rows |
| Schema | **Verified** `ID`, `Resume_str`, `Resume_html`, `Category` |
| Role | Same corpus as C1 **without the PDFs** — no Kaggle account needed. `Resume_html` still exercises real markup parsing |
| Caveats | No licence declared; HTML only, no PDF |

### C3 · `ahmedheakl/resume-atlas` — **Tier 2, clustering EDA only**

| | |
|---|---|
| URL | https://huggingface.co/datasets/ahmedheakl/resume-atlas |
| Size | **Verified** 13,389 rows, 43 categories, MIT licence |
| Schema | **Verified** `Category`, `Text` |
| Provenance | Heakl et al., *ResumeAtlas* (arXiv 2406.18125) |
| **Critical caveat** | **Verified** — `Text` is already lowercased, punctuation-stripped and stopword-removed. **Unusable for NER, evidence spans or anything shown in the UI.** Restrict to §4.7 clustering EDA |

### C4 · Kaggle `saugataroyarghya/resume-dataset` — **Tier 3, reject**

| | |
|---|---|
| URL | https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset |
| Status | **Downloaded and profiled 22 Aug 2026** — `data/raw/saugataroyarghya-resume-dataset/resume_data.csv` (17 MB), kept on disk for reference; not adopted |
| Nominal size | 9,544 rows, 35 columns |
| **Verified effective size** | **344 unique candidates.** The 9,544 rows are a candidate×job cross-product — 339 of 344 candidates repeat exactly 28 times, once per one of 28 `job_position_name` values |
| Schema | Pre-parsed structured fields only (`skills`, `educational_institution_name`, `degree_names`, `professional_company_names`, etc.) plus paired job-side fields (`job_position_name`, `skills_required`, `matched_score`, …) — **no full resume body text**; the only free text is `career_objective`, populated for 171/344 unique candidates, median 210 chars |
| **Reject because** | (1) **No annotatable text** — cannot address the B1/B2 NER gap, which needs full-length documents with entity spans in running text. (2) **Data-integrity bug**: candidate-side `responsibilities` is byte-identical to the paired job's `responsibilities.1` in **100% of rows** — candidate experience was never independently populated, so anything keyed on it is unusable. (3) `matched_score` (0–0.97, mean 0.66) has no documented generation method — synthetic/engineered, not ground truth, same caveat class as A3 but less transparent. (4) **Confirmed overlap with C1**: 47 of 171 unique `career_objective` values (27.5%) exact-match by 80-char prefix against C1's `Resume_str`; `professional_company_names` carries the `"Company Name"` placeholder in 4,336/9,544 rows, the same livecareer-template artifact seen in A1/C1. (5) No licence declared; provenance undocumented on the Kaggle page itself — traced independently to a "Global Workforce Resume Dataset" curated by Neuralframe AI for the BitFest 2025 (KUET CSE) student datathon, sourced from unspecified "open-source platforms" plus proprietary data |
| Provenance note | Kaggle page has no "About this dataset" text, license, or author info at all — everything above the file-level stats came from third-party tracing, not the publisher |

#### C4-derived · `data/processed/vm-structured-onet/` — **not adopted** (assessed 23 Aug 2026)

Three JSON artifacts contributed by a teammate, assessed after C4 itself had been rejected
(22 Aug). Source code was not available, so provenance and intent were established by
profiling the files against C4 directly.

| File | Rows | What it is |
|---|---|---|
| `resumes_raw_structured.json` | 344 resumes | C4's candidate side, re-nested into a typed schema, plus a `raw_text_reconstructed` string rendered from those fields |
| `jobs.json` | 28 jobs | C4's job side, column typos fixed (`educationaL_requirements` → `education_requirement`, `experiencere_requirement` → `experience_requirement`), `age_requirement` → `age_requirement_raw` |
| `job_to_onet_candidates.json` | 28 rows / 140 pairs | Each job title matched to its 5 nearest O\*NET-SOC occupation titles with a similarity score |

**Provenance — all three derive from C4, verified:** `resumes_raw_structured.json` self-declares
`source_file: resume_data.csv`, `conversion_method: reverse_engineered_from_structured_csv`,
`original_document_available: false`; 170 of its 171 `career_objective` values are byte-identical
to C4's; `jobs.json`'s 28 titles are exactly C4's 28 `job_position_name` values (one differs only
by a stripped trailing newline). The only external ingredient anywhere is the O\*NET-SOC title
list used for the join — 98 distinct SOC codes, 21 carrying detailed `.xx` suffixes.

**Reconstructed intent.** The conversion drops exactly two C4 columns — `matched_score` and the
corrupted candidate-side `responsibilities` — which are precisely defects (2) and (3) recorded
above, found independently. The 9,544-row cross-product is collapsed to 344 + 28 with no job
linkage retained. Dropping `matched_score` removes the only candidate↔job signal, which is what
the O\*NET layer was being built to replace: normalise both sides to a standard occupation code
and join on it. The `_raw` suffix and the 100%-`manual_review_required` flag show the work was
scoped as a first pass, not a finished dataset. **This is competent work on a rejected source**;
it is not adopted because of the source, not the workmanship.

| **Not adopted because** | (1) **Inherits every C4 defect** — 344 unique candidates, no licence, no annotatable source document, and the 27.5% C1 overlap all pass straight through; nothing downstream of C4 can remove them. (2) **`raw_text_reconstructed` adds zero information** — across all 344 resumes it contains **0 tokens** not already present in the structured fields, and is rendered from a fixed vocabulary of 6 uppercase headers in only 17 distinct layouts. A model trained on it learns the template, not the language of resumes; it cannot serve R1 or R6, which need real document text. (3) **The O\*NET join is unreliable and the taxonomy is the wrong one** — the metric is `difflib.SequenceMatcher` on lowercased, punctuation-stripped titles, which pulls on shared suffixes: SOC major group 17 (Architecture & Engineering) draws 23 of 140 pairs against group 15 (Computer & Mathematical) at 9, because "Engineer" dominates the string. All 28 rows are flagged `manual_review_required` and none has been reviewed. (4) **Conversion defects on top**: `responsibilities` and `skills_required` are 1-element lists holding undelimited blobs (`"Fast typing skill IELTSInternet browsing & online work ability."`), 63 of 474 `passing_year` values are not 4-digit years, and 157 of 344 resumes carry the `"Company Name"` livecareer placeholder as an employer. |
| **Contamination risk if re-imported** | **7 of the 344 resumes overlap A1**, the core benchmark: their `career_objective` appears verbatim inside an A1 `resume_text` (60-char normalised probe against all 8,000 A1 resumes). Both corpora are livecareer-derived, so this is the same template family reaching R2 from a second direction. If these ever enter a training set or distractor pool they leak into R2 evaluation. This is the one finding that must survive the rejection. |
| Licence | Undetermined — inherits C4's undeclared status; the O\*NET release used is not recorded, so the CC-BY attribution O\*NET requires cannot be discharged |
| Disposition | Left in place at `data/processed/vm-structured-onet/`, untracked per `.gitignore`. Not registered in `src/candidate_screener/data/sources.py` and not to be added |

**What serves the same intent, already adopted.** The occupation-normalisation layer the O\*NET
join was reaching for is covered twice over by sources already on disk:

| Intent | Use instead | Why |
|---|---|---|
| Map free-text job titles to a canonical occupation | **D1 ESCO** — 3,043 occupations, 30,417 `altLabels`, 33,460 match targets against O\*NET's ~1,000 canonical titles | On the same `difflib` metric and the same 28 titles, ESCO returns the correct occupation in its top-3 for 7 of 10 tech roles (5 at rank 1) against O\*NET's 4 of 10 anywhere in the top-5. The substrate was the problem, not the method — ESCO ships alias labels precisely for title matching |
| Link candidates to jobs | **A2 Djinni `Primary Keyword`** — 41 of 41 values shared across the JD and CV sides | The both-sided join key the O\*NET detour was constructing, already present, MIT-licensed and in-domain |
| Hard vs preferred requirements | **D1 `occupationSkillRelations_en.csv`** — 67,600 `essential` / 58,451 `optional` | The distinction `education_requirement` / `experience_requirement` was groping toward, already labelled |

Caveat carried forward: string similarity alone is unsafe against any taxonomy — against ESCO,
`AI Engineer` matches `animal artificial insemination technician` at 1.000 via the genuine alt
label `ai engineer`. The `manual_review_required` gate was the right instinct and is retained in
Phase 3.5.

---

## D. Job-description corpora and skill vocabulary (R4, R5)

### D1 · ESCO v1.2.1 — **Tier 1, skill ontology**

| | |
|---|---|
| URL | https://esco.ec.europa.eu/en/use-esco/download |
| Status | **ACQUIRED 19 Aug 2026** — `data/raw/esco_dataset-v1.2.1-classification/` (Q7 closed) |
| Size | **Verified** 13,960 skills (10,734 skill/competence + 3,221 knowledge); 3,043 occupations; 1,284 digital skills |
| **Aliases** | **Verified ~86,694 alternative labels** in `skills_en.csv` — the alias inventory §4.5 normalisation needs |
| Relations | **Verified** `occupationSkillRelations_en.csv` = 126,051 occupation↔skill links, **67,600 `essential` + 58,451 `optional`** — seeds the **hard vs preferred requirement** split directly |
| Formats | CSV bundle in place (RDF, TTL, ODS, XML, JSON-LD also offered) |
| Role | Skill normalisation, alias resolution, hard-requirement checks (§4.5) |
| Alignment | ESCO's skill/knowledge division maps directly onto SkillSpan's `tags_skill` / `tags_knowledge` — use one scheme across both |
| Caveats | EU labour-market taxonomy; tech-tool coverage is thinner than a scraped tech vocabulary, hence the D2 frequency join. Confirm derived extracts may be committed (Q11) |
| **Coverage — measured** | *Superseded 19 Aug 2026:* of D2's 100 most-frequent tech skills, ESCO matches **33 exactly** (47% of posting volume) and 55 under relaxed word matching. `aws`, `azure`, `docker`, `kubernetes`, `snowflake`, `pytorch`, `terraform`, `jira` and ~37 others are **absent entirely**. ESCO alone is insufficient for this domain — see [card](cards/D1-esco.md) |

### D2 · `lukebarousse/data_jobs` — **Tier 2, skill vocabulary only**

| | |
|---|---|
| URL | https://huggingface.co/datasets/lukebarousse/data_jobs |
| Size | **Verified** 785,741 rows, Apache-2.0 |
| Schema | **Verified** `job_title_short`, `job_title`, `job_skills`, `job_type_skills`, location, salary, dates |
| **Correction to its framing** | **Verified — there is NO job-description text column.** It cannot serve as a JD corpus |
| Role | Excellent **tech-role skill frequency** source to weight and extend the ESCO vocabulary toward data/software roles |
| **Verified vocabulary** | *Superseded 19 Aug 2026:* **252 distinct skills**; top: `sql` 384,849 · `python` 380,909 · `aws` 145,381 · `azure` 132,527 · `r` 130,892 |
| **Handling trap** | `job_skills` is a **stringified list**, not a list — iterating a cell yields characters and silently produces a 37-symbol vocabulary. Parse via `candidate_screener.data.profile.parse_skill_cell` |

### D3 · `datastax/linkedin_job_listings` — **Tier 3, avoid**

| | |
|---|---|
| URL | https://huggingface.co/datasets/datastax/linkedin_job_listings |
| Size | **Verified** 123,849 rows, 31 columns, full `description` text (~2.5k chars) |
| **Avoid because** | No licence declared, and it is a mirror of LinkedIn-scraped data (Kaggle `arshkon/linkedin-job-postings`). Using it contradicts decision D3's no-scraping premise |
| Superseded by | **A2 (Djinni)** — larger on the IT slice, MIT-licensed, anonymized, both-sided |

### D4 · `jacob-hugging-face/job-descriptions` — **Tier 3, reject**

**Verified** 853 rows under a **Llama 2 licence**; text is already lowercased and
punctuation-stripped. Too small, restrictively licensed, and pre-normalised.

---

## E. Summary — what gets adopted

| Requirement | Adopted source(s) | Licence |
|---|---|---|
| R1 NER spans | **B2 SkillSpan** (primary) + **B1 DataTurks** (resume-native seed, de-identified) | CC-BY-4.0 / undeclared |
| R2 Fit pairs | **A1 cnamuangtoun** | Undeclared (Q1) |
| R3 Retrieval pools | Constructed from **A1**; in-domain pools from **A2** | — |
| R4 In-domain eval | **A2 Djinni** + team-adjudicated labels + synthetic resumes (D3) | MIT |
| R5 Skill vocabulary | **D1 ESCO** + **D2 data_jobs** skill frequencies | EU terms / Apache-2.0 |
| R6 Raw formats | **C1 Kaggle PDFs**, fallback **C2 HTML** | Publisher terms |
| R7 Clustering EDA | **C3 ResumeAtlas** | MIT |

**Rejected and why:** A4 (derivative + broken separator + circular), B4 (Danish),
C4 (only 344 unique candidates behind a 9,544-row cross-product, no annotatable text, corrupted
`responsibilities` field, confirmed 27.5% overlap with C1, no licence) **and its derivatives in
`data/processed/vm-structured-onet/`** (inherit all of the above; reconstructed resume text adds
0 novel tokens; unreviewed O\*NET join — see C4), D3 (unlicensed LinkedIn scrape), D4 (tiny,
Llama-2 licence, pre-normalised).
