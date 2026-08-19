# Data Requirements and Profiling Findings

## 1. Data requirements derived from the proposal

Reading v3 end-to-end, the system needs seven distinct kinds of data. The proposal names sources
for five of them.

| # | Requirement | Driven by | Supervision needed | Named in v3? |
|---|---|---|---|---|
| R1 | Resume text with **entity spans** (skills, tools, titles, degrees, experience) | §4.3, §5.3.1 | Span-level BIO labels | Yes — DataTurks |
| R2 | **Resume–JD pairs with fit labels** for the 4-stage progression | §4.4, §5.3.2 | 3-class pair label | Yes — cnamuangtoun |
| R3 | **Candidate pools per JD** so Recall@k / Precision@k mean something | §5.3.1, §7 | Ranked or binary relevance over a pool | **No** |
| R4 | **In-domain tech/data/software JDs + resumes** for end-to-end evaluation | §5.1, A.7 | Human-adjudicated fit | Partly — "self-collected" |
| R5 | **Skill vocabulary / ontology** for normalisation and hard-requirement checks | §4.5, §5.1 | Controlled term list + aliases | Loosely — "public skill lists" |
| R6 | **Original-format documents** (PDF/DOCX/HTML) to exercise ingestion | §5.2 | None — raw files | **No** |
| R7 | Resume corpus with role labels for optional clustering EDA | §4.7 | Category label | Yes — Kaggle 962 / ResumeAtlas |

R3 and R6 are unmet by the named sources. R4 has no lawful collection path as written. These are
addressed in the catalog.

## 2. What the profiling actually found

I downloaded and profiled the three named sources. The headline results change the project plan.

### 2.1 The core matching dataset is far smaller than 8,000 examples

`cnamuangtoun/resume-job-description-fit` — **Verified 19 Aug 2026**:

| Metric | Train | Test |
|---|---|---|
| Rows | 6,241 | 1,759 |
| **Unique resumes** | **642** | **477** |
| **Unique JDs** | **280** | **71** |
| Label mix (No / Potential / Good) | 50.4% / 24.9% / 24.7% | 48.7% / 25.2% / 26.0% |
| Resume length, median chars | 5,134 | 5,080 |
| JD length, median chars | 2,384 | 2,401 |

The 8,000 rows are a near-complete cross-product of a small pool of documents. The **effective
sample size is ~642 resumes and ~280 JDs**, and every metric's confidence interval must be
computed against that, not against 8,000. Test-set retrieval metrics rest on **71 JDs**, of which
only **28 have at least one Good Fit resume**.

> **Implication for §7.** A Recall@10 figure computed over 28–71 queries has an error bar of
> roughly ±10 points. The ">90% Recall@10" target cannot be distinguished from ">80%" at this
> sample size. This is the strongest single argument for decision D2.

### 2.2 The published train/test split leaks resumes, violating §5.2

§5.2 commits to "split data at candidate level … to avoid leakage". The shipped split does not do this:

- **476 of the 477 test resumes (99.8%) also appear in the training split**, paired with different JDs.
- JDs are cleanly disjoint (0 overlap) — it is a **JD-level split, not a candidate-level split**.
- 0 exact (resume, JD) pairs are shared, so it is not a trivially broken benchmark — but a
  fine-tuned cross-encoder will have memorised essentially every test resume.

**Action:** either re-split at candidate level for the project's own experiments (and report both),
or state plainly in the report that the split is JD-disjoint only. Do not claim candidate-level
holdout while using the shipped split.

### 2.3 Label noise is present but small

6 (resume, JD) pairs carry conflicting labels across the corpus. Low in absolute terms, but it
confirms the labels are not authoritative ground truth, which supports treating >85% Macro-F1 as
an unrealistic ceiling.

### 2.4 The "retrieval calibration" dataset is a derivative of the core dataset — and is broken

The proposal (§5.1, row 3) treats `0xnbk/resume-ats-score-v1-en` as an independent continuous
signal for calibrating retrieval scores. **Verified — it is neither independent nor usable:**

1. **It is the same data.** After whitespace normalisation, **97.2% of its 6,374 rows (6,193) begin
   with a resume text that is exactly one of the cnamuangtoun resumes**, and the remainder of the
   text is exactly one of the cnamuangtoun JDs. In a 300-row sample, 295 contained a fit-dataset
   resume and 299 contained a fit-dataset JD.
2. **The documented separator does not exist.** The dataset card states resume and JD are joined by
   `[SEP]`. **Zero of 6,374 rows contain that token** — or any other delimiter. The resume and JD
   are concatenated with nothing between them, so the boundary cannot be recovered from the file
   alone.
3. **The score is circular for this use.** Its card states the `ats_score` was computed from
   semantic similarity. Calibrating an SBERT cosine score against a score that was itself produced
   by an embedding model, on the same pairs, measures nothing.

> **Recommendation:** remove `0xnbk/resume-ats-score-v1-en` from §5.1. If a continuous signal is
> wanted, derive it from the project's own Stage-3 scores. If the team still wants it, the
> boundary can be recovered by prefix-matching against cnamuangtoun (script provided) — but the
> circularity objection stands regardless.

### 2.5 The NER seed set is thin exactly where the project needs it

DataTurks (obtained from the GitHub mirror, no Kaggle account needed) — **Verified**:

| Metric | Value |
|---|---|
| Documents | 220 (200 train / **20 test**) |
| Total annotated spans | 3,556 |
| **Skills spans** | **472** |
| Companies / Designation / Location | 729 / 521 / 430 |
| College Name / Degree / Graduation Year | 330 / 298 / 254 |
| **Email Address / Name spans (direct PII)** | **252 / 224** |
| Years of Experience | 44 |
| **Span offset/text mismatches** | **222 (6.2%)** |
| Repo licence | **None declared** |

Three problems:

- **The §7 "Extraction F1 >85%" target would be measured on 20 documents.** That is not a
  defensible evaluation set.
- **Skills — the project's primary entity — has only 472 spans**, and "Years of Experience" only 44.
- **6.2% of spans have broken character offsets** (the annotation's stored text does not match the
  slice its offsets point at). These must be repaired or dropped during conversion to BIO, or NER
  scores will be silently wrong.

Per decision D5 this set is kept as the resume-native seed, but it cannot carry the NER
evaluation alone. See catalog §B for the supplements.

### 2.6 ResumeAtlas cannot be used for extraction or evidence

`ahmedheakl/resume-atlas` (13,389 rows, 43 categories, MIT) — **Verified**: the `Text` field is
already lowercased, punctuation-stripped and stopword-removed, e.g.
`education omba executive leadership university texas 20162018 bachelor science accounting …`.

It is therefore usable **only** for the optional clustering EDA (R7). It cannot support NER,
evidence spans or anything the recruiter UI displays. §5.1's last row should say so explicitly.

### 2.7 The §7 Recall@10 target is arithmetically unreachable, not merely ambitious

This is the strongest single finding in the report, and it is arithmetic rather than empirical.

Relevance density in this dataset is very high — **Verified** on the test split (28 JDs with at
least one Good Fit):

| Good Fit resumes per JD | min | Q1 | median | Q3 | max | mean |
|---|---|---|---|---|---|---|
| Test split | 1 | 3 | **18** | 24 | 48 | 16.4 |
| Train split | 1 | 2 | 10 | 23 | 45 | 13.4 |

A query with 18 relevant documents **cannot** achieve Recall@10 above 10/18 = **0.56**, no matter
how good the model is. Concretely:

| Metric | Attainable for … | Requires |
|---|---|---|
| Recall@10 ≥ 0.90 | **43% of JDs** | ≤ 11 relevant |
| Recall@20 ≥ 0.90 | 61% of JDs | ≤ 22 relevant |
| **Recall@50 ≥ 0.90** | **100% of JDs** | ≤ 56 relevant |

> **§7's "retrieval Recall@10 >90%" is impossible for 57% of test queries by construction.**
> Reporting it would guarantee an apparent failure that says nothing about model quality.

**Recommended metric set** (decision D9): report **Recall@50** as the recall figure, and use
**Precision@10 and nDCG@10** as the head-of-ranking measures. When ~18 of 100 pool candidates are
relevant, precision-at-head and graded ranking quality are the measures that actually discriminate
between Stages 1–4; recall at a small *k* does not.

## 3. Dataset cards that were found to be wrong

Flagged because the team will otherwise plan against them:

| Source | Card claim | Verified reality |
|---|---|---|
| `0xnbk/resume-ats-score-v1-en` | Resume and JD separated by `[SEP]` | No separator in any of 6,374 rows |
| `0xnbk/resume-ats-score-v1-en` | Derived from "an existing source", framed as standalone | 97.2% identical to `cnamuangtoun` |
| `lukebarousse/data_jobs` | Presented as a job-postings dataset | Contains **no description text** — titles and extracted skills only |

## 4. How the unmet requirements are now met

| Gap | Resolution | Source |
|---|---|---|
| R3 — no candidate pools | Build pools from the fit dataset: for each JD, its labelled resumes plus distractors sampled from the 642/477 unique resume pool, padded to a target pool size | `cnamuangtoun` + protocol in `03-acquisition-action-plan.md` |
| R4 — in-domain data with no lawful path | **Djinni Recruitment Dataset (MIT)** — 141,897 IT job descriptions + 210,250 pre-anonymized candidate CVs from a real IT hiring platform, peer-reviewed at LREC-COLING 2024 | `lang-uk/recruitment-dataset-*` |
| R5 — skill vocabulary | ESCO v1.2.1 + SkillSpan span labels + `lukebarousse/data_jobs` skill frequencies for tech-role weighting | catalog §D |
| R6 — original-format files | Kaggle `snehaanbhawal/resume-dataset` ships the **PDFs**; the HF mirror `opensporks/resumes` ships the **raw HTML** | catalog §C |

### The Djinni find is the most consequential

It is IT-domain (the project's actual target), already anonymized by the publishers, MIT-licensed,
and covers **both sides** of the match. It has **no fit labels** — but both the CV and JD tables
share a `Primary Keyword` role field, `Experience Years` and `English Level`, which gives a
principled way to construct in-domain candidate pools before the team adjudicates fit labels on a
sample (decision D4). This replaces the "200+ self-collected JDs" plan that had no lawful
execution path.

**Verified:** all 41 CV-side `Primary Keyword` families also occur on the JD side (45 JD-side
values in total), and the largest shared families are `JavaScript, Java, DevOps, .NET,
QA Automation, Node.js, PHP, Python, Project Manager` — the join works, and it lands squarely
in the project's technology/data/software target domain.

**Stated limitation (decision D11):** Djinni is a Ukrainian/Eastern-European IT hiring
platform. The team accepts this as in-domain for the technology/data/software task, and the
report must state plainly that in-domain generalisation was demonstrated on that labour
market rather than on the deployment market. This belongs in §5.4 scope boundaries and in
the §10 risk table.

## 5. Recommended amendments to the proposal

| § | Current text | Recommended change |
|---|---|---|
| §5.1 row 3 | `0xnbk` as "retrieval calibration" | Delete the row; note the circularity in §10 risks |
| §5.1 row 4 | "200+ self-collected job descriptions" | Replace with the Djinni corpus + team-adjudicated label sample |
| §5.1 last row | ResumeAtlas for clustering | Keep, but state the text is pre-normalised and EDA-only |
| §5.2 | "Split at candidate level" | Note the shipped split is JD-disjoint only; describe the project's own re-split |
| §5.3.1 | Recall@k / Precision@k | Add the pool-construction protocol these are computed over |
| §7 | Absolute >85% / >90% targets | Rebase as deltas over the measured Stage-1 baseline (D2); keep 85% only for in-domain human agreement |
| §7 | "retrieval Recall@10 >90%" | **Replace with Recall@50 + Precision@10 + nDCG@10** — Recall@10 >90% is unreachable for 57% of queries (§2.7) |
| §7 | "Extraction F1 >85%" | State the evaluation set; 20 DataTurks test docs is insufficient |
| §10 | Risk table | Add: small effective sample size; resume leakage in the shipped split; derivative datasets |

## 6. Open questions

### Resolved 19 August 2026

| # | Question | Resolution |
|---|---|---|
| Q1 | Is the undeclared licence on the core benchmark acceptable? | **Accepted — proceed.** Record the licence status in the report's data section |
| Q2 | Re-split at candidate level, or keep the shipped split? | **Re-split at candidate level.** Results are explicitly **not comparable** to published numbers on the shipped split; state this wherever Stage 1–4 results appear |
| Q3 | What pool size for retrieval? | **Decided from the data (§2.7).** Pool N = 100; report Recall@50, Precision@10, nDCG@10. Recall@10 is dropped as unreachable |
| Q4 | Kaggle account for the PDF corpus? | **Available. Not blocking** — see below |
| Q5 | Is the Djinni EE/Ukrainian IT market acceptable as in-domain? | **Acceptable, stated clearly** as a limitation in §5.4 and §10 |
| Q6 | How many pairs, and who annotates? | **200 pairs**, annotated by business-analytics master's students with software and data-science backgrounds |
| Q7 | Who acquires ESCO? | **Done.** ESCO v1.2.1 is in place at `data/raw/esco_dataset-v1.2.1-classification/` |

**On Q4 — Kaggle is not blocking.** It is needed only for step 1.6, the PDF variant of the
livecareer corpus (C1), which feeds the §5.2 ingestion/parsing demonstration. It is not on the
critical path: the matching progression runs on A1, and the in-domain evaluation runs on A2, both
already downloaded. The HTML mirror (C2) is also already available as a fallback. Pull the PDFs
whenever convenient before the ingestion component is built (§8 weeks 3–4).

### Remaining

| # | Question | Blocks | Owner |
|---|---|---|---|
| Q8 | The candidate-level re-split costs ~30% of pairs and yields a test set of ~560 pairs / 30 JDs with Good Fit (§3.2 of the action plan). Is that accepted, or should the split fraction be tuned? | Phase 3.2 | Team |
| Q9 | Which of the 41 shared Djinni role families are in scope for the 200-pair in-domain set? A spread across ~6–8 families is suggested | Phase 3.4 | Team |
| Q10 | Do the 200 in-domain pairs use 3-class fit labels identical to A1's, so the two evaluations are directly interpretable together? | Annotation guide | Team |
| Q11 | ESCO redistribution — the raw CSVs are in `data/raw/`, which is git-ignored. Confirm derived vocabulary extracts may be committed | Phase 3.5 | Team |
