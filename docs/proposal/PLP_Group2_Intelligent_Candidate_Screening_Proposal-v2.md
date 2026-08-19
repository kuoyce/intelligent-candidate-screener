National University of Singapore | Institute of Systems Science
Graduate Certificate in Practical Language Processing
Practice Module Project Proposal

# Intelligent Candidate Screening System

**Project Group:** 2
**Version:** v2 — 16 August 2026 (supersedes the original submission; see summary of changes below)

---

## Summary of Changes from v1

This revision replaces the original Task 2 design (a five-class role classifier — Data Scientist, ML Engineer, Software Engineer, DevOps Engineer, Business Analyst — trained on an assumed internal dataset of 10,000+ resumes) with a **retrieve-then-rank architecture**: a bi-encoder retrieves candidates by resume–job-description similarity, and a fine-tuned cross-encoder classifies each retrieved pair as *Good Fit / Potential Fit / No Fit*.

The reasons, in brief:

- **No public dataset uses the proposed five-role taxonomy**, and one class (ML Engineer) has no clean public label at all. Building it required inventing a label-mapping from other taxonomies, which is hard to defend and depends on internal data that may not be accessible in time.
- **The taxonomy wasn't load-bearing for the actual deliverable.** The system's core promise is "rank candidates against a job description" — a resume–job-description matching problem, not a resume-only categorisation problem. A role label was an extra layer sitting beside the ranking logic, not feeding it.
- **Pairwise fit classification is still a text-classification task** (satisfying the module's requirement to demonstrate classification), but it classifies (resume, job description) pairs into a fit outcome instead of a resume alone into a role. This removes the taxonomy problem entirely, has ready-made public training data, and doubles as the system's rank stage — collapsing what were two loosely-connected components (a role classifier and a separate matching model) into one coherent, standard architecture.
- **It is pedagogically a harder, more meaningful task for the DNN component to earn its keep on.** A five-way role classifier trained on resumes with disjoint vocabularies is comparatively easy for a classical baseline to solve well, which weakens the case for fine-tuning a transformer at all. Fit classification on resume–JD pairs requires the model to reason about semantic overlap rather than just keyword frequency, which is exactly where a fine-tuned encoder should outperform a classical baseline.

Everything else in the original proposal — the business problem, the information-extraction task, the conversational UI, and the privacy/evaluation approach — is unchanged.

---

## 1. Introduction

Recruitment teams receive large volumes of resumes in PDF and DOCX formats, but initial screening remains heavily manual. A recruiter may spend 20–30 minutes reviewing one application, making it difficult to process hundreds of candidates consistently. Important evidence such as transferable skills, technical tools and relevant experience may also be missed when wording differs between a resume and a job description.

This project proposes an Intelligent Candidate Screening System that analyses resume and job-description text, ranks applicants against role requirements and explains each score through a conversational recruiter interface.

## 2. Industry Overview

Talent-acquisition teams increasingly use applicant-tracking systems, text analytics and machine learning to manage high application volumes. However, simple keyword filters may overlook contextual or transferable skills, while opaque automated scores can reduce recruiter trust. A practical screening tool therefore needs both semantic matching and evidence-based explanations, together with privacy, fairness and human-oversight controls.

## 3. Business Problem and Objectives

Manual screening creates long lead times, inconsistent evaluation criteria and high initial false-positive rates. Recruiters also have limited time to compare several candidates against the same required and preferred skills. The proposed platform addresses these gaps by standardising the first-pass analysis and allowing recruiters to interrogate the results conversationally.

The project aims to:

- Reduce average screening time from 25 minutes to approximately 2 minutes per candidate.
- Achieve at least 85% candidate–role fit agreement with human judgement, and process 500+ applications per role.
- Extract skills, tools, qualifications, job titles and experience evidence from resumes.
- Retrieve and rank candidates against a job description, showing matched requirements, gaps and confidence.
- Provide a conversational UI for shortlist, comparison and explanation requests.

## 4. Project Design

### 4.1 Solution Architecture

*Figure 4.1 Proposed MVP architecture* — Resume/JD ingestion → text preparation → (a) information extraction, (b) retrieve-then-rank matching → ranking & evidence layer → conversational UI.

### 4.2 Language-Processing Components

**Task 1 — Information extraction / sequence labelling.**
Identify hard and soft skills, tools, frameworks, certifications, job titles and experience durations. The baseline uses spaCy rules and phrase matching; the advanced comparison uses fine-tuned BERT NER. Skill normalisation compares direct string matching with Sentence-BERT similarity.

**Task 2 — Candidate–job matching via retrieve-then-rank.**
This task replaces the original standalone role classifier. It has two stages, mirroring a standard information-retrieval pattern:

- *Retrieve:* a bi-encoder (two-tower Sentence-BERT model) embeds resumes and job descriptions independently into a shared vector space; cosine similarity produces a fast, approximate shortlist. Baseline: TF-IDF + skill-overlap cosine similarity.
- *Rank:* a fine-tuned BERT cross-encoder classifies each (resume, job description) pair from the shortlist into **Good Fit / Potential Fit / No Fit**, using both texts jointly rather than independently. Baseline: a rule-based skill-overlap threshold. This is the project's deep-neural-network component and its text-classification component at once.

At the scale targeted by this pilot (500 applications per role), brute-force cross-encoder scoring is computationally trivial — the retrieve stage is included to demonstrate the retrieve-then-rank pattern taught in the module, not because it is strictly necessary at this volume. This is stated plainly so the design choice isn't mistaken for a scalability claim it isn't making.

**Ranking and explanation.**
Combine the cross-encoder's fit classification with required/preferred skill coverage, experience relevance and career-trajectory features extracted in Task 1. The platform will display the evidence used, identified gaps, confidence and suggested interview questions through constrained templates.

**Conversational UI.**
A Streamlit or Gradio chat interface will allow recruiters to upload files, request top candidates, apply job-relevant filters, compare candidates and ask why a score was assigned. Answers will be grounded in extracted resume evidence and model outputs.

**Optional exploratory addendum — clustering.**
As a small, clearly-scoped side analysis (not a pipeline dependency), the team may cluster the IT/tech-domain resume corpus using Sentence-BERT embeddings to sanity-check whether natural groupings resemble common role families, and report this as supporting evidence in the write-up. This is presented as supplementary EDA, not a load-bearing architectural stage — an earlier design considered inserting clustering between retrieval and ranking, but this was dropped because clustering and retrieval both perform the same narrowing function, and stacking them added a component the ranking stage doesn't actually consume without doing independent, useful work.

### 4.3 Development Methodology

The team will use an Agile-inspired, iterative approach. Each sprint will produce a testable component, followed by error analysis and refinement. The classical baseline will be completed before DNN fine-tuning so that improvements can be measured rather than assumed.

### 4.4 Software Development Lifecycle

1. **Requirements and evaluation-protocol definition** — confirm recruiter needs, fit-label definitions and success criteria (no role taxonomy to define, since fit classification replaces role classification).
2. **Data preparation** — extract text, de-identify personal information, clean and align resume–job-description pairs.
3. **Baseline modelling** — build spaCy extraction, TF-IDF/skill-overlap retrieval, and rule-based fit classification.
4. **DNN modelling** — fine-tune and validate the BERT NER model, the bi-encoder retriever, and the cross-encoder fit classifier.
5. **Platform integration** — connect retrieval, ranking and evidence services to the conversational UI.
6. **Testing and reporting** — evaluate performance, fairness, usability and reproducibility.

## 5. Scope of Work

### 5.1 Data Source

Data is sourced per task rather than from one assumed internal dataset, since no single public dataset covers everything the original plan required.

| Task | Primary source(s) | Role in the project |
|---|---|---|
| Information extraction | [DataTurks Resume Entities for NER](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner) — 220 resumes, span-level labels for Skills, Designation, Companies, Degree, College, Experience | Gold seed/validation set for both the spaCy baseline and BERT NER fine-tuning. Small size means it will likely need augmentation via weak/silver labelling over a larger unlabelled resume pool, or a modest batch of self-annotated resumes. |
| Retrieval + fit classification | [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) — 8,000 resume–JD pairs, 3-class Fit label | Primary training data for both the bi-encoder retriever and the cross-encoder fit classifier. |
| Retrieval calibration | [0xnbk/resume-ats-score-v1-en](https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en) — 6,374 rows, continuous ATS-style score (Apache 2.0) | Supplementary continuous signal for calibrating and sanity-checking the retrieval/ranking scores. |
| In-domain evaluation | 200+ self-collected technology/data/software job descriptions (as in v1), paired with a sample of resumes | Both public pair datasets above are general-industry, not IT-specific. A small self-collected, in-domain held-out test set is needed to confirm the model still performs on the project's actual target domain, and earns the module's bonus credit for self-collected data. |
| Skill vocabulary | Public skill lists, developer surveys, technology repositories, IEEE/ACM terminology (unchanged from v1) | Supports entity normalisation and feature engineering. |
| Optional clustering EDA | A tech-filtered subset of a public labelled resume-category set (e.g. the ~962-resume 25-category Kaggle set, or [ResuméAtlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas), MIT-licensed, 13,389 resumes / 43 categories) | Only used for the optional exploratory clustering addendum in Section 4.2; not required by the core pipeline. |

Recruiter feedback and historical shortlist labels, assumed in v1, have no public substitute. If real recruiter labels remain unavailable by the pilot stage, the operational "recruiter rating" success measure in Section 7 should either be dropped or filled with a small, clearly-labelled synthetic/team-adjudicated stand-in, flagged as such in the final report.

### 5.2 Data Preparation and Privacy

Resume and job-description files will be converted to clean text while retaining useful section boundaries. The main steps are:

- Remove or mask names, addresses, age, gender, nationality and direct identifiers.
- Detect extraction failures, empty pages, duplicated resumes and malformed PDF/DOCX content.
- Normalise case, whitespace, common skill aliases, abbreviations and date/experience expressions.
- Split data at candidate level into training, validation and held-out test sets to avoid leakage; ensure the held-out set includes the self-collected in-domain pairs, not only the general-industry public pairs.
- Review class balance across the three fit labels and document label definitions and disagreement handling.

### 5.3 Models, Outputs and Evaluation

| Component | Baseline | MVP advanced model | Measure |
|---|---|---|---|
| Entity extraction | spaCy rules | Fine-tuned BERT NER | Entity F1 |
| Candidate–JD retrieval | TF-IDF / skill-overlap + cosine | Two-tower (bi-encoder) Sentence-BERT | Recall@k, Precision@k |
| Candidate–JD fit classification (rank) | Rule-based skill-overlap threshold | Fine-tuned BERT cross-encoder (3-class fit) | Macro-F1 |
| Conversational UI | Scripted intents | Multi-turn evidence display | Task success |

### 5.4 Scope Boundaries

The MVP covers English-language resumes and job descriptions relevant to technology, data and software roles, evaluated through candidate–job fit rather than a fixed role-family label. It excludes external skill ontologies, autonomous hiring decisions, production Kubernetes deployment and unconstrained generative recommendations. These boundaries keep the work feasible while demonstrating preprocessing, information extraction, retrieval, classification, DNN fine-tuning, ranking, evaluation and conversational-UI design.

## 6. Key Deliverables

- De-identified and quality-checked resume/job-description pair dataset, with a documented fit-label schema (no role taxonomy required).
- Skill and entity extraction service with baseline and advanced-model evaluation.
- Classical and DNN candidate–job retrieval and fit-classification models with saved artefacts.
- Ranked shortlist output containing confidence, matched requirements, gaps and source evidence.
- Conversational MVP, source code, run documentation, evaluation results, report, slides and recorded demonstration.

## 7. Success Measures

| Stage | Technical target | Operational target | Validation |
|---|---|---|---|
| Baseline | Extraction F1 >85%; retrieval Recall@10 >80% | <5 seconds per resume | Held-out test set |
| Advanced | Fit-classification Macro-F1 >85%; retrieval Recall@10 >90% | Stable core-intent responses | Model comparison, in-domain held-out set |
| Pilot | Review performance gaps across evaluated groups | Shortlisting time >80% lower; recruiter rating >4/5 | 5 recruiters; 200 applications |

## 8. Effort Estimates and Timeline

| Weeks | Effort | Activity | Output |
|---|---|---|---|
| 1–2 | 6 days | Requirements, collection, de-identification and fit-label guide | Dataset and label guide |
| 3–4 | 5 days | Preprocessing and entity extraction | Extraction baseline/API |
| 5–6 | 6 days | Bi-encoder retrieval baseline + fine-tuning, cross-encoder fit-classification baseline + fine-tuning | Retrieval and fit-classification models, with comparison |
| 7–8 | 4 days | Ranking and evidence templates | Explainable ranked output |
| 9–10 | 4 days | Conversational UI and API integration | Working MVP |
| 11 | 3 days | Performance, fairness and usability testing | Evaluation findings |
| 12 | 2 days | Documentation, report and demo | Final submission |

The total estimated effort is 30 team-days, approximately 15 days per member. The roadmap may be adjusted after data access and label quality are confirmed.

## 9. Expected Value and ROI Assumptions

At 500 candidates per month, indicative annual value is approximately $750,000: recruiter time saved of $115,000 (6,000 candidates x 23 minutes x $50/hour), faster hiring value of $450,000 (15 days x 6 roles x $5,000/day) and retention-related savings of $200,000. Against an estimated first-year implementation cost of $150,000, the indicative net ROI is approximately 400%. These figures remain assumptions to be validated during the pilot, and are independent of the Task 2 redesign above.

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Historical-data bias | High | PII removal, subgroup evaluation, balanced review data and human oversight. |
| Resume-format variability | Medium | Robust parsing, extraction validation and fallback handling. |
| Public fit-pair data is general-industry, not IT-specific | Medium | Supplement with self-collected in-domain resume–JD pairs; evaluate primarily on the in-domain held-out set, not the general-industry test split. |
| Emerging or unseen skills | Medium | Versioned vocabulary, semantic normalisation and periodic review. |
| Unsupported explanations | Medium | Evidence snippets, constrained templates and confidence thresholds. |
| Privacy and security | High | De-identification, access control, retention limits and encrypted storage. |

## 11. Next Steps

- Confirm project group members, dataset ownership, privacy approval and access arrangements.
- Finalise the fit-label definitions (Good Fit / Potential Fit / No Fit), annotation guide and evaluation protocol — no role-taxonomy definition needed under this design.
- Prepare a small de-identified sample and begin the extraction/retrieval/fit-classification baselines.
- Collect the 200+ in-domain job descriptions and a matching resume sample early, since the in-domain held-out set is now the primary evidence that the system works on the project's actual target domain.
- Validate technical feasibility before committing to the full dataset and pilot targets.
