National University of Singapore | Institute of Systems Science
Graduate Certificate in Practical Language Processing
Practice Module Project Proposal

# Intelligent Candidate Screening System

**Project Group:** 2  
**Version:** v3 — 16 August 2026 (supersedes v2)

---

## Summary of Changes from v2

This revision reframes the project around a central research question: **does increasing model sophistication provide enough improvement in candidate–job matching to justify the additional complexity, cost and loss of transparency?**

The proposed system continues to use the retrieve-then-rank architecture introduced in v2, but the evaluation design is now explicitly progressive. Rather than assuming the most sophisticated model is the best solution, the project will establish a classical baseline and progressively introduce semantic retrieval and neural re-ranking. This allows the team to determine which techniques are worth adopting rather than adding complexity for its own sake.

The main changes are:

- **A new research objective** is introduced alongside the existing system objectives: evaluate the incremental benefit of progressively more sophisticated candidate–job matching techniques.
- **Information extraction and candidate matching are treated as separate experiments.** NER is a supporting component of the final system and is evaluated on a suitable entity-labelled dataset; it is not forced into the matching-model progression where the necessary supervision is unavailable.
- **The candidate–job matching experiment uses one common resume–job-description dataset across the progression.** The same held-out pairs can therefore be used to compare a rule/skill-overlap baseline, TF-IDF retrieval, Sentence-BERT retrieval and cross-encoder re-ranking.
- **A separate in-domain evaluation set provides the common end-to-end test.** This avoids requiring one public dataset to contain NER spans, candidate–JD pairs and human suitability labels simultaneously, while still allowing the integrated system to be evaluated on the project's target domain.
- **The contribution is positioned as evidence-based technology selection rather than algorithmic novelty.** The project aims to learn from established ATS, information-retrieval and NLP approaches, then demonstrate where a practical hybrid approach performs better than cheaper alternatives.
- **The optional clustering analysis is retained only as supplementary EDA.** It is not part of the core experimental pipeline because it does not provide a necessary downstream signal for ranking.

---

## 1. Introduction

Recruitment teams receive large volumes of resumes in PDF and DOCX formats, but initial screening remains heavily manual. A recruiter may spend 20–30 minutes reviewing one application, making it difficult to process hundreds of candidates consistently. Important evidence such as transferable skills, technical tools and relevant experience may also be missed when wording differs between a resume and a job description.

This project proposes an Intelligent Candidate Screening System that analyses resume and job-description text, ranks applicants against role requirements and explains each result through a conversational recruiter interface.

The project is deliberately not positioned as an attempt to invent a fundamentally new candidate-matching algorithm. Candidate screening has already been addressed through increasingly sophisticated approaches, ranging from rules and keyword matching to information extraction, semantic retrieval, transformer-based re-ranking and large language models. The project therefore adopts a **reuse-before-reinvent** principle: established techniques should be leveraged where they already solve the problem effectively, while additional complexity should be introduced only when measurable improvement justifies it.

---

## 2. Industry and Technical Overview

Talent-acquisition systems have evolved from manual review and keyword filtering toward structured skills matching, semantic search and machine-learning-assisted ranking. Traditional ATS approaches remain useful because explicit requirements, exact skills and hard constraints are cheap to process and easy to explain. However, lexical matching can miss transferable or semantically equivalent experience when the wording of a resume differs from the wording of a job description.

Modern NLP addresses this limitation through embeddings and transformer-based models. A bi-encoder such as Sentence-BERT can independently embed resumes and job descriptions and perform efficient semantic retrieval. A cross-encoder can then jointly process a resume and job description for a more detailed relevance decision, at a higher computational cost. This retrieve-then-rank pattern is a standard information-retrieval strategy and is increasingly relevant to modern candidate-matching systems.

The latest generation of recruitment systems also incorporates AI-assisted search and, increasingly, large language models. These approaches can reason over broader context and produce natural-language explanations, but their additional complexity does not guarantee agreement with human judgement. For a recruitment use case, transparency, evidence grounding, fairness and human oversight therefore remain important considerations rather than optional features.

This project uses the existing technology landscape as a starting point rather than assuming that a newer model is inherently better. The research asks which combination of established techniques provides an effective balance of matching quality, speed, transparency and implementation effort for the project's scope.

---

## 3. Business Problem, Research Question and Objectives

### 3.1 Business Problem

Manual screening creates long lead times, inconsistent evaluation criteria and high initial false-positive rates. Recruiters also have limited time to compare several candidates against the same required and preferred skills. The proposed platform addresses these gaps by standardising first-pass analysis and allowing recruiters to interrogate the results conversationally.

### 3.2 Research Question

> **Does progressively increasing the sophistication of candidate–job matching provide a meaningful improvement in screening quality, and at what point does the additional complexity stop being worthwhile?**

The project therefore treats model selection as an empirical question rather than assuming that the most advanced model will produce the best practical solution.

### 3.3 Research Objectives

1. **Learn from existing solutions.** Identify established techniques from traditional ATS, information extraction, information retrieval, semantic matching and modern neural approaches that can be reused or adapted for candidate screening.
2. **Measure the value of additional sophistication.** Compare classical and neural matching approaches under the same evaluation protocol and assess the trade-off between performance, speed, complexity and explainability.
3. **Validate the integrated approach in-domain.** Determine whether the selected approach generalises to technology, data and software recruitment using a held-out in-domain evaluation set and human judgement where feasible.

### 3.4 System Objectives

The project aims to:

- Reduce average screening time from approximately 25 minutes to approximately 2 minutes per candidate.
- Achieve at least 85% candidate–role fit agreement with human judgement, while supporting 500+ applications per role in the pilot scenario.
- Extract skills, tools, qualifications, job titles and experience evidence from resumes.
- Retrieve and rank candidates against a job description, showing matched requirements, gaps, confidence and supporting evidence.
- Provide a conversational UI for shortlist, comparison and explanation requests.

---

## 4. Project Design

### 4.1 Design Principle

The project follows a **classical-first, evidence-driven progression**:

> **Start with the cheapest reasonable solution → measure its performance → add semantic or neural capability only where it addresses a demonstrated limitation.**

This creates two complementary evaluation questions:

- **What can we copy or adapt from established solutions?**
- **Where does the proposed approach do better, and is the improvement worth the additional complexity?**

The architecture is therefore a hybrid rather than an attempt to replace established recruitment techniques with one monolithic model.

### 4.2 Solution Architecture

*Figure 4.1 Proposed MVP architecture* — Resume/JD ingestion → text preparation → (a) information extraction, (b) candidate–job matching → ranking & evidence layer → conversational UI.

The candidate–job matching path is evaluated progressively, while information extraction is evaluated as a separate supporting component:

```text
Resume + JD
    │
    ├────────────── Information Extraction ──────────────┐
    │                Rules → BERT NER                    │
    │                                                     │
    └──────────── Candidate–Job Matching ─────────────────┤
                     │                                   │
             Classical baseline                          │
        Rules / skill overlap                            │
                     ↓                                   │
             TF-IDF retrieval                            │
                     ↓                                   │
          Sentence-BERT retrieval                        │
                     ↓                                   │
       Cross-encoder re-ranking                           │
                     │                                   │
                     └──────────┬────────────────────────┘
                                ↓
                      Ranking & evidence layer
                                ↓
                     Conversational recruiter UI
```

The final MVP does not require every model component to be trained on the same dataset. Instead, each supervised component is developed using the dataset that provides the appropriate labels, and the resulting components are integrated and evaluated on a smaller common in-domain set.

### 4.3 Task 1 — Information Extraction / Sequence Labelling

The first task identifies hard and soft skills, tools, frameworks, certifications, job titles and experience durations.

The baseline uses spaCy rules and phrase matching. The advanced comparison uses fine-tuned BERT NER. Skill normalisation compares direct string matching with Sentence-BERT similarity.

NER is treated as a **component-level evaluation** rather than part of the main matching-sophistication progression. The reason is methodological: public NER datasets provide entity-level labels, while candidate–JD fit datasets generally do not. The project therefore evaluates NER where appropriate, then uses the resulting structured evidence as an input to the integrated screening system.

### 4.4 Task 2 — Candidate–Job Matching via Progressive Retrieve-and-Rank

Task 2 is the main research experiment. It evaluates increasingly sophisticated approaches against the same resume–job-description pairs.

#### Stage 1 — Classical baseline

A rule-based skill-overlap method provides the simplest practical benchmark. Explicit requirements, known skills and overlap thresholds produce an initial fit assessment.

#### Stage 2 — Lexical retrieval

TF-IDF with skill-overlap and cosine similarity provides a stronger classical information-retrieval baseline. It remains inexpensive, fast and interpretable, while accounting for the importance of terms beyond simple binary matching.

#### Stage 3 — Semantic retrieval

A two-tower Sentence-BERT model independently embeds resumes and job descriptions into a shared vector space. Cosine similarity retrieves semantically relevant candidates even when the exact wording differs.

#### Stage 4 — Neural re-ranking

A BERT cross-encoder jointly processes each retrieved resume–JD pair and classifies it as **Good Fit / Potential Fit / No Fit**. This is the project's main deep-neural-network and text-classification component.

The retrieval stage limits expensive cross-encoder scoring to a shortlist. At the target pilot scale of approximately 500 applications per role, brute-force cross-encoding is also feasible; retrieve-then-rank is therefore evaluated as an established information-retrieval pattern rather than being presented as a demonstrated scalability necessity.

### 4.5 Ranking and Evidence

The final ranking layer combines the matching output with structured evidence such as required/preferred skill coverage, experience relevance and evidence extracted from the resume. The system will display:

- matched requirements;
- missing or weakly supported requirements;
- relevant experience evidence;
- confidence or fit category; and
- source evidence supporting the result.

The system should distinguish **hard requirements** from general semantic similarity where possible. This is intended to preserve one of the strengths of traditional ATS systems—explicit requirement handling—while benefiting from semantic matching for transferable or differently worded experience.

### 4.6 Conversational UI

A Streamlit or Gradio chat interface will allow recruiters to upload files, request top candidates, apply job-relevant filters, compare candidates and ask why a result was assigned.

The conversational layer is not the primary decision model. Responses will be grounded in extracted resume evidence and model outputs and presented through constrained templates where practical. This reduces the risk that a generative component invents evidence that is not present in the source documents.

### 4.7 Optional Exploratory Clustering

As a small, clearly scoped side analysis, the team may cluster the IT/tech-domain resume corpus using Sentence-BERT embeddings to sanity-check whether natural groupings resemble common role families.

Clustering is not a pipeline dependency and will not be used to justify the candidate-ranking architecture. Retrieval and clustering both perform a narrowing function; stacking them into the core pipeline would add complexity without a clear downstream benefit.

### 4.8 Development Methodology

The team will use an Agile-inspired, iterative approach. Each sprint will produce a testable component, followed by error analysis and refinement.

The classical matching baseline will be completed before the neural matching stages so that improvements can be measured rather than assumed. Where the advanced model does not provide a meaningful improvement over a simpler baseline, the simpler method will be retained for the relevant function.

### 4.9 Software Development Lifecycle

1. **Requirements and evaluation-protocol definition** — confirm recruiter needs, fit-label definitions, research metrics and success criteria.
2. **Data preparation** — extract text, de-identify personal information, clean datasets and define the common in-domain evaluation protocol.
3. **Component baselines** — build spaCy extraction, rule-based skill matching, TF-IDF retrieval and rule-based fit classification.
4. **Progressive model development** — fine-tune BERT NER, develop Sentence-BERT retrieval and fine-tune the cross-encoder fit classifier.
5. **Model comparison and selection** — compare stages using the same matching dataset, evaluate complexity/performance trade-offs and select the components that provide sufficient improvement.
6. **Platform integration** — connect extraction, retrieval, ranking and evidence services to the conversational UI.
7. **End-to-end testing and reporting** — evaluate the integrated system on the in-domain held-out set, including fairness, usability and human agreement where feasible.

---

## 5. Scope of Work

### 5.1 Data Strategy

No single public dataset provides all of the supervision required for NER, candidate–JD matching and human suitability judgement. Forcing all project components onto one dataset would require substantial new annotation effort and would not necessarily produce a better evaluation.

The project therefore uses a **task-specific development strategy plus a common in-domain evaluation set**:

- Task-specific datasets are used where their labels are appropriate for training or component evaluation.
- The resume–JD fit dataset is used consistently across the candidate-matching progression so that the classical-to-neural comparison is fair.
- A smaller self-collected in-domain dataset is used to evaluate the integrated system in the project's target technology/data/software domain.

| Task | Primary source(s) | Role in the project |
|---|---|---|
| Information extraction | [DataTurks Resume Entities for NER](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner) — 220 resumes, span-level labels for Skills, Designation, Companies, Degree, College, Experience | Gold seed/validation set for both the spaCy baseline and BERT NER fine-tuning. Small size means it may require augmentation through weak/silver labelling or a modest batch of self-annotated resumes. |
| Candidate–JD matching progression | [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) — 8,000 resume–JD pairs, 3-class Fit label | Common development/evaluation dataset for the classical-to-neural matching progression, including rule/skill-overlap, TF-IDF, Sentence-BERT and cross-encoder approaches. |
| Retrieval calibration | [0xnbk/resume-ats-score-v1-en](https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en) — 6,374 rows, continuous ATS-style score (Apache 2.0) | Supplementary continuous signal for calibrating and sanity-checking retrieval/ranking scores. It is not treated as the main evidence for human suitability. |
| In-domain end-to-end evaluation | 200+ self-collected technology/data/software job descriptions, paired with a sample of resumes | Common held-out evaluation set for the integrated screening system. Human/team-adjudicated fit labels will be used where feasible and clearly distinguished from synthetic or proxy labels if real recruiter labels cannot be obtained. |
| Skill vocabulary | Public skill lists, developer surveys, technology repositories, IEEE/ACM terminology | Supports entity normalisation, hard-requirement checks and feature engineering. |
| Optional clustering EDA | A tech-filtered subset of a public labelled resume-category set, such as the ~962-resume 25-category Kaggle set or [ResuméAtlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas) | Exploratory analysis only; not required by the core pipeline. |

The key distinction is that **different datasets may be used to train different components, but the main candidate-matching progression is run on one common resume–JD dataset, and the integrated system is evaluated on a common in-domain set.**

### 5.2 Data Preparation and Privacy

Resume and job-description files will be converted to clean text while retaining useful section boundaries. The main steps are:

- Remove or mask names, addresses, age, gender, nationality and direct identifiers.
- Detect extraction failures, empty pages, duplicated resumes and malformed PDF/DOCX content.
- Normalise case, whitespace, common skill aliases, abbreviations and date/experience expressions.
- Split data at candidate level into training, validation and held-out test sets to avoid leakage.
- Keep the in-domain evaluation set isolated from model training and tuning.
- Review class balance across the three fit labels and document label definitions and disagreement handling.
- Preserve evidence spans and section information where possible so that explanations can be grounded in the source resume.

### 5.3 Models, Outputs and Evaluation

#### 5.3.1 Component-Level Evaluation

| Component | Baseline | Advanced model | Primary measure |
|---|---|---|---|
| Entity extraction | spaCy rules / phrase matching | Fine-tuned BERT NER | Entity F1 |
| Skill normalisation | Direct/string matching | Sentence-BERT similarity | Precision/Recall or matching accuracy |
| Candidate–JD retrieval | TF-IDF + skill-overlap + cosine | Sentence-BERT bi-encoder | Recall@k, Precision@k |
| Candidate–JD fit classification | Rule-based skill-overlap threshold | Fine-tuned BERT cross-encoder | Macro-F1 |
| Conversational UI | Scripted intents | Multi-turn evidence display | Task success |

#### 5.3.2 Progressive Matching Experiment

The main research comparison will use the same resume–JD dataset and progressively add model sophistication:

| Stage | Matching approach | Purpose |
|---|---|---|
| 1 | Rule-based skill overlap | Establish the cheapest interpretable benchmark |
| 2 | TF-IDF + skill-overlap | Test whether classical lexical retrieval improves the baseline |
| 3 | Sentence-BERT retrieval | Measure the benefit of semantic matching when wording differs |
| 4 | Sentence-BERT retrieval + cross-encoder re-ranking | Measure the benefit of joint transformer-based comparison |
| Optional | LLM-based comparison | Exploratory comparator only if time and resources permit |

For each stage, performance will be considered alongside implementation complexity and runtime. The objective is not to maximise model sophistication; it is to identify the **simplest approach that achieves an acceptable level of screening quality**.

#### 5.3.3 Integrated End-to-End Evaluation

The final selected classical and advanced pipelines will be evaluated on the in-domain technology/data/software set. The comparison will focus on:

- candidate–JD fit agreement with human or team-adjudicated judgement;
- retrieval quality and ranking quality;
- evidence completeness and faithfulness;
- screening time;
- response latency; and
- qualitative recruiter usefulness.

Where a model component is trained on one public dataset and evaluated on another, the report will clearly separate **component performance**, **in-domain generalisation** and **end-to-end system performance** rather than treating them as interchangeable measures.

### 5.4 Scope Boundaries

The MVP covers English-language resumes and job descriptions relevant to technology, data and software roles, evaluated through candidate–job fit rather than a fixed role-family label.

It excludes:

- autonomous hiring or rejection decisions;
- a claim of production-ready recruitment deployment;
- building a comprehensive external skills ontology from scratch;
- production Kubernetes deployment;
- unconstrained generative recommendations; and
- any requirement to create one universal dataset containing all NER, suitability and ranking labels.

These boundaries keep the work feasible while demonstrating preprocessing, information extraction, retrieval, classification, DNN fine-tuning, ranking, evaluation and conversational-UI design.

---

## 6. Key Deliverables

- De-identified and quality-checked resume/job-description datasets, with documented fit-label and evaluation protocols.
- Skill and entity extraction service with baseline and advanced-model evaluation.
- Classical and neural candidate–job matching models, with a documented progression from simple to sophisticated approaches.
- Comparative evaluation showing performance, runtime and complexity trade-offs across matching stages.
- Selected integrated screening pipeline containing retrieval, ranking, evidence and confidence outputs.
- Ranked shortlist output containing matched requirements, gaps and source evidence.
- Conversational MVP, source code, run documentation, evaluation results, report, slides and recorded demonstration.

---

## 7. Success Measures

The success criteria are designed to evaluate both **absolute system performance** and **whether additional model sophistication is justified**.

| Stage | Technical target | Operational / research target | Validation |
|---|---|---|---|
| Classical baseline | Extraction F1 >85%; retrieval Recall@10 >80% | Establish runtime and quality baseline | Held-out component datasets |
| Advanced matching | Fit-classification Macro-F1 >85%; retrieval Recall@10 >90% | Demonstrate measurable improvement over the classical matching baseline | Same resume–JD test set |
| Model-selection conclusion | Identify a preferred stage based on quality vs complexity | Document where added sophistication provides or fails to provide meaningful benefit | Comparative experiment |
| In-domain pilot | Review performance gaps across evaluated groups | Shortlisting time >80% lower; recruiter/team-adjudicated agreement target ≥85% where feasible | 200+ in-domain applications/pairs |
| Conversational UI | Stable core-intent responses | Recruiter task success and evidence-grounded answers | Scenario-based testing |

The project will not treat failure of a more sophisticated model as a negative outcome. If a simpler method performs comparably or better for a component, the result will be reported as evidence that additional complexity was not justified.

---

## 8. Effort Estimates and Timeline

| Weeks | Effort | Activity | Output |
|---|---|---|---|
| 1–2 | 6 days | Requirements, data collection, de-identification and evaluation-protocol definition | Dataset plan and label/evaluation guide |
| 3–4 | 5 days | Preprocessing and entity extraction baseline/advanced comparison | Extraction component |
| 5 | 3 days | Rule-based and TF-IDF candidate-matching baselines | Classical matching benchmark |
| 6–7 | 6 days | Sentence-BERT retrieval and cross-encoder fit classification | Neural matching models |
| 8 | 3 days | Comparative model evaluation and ranking/evidence layer | Model-selection findings and explainable ranking |
| 9–10 | 4 days | Conversational UI and API integration | Working MVP |
| 11 | 2 days | In-domain, performance, fairness and usability testing | Evaluation findings |
| 12 | 1 day | Documentation, report and demo | Final submission |

The total estimated effort is approximately 30 team-days, or approximately 15 days per member. The roadmap may be adjusted after data access and label quality are confirmed.

---

## 9. Expected Value and ROI Assumptions

At 500 candidates per month, indicative annual value is approximately $750,000: recruiter time saved of $115,000 (6,000 candidates × 23 minutes × $50/hour), faster hiring value of $450,000 (15 days × 6 roles × $5,000/day) and retention-related savings of $200,000. Against an estimated first-year implementation cost of $150,000, the indicative net ROI is approximately 400%.

These figures are assumptions to be validated during the pilot rather than established financial claims. The ROI case also depends on the system reducing real screening effort without creating additional review burden through false positives, unsupported explanations or manual verification.

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No single dataset contains all required labels | High | Use task-specific datasets for component development and a common in-domain set for end-to-end evaluation. |
| Model sophistication does not improve performance | Medium | Treat this as a valid research result; retain the simpler model where its performance/complexity trade-off is better. |
| Historical-data bias | High | PII removal, subgroup evaluation, balanced review data and human oversight. |
| Resume-format variability | Medium | Robust parsing, extraction validation and fallback handling. |
| Public fit-pair data is general-industry, not IT-specific | Medium | Supplement with self-collected in-domain resume–JD pairs; evaluate primarily on the in-domain held-out set. |
| Emerging or unseen skills | Medium | Versioned vocabulary, semantic normalisation and periodic review. |
| Unsupported explanations | Medium | Evidence snippets, constrained templates and confidence thresholds. |
| LLM or generative component produces unsupported judgement | Medium | Keep generative functionality outside the core ranking decision and ground responses in extracted evidence and model outputs. |
| Privacy and security | High | De-identification, access control, retention limits and encrypted storage. |

---

## 11. Next Steps

- Confirm project group members, dataset ownership, privacy approval and access arrangements.
- Finalise the fit-label definitions (Good Fit / Potential Fit / No Fit), annotation guide and evaluation protocol.
- Prepare a small de-identified sample and establish the rule-based and TF-IDF matching baselines before neural modelling.
- Collect the 200+ in-domain job descriptions and matching resume sample early, since this set is the primary evidence for end-to-end performance in the project's actual target domain.
- Define a consistent comparison protocol so that classical and neural matching stages use the same candidate–JD test pairs.
- Validate technical feasibility before committing to the full dataset and pilot targets.

---

# Appendix A — Rationale Behind the Project Design

This appendix records the reasoning behind the v3 design. It is intentionally more detailed than the main proposal so that the project can explain why the methodology was chosen without making the core proposal overly long.

## A.1 Why the project should not claim novelty for its own sake

The project team discussed a principle of **“if someone has already solved this problem, why not leverage it?”** Candidate screening is not a new problem. Traditional ATS products already use keywords, structured candidate attributes and skill-based filters. Modern systems increasingly add semantic matching and AI-assisted search.

The project therefore should not frame novelty as inventing a new neural architecture. A more defensible contribution is to:

1. identify established approaches;
2. implement a clear classical baseline;
3. progressively introduce more sophisticated techniques;
4. measure the incremental benefit of each addition; and
5. determine which combination provides the best practical trade-off.

This is particularly appropriate for a practice-module project because it demonstrates understanding of both established and modern NLP methods while keeping the business objective central.

## A.2 Why progressive sophistication is still possible with multiple datasets

At first glance, progressive evaluation appears to require one dataset containing every label used by every stage of the system. That would be unrealistic here because the project needs different forms of supervision:

- NER requires span-level entity annotations;
- candidate matching requires paired resume–JD examples;
- human agreement requires suitability judgements; and
- the project's target domain requires technology/data/software examples.

There is no need for these labels to coexist in one dataset.

The important methodological requirement is narrower: **models being directly compared against each other must use the same underlying task and evaluation examples.**

This is satisfied for the main matching experiment because the resume–JD fit dataset can support the progression:

```text
Rule / skill overlap
        ↓
TF-IDF retrieval
        ↓
Sentence-BERT retrieval
        ↓
Cross-encoder re-ranking
```

The NER experiment is independent:

```text
spaCy rules / phrase matching
        vs.
Fine-tuned BERT NER
```

The resulting NER component can then feed structured evidence into the integrated system.

Finally, a separate in-domain set evaluates the integrated system rather than attempting to train every component simultaneously.

This distinction can be described as:

> **task-specific model development + common matching benchmark + common in-domain system evaluation.**

That is a cleaner design than attempting to construct an artificial universal dataset simply to make every component use identical examples.

## A.3 Why NER should not be treated as one stage in the sophistication ladder

The earlier design risked creating an artificial progression such as:

> rules → NER → SBERT → cross-encoder.

This is conceptually misleading because NER and candidate matching solve different problems. NER extracts information from a resume; SBERT and cross-encoders compare a resume with a job description.

The revised design therefore evaluates NER separately and treats it as a supporting capability. Its value is measured through extraction quality and usefulness of the resulting evidence rather than by forcing it into a single matching score progression.

## A.4 Why the classical baseline matters even if the final system uses transformers

A classical baseline is not merely a requirement to demonstrate an older technique.

Keyword, rule-based and TF-IDF methods have practical strengths:

- low computational cost;
- fast inference;
- deterministic behaviour;
- easy debugging;
- easy explanation; and
- strong performance for explicit requirements.

For example, when a JD says “AWS certification required”, exact matching can be preferable to a semantic model that decides another cloud-related skill is sufficiently similar.

The project should therefore not assume that semantic matching replaces lexical matching. Instead, it can test whether the best system is a hybrid in which explicit requirements and semantic similarity complement one another.

## A.5 Why retrieve-then-rank is preferable to using the most expensive model everywhere

A cross-encoder jointly processes a candidate and a job description, allowing richer interaction between the two texts. However, doing that for every candidate is more expensive than independently embedding candidates and jobs.

A retrieve-then-rank architecture therefore provides a natural compromise:

```text
All candidates
     ↓
Cheap retrieval
     ↓
Shortlist
     ↓
Expensive detailed comparison
     ↓
Final ranking
```

At the project's 500-applications-per-role scale, the computational saving is not the main reason for the architecture; brute-force scoring may still be practical. The value is that the project can demonstrate a standard information-retrieval pattern and explicitly measure whether the more expensive re-ranking stage provides meaningful benefit.

The proposal therefore avoids claiming scalability that has not been demonstrated.

## A.6 Why an LLM is not automatically the final answer

A natural question is why the system should not simply give the resume and JD to a large language model and ask for a fit judgement.

The project discussion identified several concerns:

- cost and latency can be higher than classical retrieval;
- the output may be less deterministic;
- explanations can sound convincing without being faithfully grounded in the resume;
- model judgement is not automatically equivalent to human recruiter judgement; and
- a generative model makes it harder to isolate which capability is responsible for a performance improvement.

LLMs may still be useful for the conversational layer, summarisation and explanation. They can also be included as an optional comparator if time and resources permit. They should not, however, be assumed to be the correct core ranking mechanism simply because they are the newest technology.

## A.7 Why the in-domain dataset is important

The public resume–JD fit data is useful for supervised development, but it is not necessarily representative of the project's intended technology/data/software domain.

A small self-collected in-domain dataset therefore serves a different purpose from the public training dataset. It tests whether the model generalises to the actual use case and makes the final claim more defensible.

The in-domain set should ideally include:

- realistic technology/data/software job descriptions;
- resumes with varied wording and experience levels;
- enough positive and negative examples to test retrieval and ranking;
- held-out examples that are never used for tuning; and
- human or team-adjudicated fit judgements where feasible.

If genuine recruiter labels are unavailable, the report should explicitly identify any synthetic or team-adjudicated labels rather than presenting them as historical recruiter decisions.

## A.8 What “do better” means in this project

“Better” should not simply mean a higher F1 score.

The project should consider at least four dimensions:

1. **Quality** — Does the system retrieve and rank suitable candidates more accurately?
2. **Efficiency** — Does the improvement justify additional inference time or implementation effort?
3. **Explainability** — Can a recruiter understand the reason for the recommendation and verify it against source evidence?
4. **Robustness / domain fit** — Does the improvement hold on technology/data/software resumes rather than only the public benchmark?

This supports the project's central practical question:

> **What is the simplest system that provides sufficient value?**

A sophisticated model that improves Macro-F1 by a small amount but substantially increases complexity may not be the preferred solution. Conversely, a neural model that materially improves ranking on the in-domain set and remains operationally feasible has a strong case for adoption.

## A.9 Why evidence-grounded explanations are part of the solution

A recruiter is unlikely to trust a score such as “87% match” without knowing what drove it.

The system should therefore show evidence such as:

| Requirement | Evidence |
|---|---|
| Python | Explicit experience in Python development |
| Machine learning | Relevant modelling experience |
| SQL | Evidence from previous roles/projects |
| AWS | No explicit evidence found |
| Leadership | Evidence of team or project leadership |

This also provides a practical safeguard: the recruiter can inspect the evidence instead of treating the model output as a final decision.

## A.10 Why human oversight remains part of the design

The system is intended to support first-pass screening, not to make autonomous hiring decisions. Human review remains important because candidate suitability can depend on contextual factors that are difficult to capture in a fixed benchmark.

The system should therefore be framed as:

> **decision support for recruiters, rather than an autonomous recruiter.**

This is consistent with the project's emphasis on evidence, fairness, privacy and recruiter interrogation through the conversational UI.

## A.11 Summary of the final research logic

The overall reasoning can be reduced to the following chain:

```text
Existing recruitment systems already use matching
                ↓
So we should reuse established ideas
                ↓
Start with a cheap classical baseline
                ↓
Add semantic retrieval if lexical matching misses relevant candidates
                ↓
Add cross-encoder ranking if deeper comparison improves fit judgement
                ↓
Measure the incremental benefit at every stage
                ↓
Validate the selected approach in the target domain
                ↓
Use evidence and human oversight for the final recruiter experience
```

The project is therefore not asking:

> **“Can we build an AI recruiter?”**

It is asking:

> **“Which existing NLP techniques actually earn their place in a practical candidate-screening system, and how much better are they than simpler alternatives?”**

That distinction is the central rationale for the v3 proposal.
