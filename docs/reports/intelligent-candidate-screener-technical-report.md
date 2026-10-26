# Intelligent Candidate Screener: Technical Report

**Project:** NUS ISS PLP Practice Module, Group 2  
**Review date:** 24 October 2026  
**Committed application baseline:** `409349e`  
**Concurrent development observed:** v2 plan approval `320c3fa` and uncommitted implementation on `feat/hiring-desk-v2-conversation`  
**Assessment basis:** proposals, dated plans, decision history, source code, recorded experiments and retained validation logs.

## 1. Executive Summary

The Intelligent Candidate Screener is a research-led decision-support system for reviewing resumes against job descriptions. It combines document preparation, information extraction, relevance scoring, ranking, source evidence and a recruiter interface. Its central research question is whether progressively more sophisticated matching methods provide sufficient improvement to justify their additional computational and implementation cost.

The repository contains three related delivery paths: an experimental ranking framework, the authenticated Hiring Desk application, and an earlier upload-based screening prototype. These paths share utilities and evidence principles but currently use different scoring integrations. The research framework evaluates rules, lexical retrieval, sentence embeddings, learned combinations and neural reranking. Hiring Desk primarily reviews supplied CSV rankings and supports a separate adapter for ranking newly authored job descriptions. The upload prototype uses a submission-level TF-IDF illustration with optional frozen reranking.

The latest reported research configuration is L1, a constrained linear model trained on the document-disjoint `chat-3` collection, followed by a Qwen3 reranker applied to its top 50 candidates. Its adoption includes an explicitly documented exception: the additional reranker did not meet the registered Recall@10 improvement criterion. This distinction is essential to interpreting the research contribution. A technically richer system was selected by the owner, but the evidence does not establish that its additional cost is justified under the original acceptance rule.

The committed Hiring Desk offers JD browsing, saved reviews, guided JD creation, applicant evidence, comparison tables, alerts and a demonstration email handoff. Its default v1 chatbot uses RoBERTa-SQuAD2 and cached GLiNER-medium entities. It provides source-linked extraction and structured comparison; it does not reliably produce a reasoned answer to “Why is candidate 1 better than candidate 2?” V2 conversation implementation was approved and began during this review, but was not release-validated. Corporate identity, production mail delivery, portable research-model deployment and realistic conversational-quality validation remain incomplete.

The principal value is a traceable framework for assessing recruitment matching methods and a working local review workflow. The repository supports a prototype assessment rather than a production-readiness claim. Proposal targets for recruiter time savings, screening agreement and financial benefit should be treated as objectives or assumptions, not demonstrated business outcomes.

## 2. Project Context & Objectives

### 2.1 Problem Statement and Intended Value

Manual resume screening consumes recruiter time and can produce inconsistent comparisons across candidates. Keyword-only approaches can miss transferable or differently worded experience, while semantic similarity alone may overlook explicit job requirements. An effective screening assistant must therefore combine useful ordering with inspectable evidence and retain human control over hiring decisions.

Proposal v3 adopts a classical-first progression: establish a low-cost baseline, measure it, and add neural components only when their benefit is supported. Information extraction is evaluated separately from candidate–JD matching because the datasets provide different forms of supervision. The intended business benefit is faster, more consistent first-pass review; the academic contribution is an empirical analysis of quality, runtime, transparency and complexity. [Proposal v3](../proposal/PLP_Group2_Intelligent_Candidate_Screening_Proposal-v3.md)

### 2.2 Scope and Boundaries

The core scope covers English-language technology, software and data recruitment. Inputs include JD text and candidate text, with PDF/HTML/text conversion in the earlier upload path. An additional French interview prototype transcribes and translates media into reviewer-confirmed English text; this is an exploratory extension, not evidence of general multilingual screening quality.

The project excludes autonomous hiring or rejection, unconstrained generative recommendations and production Kubernetes deployment. It does not implement a complete applicant-tracking system, corporate identity platform or external recruiting integration. In the current portal, original applicant PDFs are not reconstructed from CSV-derived text previews.

### 2.3 Objectives and Success Criteria

| Objective | Proposal or protocol measure | Interpretation at review |
|---|---|---|
| Reduce screening effort | Approximately 25 to 2 minutes per candidate; more than 80% time reduction | Business target; computational latency does not establish recruiter time saved |
| Improve suitability assessment | At least 85% agreement; classification macro-F1 above 85% | Proposal targets; later protocols distinguish binary classification from ordering |
| Retrieve suitable applicants | Proposal Recall@10 targets above 80% and 90% | Later registered retrieval experiments use Recall@50 in 500-candidate pools; the metric change must remain explicit |
| Extract useful information | Strict entity-span F1, including an 85% proposal target | Accuracy requires compatible held-out gold; extracted counts and exact offsets are insufficient |
| Select an appropriate model | Paired performance differences, uncertainty and cost | Prefer a simpler model unless additional sophistication earns its complexity |
| Support recruiter review | Scenario completion, evidence faithfulness and useful conversation | Workflow tests exist; realistic task-success and usefulness studies remain limited |
| Characterize generalization | Strata, role-family and other available group analyses | Dataset slices are not a substitute for demographic fairness evaluation |

The evaluation protocol subsequently merged `Good Fit` and `Potential Fit` into binary `Fit`; `No Fit` remains separate. Human grades may still support ordering analysis. Inter-annotator agreement is reported as context, with no success target. These amendments prevent the original three-class objectives from being presented as unchanged implementation criteria. [Registered criteria](../../plan/2026-09-26-top-down-reset/06-registration.md), [decision log](../decisions.md)

### 2.4 Data Resources and Supervision

| Resource | Purpose | Material qualification |
|---|---|---|
| A2: Djinni recruitment data | Primary in-domain benchmark; train-region model development | English IT profiles from the Ukrainian/Eastern-European market; applicability to other markets is unproven |
| A1: `cnamuangtoun` fit pairs | Model-development resource and secondary benchmark | Its supplied fit labels do not reproduce under the project's recheck; every A1 performance claim requires this caveat |
| DataTurks and SkillSpan | Resume/job-posting extraction supervision and evaluation | Entity schemas and document types differ from the integrated screening task |
| ESCO and job-skills resources | Vocabulary, aliases and requirement matching | Vocabulary coverage is incomplete and is not a universal skills ontology |
| Livecareer PDF/HTML and ResumeAtlas | Parsing, document robustness and supporting extraction work | Real-document privacy, licensing and contamination controls apply |
| Operational workspace | Reviewer accounts, JD revisions, sessions, ranking requests and local email captures | Application state is excluded from benchmark annotations and training |

The measured A2 source inventory contains 141,897 English JDs and 210,250 English CVs. The frozen final reference comprises 600 pairs: 40 generic JDs and 20 targeted JDs, each with ten judged candidates. These are inventory counts, not performance estimates. Candidate pools used in retrieval experiments also contain distractors whose relevance is not exhaustively known. [A2 card](../data/cards/A2-djinni.md), [data catalog](../data/data-catalog.md)

The final reference uses a fixed hierarchy: agreement between two humans takes precedence; an LLM breaks human ties; a single human's label is retained; only cases without human judgment can use an LLM-only label. In the recorded final set, 219 pairs have human agreement, 107 use the LLM tie-break and 274 have one human label; none are LLM-only. This is a mixed-provenance reference, not 600 independently double-annotated human decisions. [Final-reference record](../../output/annotation/a2-final-report.json)

## 3. System Architecture & Design Decisions

### 3.1 Architectural Overview

The application is a Python package with a `src/` layout. Its main separation is between data preparation, experimental scoring, evidence services, conversational inference and presentation. Streamlit provides the current local UI; reusable services own authorization, state transitions and inference contracts.

```text
Research path
Source registry -> fetch/verify/profile -> document partitions and text_v2
                                            |
                       training resources -> fitted models/features
                                            |
             evaluation candidate pools -> L1 -> top 50 -> C2 fusion -> top 10
                                            |
                             frozen references -> metrics, CIs and records

Hiring Desk path
Existing ranking CSVs + authorized source-text cache ----+
                                                       |
Guided JD creation -> revision/publish -> ranking adapter+-> review snapshot
                                                              |
                             +--------------------------------+----------+
                             |                                           |
                    GLiNER entity cache                      focused RoBERTa QA
                             +--------------------+----------------------+
                                                  |
                               comparison/evidence -> recruiter review
                                                  |
                               saved report/package -> local email capture
```

Research evaluation and operational review deliberately use separate references and storage. A chat response cannot overwrite a rank, create a Fit label or authorize an operational action.

### 3.2 Current Execution Paths

| Path | Current scoring behavior | Status |
|---|---|---|
| Research-reported system, D64 | L1 on F12 ranks a 500-CV pool; min-max L1 and weighted C2 reorder its top 50 | Implemented research configuration; adoption exception documented |
| Hiring Desk CSV reviews | Reads existing `x2_score` and `x2_rank`; does not regenerate them | Implemented and independent of live model availability |
| Newly created operational JDs | Validated `submit`/`status` adapter returns model-ranked candidates and provenance | Adapter implemented; actual teammate/model binding remains pending |
| Upload-based prototype | Submission-fitted TF-IDF illustration, with optional verified frozen cross-encoder | Separate historical application path |
| Primary v1 chat | GLiNER-medium evidence plus RoBERTa-SQuAD2; application-built comparisons | Implemented; no primary Qwen composition |
| V2 conversation under development | Explicit dialogue state, checked comparison relations and bounded Qwen relation selection | Application implementation approved; uncommitted work and feature-flag wiring observed; training remains separately gated |

The similarly named Qwen components have different purposes. Research C2 is `Qwen/Qwen3-Reranker-0.6B`, which scores JD–CV relevance. Historical conversational Qwen uses an instruction model as an answer/excerpt interface. Disabling the latter in v1 does not remove C2 from the research configuration.

### 3.3 Key Decisions and Their Rationale

| Decision | Architectural consequence | Rationale and trade-off |
|---|---|---|
| D40–D41: A2 primary; one frozen final reference | All comparable human-reference system results share the same adjudication rule | Improves comparability; reference uncertainty remains |
| D19, D46: document-disjoint development; no final-set tuning | Training and selection are isolated from final evaluation documents | Reduces leakage; requires explicit provenance for every training series |
| D53–D54: text-based inference with a JD-title exception | CV role/experience metadata cannot silently drive inference | Prevents oracle-field leakage; extracted features remain imperfect |
| D48: cheapest unbeaten rung | Complexity requires paired evidence of improvement | Protects against selecting a noisy point-estimate winner |
| D62: `chat-3` training, `chat-1` test-only | Stage-2 selection uses train-region JDs and JD-level folds | Restores separation after earlier exploratory stage-2 work |
| D64: add C2 despite the missed acceptance bar | Report the selected system and the unmet criterion together | Owner-directed, post hoc selection rather than a registered success |
| D60–D61 (entity/app): preserve CSV review and source-grounded chat | Ranking, evidence and conversational models remain separate services | Keeps review available and limits unsupported model authority |
| D55 and D58 (research): tiered, cost-aware verification | Cached checks support iteration; cold checks establish reproduction | Extensive model verification can be expensive and is not equivalent to a unit test |

D56–D61 were independently allocated in two development histories. The merged repository qualifies them as `(research)` or `(entity/app)`. Unqualified citations to those numbers are ambiguous unless their linked plan establishes the intended meaning. [Decision log](../decisions.md)

### 3.4 Data Flow, State and Trust Boundaries

Data acquisition is registry-driven. Verification records file digests and expected counts; profiling establishes measured contents. Derived text, partitions, pools and weak pairs are reconstructed through package commands. Training modules must not consume the final annotation files, and contaminated documents are excluded where specified.

Portal ingestion selects permitted CSV columns and reads only the required IDs/text from the existing source cache. Aggregate extraction cells without offsets remain unverified mentions. They are not promoted to source citations merely because a model produced them.

Each active review is bound to a snapshot and source fingerprint. A changed JD, ranking or source file makes an old review historical. Chat submissions use turn identities and optimistic revisions to prevent duplicate saves or overwriting a newer review. Authorization and source freshness are checked again after inference, so a long-running model operation cannot bypass revocation or replace a changed snapshot.

GLiNER caches are private to the principal, workspace, source digest and inference configuration. Only model resources are shared; cached content is not a permission grant. SQLite stores operational records and additive schema migrations. Reviewer-authored drafts and history need private backups because dataset fetch scripts cannot recreate them.

## 4. Implementation Details & Code Analysis

### 4.1 Core Module Responsibilities

| Package or module | Responsibility |
|---|---|
| `data/` | Source registry, acquisition, verification, profiling, partitions, weak pairs and candidate pools |
| `baselines/` | Frozen classical TF-IDF/BM25/skill baselines and reproducibility records |
| `matching/` | Common scorer interface, lexical/semantic systems, extracted features, learned combinations and selection rules |
| `rerank/` | Trained and pretrained rerankers, constrained learners, scoped scoring collections and cached neural scores |
| `evaluation/` | Classification, ordering, retrieval, paired bootstrap comparisons, error analysis and registered checks |
| `extraction/` | Vocabulary/rules, NER, role/experience extraction, GLiNER runtimes, chunking and source/coverage validation |
| `evidence/`, `pipeline.py` | Requirements, verbatim support, coverage and the draft text-screening pipeline |
| `app/portal_*` | Authenticated review, JD authoring, ranking adapter, snapshots, chat, alerts, reports and mail capture |
| `llm/` | Local extractive QA, historical Qwen clients/services and optional QA-training utilities |
| `annotation/`, `entity_annotation/` | Frozen Fit-label instrument and separately governed human entity-span annotation |
| `multimodal/` | Media decoding, transcription/translation, reviewed text and translation provenance |
| `checks.py`, `tests/`, `notebooks/` | Reproduction runner, regression tests and thin reporting/exploration notebooks |

### 4.2 Matching and Ranking Algorithms

Rules and skill overlap provide interpretable starting points. TF-IDF uses term-weighted lexical vectors, while BM25 adds document-length normalization and term-frequency saturation. Sentence-BERT independently embeds JD/CV content and compares semantic representations. Cross-encoders jointly process a pair and are evaluated as more expensive alternatives.

All ladder systems conform to one small interface. This permits the same evaluation harness to fit a method, obtain aligned pair scores and inspect its provenance/cost card.

```python
class System(Protocol):
    name: str
    rung: str

    def fit(self, train: pd.DataFrame | None = None) -> None: ...
    def score(self, pairs: pd.DataFrame) -> pd.Series: ...
    def card(self) -> dict: ...
```

*Source: `src/candidate_screener/matching/base.py`; excerpt omits imports.* The shared implementation aligns outputs to pair identifiers and rejects missing score values; application adapters additionally require finite, candidate-aligned results. These checks prevent silent reordering and invalid scores from producing plausible rankings.

The extracted-feature X2 baseline combines inferred role compatibility, experience compatibility and lexical similarity. The later F12 representation includes role and experience signals, TF-IDF, requirement coverage, concept counts/matches, skill overlap, experience descriptors and weakly fine-tuned SBERT similarity.

L1 fits ridge regression on features standardized within the training fold. Match-signal coefficients are constrained to be non-negative; count/experience descriptors can have either sign. The implementation solves bounded least squares on the ridge-augmented matrix. L2 is a monotonic gradient-boosting alternative. JD-level out-of-fold evaluation selects among the registered feature/learner options; the recorded winner is L1 on F12. [Learner code](../../src/candidate_screener/rerank/learners.py), [selection record](../../output/rerank/chat3-select.json)

The D64 reranking step uses the following score within L1's top 50:

```text
fused_score = minmax(L1_score) + 0.25 * minmax(C2_score)
```

The code preserves the shortlist ahead of candidates outside it and uses deterministic tie handling. C2 uses the fixed instruction, “Given a job description, retrieve the CVs of candidates who fit its requirements.” Its configuration pins the model revision, text variant and inference settings. This is relevance scoring, not conversational explanation or calibrated hiring probability. [Fusion code](../../src/candidate_screener/evaluation/c2_validation.py), [reranker configuration](../../src/candidate_screener/rerank/cross.py)

The upload pipeline's displayed confidence is an ordering percentile, not a calibrated probability. Binary classification remains unavailable unless a model has an explicit validated binary threshold and its validation lineage; an ordering score alone cannot supply a Fit/No Fit decision.

### 4.3 Extraction, Chunking and Evidence

The extraction subsystem supports canonical vocabulary/rule matching, a learned job-posting tagger, role-family classification, experience rules and zero-shot GLiNER. The medium model is `gliner-community/gliner_medium-v2.5`. Its configured types are knowledge, skills, titles, degrees and experience duration; certifications retain a text-retrieval route.

Sentence-first chunking preserves the original text and calculates windows against the model's prepared-input budget, including label prompts. Oversized content uses overlapping fallback windows; unrepresentable or interrupted ranges are quarantined and reported. Complete coverage is verified from processed intervals rather than inferred from a successful process exit.

Every accepted entity retains its source field, exact half-open offsets, surface text, label, confidence and context state. The schema distinguishes negation, learning context and an unqualified mention; “unqualified” here means no detected contextual qualifier, not an assessment that the person lacks qualifications.

The evidence layer labels requirements `matched`, `supported` or `missing`. Exact/alias matches receive verbatim spans. Optional semantic support selects a source sentence using a validation-calibrated similarity threshold. “Missing” means no support was found by that process; it does not establish that the candidate lacks the skill.

```python
def assert_faithful(evidence: list[Evidence], cv_text: str) -> None:
    for ev in evidence:
        if ev.status == "missing":
            continue
        if ev.start is None or cv_text[ev.start:ev.end] != ev.quote:
            raise ValueError(
                "Evidence is not a verbatim slice of the CV"
            )
```

*Condensed from `src/candidate_screener/evidence/evidence.py`; exception wording shortened.* This invariant establishes source traceability. It cannot establish that a selected sentence semantically entails a requirement, that a credential is current, or that a duration applies to a specific tool.

### 4.4 Conversational QA and Prompt Engineering

Primary v1 chat resolves candidate references and categories, retrieves relevant authorized passages and applies RoBERTa-SQuAD2 to focused factual questions. The QA model predicts answer start/end positions in context. Fast-tokenizer offset mappings exclude question/special tokens; overflowing context uses overlapping windows. A null-answer score permits abstention.

The portal then validates returned text, offsets, document owner, snapshot and source digest before creating a citation. Surrounding source context remains visible for negation, unfinished degrees, validity dates and duration statements. Total experience is not accepted as skill-specific tenure merely because the skill occurs nearby.

Broad comparisons assemble six evidence categories in application code and can return before background entity extraction completes. Focused comparisons answer each requested document independently. The result is a comparison of documented evidence, not an explanation of the ranking model's causal reasoning.

Historical Qwen prompt engineering used a restrictive evidence-ID selection contract, bounded input/output and validation against server-owned source records. That approach constrained unsupported text but also constrained answer usefulness. Primary v1 no longer invokes this generator; historical answers retain their recorded provenance. The presence of `qa_train.py` or Qwen runtime code does not establish that an adapted conversational model has been trained and promoted.

The default v1 resolver remains largely rule-based. Broad “why better?” questions, implicit references and multi-turn comparison reasoning are not reliably supported. The newly approved v2 work addresses dialogue state and checked comparison relations first. New local modules include `conversation_schema`, `conversation_service`, `comparison_facts`, `conversation_render` and `llm/conversation_qwen`; the observed service binding is opt-in through `HR_PORTAL_CHAT_VERSION=v2`.

V2's generator contract is narrower than free-form explanation: Qwen selects registered relation IDs, while application code writes factual clauses. It must preserve required relations, account for requested candidates and retain qualifications/uncertainty. The fresh suite exposed a failing critical-qualification omission test in this implementation. Its existence therefore establishes work in progress, not a validated remedy for the conversation limitation. Optional training and resource experiments retain separate approval gates. [Current extractive scope](../../plan/2026-10-24-grounded-qwen-comparisons/05-extractive-implementation.md), [v2 plan](../../plan/2026-10-24-hiring-desk-v2/README.md)

### 4.5 Portal Workflows, Authorization and Reliability

Hiring Desk provides selectable JD cards, full redacted JDs, job switching, saved per-job conversations, candidate previews and review actions. Guided authoring saves title, department, responsibilities, skills, knowledge, experience and qualifications as versioned operational JDs. Published JDs can request ranking through a trusted adapter; response identity, JD revision, model/pool provenance and candidate results must validate before a snapshot becomes ready.

Local access uses salted PBKDF2-HMAC token hashes, constant-time comparison, expiry and revocation checks. The default provisioned account lifetime is 40 days. This is local-token authentication, not corporate SSO or a conventional enterprise password service. Logout is a separate button at the bottom of the left drawer.

The controller revalidates a comparison after inference:

```python
result = self.qa_service.compare(snapshot, list(candidate_ids))
if self.snapshot(session_id).digest != snapshot.digest:
    raise StaleReview("Sources changed during comparison")
return result
```

*Source: `src/candidate_screener/app/portal_controller.py`.* The second `snapshot` call also enforces access/freshness checks. Together with candidate-membership checks before inference, this prevents an answer being accepted against a revoked or superseded review.

SQLite transactions and revision checks protect updates. Turn IDs make retries idempotent. Model workers are serialized, load lazily and have bounded I/O deadlines and process cleanup. Missing weights or failed extraction produce explicit unavailable/partial/failure states; application startup does not train replacement models.

Alerts record actual local publication, ranking and capture events, with read/unread controls. Department handoff requires package/email preview and explicit confirmation. New packages contain ten redacted CV text/profile records; historical eight-CV artifacts remain readable. The mail transport captures private MIME files for example Tech/Business recipients. Capture is not external mailbox delivery.

### 4.6 Privacy and Human Oversight

Display paths use the redaction layer, and reports/cache records preserve redacted-coordinate provenance. Mechanical redaction is a safeguard rather than a guarantee of anonymity. The current regex rules do not provide general personal-name removal and do not establish complete removal of protected attributes or indirect identifiers.

The separate entity-annotation workflow keeps suggestions apart from reviewed human spans. Assistance is restricted to authorized training documents, with provenance retained; development/test and independent annotator views remain blind. Fit judgments, entity gold, model suggestions and operational reviewer decisions are distinct data products.

The frozen entity campaign assigns 120 A1 documents, with 80/20/20 train/development/test documents and four entity types. Local journal metadata reviewed for this report records 18 fully reviewed training documents and 212 spans, including seven assisted documents with 78 spans. There are no completed development/test or independent-annotator reviews in that observed journal. A local ten-document adaptation manifest records three training epochs and 36 optimizer steps after all 130 source spans became representable under the amended tokenizer policy. These observations supersede older prose that still calls the pilot pending; they do not establish held-out entity accuracy. The local adapted checkpoint is separate from the pinned zero-shot medium model used by portal enrichment.

The media prototype records whether evidence comes from written text or a reviewed English translation and preserves segment lineage when available. Translated English quotations must not be represented as words directly spoken in the original French recording.

## 5. Project Timeline & Deployment Plan

### 5.1 Planned and Recorded Implementation Phases

Proposal v3 estimates twelve weeks and approximately 30 team-days. The repository records an iterative sequence rather than evidence that every proposed milestone or acceptance target was completed on schedule.

| Period | Recorded phase | Principal outcome or qualification |
|---|---|---|
| August 2026 | Data strategy, acquisition and derived artifacts | Source registry, manifests, profiling, corrected splits and candidate pools |
| Late August–September | Baselines, annotation and judge calibration | Classical reference implementation, annotation tooling and documented disagreement |
| 26–27 September | Top-down reset | Frozen binary reference, common matching ladder and cost-aware selection rule |
| 6–9 October | Text leakage controls, extraction and reranking | `text_v2`, inferred role/experience features and registered stage comparisons; retrieval bars were missed |
| 11–17 October | Evidence, extraction inspection and error analysis | Source validation, GLiNER-medium campaigns, human entity workflow and exploratory model diagnostics |
| 18–22 October | Conversational RAG and ranking redesign | Authored QA scenarios; classifier/reranker experiments; chat-3 training/test separation; L1 and C2 decisions |
| 23–24 October | Hiring Desk v1 | JD navigation/creation, saved review, ten-CV handoff, alerts, local mail capture and extractive QA |
| 24 October | Research/application integration | Histories merged; decision-ID collisions qualified; local extraction notebooks renamed 27–29; outstanding validation recorded |
| 24 October, concurrent with this review | V2 conversation implementation | Approved plan and local dialogue/comparison modules; validation and optional model refinement incomplete |

Earlier design documents are historical evidence, not automatic statements of the current system. In particular, the README's September results and the 11 October ownership walkthrough predate later metadata restrictions, ranking decisions and portal changes.

### 5.2 Runtime and Local Deployment

The project uses Python 3.12, an installed package and `uv` for environment management. The base stack includes pandas, PyArrow, scikit-learn, dataset/Hugging Face utilities and PDF parsing. Streamlit is optional application infrastructure; Torch and transformer dependencies are optional for the base test environment.

Two isolated inference environments pin Torch 2.2.2, Transformers 4.48.3 and GLiNER 0.2.21; the medium runtime additionally pins spaCy and related compatibility dependencies. Their declared environment constraints include macOS x86-64. A successful local setup is not proof of a clean Linux, ARM or hosted deployment. [Runtime manifests](../../runtime/transformer/pyproject.toml), [medium runtime](../../runtime/gliner-medium/pyproject.toml)

The local application entry point is:

```bash
uv sync --group app
uv run streamlit run src/candidate_screener/app/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8503 --server.headless true
```

This command does not provision missing datasets, reviewer accounts or model checkpoints. Those assets require explicit preparation. `HR_PORTAL_DATA` and `HR_PORTAL_ACCOUNTS` select local sources/account configuration; `HR_PORTAL_RANKING_ADAPTER` and `HR_PORTAL_CANDIDATE_POOL` configure the external ranking boundary. Reviewer state and model bytes are excluded from version control.

The verified deployment is a local Streamlit prototype. The reviewed repository contains no implemented production container/orchestration deployment or continuous-integration workflow. Department mail, corporate SSO and enterprise retention/encryption policies are not supplied by the local services.

### 5.3 Proposed Deployment Sequence

The 22 October model-bundle/hosting plan is explicitly on hold. It proposes portable exports for L1 coefficients, role-family features, SBERT, TF-IDF and concept statistics, plus pinned C2 assets. Its motivation remains valid: parts of the research feature path depend on ignored local artifacts or feature rebuilding and do not constitute a portable application scorer.

A defensible deployment sequence is:

1. Fix the target host, resource budget and application scorer contract; choose whether C2's cost is acceptable.
2. Export a versioned model bundle with artifact hashes, library versions, feature configuration and training lineage; remove dependencies on rebuilding raw research data at startup.
3. Review redistribution rights and learned vocabularies for identifying tokens before publishing artifacts.
4. Bind the new-JD ranking adapter to the intended frozen scorer and verify parity against its recorded outputs.
5. Run clean-environment and concurrent-user tests, then connect identity, durable storage, backup and mail delivery as separately configured services.
6. Release with explicit feature flags, rollback and monitoring for availability, latency, stale snapshots and evidence failures.

This sequence is a deployment recommendation grounded in the existing plan, not an implemented cloud architecture. C2's recorded CPU cost is approximately five minutes for a 50-pair shortlist on the project machine; the plan therefore considers L1-only service, precomputed demonstrations or a separate accelerator-backed endpoint. Those alternatives require a deployment decision and measurement. [Hosting plan](../../plan/2026-10-22-model-bundle-and-hosting/README.md)

### 5.4 Testing and Reproducibility Strategy

Tests emphasize invariants that can fail silently: ID-aligned scoring, document-disjoint splits, exact source slices, no oracle metadata, cache ownership, stale review rejection, retry idempotency, credential context and bounded worker responses. Normal tests use synthetic corpora, with explicit skips for unavailable optional models or data. Streamlit tests and retained browser walkthroughs exercise review switching, authoring, alerts, comparison and handoff.

The check runner selects registered reproduction checks through dependency analysis. Warm checks reuse cached model outputs; cold checks rebuild the relevant experiment, including training/inference where applicable. A passing warm check does not establish training reproducibility. D55 requires cold proof at specified release/evaluation boundaries, while D58 (research) requires owner confirmation before long runs. The current merged snapshot still has pending cold validation.

For this report, the full suite was run once and its complete output retained. No model retraining, new annotation, new benchmark scoring or deployment was performed as part of the report task.

| Validation record | Result | Interpretation |
|---|---|---|
| Committed integration record | 1,678 passed; 11 skipped; two baseline failures | Recorded result for the integration, not a completely green release |
| Report-time run: `logs/technical-report-pytest.log` | 1,693 passed; 11 skipped; three failures; 217.84 seconds | Executed while concurrent v2 work was changing; not an isolated release validation |
| Prior native v1 smoke | Installed GLiNER/RoBERTa exercised on invented JD/CV text with exact citation checks | Establishes local execution and selected cases, not domain accuracy |
| Prior CSV compatibility check | All 60 existing jobs retained ten candidates, order and scores | Operational preservation check, not a relevance-quality estimate |

Two failures concern the frozen baseline vocabulary digest and TF-IDF results. The third report-time failure is `test_model_cannot_omit_a_critical_qualification_even_when_both_owners_remain` in the new v2 composer tests: the expected rejection was not raised. This is material because retaining candidate coverage alone does not guarantee retention of a critical qualification. The report records the observed failure without modifying another implementation's code or declaring later fixes validated. [Integration record](../../plan/2026-10-24-chat3-integration/README.md), [v1 verification](../../plan/2026-10-24-grounded-qwen-comparisons/06-implementation-record.md)

## 6. Evaluation, Challenges, & Future Work

### 6.1 Evaluation Methodology

Two evaluation questions must remain separate:

- **Candidate fit against the frozen final set:** binary AUC/classification and human-grade ordering on the annotated A2 pairs.
- **Top-ten ordering against the chat reference:** recovery of a tie-aware judge top ten and nDCG from numerical chat scores on the scored shortlist/pool candidates.

The later chat-stage Recall@10 is not conventional recall of every qualified applicant in the 500-CV pool. Boundary ties receive fractional credit, and unjudged distractors prevent an exhaustive production-relevance interpretation. The C2 reference uses the latest `chat-1` plus `chat-2` scores; the earlier chat-3 test record used `chat-1` alone. Comparing those headline values directly would conflate model and reference changes.

The implementation uses 10,000 bootstrap resamples over JDs, percentile 95% confidence intervals and paired per-JD differences. Generic, targeted and total scopes are reported together. AUC excludes queries without both classes, so scoreable query counts can differ from the full 60. Human ordering and any LLM-related human diagnostics require the `llm_free` subset beside full results to expose dependence on tie-breaking labels.

### 6.2 Recorded Research Findings

Early text-only retrieval and trained-reranker experiments did not establish that every additional model layer was beneficial. The extracted stage-1 rungs missed their registered shortlist-recall bars, and the earlier cross-encoder training did not justify replacing weakly fine-tuned SBERT ordering. Later work changed supervision and task design: chat-3 supplies 4,800 training/selection pairs from 48 train-region JDs, split into 32 generic and 16 targeted queries, with JD-fold selection of L1/F12.

The following results concern the **later chat-reference ranking task**, not binary agreement with the frozen human-led final reference. Cells contain an estimate and 95% CI; query counts are generic **n=40**, targeted **n=20**, total **n=60**.

| System/metric | Generic | Targeted | Total |
|---|---|---|---|
| X2 Recall@10 | 0.4187 [0.3591, 0.4779] | 0.4138 [0.3173, 0.5055] | 0.4171 [0.3659, 0.4667] |
| L1 Recall@10 | 0.4843 [0.4353, 0.5340] | 0.4563 [0.3755, 0.5383] | 0.4750 [0.4321, 0.5165] |
| L1 + C2 Recall@10 | 0.4872 [0.4342, 0.5383] | 0.4670 [0.3858, 0.5472] | 0.4805 [0.4355, 0.5233] |
| L1 nDCG@10 | 0.8385 [0.7979, 0.8736] | 0.7829 [0.7135, 0.8470] | 0.8199 [0.7840, 0.8529] |
| L1 + C2 nDCG@10 | 0.8460 [0.8050, 0.8812] | 0.8034 [0.7405, 0.8599] | 0.8318 [0.7974, 0.8627] |
| Paired Recall@10 change, C2 versus L1 | +0.0029 [−0.0175, +0.0242] | +0.0107 [−0.0110, +0.0330] | +0.0055 [−0.0101, +0.0211] |
| Paired nDCG@10 change, C2 versus L1 | +0.0076 [+0.0007, +0.0143] | +0.0205 [+0.0070, +0.0357] | +0.0119 [+0.0053, +0.0186] |

Source: [`output/rerank/c2-validation-test.json`](../../output/rerank/c2-validation-test.json), reference A. Estimates are rounded for presentation; the record preserves full precision.

The registered C2 adoption criterion required a total Recall@10 improvement of at least **0.02**, with a confidence interval entirely above zero. The recorded paired change meets neither requirement. D64 nevertheless adopts the configuration because the owner accepted its higher point estimates and improved nDCG. The nDCG result supports better ordering under this chat reference; it does not establish the registered retrieval improvement or justify an unqualified superiority claim.

The same judge family supplies training targets and the main chat-stage reference. Document-disjoint training improves experimental separation but does not provide independent human validation. Supplemental diagnostics include exploratory AUC against the D41 adjudicated labels and exploratory pairwise ordering against one human annotator's grades. Both include full and `llm_free` views and do not establish an incremental C2 benefit. They do not supply an independent, representative human validation of the deployed workflow.

Reference uncertainty is also material. Human–human agreement in the frozen annotation report is:

| Scope | Judged pairs / JDs | Agreement, 95% CI | Cohen's kappa, 95% CI |
|---|---|---|---|
| Generic | 126 / 39 | 0.738 [0.638, 0.831] | 0.419 [0.265, 0.582] |
| Targeted | 200 / 20 | 0.630 [0.550, 0.715] | 0.292 [0.148, 0.431] |
| Total | 326 / 59 | 0.672 [0.607, 0.737] | 0.353 [0.249, 0.457] |

These are annotation-context measures, not system accuracy, a model ceiling or an acceptance target. [Annotation report](../../output/annotation/a2-final-report.json)

### 6.3 Extraction and Conversational Evaluation

GLiNER-medium campaigns demonstrate operational extraction, auditable coverage and source alignment. Greater mention counts than a smaller model cannot establish better precision or recall, particularly when model size and chunking strategy changed together. The observed human entity journal lacks reviewed held-out documents, and assisted training reviews cannot supply blind evaluation gold. A domain extraction-accuracy claim is therefore premature.

The v1 native smoke confirmed selected positive spans, no-answer behavior, qualifier retention and reuse of entity extraction. It also showed that RoBERTa may abstain even when a useful source excerpt exists. Exact source validation prevents fabricated quotations but does not guarantee that retrieval found the best passage or that the answer satisfies the recruiter's question.

Synthetic unit tests and browser walkthroughs establish controlled workflow behavior. They do not establish useful free-form comparison, robust recruiter task completion or production response-time distributions. In particular, candidate “better fit” explanations require explicit job criteria and checked evidence relationships; they cannot be inferred from an ordering score alone.

### 6.4 Limitations and Technical Debt

| Area | Limitation | Consequence |
|---|---|---|
| Research validity | Post hoc C2 adoption and evolving references | Claims must identify the task, reference and selection exception |
| Generalization | One regional IT market, compact profiles and constructed pools | Results do not establish performance across professions, markets or full-length resumes |
| Label quality | Mixed-provenance reference; A1 label unreliability | Supervision and human agreement require explicit caveats |
| Extraction | Incomplete held-out entity gold; heuristic contextual interpretation | Confidence and complete coverage cannot be presented as validated competence extraction |
| Conversation | V1 cannot reliably synthesize comparative reasons; v2 is unfinished | A source-linked table is not equivalent to a useful comparative conclusion |
| Integration | Research winner, CSV reviews and upload scorer differ | UI demonstrations cannot be attributed automatically to the reported research system |
| Reproducibility | Two baseline golden failures and pending merged cold checks | A committed snapshot is not a fully reproduced release |
| Runtime | Large optional artifacts, CPU inference cost and platform-specific environments | Hosted performance and clean installation remain deployment risks |
| Privacy and governance | Regex redaction, learned vocabulary leakage, licensing and local identity | Public model export and enterprise use need additional review and controls |
| Documentation | Some README, ownership and progress records predate current code/local artifacts | Historical status must be resolved through dated decisions, execution records and manifests |

The source catalog documents a historically acquired working set, but the current checkout does not contain every optional corpus. Missing-data skips and reconstructibility instructions should be preserved rather than treating historical acquisition as current deployment readiness. No demographic fairness conclusion follows from role/experience/language-level slices alone.

### 6.5 Prioritized Future Roadmap

| Priority | Work item | Required evidence before promotion |
|---|---|---|
| 1 | Resolve baseline reproduction debt and the observed v2 qualification-omission failure | Named regressions, a stable full-suite result and approved applicable reproduction checks |
| 2 | Complete v2 dialogue state and evidence-based comparison | Remembered participants/criteria, source reopening, qualified conclusions and safe clarification across complete conversations |
| 3 | Measure conversational usefulness independently | Frozen scenario-family splits, blinded reviewer assessment, uncertainty and separate model/fallback success rates |
| 4 | Bind and package the intended ranking system | Adapter contract tests, artifact-only loading, provenance, clean-host parity and realistic latency/memory measurements |
| 5 | Establish held-out entity/QA quality | Blind entity dev/test reviews, span/no-answer metrics and explicit qualification-retention evaluation |
| 6 | Refine models only after failure diagnosis | Retrieval errors separated from reader/state/composition errors; one measured intervention and resource plan at a time |
| 7 | Add enterprise integrations | Approved SSO, mail, storage/retention, audit and backup requirements with end-to-end recovery tests |
| 8 | Validate business and fairness outcomes | Representative pilot, measured recruiter effort, broader independent judgments and an appropriate fairness study |

Qwen or RoBERTa fine-tuning should be conditional on the relevant failure mode. Training a reader cannot repair lost dialogue state or missing passages; training a generator cannot supply evidence absent from the CV. V2's application implementation is approved, while additional training, capacity experiments and resource-intensive work retain separate gates.

### 6.6 Source and Status Register

| Material | Role in this report |
|---|---|
| [Proposal v3](../proposal/PLP_Group2_Intelligent_Candidate_Screening_Proposal-v3.md), [v2](../proposal/PLP_Group2_Intelligent_Candidate_Screening_Proposal-v2.md) | Problem, objectives, scope, original success targets and timeline |
| [Decision log](../decisions.md), [AGENTS rules](../../AGENTS.md) | Current methodological constraints, decision provenance and verification policy |
| [Data catalog](../data/data-catalog.md), [profile metrics](../data/profile-metrics.json), [source cards](../data/cards/) | Measured corpus inventory, provenance, licensing and limitations |
| [Final-reference record](../../output/annotation/a2-final-report.json) | Label hierarchy, annotation provenance and agreement intervals |
| [Extraction/rerank plan](../../plan/2026-10-06-extraction-and-rerank/README.md), [chat-3 plan](../../plan/2026-10-21-chat3-holdout/README.md) | Research progression and training/test boundary changes |
| [C2 plan](../../plan/2026-10-22-c2-validation/README.md), [test record](../../output/rerank/c2-validation-test.json) | Selected fusion, measured results and missed adoption criterion |
| [V1 plan](../../plan/2026-10-23-hiring-desk-v1/README.md), [extractive implementation](../../plan/2026-10-24-grounded-qwen-comparisons/06-implementation-record.md) | Delivered portal and default chat behavior |
| [V2 plan](../../plan/2026-10-24-hiring-desk-v2/README.md) | Newly approved conversation scope; implementation still in progress during review |
| [Entity campaign](../../plan/2026-10-16-entity-gold-annotation/README.md), [tokenizer amendment](../../plan/2026-10-17-character-boundary-tokenization/README.md) | Separate human-span governance and adaptation policy |
| Local entity manifest/journal and `medium-character-ten-v2/finetune-manifest.json` under ignored `data/processed/entity-annotation/` | Aggregate local progress observations; not independently recoverable from checked-in prose alone |
| [Hosting plan](../../plan/2026-10-22-model-bundle-and-hosting/README.md) | Proposed portable deployment, explicitly on hold |
| [Integration record](../../plan/2026-10-24-chat3-integration/README.md), `logs/technical-report-pytest.log` | Recorded integration debt and fresh report-time test result |

The report distinguishes committed implementation, observed local artifacts, concurrent uncommitted development and proposed work. It does not certify production readiness, independent recruitment accuracy or completion of outstanding reproduction checks.
