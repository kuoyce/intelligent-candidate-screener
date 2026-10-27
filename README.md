# Intelligent Candidate Screening System

NUS ISS PLP Practice Module, Group 2. A resume–job-description screening system: ingest
resumes in their original formats, extract structured evidence, and rank candidates against a
job description with the evidence shown back to the recruiter.

**Current phase: the top-down reset** (`plan/2026-09-26-top-down-reset/`). This means:
- one final Fit / No Fit annotation set for the in-domain A2 corpus;
- the matching ladder (rules → TF-IDF/BM25 → SBERT → cross-encoder) scored against it;
- then extraction, the evidence layer and the recruiter UI.

Decisions are logged in [`docs/decisions.md`](docs/decisions.md).

## Results (27 Sep 2026)

Against one final Fit / No Fit panel label per in-domain pair (D41), with per-JD AUC and
95% CIs over 59 JDs. Full detail: [`plan/2026-09-26-top-down-reset/09-findings.md`](plan/2026-09-26-top-down-reset/09-findings.md).

| System | Per-JD AUC | Verdict under the registered rule |
|---|---|---|
| random | 0.556 | — |
| metadata rules (role keyword + experience) | 0.778 | — |
| TF-IDF | 0.797 | — |
| **metadata + TF-IDF** | **0.849** [0.802, 0.891] | **selected (D48)**: nothing richer beats it |
| SBERT, fine-tuned on weak in-domain pairs | 0.854 [0.817, 0.890] | **selected for text-only input**: +0.057 over TF-IDF, justified |
| cross-encoder (A1-trained / weak-trained) | 0.488 / 0.661 | harmful |
| learned combination of all text models | 0.828 | not resolved |

- At 500 candidates per role, metadata + TF-IDF keeps 92.9% of the fits in a
  50-candidate shortlist, in 0.84 s.
- Re-ranking that shortlist with a cross-encoder cuts Recall@10 by a quarter to a third.
- Extraction splits by role (D49): a MiniLM tagger for job postings (SkillSpan strict
  F1 0.523 vs 0.237) and a rules matcher for resumes.

## Layout

```
src/candidate_screener/     project code
  config.py                 canonical data paths
  data/                     source registry, fetch, verify, profile, splits, pools
  baselines/                the Stage 1 classical baseline (TF-IDF, BM25) and its entrypoint
  evaluation/               retrieval and classification metrics, with their reporting guards
tests/                      pytest — runs without data/ except the golden check
notebooks/                  01-06 acquisition EDA, 07 baseline, 08 extraction readiness
output/baselines/           committed model record — figures and config, never bytes
  data/                     source registry, fetch, verify, profile, splits, pools, weak pairs
  baselines/                the frozen Phase 4 classical baseline (TF-IDF, BM25) and its record
  evaluation/               final_set (the reference), indomain (the protocol), retrieval_pilot,
                            a1_secondary, latency, metrics with their reporting guards
  matching/                 the ladder: rules, lexical, SBERT, cross-encoders, hybrids
  extraction/               skill vocabulary, rules matcher, MiniLM NER, normalisation
  inference/                the reported system (D63/D64) as an API for apps: `Screener`
  evidence/, pipeline.py    requirements, verbatim evidence, end-to-end `screen()` (deprecated draft)
  app/                      frozen screening snapshots, grounded QA + Streamlit transport
  llm/                      local extractive QA and historical Qwen service; no annotation or training
  annotation/               the frozen annotation instrument (D42)
tests/                      pytest — runs without data/ (data-dependent checks skip visibly)
notebooks/                  01-07 acquisition EDA and baseline; 08-11 the reset's findings
output/                     committed records — figures and config, never model bytes
docs/
  data/                     data catalog, per-source dataset cards, manifests
  decisions.md              the decision log (D1–D49)
  proposal/                 the module proposal this project implements
plan/                       dated plans and their implementation records
data/                       git-ignored working set (except data/vocab/, D15) — see data/README.md
```

## Setup

```bash
uv sync                     # Python 3.12, the classical stack + dev tools
uv sync --all-groups        # + ml (torch CPU, sentence-transformers) + app (streamlit)
```

## Run with Docker (any OS)

The Hiring Desk also runs in Docker on Windows, macOS or Linux, with no Python, uv or model
download on the host (D68). One image holds the code, the three environments and the pinned
pretrained weights; the data folder `data/datapack/` holds the project's own files plus the
accounts and reviews, and is passed on privately. Native development is unchanged.

```bash
mkdir logs                                             # put the data folder you were given at data/datapack/ first
docker compose build                                   # cold: ~3.5-5.5 min, 4.6 GB downloaded (27 Oct 2026)
docker compose --profile tools run --rm tools verify   # checks data/datapack/
docker compose run --rm app python -m candidate_screener.app.portal_auth --create-local-user reviewer --department benchmark
docker compose up -d                                   # http://localhost:8503
```

Requirements per OS, the data folder, accounts, backups, updates and troubleshooting:
[`docs/docker.md`](docs/docker.md).

## Screen candidates (the reset's pipeline)

```bash
uv run python -m candidate_screener.evaluation.final_set --build     # the reference (D41)
uv run python -m candidate_screener.evaluation.indomain --seed 0     # the ladder vs the final set
uv run python -m candidate_screener.evaluation.retrieval_pilot       # 500-candidate pools
uv run python -m candidate_screener.evaluation.a1_secondary          # A1, secondary (A13 caveat)
uv run python -m candidate_screener.extraction.evaluate              # rules vs NER
uv run streamlit run src/candidate_screener/app/streamlit_app.py     # the recruiter UI
```

Each record command takes `--check` to rebuild in memory and diff against the committed file.
Checks come in two tiers (D55):
- **cold** (`--check`) retrains and re-infers, and is the reproducibility proof;
- **warm** (`--check --warm`, where a module supports it) reuses cached model outputs.

One runner drives them all. It prints one line per check and keeps the full output in
`logs/checks/`:

```bash
uv run python -m candidate_screener.checks                 # warm, only checks your uncommitted changes affect
uv run python -m candidate_screener.checks --warm --all    # every live record (~15 min)
uv run python -m candidate_screener.checks --cold --all    # every record from scratch (~50 min)
uv run python -m candidate_screener.checks --list          # the registry, with measured times
```

## Using the system from an app

`candidate_screener.inference` serves the reported system to an app (plan
`2026-10-23-inference-api`). It imports no Streamlit and fits nothing: every part (role model,
vocabulary, concept frequencies, TF-IDF space, SBERT weak, the frozen L1) is loaded, so a new JD
or an uploaded CV is inference only. It runs locally on a machine with `data/`; once per machine:

```bash
uv sync --all-groups
uv run python -m candidate_screener.inference --cache-parts   # saved concept df + TF-IDF space (~40 s)
uv run python -m candidate_screener.inference --demo <final-set jd_id>   # smoke: a top 10, no CV text
```

```python
from candidate_screener import inference

s = inference.Screener.load()                        # ~4 s; wrap in st.cache_resource
res = s.rank(jd_text, cvs, jd_title="QA Engineer")   # cvs: cv_id + cv_text only (D53)
for r in res.top(10):                                # what the UI shows
    print(r.rank, r.cv_id, r.score)
why = s.explain(res, res.top(1)[0].cv_id)            # for the recruiter: reasons, skills, years
```

- **System.** D63 by default: the frozen L1 (`output/rerank/l1-chat3-model.json`) on F12 ranks
  every CV passed in. `rerank=True` is D64: L1's top 50 reordered by min-max L1 + 0.25 × min-max
  C2 (Qwen3-Reranker-0.6B, zero-shot). C2 costs ~2.5-7 s of CPU per pair, so up to a few minutes
  per JD; `progress(done, total)` reports it. `system="X2"` gives the baseline.
- **Uploads only.** CVs come from the caller; there is no CV store or search.
- **Scaling (`scaling="background"`, the default).** `tfidf_scaled` is min-max scaled over the
  CVs scored together, and L1 was trained on 500-CV pools. The CVs passed in are therefore padded
  to 500 with a fixed sample of A2 train-region CVs that are scored but never returned. Scaling
  over the uploads alone (`scaling="pool"`) reorders small pools: at 10 CVs the top candidate
  differs about 1 time in 7 (`output/inference/pool-scaling.json`). The first background call
  takes ~18 s (the 500 background CVs are embedded once and cached), later calls ~1 s.
- **`reference_date`.** "present" in a CV's date range: today by default (uploads); A2 documents
  use 2023-09-01 (`Screener.a2_pool` returns it).
- **Output.** Frozen dataclasses (`Screening`, `Ranked`, `JDProfile`, `CVProfile`), each with
  `as_dict()` (JSON-safe); `Screening.to_frame()`. No CV text is returned: anything an app
  displays from a CV passes `annotation.redact`.
- **Demo data.** `Screener.a2_pool(jd_id)` gives a final-set JD's text, title and 500-CV pool;
  pass `jd_id=` to `rank` so the C2 scores already cached are reused.
- **Check.** `inference --check` drives the API against the committed records (~12 min).

## Acquire the data (~676 MB)

```bash
uv run python -m candidate_screener.data.fetch   --all   # download every automatable source
uv run python -m candidate_screener.data.verify  --all   # files, row counts, SHA-256 digests
uv run python -m candidate_screener.data.profile --all   # reproduce the catalog's figures
```

Two sources need a human: the Kaggle PDF corpus (account + API token) and ESCO
(free registration). `fetch` prints the exact steps for whichever is missing.
`data/README.md` has the full guide.

```bash
uv run jupyter lab notebooks/       # the EDA
```

For preliminary entity-extraction work, start with
[`08-entity-extraction-readiness.ipynb`](notebooks/08-entity-extraction-readiness.ipynb)
and its [execution plan](plan/2026-09-13-entity-extraction/README.md).
The notebook checks what is available in this checkout, reports metadata without displaying
real resumes, and demonstrates the existing keyword helper on synthetic text. Missing NER
sources are reported explicitly; the historical acquisition record does not imply that every
source is installed locally. The plan covers skills, titles, qualifications and experience
evidence, with supervised extraction and theme construction as subsequent work.

Continue with [`09-resume-evidence-and-themes.ipynb`](notebooks/09-resume-evidence-and-themes.ipynb)
and its [execution plan](plan/2026-09-19-resume-themes/README.md) to explore the proposed
evidence records, activity-backed themes and requirement comparisons. Its worked examples
are manually specified synthetic evidence, not extractor predictions or accuracy results.

For executable extraction on local Djinni data, use these notebooks in reading order:

- [`10-entity-extraction-inputs.ipynb`](notebooks/10-entity-extraction-inputs.ipynb): validate inputs and sample only the committed training region.
- [`11-rule-based-entity-extraction.ipynb`](notebooks/11-rule-based-entity-extraction.ipynb): extract skills/tools, titles and qualifications with exact field-relative evidence spans.
- [`12-entity-extraction-evaluation.ipynb`](notebooks/12-entity-extraction-evaluation.ipynb): check offsets, reproducibility and synthetic regressions, and report the status of labelled evaluation.

The [approved implementation plan](plan/2026-10-05-entity-extraction-notebooks/README.md)
uses 1,000 training CVs and 1,000 training JDs by default. Its versioned seed vocabulary
and context rules provide a limited offline baseline; document coverage is descriptive,
not extraction accuracy. DataTurks/SkillSpan gold evaluation remains blocked when those
sources are absent. Real document text and entity surfaces stay in git-ignored local
artifacts; saved notebook outputs contain aggregates and synthetic examples only.

The same extraction can run without Jupyter:

```bash
uv run python -m candidate_screener.extraction.run --sample-size 1000 --seed 0
uv run python -m candidate_screener.extraction.run --sample-size 1000 --seed 0 --check
```

`--check` verifies source/code provenance, local artifact hashes, exact offsets and
reproduced predictions; it does not measure extraction accuracy.

Notebook 11 now includes a dictionary of the 15 stored entity fields and a five-row
redacted preview of actual rule-extracted mentions. Continue with
[`13-transformer-entity-extraction.ipynb`](notebooks/13-transformer-entity-extraction.ipynb)
for locally run GLiNER predictions and a comparison with rules on the same training
documents. GLiNER extracts `KNOWLEDGE`, `SKILL`, `TITLE`, `DEGREE` and
`EXPERIENCE_YEARS`; the rules comparator supports the original four labels only.
Experience spans preserve stated durations, ranges and minimums in CV claims or
JD requirements. They do not infer tenure from dates or determine job fit.
The saved five-label run has its own aggregate record at
`output/extraction/transformer-experience-years-metrics.json`; the original
four-label record and historical notebook 14 results are retained.
After combining the extraction branches, the seed-rule inspection record is
preserved as `output/extraction/rule-inspection-metrics.json`.
`output/extraction/extraction-metrics.json` describes the separately evaluated
skill rules/NER campaign; the two records have distinct schemas and populations.
Neural scores are uncalibrated; context flags still use a separate heuristic, and
unlabelled coverage/disagreement does not establish accuracy or superiority.

The transformer stack uses a separately locked uv environment to support Intel macOS
without downgrading the project's existing dependencies:

```bash
uv sync --project runtime/transformer
uv run python -m candidate_screener.extraction.model_assets
uv run python -m candidate_screener.extraction.transformer_run --sample-size 100
uv run python -m candidate_screener.extraction.transformer_run --sample-size 100 --check
```

Inference is local, with five-per-type smoke testing followed by 100 CVs and 100 JDs
and a 30-minute inference budget. Checkpoint and tokenizer revisions/hashes, runtime
versions, original offsets and field completion are recorded. Real neural evidence
stays in `data/processed/extraction/transformer-runs/<run_id>/entities.parquet`;
notebook 13 prints the actual artifact directory and uses synthetic neural previews.
See the [transformer plan](plan/2026-10-06-transformer-entity-extraction/README.md).

For charts and real examples from the generated outputs, open
[`14-entity-extraction-eda.ipynb`](notebooks/14-entity-extraction-eda.ipynb).
It compares GLiNER and rules on the same 100 CVs and 100 JDs, explores mention
frequencies, per-document counts, scores, field/context patterns and disagreements,
and exports figures to `output/extraction/eda/`. Model scores and threshold-retention
curves describe predictions, not extraction accuracy. This notebook uses existing
artifacts and regenerates only the matching rule predictions; it performs no
additional neural inference or training.

For validation against the original CV/JD fields, open
[`27-entity-extraction-source-validation.ipynb`](notebooks/27-entity-extraction-source-validation.ipynb).
It audits historical run manifests, source hashes, exact entity slices and field
coverage, and displays bounded source context beside saved labels. Source matches
establish traceability; label accuracy and missing-entity recall require compatible
gold annotations. The [validation plan](plan/2026-10-14-entity-extraction-validation/README.md)
records this distinction and the local gold-data availability.

The approved full-fit GLiNER campaign covers every pair in
`data/processed/fit/train.parquet` and `test.parquet`, extracting each unique CV
and JD once with five fixed entity categories. Its reporting notebook is
[`28-fit-gliner-comparison.ipynb`](notebooks/28-fit-gliner-comparison.ipynb).
The [campaign record](plan/2026-10-14-fit-pair-gliner-comparison/01-implementation.md)
tracks execution and validation status. Fit-conditioned distributions are
exploratory; A1 labels do not reproduce (A13), and they are not entity-span gold.

The separate [medium-v2.5 experiment](plan/2026-10-14-gliner-medium-sentence-chunking/README.md)
uses `gliner-community/gliner_medium-v2.5` with sentence-aware, prompt-budgeted
chunks and overlapping word windows for oversized sentences. Original text and
entity offsets are preserved. It has its own runtime and campaign checkpoints;
the small-model default remains unchanged. Notebook
[`29-gliner-medium-sentence-comparison.ipynb`](notebooks/29-gliner-medium-sentence-comparison.ipynb)
compares operational results after inference completes. Model and chunk strategy
change together; mention counts do not establish extraction accuracy.

The separate [human entity annotation UI](plan/2026-10-16-entity-gold-annotation/README.md)
supports selecting text spans, four category shortcuts, editing, undo and explicit
whole-document review. Try synthetic documents locally:

```bash
uv run python -m candidate_screener.entity_annotation.ui --demo --annotator reviewer
```

Open the printed `http://127.0.0.1:8766/` URL. Demo annotations are synthetic and
cannot measure the model. Real source/categories/batch/annotators must be confirmed
before creating a campaign under D56. See the
[boundary guide](plan/2026-10-16-entity-gold-annotation/annotation-guide.md) and
[evaluation/training protocol](plan/2026-10-16-entity-gold-annotation/training-protocol.md).
The frozen Fit annotation UI and labels are unchanged.

The approved [ten-document GLiNER adaptation pilot](plan/2026-10-16-entity-ten-example-finetune/README.md)
uses saved human training spans. It validates exact tokenizer boundaries before
training and preserves original reviews. Model suggestions, once available, are
explicitly loaded only on untouched primary-annotator training documents; exports
retain assistance provenance. Independent annotation and dev/test stay blank.
The approved [character-boundary tokenizer amendment](plan/2026-10-17-character-boundary-tokenization/README.md)
preserves every saved human boundary: training-derived lexical cuts and explicit
whitespace units align all130spans in the latest ten-document snapshot. The frozen
policy must accompany training, checkpoint reload and inference. This alignment
check establishes representability; it does not establish extraction accuracy.

The [French interview prototype](notebooks/16-french-interview-translation.ipynb)
demonstrates audio decoding, French speech transcription and Marian French-to-English
text translation. Full transcripts and media remain local; the notebook reports
aggregate results and separately reviewed safe excerpts. This is an ingestion
experiment, not an adopted ranking model or a three-class fit classifier. The
[implementation plan](plan/2026-10-11-multimodal-entity-pipeline/README.md) describes
text-only pipeline changes, five-label GLiNER evidence, applicant transcript review
and the proposed chat upload flow. Live pipeline/UI integration awaits that plan's
approval. Launch notebooks from the installed project environment with
`uv run jupyter lab` after `uv sync`.

## Streamlit recruiter chatbot

The primary app opens **Hiring Desk**, with typed HR chat and dropdown review menus:

```bash
uv run streamlit run src/candidate_screener/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8503
```

Hiring Desk v1 opens the supplied CSV-backed jobs alongside published operational
JDs. The production app has no workspace/demo selector. Provision an account
in your own terminal and enter a private access token of at least24 characters at
the hidden prompt; no token is printed or stored in conversation history:

```bash
uv run python -m candidate_screener.app.portal_auth --create-local-user reviewer --department benchmark
```

Then sign in. Local token authentication is
not company SSO. Reviewer credentials default to 40 days (`--hours 960`); active
sign-ins last until account expiry, revocation or logout, without an hourly timeout.
Browser reconnects can require signing in again. Use **Log out** at the bottom of
the left sidebar to end the current sign-in.
The app reads `a2-cv-extraction.csv`, `a2-jd-extraction.csv` and
`a2-x2-scores.csv` under `data/processed/phase2-rerank-output/`, without reranking
or reading `final_label`. It supports top10 review, source-text dialogs,
five-category mention comparison, saved sessions and reports/ZIP packages with
all ten ranked CV text files and profiles. Historical eight-CV packages retain
their schema. `HR_PORTAL_ACCOUNTS` and `HR_PORTAL_DATA` provide local configuration.

**Review jobs** shows selectable JD cards, Previous/Next controls, title search,
a source-grounded summary and the full redacted JD. Review actions and chat follow
below. Switch cards at any point to resume that job's saved review; conversations
and candidate selections stay with their job. CSV rankings are never silently
recomputed and remain usable when live modeling is unavailable. Full JD/CV text
continues to come from the existing authorized source-text cache; missing text is
shown as unavailable rather than reconstructed from extraction columns.

**Create new JD** guides title, department, responsibilities, required/preferred
skills and knowledge, experience and qualifications. Each accepted answer is
saved locally. Resume/edit a draft, confirm the preview, then publish and request
candidate ranking. Operational JDs and ranking snapshots live separately from
benchmark labels and training data in the local workspace database.

New JDs rank through the inference API (`inference.Screener`, plan
[2026-10-24-hiring-desk-inference-api](plan/2026-10-24-hiring-desk-inference-api/README.md)).
Once per machine, `uv run python -m candidate_screener.inference --cache-parts`; then launch with:

```bash
uv run streamlit run src/candidate_screener/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8503
```

The entry point defaults `HR_PORTAL_RANKING_ADAPTER=candidate_screener.app.portal_inference:adapter`
and `HR_PORTAL_CANDIDATE_POOL=a2-final-cvs`. An exported value overrides either; export one as
empty (`HR_PORTAL_RANKING_ADAPTER=`) to run with live ranking unconfigured.

The pool is the 600 A2 final-set CVs (reference date 2023-09-01); no labels are read. **Rank
candidates** runs D63 and usually returns within the click (the first request also loads the model,
about half a minute). Tick **Refine the top 50 with the reranker** for D64: it runs in the
background for a few minutes (210 s for a new JD on 24 Oct 2026; seconds once its C2 scores are
cached, peak RSS 4.6 GB while the reranker loads); use **Refresh ranking status** to fetch it. Each
ranking stores the model's `explain()` reasons with its snapshot, but the review does not show them:
they compare skills and years, which the score barely weighs (L1 is driven by role-family match and
text similarity), so they would contradict the order. The model reads the JD text only: skills
outside its vocabulary (ESCO ICT + data_jobs) are invisible to it, required and preferred count
alike, and it ignores experience requirements above 20 years. Failures show in the app as a generic message; the
cause is in `logs/hiring-desk-adapter.log`. Any other adapter implementing the
[contract](plan/2026-10-23-hiring-desk-v1/05-live-ranking-contract.md) (`adapter(operation,
payload) -> dict`, schema in `app/portal_ranking.py`) can be configured instead. Without an
adapter, new JDs stay saved with an explicit unavailable state; existing CSV reviews still work. No
model is trained at startup.

**Alerts** records JD publication, ranking ready/failed and local email
captured/failed events. Read/unread controls and Open job preserve access checks.
**Send to hiring department** demonstrates the flow using Tech/Business example
addresses: preview the ten-applicant package, then choose **Send to demo outbox**.
The resulting private `.eml` file can be downloaded; it is captured locally, not
delivered to a mailbox. Captures live under `data/processed/hr-portal/mail-captures/`.
Original PDFs, company SSO, external applicant events and production mail delivery
remain unconnected. Additive SQLite migrations preserve old sessions/packages;
retain a private backup of the workspace directory before moving deployments.

After selecting a role, use **Ask Hiring Desk** to ask about its requirements,
compare candidates, or inspect cited CV evidence. Dropdown candidate/category
selections establish the context for the next question; explicit candidates in
your question take precedence. Saved reviews retain their bounded conversation.
All review dropdowns remain available, including CV previews and reports.
Failed submissions offer **Retry question** and **Discard question**. A pending
question is discarded with an explanation if its review or candidate context changes.

Experimental **Hiring Desk v3** (`HR_PORTAL_CHAT_VERSION=v3`) adds an inference-only
plain-prose Qwen path, bounded conversation history, explicit comparison tables,
expandable tables for “why” explanations, and supporting sources on demand.
“Find jobs by requirement” searches authorized JD text without requiring rankings;
unranked published jobs also support standalone JD questions. Cross-JD search
does not change the selected review. The existing frozen models are reused.
Unsupported generated wording falls back to checked statements; this conservative
validator does not establish general semantic entailment. V3 remains experimental
pending the [quality gates](plan/2026-10-25-hiring-desk-v3/02-implementation-and-evaluation.md).
Its system prompt is `src/candidate_screener/llm/hr_system_prompt_v3.txt`.
The optional Qwen3.5-4B CPU trial uses a separate pinned GGUF runtime:
`uv run python -m candidate_screener.llm.gguf_assets` provisions the assets, then
`uv run python -m candidate_screener.llm.gguf_server --manifest data/processed/chatbot/qwen35/manifest.json --launch-backend`
serves the model on loopback8001. Set
`SCREENER_QWEN_MANIFEST=data/processed/chatbot/qwen35/manifest.json` together with
`HR_PORTAL_CHAT_VERSION=v3` when starting the portal to select it. A missing selected
manifest is unavailable and never silently selects the old model. Unset the manifest
variable to return to the existing Qwen service on8000. No Torch upgrade is needed.
On the Intel CPU, the Qwen3.5 trial allows up to60seconds per turn; extraction and
the existing0.5B model retain their30-second limits.
The entry point leaves `HR_PORTAL_CHAT_VERSION` at `v1`; export `v3` to try it. **When v3 is completed (its quality gates pass), make `v3` the default in `streamlit_app.py` and update this section and AGENTS.md.** Set the version back to `v2` or `v1` to roll back; no running service is migrated
automatically. An invented-source development smoke is available as
`uv run python -m candidate_screener.app.conversation_v3_smoke --real-model --output logs/v3-smoke.json`.

An opt-in **Hiring Desk v2** adds comparative conclusions, remembered comparison
participants/criteria, and source-backed follow-ups:

```bash
HR_PORTAL_CHAT_VERSION=v2 uv run streamlit run src/candidate_screener/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8503
```

Try “Compare candidates 1 and 2 for this role”, “Who has more Python experience?”,
“What about their qualifications?”, and “Where does it say that?”. Existing local
Qwen may choose/order checked comparison points; the application writes factual
clauses. Without Qwen, the same evidence supports an explicitly labelled
application comparison. No weights are downloaded or trained on startup.
`SCREENER_QWEN_DISABLED=true` disables composition; `HR_PORTAL_CHAT_VERSION=v1`
restores the default experience. The [v2 record](plan/2026-10-24-hiring-desk-v2/04-implementation-record.md)
tracks validation and remaining independent quality review; this is an opt-in
implementation, not a claim that the proposed human quality gates have passed.

V2 also accepts “Why is 1 better than 2?” and criterion-specific variants such as
“Why is 1 better than 2 on Python experience?”. The answer explains checked
JD-linked evidence with the comparison table, preserves counterevidence and
corrects an unsupported premise. Missing CV evidence is never treated as proof
of a missing skill. Short source labels link the reasons to their source controls.
The app uses deep-red controls, near-black text and a predominantly white canvas.

The current v1 path uses the installed `deepset/roberta-base-squad2` to answer
factual JD/CV questions with exact start/end spans. Answers retain source context
and expandable citations. Unsupported facts remain insufficient evidence;
missing or failed models show their availability and labelled source excerpts.
The assistant cannot change ranks or execute report, package or delivery actions.

Both **Compare candidates** and chat requests such as **Compare 1 and 2** use an
application-built table of JD requirements and candidate evidence. GLiNER medium
extracts technology, skills, titles, degrees and experience mentions in the
background and caches them privately by source/model/configuration. Use
**Refresh extraction** while work is queued. General comparison does not wait for
QA inference; focused questions such as **Compare 1 and 2 on Python** extract
answers independently from each requested document. Entity mentions do not
establish proficiency, completed credentials or skill-specific years of experience.

This path reuses the existing isolated `runtime/gliner-medium` and
`runtime/transformer` environments and pinned local checkpoints. Models load on
demand; missing assets never trigger downloads or training at startup. Primary
v1 answers no longer invoke Qwen. Earlier saved Qwen answers retain their recorded
provenance. The latest scope and verification are recorded in the
[extractive implementation plan](plan/2026-10-24-grounded-qwen-comparisons/05-extractive-implementation.md).

The earlier free-form/upload prototype is preserved separately:

```bash
uv run streamlit run src/candidate_screener/app/legacy_streamlit_app.py
```

The historical workshop setup below describes that prototype. Current portal
status is in the [extractive implementation record](plan/2026-10-24-grounded-qwen-comparisons/06-implementation-record.md).

The [19 October conversation-design plan](src/llm-plan/2026-10-19-hr-conversation-design/README.md)
defines the authenticated HR portal workflow, notifications,
stored sessions, top10 review and confirmed top8 department handoff. It preserves
the earlier implementation record and distinguishes planned integrations from
the current local prototype.

The earlier [HR conversational RAG upgrade](plan/2026-10-18-hr-conversational-rag/README.md)
introduced question-specific CV/JD passage retrieval, stable candidate follow-ups,
explicit filter previews and cited source mentions. The v1 path above supersedes
its generator selection;
synthetic development measurements have not established production QA accuracy.
The extractive baseline is provisioned explicitly, never downloaded at UI startup:

```bash
PYTHONPATH=src uv run --project runtime/transformer --no-sync python -m candidate_screener.llm.qa_assets
uv run streamlit run src/candidate_screener/app/legacy_streamlit_app.py
```

Question examples: “Who best suits this role?”, “Compare 1 and 2”, “What
certifications does applicant 1 hold?”, and “What about applicant 2?”. Source
mentions retain context; a mention does not establish an active qualification,
and missing evidence does not prove a qualification is absent. The QA feasibility
corpus is authored synthetic data with a frozen family-disjoint manifest under
`output/chatbot-rag/`; it does not use the human GLiNER labels or frozen Fit labels.

The [approved chatbot plan](plan/2026-10-12-streamlit-llm-chatbot/README.md) adds
session-only recruiter QA, immutable screening snapshots, frozen cross-encoder
loading and an optional Qwen service. Existing written CV and reviewed French
interview ingestion remains available.

The user-supplied workshop selects `Qwen/Qwen2.5-0.5B-Instruct` for local
generation. Provision it once and start the loopback service in a separate terminal:

```bash
uv run python -m candidate_screener.llm.qwen_assets
uv sync --project runtime/transformer
PYTHONPATH=src uv run --project runtime/transformer --no-sync python -m candidate_screener.llm.local_server --manifest data/processed/chatbot/qwen/manifest.json
```

For the historical prototype, start Streamlit as below. The provisioned manifest supplies local Qwen
settings automatically. The isolated service uses CPU, four threads and greedy
decoding; the app counts tokens through `/v1/tokenize`, so its environment needs
no Torch for Qwen. Loading is offline; only provisioning downloads public files.
`SCREENER_QWEN_DISABLED=true` forces templates.

For the requested backend in `rerank/models.py`, prepare the recorded selected R2:

```bash
uv run python -m candidate_screener.app.rerank_assets
```

Streamlit and the local service discover the exported manifest automatically;
restart the service after preparation. The selected directory
`data/processed/rerank/r2-60412b3f1f1e` is absent in this checkout; restore its
complete trained checkpoint and tokenizer before preparation. No training is
started. R2 has no validated binary threshold, so classification remains
unavailable. App use does not change the research selection of SBERT weak ordering.

```bash
uv sync --group app
uv run streamlit run src/candidate_screener/app/legacy_streamlit_app.py
uv run python -m candidate_screener.app.chat_smoke
```

With no model configuration, select **Illustration TF-IDF** and **Screen
demonstration** to use invented applicants and grounded template answers. The
illustration fits unsupervised IDF on submitted text; it is not an adopted benchmark
model. Ask `top 5`, `explain 1`, `compare 1 and 2`, `entities 1`, or `source 1`.
With Qwen, other phrasing is mapped to supported screening operations and the model
selects evidence. Backend facts, scores and quotes are rendered from validated
references. Generated narrative is excluded from the MVP. Invalid responses use
templates.

Provision an approved local Transformers sequence-classification checkpoint and
its tokenizer in the same directory. Native loading requires Torch/Transformers
in the app environment (`uv sync --group app --group ml` on a supported host).
On Intel macOS, the local service also supplies frozen CE inference through
`/v1/rerank`; it discovers the verified exported manifest at startup.
`SCREENER_CE_SERVICE_URL` selects this loopback transport, so the app needs no
main-environment Torch for R2. Set it to an empty string for native loading, or
pass `--reranker-manifest` to the service for another frozen manifest. App startup
performs no weights download or training.

Set `SCREENER_CE_MANIFEST` to a local JSON file with this schema:

```json
{
  "name": "Approved cross-encoder",
  "checkpoint": "checkpoint-directory",
  "tokenizer": "checkpoint-directory",
  "sha256": "replace-with-checkpoint_digest-result",
  "text_variant": "text_v2",
  "score_semantics": "ordering_logit",
  "selection_record": "path-or-id-of-approved-selection-record",
  "max_length": 384,
  "truncation_policy": "longest_first",
  "batch_size": 8
}
```

Paths resolve relative to the manifest. Compute the digest using
`candidate_screener.app.screening_service.checkpoint_digest(Path(...))`; it covers
relative filenames and bytes, including tokenizer files. Symlinked artifacts are
rejected. Length and truncation must match the approved artifact; this example is
not a model selection. The default shortlist is 50, the pool limit is 500, and
scoring documents are limited to 200,000 characters. The retrieval-only tail is
labelled **not reranked**. Retrieval scores and CE logits are separate fields.

Fit / No Fit requires `score_semantics: "binary_logit"`, a finite
`binary_threshold` **in raw logit units**, and `threshold_validation_record`.
Never copy a probability threshold into a logit manifest. For a model whose
documented output is `sigmoid(logit)`, an interior probability threshold `p` maps
to `log(p / (1-p))`; record that equivalence and the original validation lineage.
Ordering checkpoints, including R2, acquire no binary threshold through this
wrapper. Manifest declarations and file integrity are checked; statistical quality
of a supplied threshold is not independently revalidated. Without a validated
threshold, classification is **unavailable**. Percentiles are not probabilities.

Configure Qwen through environment variables, or matching string keys under
`[chatbot]` in `.streamlit/secrets.toml` (never commit secrets):

| Setting | Meaning |
|---|---|
| `SCREENER_QWEN_BASE_URL` | OpenAI-compatible URL, e.g. `http://127.0.0.1:8000/v1` |
| `SCREENER_QWEN_MODEL` / `SCREENER_QWEN_REVISION` | Exact served model and provisioned revision |
| `SCREENER_QWEN_SHA256` | Expected service digest; local provisioning supplies it automatically |
| `SCREENER_QWEN_TOKENIZER` | Matching tokenizer directory, or `service` for local `/tokenize` |
| `SCREENER_QWEN_RUNTIME` / `SCREENER_QWEN_QUANTIZATION` | Runtime name/version and quantization, or `none` |
| `SCREENER_QWEN_API_KEY` | Optional credential |
| `SCREENER_QWEN_CONTEXT_TOKENS` / `SCREENER_QWEN_OUTPUT_TOKENS` | Defaults 4096 / 512; match deployed model |
| `SCREENER_QWEN_TIMEOUT` / `SCREENER_QWEN_TEMPERATURE` | Defaults 30 seconds / 0.2; validate on selected model |
| `SCREENER_QWEN_JSON_MODE` | Default `true`; disable for runtimes lacking JSON response format |
| `SCREENER_QWEN_ENABLE_THINKING` | Optional `true`/`false` for compatible models/templates only |
| `SCREENER_QWEN_ALLOW_REMOTE` | Default `false`; explicit data-boundary decision and HTTPS required |
| `SCREENER_QWEN_DISABLED` | `true` forces templates and suppresses model autodiscovery |

Qwen supports OpenAI-compatible serving; chat templates and thinking options vary.
See the [official Qwen2.5 model and generation example](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct#quickstart)
before provisioning. The client counts tokens locally, reserves output space,
limits response bytes and retries transient errors at most once. HTTP redirects
and inherited proxies are disabled. Model revision/runtime declarations are
configuration provenance for external services. The provisioned local service's
revision, digest and token limits are checked against `/v1/models`, and its actual
runtime versions are recorded.

```bash
uv run python -m candidate_screener.app.chat_smoke --live --freeform
```

Live smoke uses invented documents and two QA requests (plus a free-form question
with `--freeform`), reports load/answer time,
token counts, client peak RSS and fallbacks, and exits unsuccessfully on missing
configuration or generated-answer fallback. Measure service memory separately.
The report separates Qwen QA from CE availability. A Qwen-only trial uses explicit
TF-IDF illustration retrieval if no CE manifest is configured; it never proves
R2 inference. Measure checkpoint/service compatibility separately.

Applicant material and chat context stay in session memory; shared resources hold
frozen models only. Document, JD/title or configuration changes invalidate results
and history. Reset clears session material. Redaction masks contact identifiers
but does not anonymize names. No persistence, public hosting, new annotation or
model adoption is included.

## Build the derived artefacts

```bash
uv run python -m candidate_screener.data.build  --all --seed 0   # splits, pools, vocabulary
uv run python -m candidate_screener.data.verify --derived        # acceptance test on them
```

Builders are pure functions of (`data/raw/`, seed). Nothing built is committed; what *is*
committed is the manifest naming each document by content hash, so the artefacts are
reconstructible byte-for-byte — see [`docs/data/manifests/`](docs/data/manifests/).

## Run the baseline

```bash
uv run python -m candidate_screener.baselines.run --seed 0   # fit, score, write the record
uv run python -m candidate_screener.baselines.run --check    # assert it still reproduces
uv run python -m candidate_screener.evaluation.retrieval --seed 0  # the same model, ranked over the pools
uv run pytest                                                # unit + regression suite
```

The Stage 1 classical baseline (TF-IDF cosine and BM25, each over a single-feature logistic
regression) lives in `src/candidate_screener/baselines/`, and its figures are committed to
[`output/baselines/`](output/baselines/README.md). The fitted models are *not* committed — they
are a regenerable cache under git-ignored `data/processed/baselines/`. `--check` refits in
memory and diffs against the committed record, so a run that quietly differs fails loudly.

`pytest` needs no data: every test but the end-to-end golden check runs on inline synthetic
corpora and the golden check skips, visibly, when `data/` is absent.

## Where to look

| Question | Read |
|---|---|
| What data do we have, under what licence, with what defects? | [`docs/data/data-catalog.md`](docs/data/data-catalog.md) and [`docs/data/cards/`](docs/data/cards/) |
| Why these sources, and what did profiling change? | [`plan/2026-08-19-data-strategy/01-requirements-and-findings.md`](plan/2026-08-19-data-strategy/01-requirements-and-findings.md) |
| What happens next with the data? | [`plan/2026-08-23-derived-artefacts/`](plan/2026-08-23-derived-artefacts/README.md) — Phase 3, supersedes the acquisition plan's §3 |
| What does the baseline score, and what is it comparable to? | [`output/baselines/README.md`](output/baselines/README.md) — the committed figures, and why accuracy sits below the majority floor |
| What was actually built, and where did it deviate? | [`plan/2026-08-19-data-strategy/04-acquisition-implementation.md`](plan/2026-08-19-data-strategy/04-acquisition-implementation.md) |
| How should an agent work in this repo? | [`AGENTS.md`](AGENTS.md) |

## Three findings that govern the evaluation design

1. **The core benchmark's effective size is ~640 resumes and ~280 JDs**, not 8,000 examples —
   every confidence interval must be computed against the document counts.
2. **Its shipped split leaks 99.8% of test resumes into train.** The project re-splits so that
   resumes *and* JDs are disjoint, and results are therefore not comparable to published
   numbers on the shipped split. That evaluation is **31 queries**, not 659 pairs — and which
   31 you get moves by a factor of two across random seeds, so the seed is fixed and
   committed.
3. **Every metric ceiling is a property of the split, not of the corpus.** On the shipped split
   the median query has 18 relevant resumes and Recall@10 is unreachable; on the leak-free split
   it has 6, and Recall@10 reaches 0.90 for 93.5% of queries while Recall@50 saturates. The
   adopted metrics are **Recall@10, Precision@5 and nDCG@10**, each reported with its *n* and a
   bootstrap CI — enforced in `candidate_screener.evaluation.metrics`, not by convention.
4. **Precision@k on the pools is capped below 1.0, and Recall@10 is not.** Many queries have
   fewer than *k* judged-relevant resumes, so a perfect ranker cannot fill the top *k*:
   Precision@5 tops out at **0.7806** strict, Recall@10 at 0.983. The cap is a property of the
   split — identical at every pool depth — so every precision figure is reported with its
   ceiling beside it *(D26)*. TF-IDF's Recall@10 is **0.298** [0.216, 0.391] against the
   proposal's 0.80 target; that gap is the finding, not a defect to tune away before reporting.

   *Superseded 26 Sep 2026 (D40).* The attainable ceilings (0.7806 / 0.7188 / 0.983) are **retired, not restated**. They are computed from A1 labels that do not reproduce (A13), and A1 is now a secondary benchmark. A1 figures are quoted with *n*, CI and the A13 line, with no "% of attainable". The primary benchmark is the A2 final set (D41). See `plan/2026-09-26-top-down-reset/`.

## The annotation session

One session, ~250 judgements, ~2.5 team-days *(decision D25)* — 200 in-domain Djinni pairs and
50 already-judged A1 pairs mixed in blind to test whether A1's own labels hold up. Read
[`docs/annotation-guide.md`](docs/annotation-guide.md) first.

```bash
uv run python -m candidate_screener.data.build --task indomain judging-queue --seed 0
uv run python -m candidate_screener.annotation.ui --annotator <your name>   # label
uv run python -m candidate_screener.annotation.queue --progress             # how far in
# dispatch:  data/processed/indomain/judging-queue.csv   (git-ignored, redacted, blind)
# key:       docs/data/manifests/judging-queue.csv       (committed, ids only, no text)
# labels:    docs/data/manifests/judgements.csv          (committed, header until the session runs)
```

The dispatch file carries three columns and nothing else. An annotator who could see why a pair
was selected would know which ones already have a label and would anchor on the expected answer.

**Pairs are labelled in a local UI** *(decisions D30, D32)*: one screen per query, three
buttons per candidate, and the top-1 shortlist question asked once over a complete field — which
the shuffled flat file could not do, since it scatters a JD's candidates and carries no key to
put them back together. Both corpora are served here and interleave: in-domain groups are 10
candidates, A1 recheck groups 1 to 4. Stop whenever; re-run to resume, since `judgements.csv`
is the only progress there is. Creating a batch picks job titles from the **41** reachable in
the corpus — the generic batch covers 22, which is what an unstratified draw of 40 JDs landed
on rather than a designed scope.

**Each JD gets 10 candidates, 8 of them from its own role family** *(decision D31)*. The first
cut drew 5 from one undifferentiated pool, which put only **5.0%** of pairs in the JD's own
family; a pilot labelled **28 of its first 30 pairs `No Fit`**. A set of obvious negatives costs
the same hours as one that discriminates and tells you nothing about which system is better. The
realised rate is now 80.0%, and every in-domain precision figure has to say that this pool is
deliberately enriched — it is a comparison instrument between systems, not an estimate of
production precision.

### Growing it, pausing it, re-scoping it *(D28)*

The set is a **frozen, append-only batch campaign** — `docs/data/manifests/indomain-batches.json`
is its config. Batch *N* draws only from what batches 1..*N*-1 left behind, so appending can
never change an existing `pair_id`, and labels already collected stay valid.

```bash
uv run python -m candidate_screener.annotation.sample --add-batch --n-jds 20        # wider
uv run python -m candidate_screener.annotation.sample --add-batch --reuse-jds       # deeper
uv run python -m candidate_screener.annotation.sample --add-batch --keywords "Data Science"
uv run python -m candidate_screener.annotation.queue  --build                       # resume
uv run python -m candidate_screener.annotation.queue  --progress                    # where you are
```

`queue --build` is the resume command: it emits only pairs with no row in `judgements.csv`, so
a session that stops at pair 130 of 250 restarts with 120 and no bookkeeping.

**Figures are quoted per stratum, and there are exactly two** *(D29)*. `generic` is the
unstratified batch — *"a typical posting from this board"*. `targeted` pools **every**
keyword-scoped batch — *"the job families we chose to cover"*. Batches within a stratum are
pooled freely; the two strata are never averaged together, because their average describes a
mixture that exists nowhere. `verify --derived` prints the per-stratum counts on every run.

The session is **two annotators, both covering every title**, working from one queue split by
hand, with 30% double-labelled for κ. Assigning titles by expertise was considered and
rejected: it confounds annotator with title, and the confound reaches the pooled figure.

## Data handling rules

- Nothing under `data/` is committed. Derived artefacts are reproducible from the scripts.
- **DataTurks (B1) contains direct PII** — `Name` and `Email Address` are annotated entity
  classes. De-identify before use, and never commit un-redacted content, notebook outputs
  included.
- **44 livecareer (C1/C2) resumes overlap the A1 benchmark.** Exclude
  `data/interim/c1_a1_contamination.csv` from any NER training set or distractor pool built
  off that corpus.
