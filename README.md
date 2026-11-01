# Intelligent Candidate Screening System

NUS ISS PLP Practice Module, Group 2. The **Hiring Desk** ranks the resumes in a candidate pool
against a job description (JD). It shows the evidence for each ranking, and a recruiter can ask
about the role and its candidates in plain English.

The research question (proposal v3, [`docs/proposal/`](docs/proposal/)) is **whether more
sophisticated matching earns its complexity**. The matching ladder runs rules → TF-IDF/BM25 →
Sentence-BERT → cross-encoder. Every step up is scored against the same human-built reference,
and a step is kept only where it beats the simpler one.

**Where it stands (1 Nov 2026):**

- **Ranking:** new JDs are ranked by the reported system, **D64**: a learned ranker (L1) orders the
  600-CV pool, then a zero-shot cross-encoder (C2) reorders the top 50. Without that second step it
  is D63.
- **Chat:** the default is **v3**, in which Qwen3.5-4B writes answers from checked source evidence
  (D69).
- **Running it:** the whole application runs in Docker on Windows, macOS and Linux, or natively with
  `uv`.

Every decision and its reason is in [`docs/decisions.md`](docs/decisions.md). The plans and their
implementation records are under [`plan/`](plan/).

## Results

### The reported system (D63/D64), 22 Oct 2026

Measured on the 60 final-set JDs, each with a pool of 500 CVs. The reference is the LLM's chat
top 10 (`chat-1` + `chat-2`, D62), not the human labels. The record is
[`output/rerank/c2-validation-test.json`](output/rerank/c2-validation-test.json), and the test
is marked *exploratory* in [plan 2026-10-22-c2-validation](plan/2026-10-22-c2-validation/README.md).
95% CIs are bootstrapped over JDs: *n* = 40 generic, 20 targeted, 60 in total.

| System | Recall@10, generic | Recall@10, targeted | Recall@10, total [CI] | nDCG@10, total [CI] |
|---|---:|---:|---:|---:|
| X2: metadata + TF-IDF (the baseline) | 0.419 | 0.414 | 0.417 [0.366, 0.467] | 0.775 [0.725, 0.820] |
| **D63:** L1 alone | 0.484 | 0.456 | 0.475 [0.432, 0.517] | 0.820 [0.784, 0.853] |
| **D64:** L1, then C2 on the top 50 | 0.487 | 0.467 | 0.480 [0.436, 0.523] | 0.832 [0.797, 0.863] |

- **L1 over the baseline:** L1 finds more of the reference's top 10 than X2. The paired delta is
  +0.058 [+0.029, +0.088].
- **C2 missed its registered bar.** It was required to add at least +0.02 Recall@10 with the CI
  above 0. It added +0.005 [−0.010, +0.021], although it did order the top 10 better: nDCG@10
  +0.012 [+0.005, +0.019]. The owner adopted it on that ground (D64). The decision log records
  this as an override of the bar.
- **On the human labels** (422 judged pairs), C2 does not change the per-JD AUC: −0.003
  [−0.021, +0.013] against L1.

### The matching ladder on the human final set (the reset, 27 Sep 2026)

One final Fit / No Fit label per in-domain pair (D41), with per-JD AUC and 95% CIs over 59 JDs.
Full detail is in
[`plan/2026-09-26-top-down-reset/09-findings.md`](plan/2026-09-26-top-down-reset/09-findings.md).

| System | Per-JD AUC | Verdict under the registered rule |
|---|---|---|
| random | 0.556 | — |
| metadata rules (role keyword + experience) | 0.778 | — |
| TF-IDF | 0.797 | — |
| **metadata + TF-IDF** | **0.849** [0.802, 0.891] | **selected (D48)**: nothing richer beats it |
| SBERT, fine-tuned on weak in-domain pairs | 0.854 [0.817, 0.890] | **selected for text-only input**: +0.057 over TF-IDF, justified |
| cross-encoder (A1-trained / weak-trained) | 0.488 / 0.661 | harmful |
| learned combination of all text models | 0.828 | not resolved |

- With 500 candidates per role, metadata + TF-IDF keeps 92.9% of the fits in a 50-candidate
  shortlist, in 0.84 s.
- Re-ranking that shortlist with a cross-encoder cuts Recall@10 by a quarter to a third.
- Extraction splits by role (D49): a MiniLM tagger for job postings (SkillSpan strict F1 0.523 vs
  0.237) and a rules matcher for resumes.

## Run with Docker (any OS)

The Hiring Desk also runs in Docker on Windows, macOS or Linux, with no Python, uv or model
download on the host (D68). One image holds the code, the three environments and the pinned
pretrained weights; the data folder `data/datapack/` holds the project's own files plus the
accounts and reviews, and is passed on privately. Native development is unchanged.

```bash
mkdir logs                                             # put the data folder you were given at data/datapack/ first
docker compose build                                   # app + llm4b (Qwen3.5-4B, 2.74 GB model); app alone cold: ~3.5-5.5 min, 4.6 GB (27 Oct 2026)
docker compose --profile tools run --rm tools verify   # checks data/datapack/
docker compose run --rm app python -m candidate_screener.app.portal_auth --create-local-user reviewer --department benchmark
docker compose up -d                                   # http://localhost:8503, chat v3 with Qwen3.5-4B (give Docker ~15 GB)
```

Chat v3 with Qwen3.5-4B is the default (D69). On a machine that cannot give Docker about 15 GB, use
the lighter v1 setup: `HR_PORTAL_CHAT_VERSION=v1` in `.env`, then `docker compose build app` and
`docker compose up -d app` (about 10 GB; the 4B is neither built nor started).

Requirements per OS, the data folder, accounts, backups, updates and troubleshooting:
[`docs/docker.md`](docs/docker.md).

## Set up a fresh clone with only the data folder

Tested end to end on 31 Oct 2026 in a new clone that held only the handed-over data folder
(plan [2026-10-31-tests-and-fresh-setup](plan/2026-10-31-tests-and-fresh-setup/README.md)). The
times below are from that machine with warm uv and Docker caches; a new machine also downloads the
packages, images and models.

**1. Code and data.** The owner hands over the data folder: the inputs only, never the accounts,
reviews or caches.

```bash
git clone --branch <branch> https://github.com/kuoyce/intelligent-candidate-screener.git intelligent-candidate-screener-clone
# owner, in their own checkout:
uv run python -m candidate_screener.datapack handover <path to the clone>/data/datapack
# in the clone:
mkdir logs
cp .env.example .env        # optional; set HOST_UID/HOST_GID if `id -u`/`id -g` are not 1000
```

`handover` verifies the copy. The last line reads `OK: 24 inputs (256.8 MB) match the manifest
and what the code expects`.

**2a. Run it in Docker** (the operator path; give Docker about 15 GB):

```bash
docker compose build                                   # app + llm4b; 8 s here with a warm cache
docker compose --profile tools run --rm tools verify   # OK, 4 s
docker compose up -d --wait                            # app and llm4b healthy after ~20 s
```

Then open http://localhost:8503. With no account yet, the page is **Create the first account**;
fill it in, then sign in. If an old `.env` still has
`COMPOSE_FILE=compose.yaml:compose.llm4b.yaml`, delete that line: the file no longer exists.

**2b. Run it natively** (the developer path). The data folder's layout mirrors `data/`, so link
its folders in. Keep the models in their own folder with `SCREENER_MODELS_ROOT`, so the data
folder stays exactly as handed over:

```bash
uv sync --all-groups
uv sync --project runtime/transformer
uv sync --project runtime/gliner-medium
mkdir -p data/processed
for d in data/datapack/processed/*/; do ln -s "../datapack/processed/$(basename "$d")" "data/processed/$(basename "$d")"; done
ln -s datapack/vocab data/vocab
export SCREENER_MODELS_ROOT=data/models     # every command below, and the app, need this
# the pretrained models, at the Docker image's pins (about 6 GB; 102 s here)
uv run python -m candidate_screener.llm.qa_assets --revision adc3b06f79f797d1c575d5479d6f5efe54a9e3b4
uv run python -m candidate_screener.extraction.medium_assets --encoder-revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3
PYTHONPATH=src uv run --project runtime/gliner-medium --no-sync python -c 'import json; from candidate_screener.extraction.medium_model import runtime_versions; from candidate_screener.extraction.medium_assets import MODEL_DIR; (MODEL_DIR/"runtime-versions.json").write_text(json.dumps(runtime_versions(),indent=2,sort_keys=True)+"\n")'
uv run python -m candidate_screener.llm.qwen_assets --revision 7ae557604adf67be50417f59c2c2f167def9a775
uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-Reranker-0.6B", revision="e61197ed45024b0ed8a2d74b80b4d909f1255473")'   # C2 (D64) loads offline only
uv run python -m candidate_screener.llm.gguf_assets          # Qwen3.5-4B for chat v3 (2.74 GB)
```

Then start the app, with the 4B service in a second terminal (same `SCREENER_MODELS_ROOT`):

```bash
uv run python -m candidate_screener.llm.gguf_server --manifest data/models/chatbot/qwen35/manifest.json --launch-backend   # ready in ~20 s
uv run streamlit run src/candidate_screener/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8503
```

Notes:

- **No cache step is needed.** `inference --cache-parts` is not needed, because the data folder
  already holds `processed/inference/parts/`.
- **Ranking works offline.** With `HF_HUB_OFFLINE=1`, loading the Screener took 22.9 s, D63
  ranked an invented JD in 47 s, and D64 in 149 s.
- **Native and Docker share one workspace.** The linked `data/processed/hr-portal` is the same
  folder Docker uses, so they share accounts and reviews. Native shows the first-account form
  only if Docker has not created one.
- **Do not run both at once.** Docker and native both use port 8503 and the 4B's ports, and
  together they exceed the memory of a 16 GB machine.
- **Expect fallbacks.** On a loaded 16 GB machine, the 4B often exceeds its 60 s budget on real
  JDs, so a v3 answer may be the app's marked fallback (`deadline_exceeded` or
  `unsupported_paraphrase` under **Answer details**). That is the designed behaviour, not a
  setup fault.


## Using the Hiring Desk

Open http://localhost:8503, in Docker or natively. A step-by-step walkthrough, from the build to a
first chat answer, is in [`docs/user-guide.md`](docs/user-guide.md).

- **Accounts.**
  - The first visit with no accounts shows **Create the first account**: a login ID, departments
    (pre-filled `benchmark`), an access token of 24 to 512 characters entered twice, and the
    expiry (default and maximum 40 days).
  - Further accounts are made from the command line:
    `python -m candidate_screener.app.portal_auth --create-local-user <id> --department benchmark`
    (prefix `docker compose run --rm app` in Docker).
  - This is local token sign-in, not company SSO. Serve it on loopback only.
- **Review jobs.** The cards hold the 60 benchmark JDs (from the CSV export) and any published JDs.
  Each card shows the JD summary, its skills and experience, the full redacted JD, and the ranked
  applicants. The **Review action** menu has five entries:
  - **Top10 ranked CVs**;
  - **Open CV** (the redacted text preview);
  - **Generate session report**;
  - **Send to hiring department**, a local demonstration that is not mailbox delivery;
  - **Review another job**.
- **Create new JD.** A guided form takes the title, department, responsibilities, skills,
  experience and qualifications. Confirm the preview, then publish and rank.
  - D63 ranks the 600 A2 final-set CVs in seconds to a minute: 35-43 s with empty caches, 14 s
    once they are filled.
  - **Refine the top 50 with the reranker** runs D64 in the background (94-121 s), then
    **Refresh ranking status** collects it.
  - The model sees only the JD's text and title and each CV's text (D53, D54).
- **Alerts** record JD publication, rankings ready or failed, and department captures.

Every screen, setting and flow in detail: [`docs/hiring-desk.md`](docs/hiring-desk.md).

## The v3 chat assistant

**Ask Hiring Desk**, beside each review, answers questions about the selected JD and its ranked
candidates. It is the default chat (D69): the panel's caption reads "Hiring Desk v3".

### How an answer is built

```
question ──► 1. find evidence ─► 2. check it ─► 3. write prose ─► 4. check the wording ─► answer
             RoBERTa-SQuAD2      comparison      Qwen3.5-4B          every claim supported,      or the app's
             spans; GLiNER-      table, checked  (local, CPU):       owned by the right          own checked
             medium mentions     relations       evidence matrix +   candidate, qualifiers       wording
             (JD and CVs)        (before any     selected passages + kept                        (marked
                                 generation)     bounded history                                 fallback)
```

1. **Find the evidence.** The frozen extractive reader (RoBERTa-SQuAD2) finds answer spans in the
   JD and the CVs, and GLiNER-medium supplies skill, knowledge, title, degree and experience
   mentions. Both run locally.
2. **Check it.** Before any text is generated, the application builds the comparison table and the
   checked relations between candidates. Examples: a completed degree against one in progress, an
   explicit negative ("No commercial Python experience"), or an expired certificate.
3. **Write the prose.** Qwen3.5-4B receives a compact evidence matrix, the selected source passages
   and a bounded slice of the conversation history, never whole CVs. Its system prompt is
   [`src/candidate_screener/llm/hr_system_prompt_v3.txt`](src/candidate_screener/llm/hr_system_prompt_v3.txt).
4. **Check the wording.** The application accepts the text only if every claim matches the checked
   evidence, belongs to the right candidate, and keeps the source's qualifiers (ongoing, expired,
   negated). Otherwise it answers with its own checked wording, which is the **marked fallback**.
   Comparisons then keep v2's checked comparison.

The history is conversation context only, never evidence. Each answer is re-checked against the
current sources.

### What to ask

| Kind | Example | What comes back |
|---|---|---|
| The role | "What does this role require?" | The JD's requirements, quoted from the JD |
| One candidate | "What degree does candidate 1 have?" | The stated degree with its status (completed, in progress) |
| Comparison | "Compare candidates 1 and 2", then "What about their qualifications?" | A comparison table plus prose; follow-ups keep the table |
| Why | "Why is 1 ranked above 2?" | One answer line with supporting points and any counterevidence, not a new Fit call |
| Fit summary | "Summarise why candidate 1 is a good fit to this JD" | Strengths with quotes, and the JD requirements that are not confirmed |
| Find jobs | **Find jobs by requirement**, or a question on a published JD that is not ranked yet | Matching JD passages; the selected review does not change |
| Where it says so | "Where does it say the Master degree is in progress?" | The source quote, instantly |

**What it will not do.**

- It cannot change ranks or scores, or make a Fit call.
- It does not infer proficiency or tenure. Entity mentions cannot establish how long someone used
  a skill.
- A missing mention is not evidence of absence, and answers say so.
- Flags describe source support, not independent verification.

### Reading an answer

Open **Answer details** under an answer.

- `generation_mode: qwen_prose` means the 4B's text was accepted.
- `qwen_fallback` means the application's checked wording was shown, and `fallback_reason` says why:

  | Reason | Meaning |
  |---|---|
  | `deadline_exceeded` | The 4B ran past the 60 s turn limit |
  | `provider_unavailable`, `worker_unavailable`, `worker_failed` | The model service was not reachable or failed |
  | `unsupported_paraphrase` | A claim was not backed by the checked evidence |
  | `qualification_omission`, `participant_omission`, `counterfact_omission` | A qualifier, a candidate or a counterpoint was left out |
  | `redaction_required`, `invalid_prose_format`, `non_prose_output`, `invalid_response_size`, `context_overflow` | The output's form was not usable |

**Show supporting sources** opens the quoted passages behind the answer.

### Model, speed and memory

- **Runtime.** Qwen3.5-4B (Unsloth Q4_K_M GGUF, pinned revision and SHA-256) runs on the published
  llama.cpp b11896 CPU build for macOS or Linux, x64 or arm64. It uses 4 threads and has thinking
  disabled.
- **Where it lives.** In Docker it is the `llm4b` service. Natively, `gguf_server` runs it on
  127.0.0.1:8001. The app never downloads, starts or trains a model.
- **Speed.** Each turn has a **60 s limit** (D67). A v3 prompt is 1,300-1,700 tokens, and reading
  it is about 85-90% of a turn. Turns took 34-57 s on an idle machine (28 Oct 2026). While a
  ranking runs, or on a loaded 16 GB machine, many turns reach the limit and fall back (31 Oct
  2026).
- **Memory.** The 4B peaks at about 4.7 GB, and Docker needs about 15 GB for the default setup.
- **Faster options.** v2 (Qwen2.5-0.5B, `--profile llm`) answers in seconds. v1 is extractive
  answers only.
- **Evidence so far.** In the invented-source smoke test (`conversation_v3_smoke --real-model`), 1
  of 4 factual turns came from the 4B and the rest were checked fallbacks. These are development
  cases, not a held-out accuracy figure. The quality gates are in
  [plan 2026-10-25-hiring-desk-v3](plan/2026-10-25-hiring-desk-v3/02-implementation-and-evaluation.md).

### Settings

| Setting | Effect |
|---|---|
| `HR_PORTAL_CHAT_VERSION` | `v3` (the default), `v2` (the 0.5B orders checked comparisons), or `v1` (extractive answers only; the rollback). Any other value behaves as `v1` |
| `SCREENER_QWEN_MANIFEST` | Selects a model manifest. Unset natively: the provisioned 4B, else the 0.5B. In Docker, `compose.yaml` sets it |
| `SCREENER_QWEN_DISABLED=true` | No generation: every answer is the checked application wording |

The design, prompts and records: the plans
[2026-10-25-hiring-desk-v3](plan/2026-10-25-hiring-desk-v3/README.md) (v3),
[2026-10-28-gguf-cross-platform](plan/2026-10-28-gguf-cross-platform/README.md) (the 4B on every
platform) and [2026-10-31-v3-default](plan/2026-10-31-v3-default/README.md) (the default).

## Using the ranking from code

`candidate_screener.inference.Screener` serves the reported system to any app. It loads its parts
and never fits:

```python
from candidate_screener import inference

s = inference.Screener.load()                        # a few seconds
res = s.rank(jd_text, cvs, jd_title="QA Engineer")   # cvs: cv_id + cv_text only (D53); rerank=True for D64
for r in res.top(10):
    print(r.rank, r.cv_id, r.score)
why = s.explain(res, res.top(1)[0].cv_id)            # reasons, skills, years
```

Scaling, reference dates, the output types and the demo data are covered in
[`docs/inference-api.md`](docs/inference-api.md).

## Repository layout

```
src/candidate_screener/
  config.py                 canonical data paths
  data/                     source registry, fetch, verify, profile, splits, pools, weak pairs
  baselines/                the frozen Phase 4 classical baseline (TF-IDF, BM25)
  matching/                 the ladder: rules, lexical, SBERT, cross-encoders, hybrids
  rerank/                   the learned ranker (L1) and the cross-encoder (C2)
  inference/                the reported system (D63/D64) as an API: `Screener`
  extraction/               skill vocabulary, rules matcher, MiniLM NER, GLiNER
  evaluation/               the final set (the reference), the protocol, metrics and reports
  app/                      the Hiring Desk: services, chat v1/v2/v3, Streamlit
  llm/                      local extractive QA, Qwen runtimes (0.5B service, 4B GGUF)
  annotation/               the frozen Fit annotation instrument (D42)
  entity_annotation/        the entity-span annotation UI (D56)
  datapack.py               the Docker data folder: create, verify, handover
  checks.py                 the runner for every committed record's `--check`
tests/                      pytest; runs without data/
notebooks/                  numbered, one topic each; index in notebooks/README.md
output/                     committed records: figures and config, never model bytes
docs/                       decisions, Docker guide, Hiring Desk guide, data catalog, proposal
plan/                       dated plans and their implementation records
runtime/                    separately locked environments (transformer, gliner-medium)
data/                       git-ignored working set; see data/README.md
Dockerfile, compose.yaml    the Docker images and services (D68)
```

## Data handling rules

- Nothing under `data/` is committed. Derived artefacts are reproducible from the scripts.
- **DataTurks (B1) contains direct PII**: `Name` and `Email Address` are annotated entity classes.
  De-identify before use, and never commit un-redacted content, notebook outputs included.
- **44 livecareer (C1/C2) resumes overlap the A1 benchmark.** Exclude
  `data/interim/c1_a1_contamination.csv` from any NER training set or distractor pool built off
  that corpus.
- Every document shown to a person passes `annotation.redact`.

## Further reading

| Topic | Where |
|---|---|
| Every decision and why | [`docs/decisions.md`](docs/decisions.md) |
| **User guide: start it and use it, click by click** | [`docs/user-guide.md`](docs/user-guide.md) |
| Running in Docker: requirements, the data folder, accounts, backups, troubleshooting | [`docs/docker.md`](docs/docker.md) |
| The Hiring Desk in detail | [`docs/hiring-desk.md`](docs/hiring-desk.md) |
| The inference API in detail | [`docs/inference-api.md`](docs/inference-api.md) |
| Development: tests, checks, data acquisition, baseline, annotation | [`docs/development.md`](docs/development.md), [`AGENTS.md`](AGENTS.md) |
| The notebooks | [`notebooks/README.md`](notebooks/README.md) |
| Earlier chatbot prototypes | [`docs/chatbot-history.md`](docs/chatbot-history.md) |
| The data | [`data/README.md`](data/README.md), [`docs/data/data-catalog.md`](docs/data/data-catalog.md) |
