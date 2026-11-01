# Hiring Desk: using the recruiter app

This page covers the current Hiring Desk, the Streamlit recruiter app: starting it, the first
and further accounts, sign-in and expiry, reviewing jobs, creating a new JD and ranking it
through the inference API (D63/D64), alerts, sending a package to the hiring department, and
the three chat versions (v3, the default since 31 Oct 2026; v2; and the v1 extractive path).
Running it in Docker is in [`docker.md`](docker.md); the earlier prototypes and workshop
setups are in [`chatbot-history.md`](chatbot-history.md). Moved from README.md on 1 Nov 2026.

## Evaluation

The preliminary CUI evaluation harness and reviewer workflow are documented in
[cui-evaluation.md](cui-evaluation.md). Its synthetic development runs
verify the harness; independent held-out quality acceptance remains pending.

## Starting the app

The primary app opens **Hiring Desk**, with typed HR chat and dropdown review menus:

```bash
uv run streamlit run src/candidate_screener/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8503
```

Hiring Desk v1 opens the supplied CSV-backed jobs alongside published operational
JDs. The production app has no workspace/demo selector.

## Accounts and sign-in

**The first account.** While no account exists (no `accounts.json`, or one with no
users), the app opens a **Create the first account** screen instead of sign-in: a
login ID, the departments (comma-separated, pre-filled with `benchmark`), the access token twice
(24 to 512 characters, hidden) and the expiry in days (default and maximum 40).
The check that no account exists runs under the account file's lock, so of two
simultaneous submissions only one succeeds. Mismatched or short tokens create
nothing. Once an account exists the screen is gone for good and sign-in is shown.
Accepted risk (owner's choice): while no account exists, whoever opens the page first
creates the first account. Serve the app on loopback only (the commands here use
`127.0.0.1`; Docker publishes the port on host loopback); if the port is ever exposed on
a network, a one-time setup code must be added first.

**Further accounts** (there is no administrator role in the app): run the command in
your own terminal and enter a private access token of 24 to 512 characters at
the hidden prompt; no token is printed or stored in conversation history. Two
different entries, a short token or an existing login ID print an error and
create nothing:

```bash
uv run python -m candidate_screener.app.portal_auth --create-local-user reviewer --department benchmark
```

Then sign in. Local token authentication is
not company SSO. Reviewer credentials default to 40 days (`--hours 960`); active
sign-ins last until account expiry, revocation or logout, without an hourly timeout.
Browser reconnects can require signing in again. Use **Log out** at the bottom of
the left sidebar to end the current sign-in.

## The benchmark jobs (CSV export)

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

## Create new JD and live ranking

**Create new JD** guides title, department, responsibilities, required/preferred
skills and knowledge, experience and qualifications. Each accepted answer is
saved locally. Resume/edit a draft, confirm the preview, then publish and request
candidate ranking. Operational JDs and ranking snapshots live separately from
benchmark labels and training data in the local workspace database.

New JDs rank through the inference API (`inference.Screener`, plan
[2026-10-24-hiring-desk-inference-api](../plan/2026-10-24-hiring-desk-inference-api/README.md);
the API itself is described in [`inference-api.md`](inference-api.md)).
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
alike, and it ignores experience requirements above 20 years. Failures show in the app as a generic
message; the cause is in `logs/hiring-desk-adapter.log`. Any other adapter implementing the
[contract](../plan/2026-10-23-hiring-desk-v1/05-live-ranking-contract.md) (`adapter(operation,
payload) -> dict`, schema in `app/portal_ranking.py`) can be configured instead. Without an
adapter, new JDs stay saved with an explicit unavailable state; existing CSV reviews still work. No
model is trained at startup.

## Alerts and sending to the hiring department

**Alerts** records JD publication, ranking ready/failed and local email
captured/failed events. Read/unread controls and Open job preserve access checks.
**Send to hiring department** demonstrates the flow using Tech/Business example
addresses: preview the ten-applicant package, then choose **Send to demo outbox**.
The resulting private `.eml` file can be downloaded; it is captured locally, not
delivered to a mailbox. Captures live under `data/processed/hr-portal/mail-captures/`.
Original PDFs, company SSO, external applicant events and production mail delivery
remain unconnected. Additive SQLite migrations preserve old sessions/packages;
retain a private backup of the workspace directory before moving deployments.

## Ask Hiring Desk (chat)

After selecting a role, use **Ask Hiring Desk** to ask about its requirements,
compare candidates, or inspect cited CV evidence. Dropdown candidate/category
selections establish the context for the next question; explicit candidates in
your question take precedence. Saved reviews retain their bounded conversation.
The **Review action** dropdown offers Top10 ranked CVs, Open CV, Generate session report,
Send to hiring department and Review another job.
Failed submissions offer **Retry question** and **Discard question**. A pending
question is discarded with an explanation if its review or candidate context changes.

### v3 (the default)

**Hiring Desk v3** (`HR_PORTAL_CHAT_VERSION=v3`, the default since 31 Oct 2026, D69) adds an
inference-only plain-prose Qwen path, bounded conversation history, explicit comparison tables,
expandable tables for “why” explanations, and supporting sources on demand.
“Find jobs by requirement” searches authorized JD text without requiring rankings;
unranked published jobs also support standalone JD questions. Cross-JD search
does not change the selected review. The existing frozen models are reused.
Unsupported generated wording falls back to checked statements. Candidate
comparisons retain v2's checked comparison when generation is rejected or
unavailable. Comparison replies and their criterion follow-ups retain the
candidate table; directed “why” replies keep it expandable. V3 retains v1's six
broad evidence categories and shows recognized JD-specific criteria alongside
them. The table and checked relations are calculated before generation; a compact
Evidence Matrix precedes Selected Passages in Qwen's prompt, so the explanation
sees the broad statuses without receiving whole CVs. Flags describe source
support, not independent verification or inferred proficiency. Narrower JD
parsing does not replace the broad table.

Single-candidate alignment summaries retain broad strengths and material JD gaps.
Where checked source evidence establishes overlap with the JD, they end with a
conditional relevance conclusion while keeping unconfirmed requirements explicit.
Source quotations share one CV attribution, and the concluding statement appears
in a separate paragraph.

Directed “why” questions return one concise answer line with supporting points
and any counterevidence, rather than a new Fit decision. For JD-only
questions, a complete authorized supporting passage can be quoted when generation
is rejected or unavailable; the reply retains its fallback provenance. This
conservative validator does not establish general semantic entailment. V3's
[quality gates](../plan/2026-10-25-hiring-desk-v3/02-implementation-and-evaluation.md)
remain pending (D66, D67).
Its system prompt is `src/candidate_screener/llm/hr_system_prompt_v3.txt`.

V3's default model is Qwen3.5-4B (owner, 31 Oct 2026, D69), on a separate pinned GGUF CPU runtime:
`uv run python -m candidate_screener.llm.gguf_assets` provisions the assets, then
`uv run python -m candidate_screener.llm.gguf_server --manifest data/processed/chatbot/qwen35/manifest.json --launch-backend`
serves the model on loopback 8001 (run it in its own terminal; the portal never starts it).
With `SCREENER_MODELS_ROOT` set, the manifest is under that folder instead, for example
`data/models/chatbot/qwen35/manifest.json`.
With no `SCREENER_QWEN_*` setting, v3 selects the provisioned manifest
`data/processed/chatbot/qwen35/manifest.json` by itself; without it, v3 uses the
existing 0.5B settings (the Qwen service on 8000). An explicit
`SCREENER_QWEN_MANIFEST` still selects a manifest, and a missing selected
manifest is unavailable and never silently selects the old model. If the 4B service
is not running, v3 answers through its marked fallback. No Torch upgrade is needed.
The commands are the same on every supported host: `gguf_assets` picks the pinned
llama.cpp b11896 runtime for macOS (x64, arm64) or Linux (x64, arm64) by itself,
and `--platform KEY` chooses one explicitly (for example, to prepare a folder for
another machine; the start check is then skipped and recorded). On the host's own
platform it first starts `llama-server --version`, before the 2.74 GB model is
fetched or a manifest written; on Linux that needs glibc 2.34 or later (x64), 2.38 or later (arm64),
and `libgomp1`. In Docker the 4B runs as its own image, the `llm4b` service that plain
`docker compose up -d` starts ([`docker.md`](docker.md) §6; plans
[`2026-10-31-v3-default`](../plan/2026-10-31-v3-default/README.md) and
[`2026-10-28-gguf-cross-platform`](../plan/2026-10-28-gguf-cross-platform/README.md)).
The Qwen3.5 trial allows up to 60 seconds per turn; extraction and
the existing 0.5B model retain their 30-second limits.

`HR_PORTAL_CHAT_VERSION` defaults to `v3` (owner, 31 Oct 2026, D69). Being the default is not
quality acceptance: v3's quality gates remain pending. Set the version to `v1` (or `v2`) to roll
back; no running service is migrated automatically. An invented-source development smoke is
available as
`uv run python -m candidate_screener.app.conversation_v3_smoke --real-model --output logs/v3-smoke.json`.

### v2 (opt-in)

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
restores the v1 experience (the rollback). The [v2 record](../plan/2026-10-24-hiring-desk-v2/04-implementation-record.md)
tracks validation and remaining independent quality review; this is an opt-in
implementation, not a claim that the proposed human quality gates have passed.

V2 also accepts “Why is 1 better than 2?” and criterion-specific variants such as
“Why is 1 better than 2 on Python experience?”. The answer explains checked
JD-linked evidence with the comparison table, preserves counterevidence and
corrects an unsupported premise. Missing CV evidence is never treated as proof
of a missing skill. Short source labels link the reasons to their source controls.
The app uses deep-red controls, near-black text and a predominantly white canvas.

### v1 (the extractive path, and the rollback)

The current v1 path uses the installed `deepset/roberta-base-squad2` to answer
factual JD/CV questions with exact start/end spans. Answers retain source context
and expandable citations. Unsupported facts remain insufficient evidence;
missing or failed models show their availability and labelled source excerpts.
The assistant cannot change ranks or execute report, package or delivery actions.

Comparison is a chat request, such as **Compare 1 and 2**; the Review action
dropdown no longer offers it. The answer uses an application-built table of JD
requirements and candidate evidence. GLiNER medium extracts technology, skills,
titles, degrees and experience mentions in the background and caches them
privately by source/model/configuration. General comparison does not wait for
QA inference; focused questions such as **Compare 1 and 2 on Python** extract
answers independently from each requested document. Entity mentions do not
establish proficiency, completed credentials or skill-specific years of experience.

This path reuses the existing isolated `runtime/gliner-medium` and
`runtime/transformer` environments and pinned local checkpoints. Models load on
demand; missing assets never trigger downloads or training at startup. Primary
v1 answers no longer invoke Qwen. Earlier saved Qwen answers retain their recorded
provenance. The latest scope and verification are recorded in the
[extractive implementation plan](../plan/2026-10-24-grounded-qwen-comparisons/05-extractive-implementation.md).
