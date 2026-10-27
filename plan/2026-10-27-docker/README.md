# Docker: run the Hiring Desk on any OS, from one Dockerfile and one data folder

Status: **approved by the owner (27 Oct 2026); phase 1 complete on `feat/docker-v1` (28 Oct 2026); v2 paused until the owner resumes it** (off
`e530659`; [`05-execution-stages.md`](05-execution-stages.md) holds the stage table and the resume
rules).

| File | Contents |
|---|---|
| this README | why, owner answers, accounts, scope, steps, verification, cost |
| [`01-design-spec.md`](01-design-spec.md) | the image, the runtime contract, compose, tests, cross-OS hazards, code changes |
| [`02-handover-guide-spec.md`](02-handover-guide-spec.md) | what the maintainer guide `docs/docker.md` must contain, and how it is accepted |
| [`03-optional-parts-log.md`](03-optional-parts-log.md) | moved on 28 Oct 2026 to [`docs/docker-optional-parts.md`](../../docs/docker-optional-parts.md), the living log of optional parts |
| [`04-app-data-folder.md`](04-app-data-folder.md) | the Docker data folder `data/datapack/`: contents, commands, who runs what, upgrades |
| [`05-execution-stages.md`](05-execution-stages.md) | execution stages, the long-job and resume rules, the account-creation screen, the implementation record |

## Why

The owner (27 Oct 2026) does not know which OS the application will run on, and asks for:

1. the application dockerized, with **everything starting from the Docker build file**;
2. room for the **frontend, still work in progress**, to keep changing;
3. **failing tests** (frontend) to be expected;
4. **proper docs for someone taking over**.

Today the Hiring Desk runs only on a machine prepared by hand: three uv environments (main,
`runtime/transformer`, `runtime/gliner-medium`) and six asset commands in the right order; a
git-ignored `data/` that differs per machine (this one lacks the chat models; the teammate's Intel
Mac cannot run SBERT); parts that assume Intel macOS (runtime locks resolved for `darwin x86_64`;
`llm/gguf_assets.py` pins a macOS llama.cpp binary).

## Owner answers (27 Oct 2026, three rounds)

| Topic | Answer |
|---|---|
| Development | **Stays local and unchanged** (native runs read `data/`, experiments included). Docker and the data folder are updated only once a change is settled |
| Data | Docker mounts **`data/datapack/`**: the project's own files copied from `data/`, plus the Docker workspace and caches. `create --with-workspace` copies the local workspace in. Upgrades purge and rebuild, keeping the workspace unless the change is breaking (`04-app-data-folder.md`) |
| Model weights | **Downloaded when the image is built** (pretrained, no fine-tuning): RoBERTa-SQuAD2, GLiNER-medium, Qwen2.5-0.5B, C2. Project-trained parts (role-family model, SBERT weak) come from the data folder |
| Optional parts | C2 / D64 and chat v2/v3 with Qwen2.5-0.5B; not Qwen3.5-4B or a workbench. A log, reviewed at each update: `03-optional-parts-log.md` |
| Architectures | amd64 and arm64; small arm64 differences are fine, no large changes. **No ARM emulation now**: arm64 is checked when someone builds on Apple Silicon |
| Runtime torch | CPU build on Linux |
| Pins | pin the two model downloads that are unpinned today |
| Branch | **`feat/docker-v1`** off `e530659`; commit and push there only. The account-creation screen gets its own branch |
| Cold run (Q17) | **skipped**: no model weights change; the approved warm run covers the `config.py` change |
| Execution | in stages, each committed and pushed, so a long job cut short by a usage limit can be restarted; no scheduled resume job (`05-execution-stages.md`) |
| Tests | the `test` image reports, does not gate; the long checks run **only on the owner's side**, at this stage and when updates are made, never at each run. The Docker healthcheck stays |
| Distribution | each person clones and builds; the data folder is passed over privately |
| Workspace | Docker starts empty unless `create --with-workspace` copies the local one |
| Other OSes | raised after the Docker build is complete, if the teammate's error persists |
| Guide | `docs/docker.md` |
| Decision log | add a short **D68** |
| Accounts | CLI for now. Then, on its own branch: **no accounts → a create screen; otherwise the sign-in screen**; no administrator role yet (see below) |
| Check run | the ~70 min warm affected run is approved |

## Are model weights compatible across operating systems?

Yes. Weight files load on any OS; what does not travel is the **software that runs them**, when a
package has no build for an OS. That is the Intel-Mac SBERT failure: SBERT needs the main `ml`
group's torch 2.14 (`uv.lock`), which lists builds for macOS arm64, Linux x86_64/aarch64 and
Windows, **none for macOS x86_64** (PyTorch's last Intel-Mac series is 2.2,
`runtime/transformer/README.md`; also recorded in
`plan/2026-10-24-hiring-desk-v2/08-onnx-live-ranking.md`). In Docker every host runs the same Linux
packages, so SBERT and C2 work on an Intel Mac too. RoBERTa, GLiNER and Qwen2.5-0.5B run on torch
2.2.2, available everywhere. The role-family model is a pickle tied to library versions, not to
the OS; the image installs the same versions from `uv.lock`. The same weights can differ in the
last bits on another CPU, which the arm64 bar below allows.

## What "everything starts from the Docker build file" means

```bash
docker compose build                                   # OS, uv, the three environments and all pretrained weights
docker compose --profile tools run --rm tools create --with-workspace   # owner, once: data/ -> data/datapack/
docker compose run --rm app python -m candidate_screener.app.portal_auth --create-local-user reviewer --department benchmark
docker compose up -d                                   # http://localhost:8503
```

(Command names indicative; steps 3-4 fix them.) A receiver puts the folder they were given at
`data/datapack/` instead of running `create`. No host Python, uv or model download. Nothing is
verified at start-up; `verify` runs by hand after an update.

## Accounts in a new Docker build

- An image never contains accounts. They live in the data folder,
  `data/datapack/processed/hr-portal/accounts.json`, so rebuilding the image leaves them alone; a
  purge removes them, unless the workspace is kept (`04-app-data-folder.md` §3).
- With no accounts, the sign-in page says so. Create one per reviewer:
  `docker compose run --rm app python -m candidate_screener.app.portal_auth --create-local-user <login> --department <dept>`.
  It asks twice, hidden, for a token of at least 24 characters, and stores only a salted PBKDF2
  hash (310,000 iterations), the departments and an expiry (default and maximum 40 days). On
  Windows, run it from PowerShell or Windows Terminal (Git Bash cannot show the hidden prompt).
- A handover never carries accounts: the next maintainer creates their own.
- Today the command only creates; renewing or disabling a login means a new login ID or editing
  `accounts.json`. The owner prefers to move account management into the frontend.

**In-app account creation (owner, 27 Oct 2026): after the Docker phase 1, on its own branch.**
With no accounts the app shows a "create the first account" screen; otherwise the sign-in screen.
No administrator role yet, and no setup code: while no account exists, whoever opens the page
first creates it, which the loopback-only port limits to someone on the same machine (accepted
risk; revisit before exposing the port on a network). Design and tests: `05-execution-stages.md`
§S10. Docker does not depend on it; the guide's account section changes when it merges.

## Measured starting point (27 Oct 2026, commit `e530659`)

| Fact | Value | Source |
|---|---|---|
| Native test suite (all groups) | 2105 passed, **17 failed**, 38 skipped, 110 s | `logs/pytest-docker-plan.log` |
| The 17 failures | 11 in `tests/test_portal_extractive.py`, 3 in `tests/test_portal_ui.py`, 3 in `tests/test_portal_v1_ui.py`: all frontend | same log |
| App's CSV inputs | byte-identical (SHA-256) to the committed `output/matching/a2-*.csv` | `sha256sum` |
| App data paths | 141 modules reachable from the entry point; every data path derives from `config.DATA` except `llm/qwen_assets.py`'s, spelled out | static survey |
| Asset modules | `llm/qa_assets.py`, `llm/qwen_assets.py`, `extraction/medium_assets.py` import only the standard library and `config`, and run as scripts | source |
| Main lock torch | 2.14.0+cpu, CPU index; no macOS x86_64 build | `uv.lock` |
| Runtime locks torch | 2.2.2 from PyPI: on linux x86_64 the CUDA build plus 12 `nvidia-*` packages; on aarch64 a CPU wheel | `runtime/*/uv.lock` |
| Model workers | started as `uv run --project runtime/<name> --no-sync …`: uv and both runtime venvs are needed at run time | `extraction/medium_runtime.py`, `llm/extractive_qa.py` |
| Qwen services | bind 127.0.0.1; the client refuses non-loopback HTTP unless `SCREENER_QWEN_ALLOW_REMOTE` and HTTPS | `llm/local_server.py`, `llm/qwen.py` |
| Paths | `config.PROJECT_ROOT` comes from the package file: editable install, source at `/app` | `config.py` |
| Asset pins | C2 and GLiNER-medium pinned in code; RoBERTa-SQuAD2 and GLiNER-medium's encoder resolved at download time; Qwen2.5-0.5B `7ae5576…` in this machine's manifest | asset modules |
| Line endings | no `.gitattributes`; Git for Windows' default (`core.autocrlf=true`) rewrites committed CSV manifests to CRLF, changing the digests in cache fingerprints | `git ls-files --eol` |
| Docker here | Engine 27.5.1, Compose 2.32.4, buildx 0.20.0 (amd64 only, no emulation); 22 cores, 15 GB RAM (WSL2) | `docker version`, `docker buildx ls` |

## Scope

**Phase 1.** The `app` image runs the Hiring Desk as natively, v1 chat by default: CSV-backed
benchmark reviews; **Create new JD** with D63 ranking on `a2-final-cvs`; **D64** (C2) on request;
v1 chat (RoBERTa-SQuAD2, GLiNER-medium); alerts, demo outbox, local sign-in (CLI accounts). Plus
the data folder commands, a `test` image, `compose.dev.yaml`, the guide, D68.

**Phase 2.** Chat v2/v3 with Qwen2.5-0.5B as the compose profile `llm` (weights already in the
image from phase 1). The account-creation screen, on its own branch (`05-execution-stages.md` S10).

**Not now (logged).** Qwen3.5-4B on Linux; a research workbench image; arm64 checks under
emulation.

**Out.** Public hosting, TLS, SSO, real mail; GPUs; the Kaggle (C1) and ESCO (D1) sources; R2
reranker checkpoints; the legacy prototype and the entity-annotation UI; changes to ranking,
models or research results; retraining in Docker; changes to native development; CI (Q22: none).

## Open questions

None. Q17 (D55's cold run before marking the plan complete) is answered: skipped, since no model
weights or training code change; the approved warm run covers the `config.py` change.

## How the work-in-progress frontend is accommodated

- **Development stays native.** Docker is refreshed only when a change is settled.
- **Layer order.** Pretrained weights and environments come first and depend on no frontend file;
  `src/candidate_screener/app/` and `.streamlit/` are copied last. A frontend change rebuilds in
  about a minute (target, V7).
- **Checking the frontend in Docker before an update:** `compose.dev.yaml` bind-mounts `src/` and
  `.streamlit/` into the running container with Streamlit's polling watcher.
- **Drift guards**, run at updates: a contract test fails when the source reads an `HR_PORTAL_*`
  or `SCREENER_*` setting the guide does not document; `verify` names any file the folder lacks.
- **The optional-parts log** is reviewed at each update.
- **Failing tests.** The `test` image runs the full suite and reports.
- **No frontend file is changed by this plan.**

## Steps

Execution order, checkpoints and the resume rules: `05-execution-stages.md` (the run-time audit
of step 1 moves after step 4, because it needs the image).

0. Approved (27 Oct 2026). Branch `feat/docker-v1` off `e530659`; commit and push the plan.
1. **Spike: measure first** (scratch Dockerfile in the session scratchpad; only the record is
   committed). (a) Run-time read/write audit: a Python audit hook on `open` during a scripted
   session (sign-in, a benchmark review, D63 and D64 on an invented JD, a v1 question, an entity
   comparison, then v2/v3 with Qwen) in a container with the owner's real `data/`; it confirms the
   folder's file list (04 §1). (b) aarch64 resolution of the three locks without emulation
   (`uv pip install --dry-run --python-platform …`). (c) `COPY --exclude` support. (d) Build time,
   image size, memory. Results amend `01-design-spec.md`; the owner re-confirms if scope changes.
2. **Repository changes** (each test shown to fail first): `.gitattributes`, `.dockerignore`,
   `.env` in `.gitignore`; `config.MODELS` and the three asset modules (04 §4); the download pins;
   the runtime lock change.
3. **The data folder module** `candidate_screener.datapack` (`create`, `verify`, `handover`,
   `purge`) with synthetic-data tests; then the owner's `create --with-workspace`.
4. **Dockerfile and compose** (`01-design-spec.md`): stages `base`, `deps`, `models`, `app`,
   `test`; services `app`, `tools`, `qwen` (profile `llm`), `test`; `compose.dev.yaml`;
   `.env.example`; `tests/test_docker_contract.py`.
5. **Verification, once, on the owner's machine** (below). Long runs announced first.
6. **Docs**: `docs/docker.md` (`02-handover-guide-spec.md`), README quick start, AGENTS.md
   (commands, "Where code goes", the rule that a new run-time file, setting or optional part
   updates the folder list, the guide or the log in the same change), `data/README.md`,
   `docs/data/cards/HR-operational-workspace.md`, `docs/project-overview.md` §7, D68,
   `docs/docker-optional-parts.md` (the log, moved), this plan's implementation record.
7. **Review**: an independent sub-agent reviews the diff against this plan; `uv run pytest`; the
   approved warm affected-check run (~70 min, started in the background per AGENTS testing rule
   9).

Sub-agents (AGENTS workflow 3): one each for steps 2, 3 and 4, one reviewer for step 7. The parent
runs steps 1 and 5. Phase 2 (the `llm` profile) follows the same verification rows for v2/v3.

## Verification: once, at this stage, on the owner's side

Nothing here runs when a container starts, and a receiver of a handover runs none of it (only
`verify`, seconds). After an update the owner repeats the rows for that kind of update: a
frontend change, V1 and V3; a lock or pin change, also V8; a data refresh, `verify` and V2b.

| # | What | Pass |
|---|---|---|
| V1 | `docker compose run --rm test` at the branch's base commit | the same pass/fail set as the native suite at that commit (today: the 17 frontend failures). A difference, e.g. a test relying on the host's HF cache, is explained in the record, never skipped or patched |
| V2 | Record fidelity (amd64): `python -m candidate_screener.inference --check` in the container with the owner's full `data/` mounted read-only | 14/14 items, as natively on 23 Oct 2026. A failing item stops the plan; the owner decides |
| V2b | Folder parity: invented JDs ranked through `portal_inference` against `a2-final-cvs`, D63 and D64: natively on `data/`, and in the container on `data/datapack/` | the same top 10 in the same order, every score within 1e-5 (5 dp). *Amended by the owner on 28 Oct 2026 after S6, from "identical at 6 dp": the 4e-6 gap comes from the app's C2 thread setting (05, S6)* |
| V3 | App smoke, scripted, invented documents only, no real text printed | sign-in with an account created in a scratch folder; a benchmark review opens with top 10 and source preview; an invented JD ranks to `ready` (D63); D64 completes; a v1 question returns an extractive answer with its source; an entity comparison returns mentions (phase 2: a v2 comparison and a v3 answer composed by Qwen, not the fallback) |
| V4 | Browser, by hand: the V3 flow at http://localhost:8503 | works on the owner's machine (other OSes later) |
| V5 | Write completeness: V3 with the image read-only and only the data folder, `logs/` and a tmpfs writable | passes |
| V6 | No run-time downloads: V3 with the container's network disabled | passes |
| V7 | Rebuild cost | (a) a change under `app/` rebuilds `app` in ≤ 1 min with no model download; (b) a lock change re-runs `deps` onwards, still with no model download (time recorded) |
| V8 | Cold build from an empty build cache | wall time, peak memory, bytes downloaded, image size recorded; they become the guide's requirements |
| V9 | Line endings: digests of a checkout made with `core.autocrlf=true` | identical to an LF checkout |
| V10 | arm64, **not run now** (no emulation, no arm64 machine). Registered now for the first arm64 run: V3, and V2b against the amd64 results | **no large changes**: every ranking score (L1, D64's fused score) within 1e-3 of amd64; the same top 10 in the same order, except candidates whose amd64 scores are within 1e-3 of each other; the same extractive answer text; the same entity mentions, except those scored within 0.01 of GLiNER's 0.5 threshold |
| V11 | Data folder commands | a missing, changed or stale file is reported by name; `create` keeps an existing workspace; `handover` output has no `processed/hr-portal/` and no caches; `purge` without `--yes` deletes nothing (unit tests, then once on the real folder) |

## Risks

| Risk | Mitigation |
|---|---|
| The folder drifts from the code (a fingerprint changes, a feature reads a new file) | `verify` after each update names the gap; `create` refreshes |
| A record does not reproduce in the container | V2; the owner decides |
| Docker Desktop's memory limit (D64 peaks at 4.6 GB, plus workers and Qwen) | V8 measures; the guide states the minimum and the OOM symptom (exit 137) |
| Bind-mount quirks (slow Windows paths, macOS file sharing, Linux owners) | run as the host user; keep the clone inside WSL2 on Windows; guide notes per OS |
| The folder holds anonymised real CV and JD text, the project's models and the workspace | passed privately; `handover` leaves the workspace out; display still redacts |
| Upstream model files change or disappear | pinned revisions; the build fails loudly |
| A breaking change is not recognised as breaking, and an old workspace meets new code | the developer states it in the commit; `verify` reports the workspace format where recorded; the app's own migrations are additive |
| Windows line endings | `.gitattributes`, V9 |

## Cost (estimate)

About 2 days of agent work: spike 0.5, repository changes 0.25, data folder module 0.5,
Dockerfile and compose 0.25-0.5, verification 0.5, docs 0.25-0.5. Machine time: image build an
estimated 20-40 min; V2 about 15 min; the warm check run about 70 min. Disk: image ~8-9 GB,
folder ~250 MB plus workspace and caches. Step 1 replaces the estimates with measurements.

## Implementation record

(Appended during implementation: commits, measurements, deviations, review.)
