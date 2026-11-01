# Docker optional parts log

Owner, 27 Oct 2026: "we might need to review this optional parts again during each subsequent
updates, please keep a log."

Review this log at **every update** of the application or the Docker setup (the release checklist
in [`docker.md`](docker.md) §10 points here): re-decide each part whose trigger has fired, and add
a row; never edit an old row. The log started in the Docker plan
([`plan/2026-10-27-docker/`](../plan/2026-10-27-docker/README.md), decision D68) and moved here on
28 Oct 2026, with its rows unchanged.

## Current state

| Part | In the Docker setup? | Size / cost | Review when |
|---|---|---|---|
| C2 reranker, D64 ("Refine the top 50") | **yes**, phase 1 | +1.2 GB image; minutes per new JD; peak RSS 4.6 GB | the reported system changes (D63/D64), or a host has < 8 GB for Docker |
| Chat v2 with Qwen2.5-0.5B (compose profile `llm`) | **yes**, for v2 only since 31 Oct 2026 (v3 uses the 4B): weights in the image from phase 1; the service with its own healthcheck in phase 2. v2 and v3 answers composed by Qwen are saved through the Hiring Desk's chat path in the container (28 Oct 2026, after the provenance-key fix) | ~1 GB image; one extra container, 3.3 GB resident at peak | v2 is retired, or the chat model changes |
| In-app account creation (frontend) | **yes**, first account only (S10, merged into `feat/docker-v1` on 28 Oct 2026): a create screen only when no account exists, no administrator role | ~0.5 day with tests | when an administrator role or network exposure is wanted |
| Qwen3.5-4B GGUF for chat v3 (D67, D69) | **yes, the default** (31 Oct 2026, D69), as a separate image: `candidate-screener/hiring-desk-llm4b:local`, Dockerfile target `llm4b` (Ubuntu 24.04, the published llama.cpp b11896 Linux build per architecture), the `llm4b` service of `compose.yaml` with no profile, so plain `docker compose build` and `up -d` include it; the default `app` image unchanged; v1 (`docker compose up -d app`) the rollback and the lighter setup ([`docker.md`](docker.md) §6). E1-E3 pass (`linux-x64`, native against Docker); E4 (Intel Mac) pending, not a gate; E5 (`linux-arm64` under emulation) passes: identical tokens to `linux-x64` | image 2.94 GB; `llm4b` peak 4.66 GB, about 11.5 GB with `app` running D64; v3 turns 39-57 s idle, most hit the 60 s budget during a ranking | v3 passes or fails its quality acceptance, the owner changes the default, a pin changes, or E4/E5/E6 report a difference |
| Research workbench image | no | dev tools plus research data in Docker | someone needs to run `--check`s or notebooks without a native setup |
| ONNX embedding backend (`HR_PORTAL_EMBEDDING_BACKEND=onnx`) | no (Docker uses torch) | — | only if a host cannot run the torch image |
| arm64 checks under emulation | done as far as emulation allows (S11, 28 Oct 2026); V10's numbers wait for a real Apple Silicon Mac | privileged `binfmt` registration on the owner's machine | a real arm64 machine is available |

## Entries

| Date | Part | Decision | By | Reason |
|---|---|---|---|---|
| 2026-10-27 | C2 / D64 | include, phase 1 | owner | part of the reported system; the UI offers it |
| 2026-10-27 | Qwen2.5-0.5B, chat v2/v3 | include, phase 2 | owner | v3 is in active development |
| 2026-10-27 | Qwen3.5-4B GGUF | exclude for now | owner | not selected; needs a code change for Linux |
| 2026-10-27 | Research workbench | exclude for now | owner | not selected |
| 2026-10-27 | Architectures | amd64 + arm64 | owner | target OS unknown; Apple Silicon should run natively |
| 2026-10-27 | Runtime torch | CPU build on Linux | owner | the CUDA libraries are unused on CPU and add several GB |
| 2026-10-27 | Pretrained model weights | in the image, downloaded at build | owner | no fine-tuning; keeps the data folder small and project-specific |
| 2026-10-27 | Qwen2.5-0.5B weights | in the image from phase 1 | agent, following the row above | the `llm` profile then needs no rebuild |
| 2026-10-27 | Account management | CLI now; in-app creation next update | owner | Docker does not depend on it; the frontend is still changing |
| 2026-10-27 | arm64 emulation | not now | owner | arm64 is checked when someone builds on Apple Silicon |
| 2026-10-27 | Account-creation screen | no accounts → create screen, else sign-in; no administrator, no setup code; own branch | owner | simplest useful flow; the port is loopback-only |
| 2026-10-27 | arm64 test under emulation | in v2 (S11): emulation registered for the test, removed after | owner ("continue with v2, test the Apple Silicon files"), read by the agent as permission | no arm64 machine is available |
| 2026-10-27 | arm64 emulation | installed by the owner; the agent never installs or uninstalls it and reminds the owner to uninstall when v2 is done | owner | owner's instruction |
| 2026-10-28 | Qwen2.5-0.5B, chat v2/v3 (`llm` profile) | keep; service verified, V3's v2/v3 rows **not passed** | agent (S9) | the `qwen` service starts, is healthy in 5.7 s and Qwen composes v2 and v3 answers in the container; the review store refuses to save any Qwen-composed answer (provenance key `output_token_budget` read as a credential), an app issue left to the owner |
| 2026-10-28 | Qwen2.5-0.5B, chat v2/v3 (`llm` profile) | keep; V3's v2/v3 rows **pass** | owner (the fix), agent (S9 rerun) | the owner renamed the provenance key to `output_max_new_tokens` in both composers; on the rebuilt image, v2 4/4 and v3 2/4 turns composed by Qwen were saved through `controller.chat` (the other two v3 turns: Qwen's wording rejected by the application, as designed); with `qwen` stopped every turn answered through the fallback |
| 2026-10-28 | arm64 test under emulation (S11) | stop; V10 waits for a real Apple Silicon Mac | owner | the arm64 image builds, starts, signs in, answers a v1 question and passes its tests apart from the known failures and four timing tests; under QEMU the models ran about 300 times slower (C2 651.6 s per pair), so no ranking or entity comparison finished |
| 2026-10-28 | First-account screen (S10) | merged into `feat/docker-v1` | owner | the guide's §4 and §7 describe it; further accounts still use the command |
| 2026-10-28 | Qwen3.5-4B GGUF (D67) | include, opt-in: a separate `llm4b` image on Ubuntu 24.04 through `compose.llm4b.yaml`, with packages pinned by version, against the dated snapshot `20261027T000000Z` on amd64 (on arm64 the live ports archive with the same pins, since the snapshot serves no `ubuntu-ports`; owner, 28 Oct 2026); `macos-arm64` pinned for native runs; own branch `feat/gguf-cross-platform`, not merged into `feat/docker-v1`; E4 run by the teammate and recorded, not a gate; E1-E3 and E5 must pass on the owner's machine; no decision-log amendment | owner (Q1-Q6 of [`plan/2026-10-28-gguf-cross-platform/`](../plan/2026-10-28-gguf-cross-platform/README.md)) | the `linux-arm64` release needs glibc 2.38 and does not start on the app's bookworm base, so a separate image avoids compiling and keeps the default image and `compose.yaml` unchanged. Results (G2b-G3): build 3 min 30 s, image 2.94 GB; healthy 8.3-12.7 s after `up`; `llm4b` 4.66 GB peak, `app` 6.85 GB with D64, about 11.5 GB together; idle v3 turns 38.7-57.1 s (native 33.9-52.9 s), during rankings three of four turns hit the 60 s budget and fell back; E1-E3 pass; E5 (arm64 under emulation) passes with identical tokens |
| 2026-10-31 | Qwen3.5-4B GGUF (D67), chat v3 | the default: `llm4b` moves into `compose.yaml` with no profile, `compose.llm4b.yaml` and `.env.example`'s `COMPOSE_FILE` line are removed, `app` defaults to `HR_PORTAL_CHAT_VERSION=v3`; the 0.5B `qwen` service stays in the `llm` profile for v2 only; v1 with `docker compose up -d app` is the rollback and the lighter setup (about 10 GB instead of 15 GB) | owner (D69, [`plan/2026-10-31-v3-default/`](../plan/2026-10-31-v3-default/README.md), approved as recommended) | one command should bring everything up: with `.env` at `v3` and the model behind an override file, plain `up -d` started only `app` and every v3 turn fell back. Not quality acceptance; no training, labels or ranking change |
