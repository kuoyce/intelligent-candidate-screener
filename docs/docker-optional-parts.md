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
| Chat v2/v3 with Qwen2.5-0.5B (compose profile `llm`) | **yes**: weights in the image from phase 1; the service with its own healthcheck in phase 2. v2 and v3 answers composed by Qwen are saved through the Hiring Desk's chat path in the container (28 Oct 2026, after the provenance-key fix) | ~1 GB image; one extra container, 3.3 GB resident at peak | v3 passes its quality gates and becomes the default, or the chat model changes |
| In-app account creation (frontend) | **after phase 1**, on its own branch (`feat/account-create-screen`): a create screen only when no account exists, no administrator role | ~0.5 day with tests | when an administrator role or network exposure is wanted |
| Qwen3.5-4B GGUF trial (D67) | no | +2.75 GB; needs a Linux llama.cpp build per architecture (code change in `llm/gguf_assets.py`) | the owner adopts the 4B model for v3 |
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
