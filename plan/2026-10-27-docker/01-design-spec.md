# 01 — Design specification

Draft v4, 27 Oct 2026, after the owner's third round of answers: native development unchanged;
`data/datapack/` is the Docker app's data folder (`04-app-data-folder.md`); pretrained model
weights are baked into the image. Items marked **(step 1)** are measured or confirmed by the
spike; the owner re-confirms if a measurement changes scope.

## 1. What runs

```
host browser ── 127.0.0.1:8503 ──► app container (runs as the host user; workdir /app = PROJECT_ROOT)
                                     ├─ Streamlit Hiring Desk (portal_ui), v1 chat by default
                                     ├─ ranking adapter thread ─► inference.Screener (D63; D64 = C2)
                                     ├─ QA worker      uv run --project runtime/transformer  (RoBERTa-SQuAD2)
                                     └─ entity worker  uv run --project runtime/gliner-medium (GLiNER-medium)

  /app/data      ◄── bind mount of ./data/datapack, read-write: project files, workspace, caches
  /app/logs      ◄── bind mount of ./logs
  /opt/models    ◄── in the image: RoBERTa-SQuAD2, GLiNER-medium, Qwen2.5-0.5B (SCREENER_MODELS_ROOT)
  /opt/hf        ◄── in the image: C2 in the Hugging Face cache layout (HF_HOME)

[profile llm]   qwen container, network_mode "service:app" ─► 127.0.0.1:8000 inside app's network
                namespace (Qwen2.5-0.5B from /opt/models, runtime/transformer)
[profile tools] the same image, for the data folder commands (create, verify, handover, purge)
```

The Qwen service shares the app container's network namespace because both the service and the
client insist on loopback (`llm/qwen.py`): no code change, one process per container. The guide
notes that restarting `app` requires restarting `qwen`.

## 2. Dockerfile stages

One `Dockerfile` at the repository root (multi-stage, BuildKit). Base images pinned by tag and
by their **multi-architecture index digest**, so one pin serves amd64 and arm64; uv pinned to the
version that wrote the locks (0.12.7 on this machine). Each person builds for their own machine
(owner), so an Apple Silicon Mac builds arm64 natively; nothing here builds arm64 under emulation
(owner: no emulation now).

| Stage | From | Copies in | Produces | Rebuilt when |
|---|---|---|---|---|
| `uv` | `ghcr.io/astral-sh/uv:0.12.7@sha256:…` | — | the uv binary | the uv pin changes |
| `base` | `python:3.12-slim-bookworm@sha256:…` | uv | OS packages (`ca-certificates`, `libgomp1` for OpenMP), fixed ENV | the base pin changes |
| `deps` | `base` | `pyproject.toml`, `uv.lock`, `.python-version`, `runtime/*/pyproject.toml`, `runtime/*/uv.lock` | `/app/.venv` (groups `app`, `ml`), `/app/runtime/transformer/.venv`, `/app/runtime/gliner-medium/.venv`; `uv sync --frozen`, uv cache as a cache mount | a lock or pyproject changes |
| `models` | `deps` | only `src/candidate_screener/__init__.py`, `config.py`, `llm/qa_assets.py`, `llm/qwen_assets.py`, `extraction/medium_assets.py` (self-contained: standard library plus `config`) | the asset commands run **by file path** with pinned revisions into `/opt/models`; C2 into `/opt/hf` with `huggingface_hub.snapshot_download`; downloads go through a cache mount, so a re-run does not download again | a pin or one of those five files changes; **never a frontend change** |
| `app` (default) | `models` | `output/`, `docs/data/`; `src/` without `app/`; then a `RUN` that writes GLiNER-medium's `runtime-versions.json` from the image's runtime (the package's own `runtime_versions()`) and compiles bytecode; then `src/candidate_screener/app/` and `.streamlit/` last | the image people run; the project installed editable | backend changes re-run the middle layers; a frontend change only the last ones (V7) |
| `test` | `deps` plus the `dev` group | the committed repository, including `data/README.md` and `data/vocab/` (tests read it), no other `data/` | `pytest` as its command | any source or test change |

Notes:

- **Editable install at `/app`.** `config.PROJECT_ROOT` comes from the package's location, so the
  source stays at `/app/src` and the venvs keep the same absolute paths in every stage (console
  scripts carry `#!/app/.venv/bin/python`). Never `--no-editable`. Everything under `/app` and
  `/opt` is world-readable, so any host user id can run it.
- **Excluding `app/` from the middle layer:** `COPY --exclude` if the pinned Dockerfile syntax
  supports it (step 1), else an explicit list of subpackages with a contract test that fails when
  a new one is not classified.
- **Build context** mirrors `.gitignore`: of `data/`, only `README.md` and `vocab/` enter it.
- **Labels.** `org.opencontainers.image.revision` from a `GIT_COMMIT` build arg (no `.git` in the
  build context).
- **arm64 without emulation (step 1):** each lock's packages are resolved for
  `aarch64-manylinux` with `uv pip install --dry-run --python-platform …`, which reads package
  metadata without running anything. If one does not resolve, the guide says amd64 only for that
  release and the optional-parts log records it.

## 3. The data folder in Docker

What it holds, its manifest, its commands and the upgrade flows: `04-app-data-folder.md`. Docker
specifics:

- compose bind-mounts `./data/datapack` at `/app/data`, **read-write**. The code's default paths
  (`data/processed/…`, `data/vocab/`) therefore find the folder with no data-path change, and the
  workspace and caches land in the host folder.
- **Nothing is verified at start-up.** `verify` runs by hand after an update. A missing file
  shows up as it would natively (an "unavailable" state, or an error in `logs/`), and the guide's
  troubleshooting points to `verify`.
- The `tools` service runs `python -m candidate_screener.datapack` in the same image, with
  `./data/datapack` mounted read-write at `/app/data` and the source mounted read-only at `/source`
  (`SCREENER_DATA_SOURCE` in `.env`: a real data directory, or a received handover copy). On the
  owner's machine `create` can equally run natively, which follows this worktree's symlinks.

## 4. Pins

| Artifact | Pin | Pinned today in |
|---|---|---|
| Python base image | `python:3.12-slim-bookworm@sha256:<index digest>` (chosen at step 4) | new |
| uv | 0.12.7 | new |
| Python packages | `uv.lock`, `runtime/*/uv.lock`, `--frozen` | committed |
| C2 | `Qwen/Qwen3-Reranker-0.6B` @ `e61197ed45024b0ed8a2d74b80b4d909f1255473` | `rerank/cross.py` |
| GLiNER-medium | `gliner-community/gliner_medium-v2.5` @ `88c3b98b57ad5e7d66fb209ed61c53f4b1fd05da` | `extraction/medium_assets.py` |
| GLiNER-medium encoder | the teammate's manifest revision if available, else chosen on approval | resolved on first download today |
| RoBERTa-SQuAD2 | `deepset/roberta-base-squad2` @ the teammate's manifest revision if available, else chosen on approval | resolved at download today |
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` | this machine's manifest |
| Data folder inputs | SHA-256 per file in `pack-manifest.json` | new |

## 5. Runtime contract (`app` stage and compose)

| Item | Value | Why |
|---|---|---|
| User | the host user: `user: "${HOST_UID:-1000}:${HOST_GID:-1000}"` | files written into the bind-mounted folder stay owned by its owner (Linux); Docker Desktop on macOS and Windows maps owners itself |
| Working directory | `/app` | equals `config.PROJECT_ROOT` |
| Command | `streamlit run src/candidate_screener/app/streamlit_app.py --server.address=0.0.0.0 --server.port=8503 --server.headless=true --browser.gatherUsageStats=false --server.fileWatcherType=none` (exec form, no wrapper) | binds inside the container; published on host loopback only |
| Port | 8503, published as `127.0.0.1:${HIRING_DESK_PORT:-8503}:8503` | local token sign-in is not SSO; no TLS |
| Fixed ENV | `SCREENER_MODELS_ROOT=/opt/models`, `HF_HOME=/opt/hf`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `UV_OFFLINE=1`, `HOME=/tmp`, `UV_CACHE_DIR=/tmp/uv-cache`, `MPLCONFIGDIR=/tmp/matplotlib`, `PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `PATH=/app/.venv/bin:$PATH` | no downloads at start-up (AGENTS); writes only to the data folder, `logs/` and `/tmp` |
| Owner-overridable (`.env`) | `HR_PORTAL_CHAT_VERSION` (default `v1`), `HIRING_DESK_PORT`, `HOST_UID`/`HOST_GID`, `SCREENER_DATA_SOURCE`; ranking adapter and pool keep `streamlit_app.py`'s defaults | same behaviour as a native start |
| PID 1 | compose `init: true` | workers start in their own sessions; something must reap them and forward SIGTERM |
| Healthcheck (kept, owner) | an HTTP request to Streamlit's `/_stcore/health` every 30 s; it verifies nothing else | `docker compose ps` shows "healthy" |
| Stop | `stop_grace_period: 30s` | the adapter thread and workers exit cleanly |

## 6. Compose

`compose.yaml` (sketch; final values after step 1):

```yaml
x-run: &run
  image: candidate-screener/hiring-desk:local
  user: "${HOST_UID:-1000}:${HOST_GID:-1000}"
  volumes:
    - ./data/datapack:/app/data
    - ./logs:/app/logs
services:
  app:
    <<: *run
    build: {context: ., target: app, args: {GIT_COMMIT: "${GIT_COMMIT:-unknown}"}}
    init: true
    ports: ["127.0.0.1:${HIRING_DESK_PORT:-8503}:8503"]
    environment: {HR_PORTAL_CHAT_VERSION: "${HR_PORTAL_CHAT_VERSION:-v1}"}
    healthcheck: {test: ["CMD", "python", "-c", "…/_stcore/health…"], interval: 30s, start_period: 90s}
    stop_grace_period: 30s
    restart: unless-stopped
  qwen:                         # phase 2
    <<: *run
    profiles: [llm]
    network_mode: "service:app"
    environment: {PYTHONPATH: /app/src}
    command: [uv, run, --project, runtime/transformer, --no-sync, python, -m,
              candidate_screener.llm.local_server, --manifest, /opt/models/chatbot/qwen/manifest.json]
  tools:
    <<: *run
    profiles: [tools]
    volumes:
      - ./data/datapack:/app/data
      - "${SCREENER_DATA_SOURCE:-./data}:/source:ro"
    entrypoint: [python, -m, candidate_screener.datapack]
  test:
    build: {context: ., target: test}
    profiles: [test]
    network_mode: none          # the suite needs no network and no data
```

- `.env.example` (committed) lists every setting a maintainer may change; `.env` is git-ignored;
  `data/datapack/` is already ignored (`data/*`).
- `compose.dev.yaml` (frontend checks in Docker before an update): bind-mounts `./src` and
  `./.streamlit` read-only over the image's copies and starts Streamlit with
  `--server.fileWatcherType=poll --server.runOnSave=true`. Day-to-day development stays native.
- An optional Hugging Face token for rate limits during the build: a build secret, never an ARG
  or ENV.

## 7. Tests

- `docker compose run --rm test`: `pytest` with all groups, no `data/`, no network. Its first run
  is V1. The `app` image does not depend on it (owner).
- `tests/test_config_models_root.py`: with `SCREENER_MODELS_ROOT` unset every path is unchanged
  (including `qwen_assets.MODEL_DIR`, today spelled out); absolute and relative roots; the three
  asset modules follow it.
- `tests/test_datapack.py` (synthetic): `create` follows symlinks and copies only listed files,
  replaces old inputs, clears caches and keeps an existing workspace; `--with-workspace` refuses
  to overwrite one without `--replace-workspace`; `verify` reports missing, changed and stale files;
  `handover` never contains `processed/hr-portal/` or caches; `purge` without `--yes` deletes
  nothing; nothing writes outside the target folder.
- `tests/test_docker_contract.py` (plain text checks, no Docker, no new dependency): every
  `HR_PORTAL_*` and `SCREENER_*` name read under `src/` appears in the guide's configuration
  table; ports are published on `127.0.0.1` only; the `app` stage sets `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `UV_OFFLINE=1` and `SCREENER_MODELS_ROOT`; `.dockerignore` excludes
  `data/*` except `README.md` and `vocab/`, and `.venv`, `**/.venv`, `logs`,
  `.streamlit/secrets.toml`, `.env`; `.gitattributes` marks `docs/data/**`, `output/**` and
  `data/vocab/**` as `-text`; the Dockerfile's model revisions equal the code's pins where the code
  has one (C2, GLiNER-medium).
- Each new test is shown to fail against a broken version (AGENTS testing rule 2).

## 8. Cross-OS hazards

| Hazard | Where | Mitigation | Checked by |
|---|---|---|---|
| CRLF checkout changes committed manifest bytes, hence digests and fingerprints | Windows (Git's default `core.autocrlf=true`) | `.gitattributes`: `docs/data/** -text`, `output/** -text`, `data/vocab/** -text`; `Dockerfile`, `*.toml`, `*.yaml` `eol=lf`; `verify` names fingerprint mismatches | V9, V11 |
| CPU architecture | Apple Silicon, Windows on ARM | built natively on each machine; lock resolution checked for aarch64 without emulation; the registered "no large changes" bar applies at the first arm64 run | step 1, V10 later |
| Docker VM memory | Docker Desktop on macOS and Windows | the guide's stated minimum (V8); exit 137 explained | V8 |
| File owners in the bind-mounted folder | Linux | run as the host user (`HOST_UID`/`HOST_GID`) | V5 |
| Bind-mount speed and sharing | Windows: slow from a `C:\` path, fast inside WSL2; macOS: the folder must be shared with Docker (default: under `/Users`) | the guide says where to keep the clone per OS | V4 (owner); others later |
| Symlinked `data/` subfolders (as in this worktree) | any | `create` follows symlinks natively; `tools` mounts a real data directory | V11 |
| Hidden prompt for the account token | Git Bash (mintty) on Windows | PowerShell or Windows Terminal | guide |
| Port 8503 in use | any | `HIRING_DESK_PORT` in `.env` | guide |
| Corporate proxy or TLS inspection | some networks | standard `HTTP(S)_PROXY` build args; adding a CA certificate | guide |
| No `.git` in the image | `checks --affected`, `inference --freeze` provenance | the image runs neither; checks run natively, on the owner's side | guide |
| Container clock is UTC | `reference_date` "today" for uploads | `TZ` documented | guide |

## 9. Security and data handling

- Nothing secret enters the build context or the image: `.dockerignore` excludes `data/*`
  (except the committed `README.md` and `vocab/`), `.streamlit/secrets.toml` and `.env`.
- The image holds code, environments and public pretrained weights, no Djinni text. The data
  folder holds Djinni-derived text (MIT, anonymised by the publishers), the project's trained
  models and this deployment's workspace. `handover` leaves the workspace and caches out; the
  folder travels privately (owner). Displayed text still passes `annotation.redact`; that code does
  not change.
- Local token sign-in is not SSO; the port is published on host loopback. Exposing it on a network
  needs TLS and an authentication decision, out of scope.
- No training, label reading or model download at start-up; no network needed at run time (V6).

## 10. Changes to existing code (small; native defaults unchanged)

1. `config.py`: `MODELS` = `SCREENER_MODELS_ROOT` (resolved against `PROJECT_ROOT` when relative),
   else `PROCESSED`.
2. `llm/qa_assets.py` (`ROOT`), `extraction/medium_assets.py` (`MODEL_DIR`), `llm/qwen_assets.py`
   (`MODEL_DIR`, today `Path(__file__).parents[3] / "data/processed/chatbot/qwen"`): built from
   `config.MODELS`; identical values when the variable is unset.
3. `llm/qa_assets.py`: `--revision`; `extraction/medium_assets.py`: `--encoder-revision` (owner: pin them).
4. `runtime/transformer/pyproject.toml`, `runtime/gliner-medium/pyproject.toml` and their locks:
   the PyTorch CPU index for torch on Linux only (owner: CPU build). Step 1 checks every place a runtime version
   is compared or recorded (`2.2.2+cpu` on linux x86_64).
5. No frontend file changes (`app/`, `.streamlit/`): the teammate's work is not touched.
6. If step 1 finds a write outside the data folder, `logs/` and `/tmp`, or a hidden download, it
   is reported to the owner, not patched quietly.

New: `candidate_screener/datapack.py`, `Dockerfile`, `.dockerignore`, `.gitattributes`,
`compose.yaml`, `compose.dev.yaml`, `.env.example`, `docs/docker.md`,
`docs/docker-optional-parts.md` (moved from this plan), `tests/test_config_models_root.py`,
`tests/test_datapack.py`, `tests/test_docker_contract.py`.

## 11. Estimates (replaced by step 1 and V8 measurements)

| | Estimate | Basis |
|---|---|---|
| Image build (cold) | 20-40 min | ~3 GB of packages and ~3.5 GB of model weights; nothing trained |
| Image size | ~8-9 GB | main env with torch CPU, two CPU-only runtimes, four pretrained models |
| Data folder | ~250 MB of inputs, plus the workspace and caches | `04-app-data-folder.md` §1 |
| Memory at run time | ~2-3 GB for D63; ~5-7 GB with D64, both chat workers and Qwen | `inference --check` 2.2 GB; D64 4.6 GB |
| Frontend-only rebuild | under 1 min | layer design (V7) |
