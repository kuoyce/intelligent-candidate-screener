# Running the Hiring Desk with Docker

This guide covers running and maintaining the containerised Hiring Desk on any operating system.
It does not describe the system itself (what it ranks, how, and how well): for that, read
[`project-overview.md`](project-overview.md), [`ownership/`](ownership/01_project_map.md) and
[`AGENTS.md`](../AGENTS.md). The plan behind this set-up is
[`plan/2026-10-27-docker/`](../plan/2026-10-27-docker/README.md), decision D68 in
[`decisions.md`](decisions.md). Since 31 Oct 2026 chat v3 with Qwen3.5-4B is the default, and plain
`docker compose up -d` starts the model service with the app (D69,
[`plan/2026-10-31-v3-default/`](../plan/2026-10-31-v3-default/README.md)); v1 is the rollback and
the lighter setup.

| Reader | Who | Sections |
|---|---|---|
| **Operator** | Receives the repository and a data folder and runs the app, on any OS. Runs no long checks; at most `verify`, which takes seconds | 1-9, 12, 14, 15 |
| **Developer** | Today the owner. Develops natively, decides when a change is settled, refreshes the image and the data folder, runs the checks on their own machine, hands over | all, especially 10, 11 and 13 |
| **New maintainer** | Takes over from the developer | start with 16 |

**Where the figures come from.** Every time, size and memory figure in this guide was measured on
the owner's machine on 27-28 Oct 2026: Windows with WSL2 (Ubuntu 24.04), 22 cores, 15.3 GiB of
memory, Docker Engine 27.5.1 and Compose 2.32.4 installed inside WSL2 (not Docker Desktop),
amd64. The records are the plan's stage log
([`05-execution-stages.md`](../plan/2026-10-27-docker/05-execution-stages.md), stages S3-S8) and
the S6 verification results (`logs/docker/s6-results.md` on that machine; `logs/` is
git-ignored). Figures without a date are from 27 Oct 2026 (stages S3-S6, named by their
verification row, such as V8); figures marked "28 Oct 2026" were taken while this guide was
written or in its fresh-clone walkthrough (S8). The figures for chat v3 with Qwen3.5-4B
([§6](#chat-v3-with-qwen35-4b-the-default)) come from a second plan,
[`plan/2026-10-28-gguf-cross-platform/`](../plan/2026-10-28-gguf-cross-platform/README.md) (its
implementation record, stages G2b-G3, 28 Oct 2026, the same machine). Your machine and network
will give other times. GB means 10^9 bytes.

**Not verified yet:** Docker Desktop on Windows or macOS; any arm64 machine (Apple Silicon); the
PowerShell forms of the commands (every command here was run in bash). Chat v2/v3 with the `llm`
profile was verified on Linux on 28 Oct 2026 by a scripted session, not yet in a browser
([§6](#chat-v2-the-llm-profile)). Chat v3 with Qwen3.5-4B (the `llm4b` service, then started
through a separate override file) was measured on Linux amd64 on 28 Oct 2026 by scripted sessions,
not in a browser, and its arm64 image only under
emulation ([§6](#chat-v3-with-qwen35-4b-the-default), [§15](#15-known-limitations)).

Contents: [1 What this is](#1-what-this-is) ·
[2 Why Docker](#2-why-docker) ·
[3 Requirements](#3-requirements) ·
[4 Quick start](#4-quick-start-operator) ·
[5 The data folder](#5-the-data-folder) ·
[6 Daily operation](#6-daily-operation-operator) ·
[7 Accounts](#7-accounts) ·
[8 Configuration reference](#8-configuration-reference) ·
[9 Backups](#9-backups-operator) ·
[10 Releasing an update](#10-releasing-an-update-developer) ·
[11 Tests](#11-tests) ·
[12 Troubleshooting](#12-troubleshooting) ·
[13 How the image and the folder are built](#13-how-the-image-and-the-folder-are-built) ·
[14 Security and privacy](#14-security-and-privacy) ·
[15 Known limitations](#15-known-limitations) ·
[16 First day](#16-first-day-for-a-new-maintainer)

## 1. What this is

The Docker set-up is one image plus one folder.

- **The image**, `candidate-screener/hiring-desk:local`, is built from the repository's
  [`Dockerfile`](../Dockerfile). It holds the code, the three Python environments (the main one
  and the two model runtimes, `runtime/transformer` and `runtime/gliner-medium`) and four
  pretrained models, downloaded at pinned revisions when the image is built: RoBERTa-SQuAD2 (v1
  chat answers), GLiNER-medium (entity comparisons), Qwen2.5-0.5B (chat v2) and
  Qwen3-Reranker-0.6B, called C2 (used by D64). Chat v3, the default, composes with Qwen3.5-4B,
  which runs from its own image, `candidate-screener/hiring-desk-llm4b:local`, in the `llm4b`
  service ([§6](#chat-v3-with-qwen35-4b-the-default)).
- **The data folder**, `data/datapack/`, is mounted into the container at `/app/data`. It holds
  the project's own files that the app reads (the A2 texts, the CSV-backed reviews, the
  role-family model and SBERT weak that the project trained, the fitted inference parts, the
  skill vocabulary), the **workspace** people create while using the app (accounts, saved
  reviews, JD drafts, mail captures) and caches.

```
host browser ── 127.0.0.1:8503 ──► app container (runs as the host user; /app is the project root)
                                     ├─ Streamlit Hiring Desk, v3 chat by default (v1 the rollback)
                                     ├─ ranking adapter thread ─► inference.Screener (D63; D64 adds C2)
                                     ├─ QA worker      uv run --project runtime/transformer   (RoBERTa-SQuAD2)
                                     └─ entity worker  uv run --project runtime/gliner-medium (GLiNER-medium)

  /app/data      ◄── ./data/datapack, read-write: project files, workspace, caches
  /app/logs      ◄── ./logs
  /opt/models    ◄── in the image: RoBERTa-SQuAD2, GLiNER-medium, Qwen2.5-0.5B (SCREENER_MODELS_ROOT)
  /opt/hf        ◄── in the image: C2, in the Hugging Face cache layout (HF_HOME)

llm4b            llm4b container (its own image; no profile) in app's network namespace
                 ─► 127.0.0.1:8001 facade ─► 127.0.0.1:8002 llama-server (Qwen3.5-4B GGUF; v3 only)
[profile llm]    qwen container in app's network namespace ─► 127.0.0.1:8000 (Qwen2.5-0.5B; v2 only)
[profile tools]  the same image, for the data folder commands (create, verify, handover, purge)
[profile test]   the test image: the test suite, with no data and no network
```

What it is **not**:

- **A public deployment.** There is no TLS, no single sign-on and no real mail. The port is
  published on the host's loopback address only.
- **The development set-up.** Development stays native (`uv`, the full `data/`). Docker and the
  data folder are refreshed only once a change is settled.
- **The research reproduction.** Nothing is trained, no label is read and no `--check` runs in
  the container. The checks run natively, on the owner's machine.

## 2. Why Docker

Model weights load on any operating system; what does not travel is the software that runs them,
when a package has no build for an OS. SBERT needs the main environment's torch 2.14
(`uv.lock`), which has builds for macOS arm64, Linux and Windows but none for Intel macOS, so
SBERT and C2 cannot run natively on an Intel Mac. In a container every host runs the same Linux
packages from the same locks, so SBERT and C2 run on an Intel Mac too, and the Hiring Desk
behaves the same everywhere. The price is an 8 GB image, built once on each machine.

## 3. Requirements

### Per operating system

| Host | Docker | Status |
|---|---|---|
| Linux, x86_64 | Docker Engine with the compose plugin (`docker compose`, Compose v2) | **verified** (Ubuntu 24.04 inside WSL2, Engine 27.5.1, Compose 2.32.4; 27-28 Oct 2026) |
| Windows 10/11 | Docker Desktop with the WSL2 back end, or Docker Engine inside a WSL2 distribution | Engine inside WSL2: **verified** (the owner's machine). Docker Desktop: expected to work, not verified |
| macOS, Intel | Docker Desktop | expected to work, not verified |
| macOS, Apple Silicon; Linux arm64 | Docker Desktop; Docker Engine | expected to work, not verified on a real machine. An arm64 image was built and run under emulation on Linux (28 Oct 2026): it builds (every lock installs, the four models download), starts, signs in and answers a v1 question, and the test suite shows only the known failures plus four timing tests emulation is too slow for. Rankings and entity mentions have not been compared with amd64 yet ([§15](#15-known-limitations)) |

**Windows.** Use Docker Desktop with the WSL2 back end, or Docker Engine inside a WSL2
distribution (the owner's set-up). Keep the clone inside the WSL2 file system (for example under
your Linux home folder) and run the commands from a WSL2 terminal: the data folder is
bind-mounted, and bind mounts from a `C:\` path are slow (a known Docker Desktop property; not
measured here). With WSL2, Docker's memory is the WSL2 virtual machine's. Set it in
`%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=15GB
```

(15 GB for the default, chat v3 with Qwen3.5-4B; `memory=10GB` is enough for the v1 setup,
[§3](#memory-disk-network-and-time).)

Then run `wsl --shutdown` in PowerShell and start Docker again. Create accounts from a terminal that
can hide what you type ([§7](#7-accounts)): with Docker Engine inside WSL2, a WSL2 terminal; with
Docker Desktop, PowerShell or a WSL2 terminal. Not Git Bash.

**macOS.** Install Docker Desktop and set its memory under Settings → Resources. The clone must sit
in a folder that Docker Desktop shares with containers (Settings → Resources → File sharing; your
home folder under `/Users` is shared by default).

**Linux.** Install Docker Engine with the compose plugin. The containers run as
`HOST_UID:HOST_GID` (default `1000:1000`), so the files they write into `data/datapack/` and
`logs/` stay yours. If `id -u` or `id -g` print something other than 1000, set both in `.env`
([§8](#8-configuration-reference)). Create `logs/` (and `data/datapack/`, if you will fill it with
`create`) before the first `docker compose run` or `up`. Otherwise Docker creates them owned by
root and the app cannot write there.

### Memory, disk, network and time

| Resource | Measured (27-28 Oct 2026) | Guidance, and how it was derived |
|---|---|---|
| Memory, building | The cold build peaked at 1.26 GB of anonymous memory; page cache filled the rest of the 10.3 GB the builder touched, and is reclaimable | Any machine that can run the app can build it. A build under a Docker Desktop memory limit was not measured |
| Memory, D63 only (sign-in, CSV reviews, ranking new JDs) | Ranking through the inference API in the container (`inference --check`, V2): 2.36 GB at peak | Give Docker **at least 4 GB**: the measured peak plus room for Streamlit, whose process holds the same ranking parts. The app with D63 alone was not measured |
| Memory, the v1 setup: all of phase 1 (D63, D64, v1 chat with both model workers; `docker compose up -d app`) | A scripted session in the container (V3): the app process 5.26 GB, the whole container 7.56 GB including page cache. One native process ranking two JDs with D63 and D64: 7.38 GB | Give Docker **at least 10 GB**: the container's 7.6 GB peak plus about a third, because one process alone reached 7.4 GB without the chat workers |
| Memory, chat v2 (Qwen2.5-0.5B, `llm` profile) | The `qwen` container (S9): the model server peaked at 3.26-3.36 GB resident; the container at 2.66-3.82 GB including page cache. The app container in chat-only sessions (no ranking): 2.66-4.26 GB | Give Docker **about 3.5 GB more** than for phase 1 when the `llm` profile runs: the server's measured peak, rounded up. A session with D64 and Qwen together was not measured |
| Memory, the default: chat v3 with Qwen3.5-4B (`app` and `llm4b`, plain `docker compose up -d`) | 28 Oct 2026 (plan 2026-10-28-gguf-cross-platform, G2b-G2c): the `llm4b` container 4.42 GB at rest and 4.66 GB at peak (`llama-server` 4.55 GB resident); the app container 51 MB at rest and 6.85 GB at peak while D64 ran beside v3 turns. Together about 11.5 GB. 31 Oct 2026 (plan 2026-10-31-v3-default, V3: plain `up -d`, four v3 turns, no ranking): peaks of 0.22 GB for `app` and 2.44 GB for `llm4b`, lower than the 4.66 GB of 28 Oct; the difference was not investigated, so the guidance keeps the higher figure | Give Docker **at least 15 GB** for the default: phase 1's 10 GB plus the `llm4b` container's 4.7 GB peak, rounded up. A machine that cannot give Docker 15 GB (for example one with 16 GB in total) uses the v1 setup instead, which needs about 10 GB ([§4](#4-quick-start-operator)). With D63 only and no ranking during chat, less suffices, but that was not measured |
| Disk | Image 7.98 GB. The Qwen3.5-4B image (`hiring-desk-llm4b`, built by default since 31 Oct 2026; not built in the v1 setup) 2.94 GB, plus its download cache mount (not measured) (28 Oct 2026). Build cache after a cold build 14.74 GB (the layers, plus a second copy of the weights and wheels in the cache mounts). Test image 4.59 GB. Data folder: 256.8 MB of inputs, plus the workspace (2.6 MB on the owner's machine) and caches (1.1 MB after the owner's browser session), both measured 28 Oct 2026 | **About 30 GB free** for the first build (`app` and `llm4b`), **35 GB** if you also build the test image: the sum of the measured sizes, rounded up (the `llm4b` download cache was not measured). It is an upper bound, because the image's layers can also appear in the build cache. Afterwards `docker builder prune` returns most of the build cache, at the price of downloading again at the next build |
| Network | The cold build downloaded 4,382 MiB (4.60 GB): base images, Debian packages, Python wheels and model weights | Needed **only to build**. The app ran every step of the smoke test with the container's network disabled (V6) |
| Time | Cold build: 3 min 23 s (V8, the measured run), and 5 min 28 s in an earlier run. The model downloads took 156 s of the measured run and 280 s of the earlier one. The `llm4b` image (amd64, 28 Oct 2026, built separately then): 3 min 30 s, of which 157 s for Ubuntu packages from the dated snapshot (about 450 kB/s) and 39 s for the runtime and model download | Your download speed decides it |

Docker versions used: Engine 27.5.1, Compose 2.32.4, buildx 0.20.0 (BuildKit 0.18.2 in the
default builder). The Dockerfile needs BuildKit features (`COPY --exclude`, symbolic `--chmod`,
optional build secrets) through its pinned syntax line, `docker/dockerfile:1@sha256:4edf897a…`,
which BuildKit pulls from Docker Hub at the first build. Older versions were not tested.

## 4. Quick start (operator)

You need Docker ([§3](#3-requirements)), git, and the data folder the developer passed to you
privately: a folder with `pack-manifest.json` at its top. The commands below are the same in
bash, zsh and PowerShell.

1. **Get the repository.** Not timed (it depends on your network).

   ```
   git clone <repository URL>
   cd <the clone's folder>
   git checkout <the branch or commit the developer named>
   ```

   On Windows, run these in a WSL2 terminal, inside the Linux file system. Line endings are
   handled: the repository's `.gitattributes` keeps every file whose digest matters byte-exact,
   even with Git for Windows' default `core.autocrlf=true` (34 of 34 digests identical, V9).
   `uv.lock`, the runtime locks and `.python-version` may get CRLF line endings there, which uv
   accepts.

2. **Put the data folder in place and create `logs/`.** Move or copy the folder you received to
   `data/datapack/`, with your file manager or a shell, so that `data/datapack/pack-manifest.json`
   exists. Then, in the clone:

   ```
   mkdir logs
   ```

3. **Settings (optional).** Copy the example settings and edit the copy:

   ```
   cp .env.example .env
   ```

   On Linux, set `HOST_UID` and `HOST_GID` if `id -u` and `id -g` are not 1000. Change
   `HIRING_DESK_PORT` if port 8503 is taken. Without a `.env`, the defaults shown in `.env.example`
   apply: chat v3 with Qwen3.5-4B, which needs about 15 GB of Docker memory ([§3](#3-requirements)).

   **The lighter v1 setup** (about 10 GB, for a machine that cannot give Docker 15 GB): set
   `HR_PORTAL_CHAT_VERSION=v1` in `.env`, then use `docker compose build app` in step 4 and
   `docker compose up -d app` in step 7. Only `app` is built and started; `llm4b` and its 2.74 GB
   model are not. v1 answers with the extractive model and entity comparisons.

4. **Build the image.**

   ```
   docker compose build
   ```

   It builds two images: `app`, and `llm4b` with Qwen3.5-4B for chat v3. The first build downloads
   the 2.74 GB model and the llama.cpp runtime for `llm4b`. The output ends with `app  Built` and
   `llm4b  Built` (31 Oct 2026, V3: 81 s for both with a warm cache).
   Measured separately before they were built together: `app` from an empty cache took 3 min 23 s
   (5 min 28 s in an earlier run) and downloaded 4.60 GB, image 7.98 GB; `llm4b` took 3 min 30 s,
   image 2.94 GB (28 Oct 2026). With nothing changed, a rebuild takes
   a few seconds (5.5 s). Build in the clone you run the app from: the image's name is shared by
   every clone on the machine ([§10](#10-releasing-an-update-developer)).

5. **Check the folder.**

   ```
   docker compose --profile tools run --rm tools verify
   ```

   The last line should read `OK: 24 inputs (256.8 MB) match the manifest and what the code
   expects` (about 2 s, 28 Oct 2026). A received copy also prints
   `workspace processed/hr-portal/: none`. Anything else: [§5](#what-verify-reports).

6. **Create your reviewer account** ([§7](#7-accounts) explains accounts). With no account yet,
   you can skip this step: the app then opens on a **Create the first account** form (step 8).
   For an account from the command (or any further account):

   ```
   docker compose run --rm app python -m candidate_screener.app.portal_auth --create-local-user reviewer --department benchmark
   ```

   Replace `reviewer` with your login ID. The command asks twice for an access token of 24 to 512
   characters, first `Choose an access token (24 to 512 characters; not displayed):`, then
   `Confirm access token:`. Nothing appears as you type; press Enter after each. It then prints
   `Local reviewer account created. This is local token authentication, not company SSO.` (about
   2 s, 28 Oct 2026). If the two entries differ (`error: Tokens differ`), the token is too short or
   too long, or the login ID exists (`error: Use a bounded login ID, token of 24 to 512 characters
   and explicit departments`, or `error: Existing login IDs cannot be overwritten`), nothing is
   created: run the command again. (The error wording is from the code and its tests; the
   walkthrough of 28 Oct 2026 saw the older traceback form.) On Windows, run it where the token can be hidden: with Docker Engine inside WSL2 (the verified
   set-up), a WSL2 terminal, for example in Windows Terminal; with Docker Desktop, any terminal
   where `docker` works (PowerShell or a WSL2 terminal). Not Git Bash.

7. **Start the app.** One command starts both services, `app` and `llm4b`:

   ```
   docker compose up -d --wait
   docker compose ps
   ```

   `--wait` returns once both are healthy. On 31 Oct 2026 (V3) plain `docker compose up -d --wait` reported both healthy after 18.8 s.
   Earlier measurements: `app` alone was healthy after about 7 s (7.6 s on 27 Oct 2026, 6.8 s and,
   with `--wait`, 6.2 s on 28 Oct 2026); with `llm4b` started beside it, both were healthy 7.5-12.7 s
   after `up -d` (28 Oct 2026). Without `--wait`, `docker compose ps` shows `Up … (health:
   starting)` for the first seconds and then `Up … (healthy)` for each. Wait for `llm4b` to be
   healthy before signing in: a v3 session started earlier answers through the marked fallback
   ([§6](#chat-v3-with-qwen35-4b-the-default)). In the v1 setup the command is
   `docker compose up -d app`. If `up` fails with `address already in use` or `port is already
   allocated`, something else holds the port: see [§12](#12-troubleshooting).

8. **Open http://localhost:8503 and sign in** with your login ID and token, on the page headed
   "Local reviewer sign-in". If you skipped step 6, the page is headed "Create the first account"
   instead: fill it in, press **Create the first account**, then sign in ([§7](#7-accounts)). If you changed `HIRING_DESK_PORT` in step 3, open
   `http://localhost:<that port>` instead; the app's log still prints `http://localhost:8503`, its
   port inside the container. Ranking a new JD (D63) took 35-43 s with empty caches and 14 s once
   they were filled. "Refine the top 50 with the reranker" (D64) took 94-121 s for 50 new
   candidate pairs; press **Refresh ranking status** to collect it (27 Oct 2026).

To stop the app: `docker compose stop`. The owner, who has the full `data/`, fills the folder in
step 2 with `uv run python -m candidate_screener.datapack create --with-workspace` instead
([§5](#5-the-data-folder)), with the native Hiring Desk stopped, so that its workspace is not
copied in the middle of a write.

## 5. The data folder

### What it holds

Exactly what the app read in the run-time audit (S4, 27 Oct 2026): 24 input files, 256.8 MB.
Nothing else under `data/` is read.

| Item | Path under `data/datapack/` | Size | Copied from |
|---|---|---:|---|
| CSV-backed benchmark reviews | `processed/phase2-rerank-output/a2-{cv-extraction,jd-extraction,x2-scores}.csv` | 0.25 MB | the committed `output/matching/` (identical bytes) |
| Source text, and the `a2-final-cvs` candidate pool | `processed/matching/a2-pairs-v2-539ddf32117c.parquet` | 0.6 MB | `data/` |
| A2 train-region CVs (background pool) and JDs | `processed/matching/a2-train-{cv,jd}-v2-20000-0.parquet` | 35 MB | `data/` |
| Role-family model, **trained by the project** | `processed/extraction/role-family/LATEST` and the `role-model-44b7fff54614.pkl` it names | 128 MB | `data/` |
| SBERT weak, **fine-tuned by the project** | `processed/matching/sbert-weak-3255167debe9/` (10 files) | 92 MB | `data/` |
| Inference parts (fitted) | `processed/inference/parts/LATEST` and the folder it names (4 files) | 0.1 MB | `data/` |
| Skill vocabulary | `vocab/skills.csv` | 1.3 MB | the committed `data/vocab/` |
| The folder's manifest | `pack-manifest.json` | — | written by `create`: each input's size, SHA-256 and origin, the fingerprints the code expects, the source commit and the date |

The app also writes into the folder while it runs:

| Written by the app | Path | Replaceable? |
|---|---|---|
| **Workspace**: accounts, saved reviews and chat history, JD drafts and revisions, ranking records, report and mail captures, entity cache | `processed/hr-portal/` (0700 folders, 0600 files) | **No.** Nothing can rebuild what people typed and saved: back it up ([§9](#9-backups-operator)). `create` keeps it, `handover` never copies it, `purge --yes` deletes it |
| Caches: SBERT embeddings and C2 scores | `processed/embeddings/`, `processed/rerank/cross/` | Yes. They are rebuilt on demand; the first ranking after they are emptied is slower. `create` empties them |

The inputs are replaceable too: `create` rebuilds them from a source, and the developer can always
make a new copy. Outside the folder, the app writes only to `logs/`, `/tmp` and `/dev/shm` (S4).

### The commands

The data folder commands are `python -m candidate_screener.datapack`
([`datapack.py`](../src/candidate_screener/datapack.py)). They run by hand, never when the app
starts. Natively they run in the main environment (`uv run …`; they only copy files). In Docker
they run through the `tools` service: the same image, with the folder at `/app/data` and the source
folder named by `SCREENER_DATA_SOURCE` in `.env` mounted read-only at `/source`.

| Command | What it does | Native form | Docker form |
|---|---|---|---|
| `create` | Fills the folder from a source (a full `data/`, following its symbolic links, or a received copy): replaces the old inputs, empties the caches, **keeps an existing workspace**, writes the manifest, then verifies the folder fully. Fills only a new or empty folder, a data folder (it has `pack-manifest.json`) or one holding nothing but a workspace and caches; it refuses anything else, such as a copy of `data/`, and names what it found. Run it with the app stopped | `uv run python -m candidate_screener.datapack create --source PATH` (default source: `data/`) | set `SCREENER_DATA_SOURCE` in `.env`, then `docker compose --profile tools run --rm tools create` |
| `create --with-workspace` | Also copies the source's `processed/hr-portal/` (the owner's local accounts and reviews). Refuses to overwrite a workspace already in the folder unless `--replace-workspace` is added. Stop the native Hiring Desk first, so its workspace is not copied in the middle of a write | `uv run python -m candidate_screener.datapack create --with-workspace` | `docker compose --profile tools run --rm tools create --with-workspace` |
| `verify` | Compares the folder with its manifest and with what the code expects; prints every problem and the workspace's format version. `--full` also compares every SHA-256 | `uv run python -m candidate_screener.datapack verify` | `docker compose --profile tools run --rm tools verify` |
| `handover DEST` | Copies the inputs and the manifest, **never the workspace or the caches**, into a new or empty folder, then verifies the copy. Refuses a destination inside the clone, where `git add -A` or the Docker build context would pick up the copy, except under the git-ignored `data/` (not `data/vocab/`). Also refuses to copy a folder that holds symbolic links or whose last `create` did not finish | `uv run python -m candidate_screener.datapack handover ../datapack-handover` (any new folder outside the clone) | not available: the container's writable mounts are all inside the clone; run it natively |
| `purge` | Lists what it would delete. With `--yes` it empties the folder, workspace included (for a breaking change) | `uv run python -m candidate_screener.datapack purge` | `docker compose --profile tools run --rm tools purge` |

What has been run on real folders: natively `create --with-workspace` (0.85 s), `verify --full`,
`handover` and `purge` without `--yes` (S5 and S6, 27 Oct 2026); in Docker `verify`,
`verify --full` and `purge` without `--yes` (1-2 s each, 28 Oct 2026). `create --source`,
`purge --yes` and every Docker `create` have unit tests only so far
([`tests/test_datapack.py`](../tests/test_datapack.py)). A Docker `create` needs a source without
symbolic links that point outside it (a worktree's `data/` with linked subfolders does not work;
run `create` natively there) and with `vocab/skills.csv` (a handover copy has it).

Exit status: 0 when clean; 1 when verification finds a problem (also at the end of `create` and
`handover`); 2 when refused (nothing was changed) or stopped by a file error (run `create` again).
Before it changes anything, `create` writes a provisional manifest, so a `create` that stops part
way (a full disk, a closed terminal) still leaves a data folder: `verify` says the last `create`
did not finish, `handover` refuses it, and the same `create` run again completes it and removes
what the stopped run left behind.

### Who runs what

`create` always does the same thing: it fills `data/datapack/` from a source folder that has the
project's files. Only the source differs. Someone with their own full `data/` (the owner) runs
`create` from it and never needs a handover. Someone who cloned the repository has an empty
`data/` (it is git-ignored, and the trained models come from runs on the owner's machine), so they
get the project's files from a copy made with `handover`, which leaves out the owner's accounts,
reviews and caches.

| Person | First time | Upgrade, not breaking | Upgrade, breaking |
|---|---|---|---|
| Has a full local `data/` (the owner) | `create --with-workspace` (the files plus the local accounts and reviews) | `create` (fresh files, empty caches; the Docker accounts and reviews are kept) | `purge --yes`, then `create --with-workspace` (or `create` to start empty) |
| Has no project data (an operator, a new maintainer) | put the received copy at `data/datapack/` ([§4](#4-quick-start-operator)), or `create` from it | receive a new copy, then `create` from it (your accounts and reviews are kept) | `purge --yes`, then `create` from the new copy |

Every upgrade also rebuilds the image and runs `verify` ([§10](#10-releasing-an-update-developer)).

### What `verify` reports

| Message | Meaning | What to do |
|---|---|---|
| `OK: 24 inputs (256.8 MB) match the manifest and what the code expects` | Clean | Nothing |
| `missing: PATH` | An input in the manifest is not in the folder | `create` again (from your source or a new copy) |
| `changed: PATH (N bytes; the manifest says M)`, or `… its SHA-256 differs …` | A file was modified: an editor, a copy tool, or a line-ending conversion on Windows | `create` again; on Windows, copy the folder with a tool that does not convert text |
| `unexpected: PATH` | A file outside the inputs, the workspace and the caches | Remove it (`create` leaves files it did not write in place) |
| `linked: PATH …` | The folder holds a symbolic link where it needs a copy | A linked file: `create` again replaces it with a copy. A linked folder: `create` refuses it, so first remove the link itself (not what it points to), then `create` again |
| `pack-manifest.json is provisional: the last create did not finish (run create again)` | A `create` stopped part way | Run the same `create` again |
| `the code expects PATH (…); the folder lacks it` | The code moved on: it now reads a file this folder does not have (a new input or a new fingerprint) | Refresh the folder from a source that has it; an operator needs a new copy from the developer |
| `stale: PATH is no longer read by the code` | An old input | `create` removes it |
| `stale: …/LATEST names X; the code's committed record names Y` | The folder's role model is not the one the code's record expects | Refresh from a source with the right model |
| `stale: …/parts.json records …: refit them for this folder` | The inference parts were fitted against another role model or vocabulary | Developer: refit natively (`inference --cache-parts`), then `create` |
| `stale: PATH differs from the committed …` | A copied review table or the vocabulary differs from the repository's | `create` again |
| `workspace: …/accounts.json has format N; the code reads M …` (or `workspace.sqlite` is newer than the code) | The workspace was written by another version of the app: a breaking change. Reported only for a file `verify` can read | Back up, stop the app, then `purge --yes` and `create`, or `create --with-workspace --replace-workspace` ([§10](#10-releasing-an-update-developer)) |
| `workspace: cannot read …/accounts.json (…): fix it or restore it from a backup` (or `…/workspace.sqlite`) | The file is damaged (for example a hand edit left invalid JSON), locked by a running app, or left by a write that was cut short. It is not a format change | **Do not purge.** Fix the file or restore the workspace from a backup ([§12](#12-troubleshooting), [§9](#9-backups-operator)) |
| `… has no pack-manifest.json: not a data folder (run create)` | The folder is empty or not a data folder. On macOS this also happens when the clone is not shared with Docker Desktop (the mount appears empty) | Put the folder in place, or `create`; on macOS check file sharing |

On Linux, if the app cannot write its workspace or logs, look at the owners of `data/datapack/`
and `logs/` ([§12](#12-troubleshooting)).

## 6. Daily operation (operator)

| Task | Command |
|---|---|
| Start (in the background) | `docker compose up -d` (`app` and `llm4b`; `--wait` to return once both are healthy). The v1 setup: `docker compose up -d app` |
| Stop, keeping the container | `docker compose stop` |
| Stop and remove the container and its network (the data folder and the image stay) | `docker compose down` |
| Restart | `docker compose restart app` |
| Status and health | `docker compose ps` (for `app`, `Up … (healthy)` when Streamlit answers; for `llm4b`, once the 4B can answer; `(health: starting)` for the first seconds after a start). It lists running containers only: a stopped app appears in `docker compose ps -a` |
| The app's output | `docker compose logs app`, or `docker compose logs -f --tail 100 app` to follow it (Ctrl+C stops following; the app keeps running) |

The app's healthcheck asks Streamlit's `/_stcore/health` every 30 s and verifies nothing else. The
`llm4b` and `qwen` services have their own (below); `tools`, which runs no server, disables it. Nothing is
checked when the container starts. The container has `restart: unless-stopped`: it comes back when
Docker starts again, unless you stopped it.

Logs on the host, in `logs/`, each created the first time its part runs (after the quick start,
`logs/` is still empty): `hiring-desk-adapter.log` (the ranking adapter; failures are logged
without document text) and `medium-worker-*.log` (the GLiNER-medium worker). The RoBERTa worker's
error output is discarded. Times in the app's alerts and in its logs are in UTC (the container's
clock); ranking uses the pool's fixed reference date, so it does not depend on the time zone.

What survives what:

| Event | Workspace (accounts, reviews) | Caches | Inputs | Rankings in progress |
|---|---|---|---|---|
| `restart`, `stop`/`start`, `down`/`up -d`, a host reboot | kept | kept | kept | lost from memory; **Refresh ranking status** reruns the job from its saved request |
| Image rebuild, then `up -d` | kept | kept | kept | as above |
| `create` (an update) | kept (replaced only with `--with-workspace --replace-workspace`) | emptied | replaced | — |
| `purge --yes` | **deleted** | deleted | deleted | — |

### Chat v2: the `llm` profile

Chat v2 (Qwen orders checked comparison relations, D65) needs the `qwen` service, Qwen2.5-0.5B on
the CPU, whose weights are already in the image. Since 31 Oct 2026 the profile serves v2 only: v3,
the default, uses Qwen3.5-4B in `llm4b` ([below](#chat-v3-with-qwen35-4b-the-default)), because
`compose.yaml` sets `SCREENER_QWEN_MANIFEST` on `app` and a selected 4B manifest never falls back
to the 0.5B model. Verified on 28 Oct 2026 (S9, a scripted session in the container, invented
documents), when v3 could still use the 0.5B: v2 and v3 answers composed by Qwen were saved and
returned through the Hiring Desk's chat path, and with `qwen` stopped both answered through their
fallback. The v3 figures below are from that run. Not yet tried by hand in a browser. The profile
also starts `app` and `llm4b`.

| Task | Command |
|---|---|
| Start (set `HR_PORTAL_CHAT_VERSION=v2` in `.env` first) | `docker compose --profile llm up -d` (`--wait` to return once all three services are healthy) |
| Status | `docker compose --profile llm ps`: `qwen` shows `(healthy)` once the model has loaded |
| Stop only Qwen | `docker compose --profile llm stop qwen` |
| Stop everything | `docker compose --profile llm down` (name the profile so `qwen` is included) |

The figures below for v3 with the 0.5B are historical (S9, 28 Oct 2026).

- **Healthcheck.** `qwen` asks its own server, `GET http://127.0.0.1:8000/v1/models`, every 30 s,
  with 120 s allowed for the model to load. The server answers only after loading, so `(healthy)`
  means Qwen can answer. Measured: the model loaded in 3-4 s and `qwen` was healthy 5.7 s after
  `up`; 15 s and 20 s while another image build used the CPU (S9).
- **Memory.** The model server peaked at 3.3 GB resident (S9); see [§3](#3-requirements).
- **Speed.** On an idle machine a turn composed by Qwen took 2-5 s (v2) and 3.4-5.3 s (v3), of
  which the model took 1.5-3.5 s. With another image build using the CPU: 3.4-17.7 s (v2) and
  4.7-26.1 s (v3); the first v3 turn includes starting the answer worker. Both versions give up
  after 30 s and answer through the fallback.
- **The restart caveat.** `qwen` uses the app container's network (both insist on loopback).
  `docker compose restart app` restarts `qwen` with it (S9). After the app restarts any other way
  (its restart policy, Docker restarting, a recreate; not tested), restart Qwen:
  `docker compose --profile llm restart qwen`. `qwen` has no restart policy of its own.
- **The app does not wait for Qwen.** It starts alone; while `qwen` is stopped or loading, v2 and
  v3 answer through their fallback (S9: every turn answered, in under 4 s).
- **Qwen or fallback?** Open **Answer details** under a v3 answer: `mode` is `qwen_prose` and
  `model` is the model that wrote it (`Qwen/Qwen3.5-4B` for the default; `Qwen/Qwen2.5-0.5B-Instruct`
  in the S9 run); `qwen_fallback`, with
  `provenance.fallback_reason` (`provider_unavailable` when the service is down), when the
  application did. A v2 answer's caption reads "Qwen selected validated comparison points" when
  Qwen composed it and "Qwen unavailable or rejected · application comparison" for the fallback.
  The fallback is also used when Qwen answers but its output fails the application's checks (S9
  saw `unsupported_paraphrase` and `qualification_omission` in v3): that is expected, not a fault.

### Chat v3 with Qwen3.5-4B (the default)

Chat v3 composes its prose with Qwen3.5-4B (the Q4_K_M GGUF of the D67 trial). It is the default
since 31 Oct 2026 (owner, D69, plan
[`2026-10-31-v3-default`](../plan/2026-10-31-v3-default/README.md)): the `llm4b` service is part of
`compose.yaml` with no profile, so `docker compose build` and `docker compose up -d` include it.
Being the default is not quality acceptance: v3 stays experimental, and v1
(`HR_PORTAL_CHAT_VERSION=v1`, `docker compose up -d app`) is the rollback. The 4B runs in its
**own image**, `candidate-screener/hiring-desk-llm4b:local` (Dockerfile target `llm4b`: Ubuntu
24.04, the published llama.cpp `b11896` build for the machine's architecture and the 2.74 GB model;
plan [`2026-10-28-gguf-cross-platform`](../plan/2026-10-28-gguf-cross-platform/README.md)); the
`app` image is unchanged by it. The figures below were measured on 28 Oct 2026 on Linux amd64 by
scripted sessions with invented documents (G2b-G3), not in a browser, when the service was still
started through a separate override file; its contents moved into `compose.yaml` unchanged.

| Task | Command |
|---|---|
| Build (part of `docker compose build`; the first build downloads the 2.74 GB model and the runtime; 3 min 30 s on its own on 28 Oct 2026) | `docker compose build llm4b` (only this image) |
| Start | `docker compose up -d` (`--wait` to return once both are healthy) |
| Status | `docker compose ps`: `llm4b` shows `(healthy)` once the model has loaded |
| Stop only the 4B | `docker compose stop llm4b` |
| Stop everything | `docker compose down` |

- **What it adds.** The `llm4b` service shares the app container's network (the facade insists on
  loopback): the facade on 127.0.0.1:8001 and `llama-server` on 127.0.0.1:8002, no published port,
  no data folder, no logs. `app` gets `SCREENER_QWEN_MANIFEST=/opt/qwen35-manifest/manifest.json`
  ([§8](#qwen-for-chat-v2v3-screener_qwen_)) and reads only the manifest's identity fields, from a
  read-only named volume. v2 is unaffected: only v3's composer reads the manifest, so v2 still uses
  `qwen` from the `llm` profile.
- **Healthcheck.** `llm4b` asks its facade, `GET http://127.0.0.1:8001/v1/models` with no proxy,
  every 30 s, with 300 s allowed for the facade's SHA-256 of the model and the model load. The
  facade answers 503 until `llama-server` serves the pinned model, so `(healthy)` means the 4B can
  answer. Measured: both services healthy 8.3 s after `up -d` (G2b), and 12.7 s and 7.5 s (G2c).
- **The manifest volume, and timing.** Docker fills the named volume
  `qwen35-manifest-e87f176-b11896` from the `llm4b` image when `llm4b` is created, seconds after
  `app`. A v3 session started in that gap uses the marked fallback until its next session, so wait
  for `(healthy)` (or use `--wait`) before signing in. The volume's name carries the pins (model
  revision `e87f176…`, llama.cpp `b11896`), so a pin change gets a fresh copy.
- **Memory.** `llm4b` 4.42 GB at rest and 4.66 GB at peak (`llama-server` 4.55 GB resident); `app`
  6.85 GB at peak with D64 running: about 11.5 GB together (G2c). Give Docker **at least 15 GB**
  ([§3](#3-requirements)).
- **Speed.** The 60 s turn budget for the 4B (D67) is unchanged. Idle (v3 prompt
  `hiring-desk-v3-summary-conclusion-7`), turns took 38.7, 55.9, 57.1 and 46.5 s in the container
  against 33.9-52.9 s natively (G2a): 4-6 s slower, with 3 s left under the budget. **While a ranking
  runs** in `app` (D63 and D64 for two invented JDs), the first turn took 58.8 s and the other three
  hit the budget (`deadline_exceeded`, 60.1 s each) and fell back, as designed; the rankings
  finished (D63 36.4 s and 2.1 s, D64 152.0 s and 128.0 s). On that machine the 4B fits the budget
  only when the CPU is otherwise idle. In both runs one of four factual turns was accepted and the
  rest fell back: a development smoke on invented sources, not a quality result.
- **Stopping.** SIGTERM stops the facade and its `llama-server` together: `stop llm4b` took 0.31 s
  (G2b) and 0.42 s (G2c). While `llm4b` is stopped or loading, v3 answers through its marked
  fallback.
- **The restart caveat.** As with `qwen`, `llm4b` uses the app container's network and has no
  restart policy of its own. After the app restarts or is recreated, restart the 4B:
  `docker compose restart llm4b` (not tested for `llm4b`).
- **The 4B or the fallback?** Open **Answer details** under a v3 answer: `mode` is `qwen_prose` and
  `model` is `Qwen/Qwen3.5-4B` when the 4B wrote it; `qwen_fallback`, with
  `provenance.fallback_reason`, when the application did (`deadline_exceeded` when the 60 s budget
  ran out, `unsupported_paraphrase` when the application's checks rejected the wording,
  `provider_unavailable` when the service is down). With the manifest selected, a missing 4B never
  silently selects the 0.5B model.
- **Checking equivalence.** The read-only probe `gguf_probe` is in the image. It sends only its six
  invented prompts and records the model identity, the chat template, the token ids and greedy
  completions. With both services running (31 s in the container on 28 Oct 2026, G3):

  ```
  docker compose exec llm4b python -m candidate_screener.llm.gguf_probe --manifest /opt/models/chatbot/qwen35/manifest.json --out /tmp/probe.json
  docker compose cp llm4b:/tmp/probe.json logs/probe-docker.json
  uv run python -m candidate_screener.llm.gguf_probe --compare REFERENCE.json logs/probe-docker.json --rule identical
  ```

  The container writes only under `/tmp`, hence the copy. The reference record goes first; use
  `--rule identical` for the same runtime build (E3) and `--rule near-tie` across builds (E4). The
  bar was registered before any run in
  [`03-equivalence.md`](../plan/2026-10-28-gguf-cross-platform/03-equivalence.md); on 28 Oct 2026,
  native against Docker on the owner's machine (`linux-x64`, both loading
  `libggml-cpu-alderlake.so` with 4 threads), E1, E2 and E3 passed. The `exec`/`cp` form above is
  the documented route; the recorded run was scripted.

## 7. Accounts

- **The first account, in the browser.** While no account exists, the app shows **Create the
  first account** instead of the sign-in form: login ID, departments (pre-filled `benchmark`,
  comma-separated), the access token twice (hidden, 24 to 512 characters) and the expiry in days
  (default and maximum 40). On success it returns to the sign-in form with "The first account was
  created. Sign in with its login ID and access token."; it does not sign you in. The form is shown
  only while `accounts.json` is missing or holds no users; a damaged file counts as having accounts,
  so the form never replaces it. If two people submit at once, one account is created and the other
  sees "An account already exists. Reload the page and sign in." There is no administrator role.
  **Accepted risk** (owner, 27 Oct 2026): while no account exists, whoever opens the page first
  creates the first account. The port is published on host loopback only, so that is someone on the
  same machine; before exposing the port on a network, create the first account by command (or add
  a one-time setup code to the app).
- **Further accounts, one per reviewer**, are created with the command in [§4](#4-quick-start-operator):

  ```
  docker compose run --rm app python -m candidate_screener.app.portal_auth --create-local-user LOGIN --department benchmark
  ```

  It runs in a one-off container (`run` publishes no port, so it works while the app is running)
  and writes `data/datapack/processed/hr-portal/accounts.json`. The running app reads that file at
  every sign-in, so there is nothing to restart.
- **The token.** The command and the form ask twice, hidden, for a token of 24 to 512 characters
  (the same limits as sign-in). It stores only a salted PBKDF2-SHA256 hash (310,000 iterations), the
  departments and the expiry; the token itself is never stored or printed. Keep it in a password
  manager.
- **The hidden prompt needs a real terminal.** On Windows with Docker Engine inside WSL2 (the
  verified set-up), use a WSL2 terminal, for example in Windows Terminal; with Docker Desktop, any
  terminal where `docker` works, PowerShell or a WSL2 terminal. Not Git Bash (mintty), which cannot
  show it. Without a terminal, Python warns `GetPassWarning: Can not control
  echo on the terminal` and `Password input may be echoed`: stop with Ctrl+C and run the command
  from a real terminal. (On 28 Oct 2026 the command was first run in the container with its input
  piped, which printed those warnings and created the account in a throw-away file. In the
  fresh-clone walkthrough the same day, a script typed the token at both prompts through a
  pseudo-terminal, in bash on the owner's WSL2 machine: nothing typed was shown, and the account was
  created. Typing by hand, and PowerShell, were not checked.)
- **Departments.** Every review the Hiring Desk opens today (the CSV-backed ones and new JDs)
  belongs to the department `benchmark`, so an account needs `--department benchmark` to open
  them. `--department` can be repeated; other names open nothing in the current app.
- **Expiry.** 40 days by default, which is also the maximum (`--hours`, default and maximum 960).
  A sign-in lasts until the account expires, the account changes, or the reviewer logs out; a
  browser reconnect can ask for the token again.
- **There is no command to renew, disable or reset an account yet.** Until there is:
  - *Disable a login*: in `accounts.json`, set that login's `"disabled"` to `true`.
  - *A lost token or an expired account*: remove that login's entry from `accounts.json`, then
    create it again with the same login ID. Saved reviews belong to the login ID, so they stay
    with it; a new login ID does not see the old one's reviews.

  Edit the file with a text editor, keep it valid JSON (`{"users": {…}, "version": 1}`), and copy
  it first ([§9](#9-backups-operator)); `verify` reports a file that is not valid JSON as one it
  cannot read. Any change to a login's entry signs that reviewer out.
- **The app's own hint.** The first-account form also shows the native command for further
  accounts (`uv run python -m …`). In Docker, use the `docker compose run` form above.
- **Accounts never travel.** The image contains none, and `handover` copies none: each operator
  creates their own. Rebuilding the image leaves accounts alone; `purge --yes` deletes them.

## 8. Configuration reference

Every setting the code under `src/` reads, plus the compose settings and the image's fixed
settings worth knowing. `tests/test_docker_contract.py` fails when the code reads an
`HR_PORTAL_*` or `SCREENER_*` name that these tables do not list.

**How settings reach the app.** Compose reads `.env` (copy it from
[`.env.example`](../.env.example); it is git-ignored) for the settings in the first table, and
passes only `HR_PORTAL_CHAT_VERSION` into the app container (plus `SCREENER_QWEN_MANIFEST`, which
[`compose.yaml`](../compose.yaml) sets on `app` for v3's Qwen3.5-4B). To pass another setting, add a
`compose.override.yaml` (or `.yml`) beside `compose.yaml`. Compose reads it automatically unless
you name files with `-f` or `COMPOSE_FILE`. Like `.env`, it is git-ignored and docker-ignored, so it stays out of
commits and images and can hold this machine's secrets. For example:

```yaml
services:
  app:
    environment:
      SCREENER_QWEN_DISABLED: "true"
```

Then run `docker compose up -d`. This route was not exercised during verification. "Safe to
change: no" means that only the default is built, tested and verified.

### Compose settings (`.env`)

| Name | Docker default | Effect | Safe to change |
|---|---|---|---|
| `HIRING_DESK_PORT` | `8503` | The host port, published on 127.0.0.1 only. The app's log still prints `http://localhost:8503` (the container's port) | Yes: any free port; then open `http://localhost:<port>` |
| `HR_PORTAL_CHAT_VERSION` | `v3` | Passed to the app. `v3`, the default (owner, 31 Oct 2026): experimental Qwen prose and JD requirement search (D66), with Qwen3.5-4B in the `llm4b` service that plain `docker compose up -d` starts; while it is not running, v3 answers through its marked fallback. `v1`: extractive answers (RoBERTa-SQuAD2) and entity comparisons (GLiNER-medium), the rollback and the lighter setup (`docker compose up -d app`, which starts `app` alone). `v2`: Qwen orders checked comparison relations (D65); it needs the `llm` profile (`docker compose --profile llm up -d`). Any other value behaves as `v1` | `v3`: yes; give Docker about 15 GB; v3 is experimental. `v1`: yes, with `docker compose up -d app`. `v2`: yes, with the `llm` profile running ([§6](#chat-v2-the-llm-profile)) |
| `HOST_UID` | `1000` | Linux: the user id the containers run as, so files written into `data/datapack/` and `logs/` stay yours. Docker Desktop on macOS and Windows maps owners itself | Yes: set it to `id -u` |
| `HOST_GID` | `1000` | Linux: the group id the containers run as | Yes: set it to `id -g` |
| `SCREENER_DATA_SOURCE` | `./data` | `tools` service only: the host folder mounted read-only at `/source`, which `create` copies from. Set it to a received copy, or to a full `data/` without symbolic links that point outside it | Yes |
| `GIT_COMMIT` | `unknown` | A build argument: the commit recorded in the image label `org.opencontainers.image.revision`. Set it per build ([§10](#10-releasing-an-update-developer)) | Yes |

### Read by the Hiring Desk (`HR_PORTAL_*`)

D63 is the default ranking of a new JD; D64 is chosen per JD in the app ("Refine the top 50 with
the reranker"). No setting switches between them.

| Name | Docker default | Effect | Safe to change |
|---|---|---|---|
| `HR_PORTAL_ACCOUNTS` | unset: `data/processed/hr-portal/accounts.json`, i.e. the data folder's | The accounts file the sign-in reads. The account command writes to its own `--accounts` option (same default), not to this setting | No: a path outside the data folder is lost when the container is recreated, and the command would need the same `--accounts` |
| `HR_PORTAL_DATA` | unset: `data/processed/phase2-rerank-output/` in the data folder | The folder of the three CSV review tables | No: `verify` covers the default location only |
| `HR_PORTAL_RANKING_ADAPTER` | unset; the entry point defaults it to `candidate_screener.app.portal_inference:adapter` | The ranking adapter for new JDs (`module:function`). An empty value leaves ranking unconfigured | No |
| `HR_PORTAL_CANDIDATE_POOL` | unset; the entry point defaults it to `a2-final-cvs` | The candidate pool new JDs are ranked against: the 600 A2 final-set CVs, the only pool the adapter serves | No |
| `HR_PORTAL_EMBEDDING_BACKEND` | unset: `torch` | The SBERT embedding back end. `onnx` needs `processed/inference/sbert-onnx/`, which the data folder does not hold, and disables D64 in the UI | No: keep `torch` |
| `HR_PORTAL_CHAT_VERSION` | `v3` | See the compose settings above | See above |

### Models and the data folder (`SCREENER_*`)

| Name | Docker default | Effect | Safe to change |
|---|---|---|---|
| `SCREENER_MODELS_ROOT` | `/opt/models` (set in the image) | Where `config.MODELS` points: the folder holding RoBERTa-SQuAD2, GLiNER-medium and Qwen2.5-0.5B. Unset (native runs), it is `data/processed` | No |
| `SCREENER_DATAPACK_SOURCE` | `/source` (`tools` service) | `datapack create`'s default source. Unset (native runs), it is `data/` | No in Docker: it must name the read-only mount (a contract test checks it). Natively, prefer `--source` |
| `SCREENER_DATAPACK_DEST` | `/app/data` (`tools` service) | The data folder every `datapack` command works on, and the only case in which the project's `data/` may be that folder (the container mounts the folder there). Unset (native runs), it is `data/datapack` | No in Docker (a contract test checks it). Natively, prefer `--dest` |

### Qwen for chat v2/v3 (`SCREENER_QWEN_*`)

In Docker, `compose.yaml` sets `SCREENER_QWEN_MANIFEST` on `app`, so v3 talks to the Qwen3.5-4B
service in `llm4b` (127.0.0.1:8001). With none of the others set, chat v2 takes its settings from
the image's Qwen manifest (`/opt/models/chatbot/qwen/manifest.json`) and talks to the `qwen`
service of the `llm` profile at `http://127.0.0.1:8000/v1`. Natively, with no `SCREENER_QWEN_*`
set, v3 uses the provisioned 4B manifest (`models/chatbot/qwen35/manifest.json`, written by
`gguf_assets`) when it exists, and the 0.5B settings otherwise. Setting `SCREENER_QWEN_BASE_URL` or `SCREENER_QWEN_MODEL` switches to
an explicitly configured service, which must then be described completely: nothing is filled in
from the local manifest. Both v2 and v3 cap the context at 4,096 tokens and force temperature 0;
v2 caps the output at 512 tokens and the timeout at 30 s, v3 the output at 256 tokens and the
timeout at 30 s (60 s for Qwen3.5-4B). The historical prototype and `app/chat_smoke.py` read these
names too.

| Name | Docker default | Effect | Safe to change |
|---|---|---|---|
| `SCREENER_QWEN_DISABLED` | unset (`false`) | `true` turns Qwen off; v2/v3 then answer through their explicit fallbacks | Yes (through an override file) |
| `SCREENER_QWEN_BASE_URL` | unset; from the manifest: `http://127.0.0.1:8000/v1` | The model service's address. The client refuses a non-loopback address unless `SCREENER_QWEN_ALLOW_REMOTE` is `true` and the address is HTTPS | No |
| `SCREENER_QWEN_MODEL` | unset; from the manifest: `Qwen/Qwen2.5-0.5B-Instruct` | The model's identity, checked against the service | No |
| `SCREENER_QWEN_REVISION` | unset; from the manifest: `7ae557604adf67be50417f59c2c2f167def9a775` | The model's pinned revision | No |
| `SCREENER_QWEN_SHA256` | unset; from the manifest: the checkpoint's digest | The expected checkpoint digest | No |
| `SCREENER_QWEN_TOKENIZER` | unset; from the local settings: `service` | `service` counts tokens with the service's own tokenizer; otherwise a local tokenizer folder | No |
| `SCREENER_QWEN_RUNTIME` | unset; from the local settings: `local Transformers service` | The runtime label recorded with answers | No |
| `SCREENER_QWEN_QUANTIZATION` | unset; from the manifest: `none` | The quantisation label | No |
| `SCREENER_QWEN_CONTEXT_TOKENS` | unset; from the manifest: `4096` | The context limit (v2/v3 cap it at 4,096) | No |
| `SCREENER_QWEN_OUTPUT_TOKENS` | unset; from the manifest: `256` | The output limit (capped as above) | No |
| `SCREENER_QWEN_TIMEOUT` | unset; from the local settings: `90` | Seconds per model call (capped as above) | No |
| `SCREENER_QWEN_TEMPERATURE` | unset; from the local settings: `0` | Sampling temperature (v2/v3 force 0) | No: no effect |
| `SCREENER_QWEN_JSON_MODE` | unset; from the local settings: `false` | Asks the service for JSON output | No |
| `SCREENER_QWEN_ENABLE_THINKING` | unset | Qwen's thinking mode, when a service supports it (the Qwen3.5-4B service sets `false`) | No |
| `SCREENER_QWEN_ALLOW_REMOTE` | unset (`false`) | Allows a non-loopback HTTPS service: document text would leave the machine | No: it changes the data boundary; an owner decision |
| `SCREENER_QWEN_API_KEY` | unset | A bearer token for an explicitly configured service | Only with a remote service; never put it in the image or a committed file |
| `SCREENER_QWEN_MANIFEST` | `/opt/qwen35-manifest/manifest.json`, set on `app` by [`compose.yaml`](../compose.yaml) | Selects the Qwen3.5-4B GGUF service's manifest for v3 (D67); v2 does not read it. The `llm4b` image holds the service and the model; `app` reads a copy of the manifest from a read-only named volume ([§6](#chat-v3-with-qwen35-4b-the-default)). A missing selected manifest is unavailable and never selects the 0.5B model. Natively, unset means: the provisioned 4B manifest if present, else the 0.5B settings | No |

### Read only by tools outside the Hiring Desk

The historical free-form prototype (`app/legacy_streamlit_app.py`) and the chat smoke tool
(`app/chat_smoke.py`) are in the image but are not run by it.

| Name | Docker default | Effect | Safe to change |
|---|---|---|---|
| `SCREENER_QA_MANIFEST` | unset | The prototype's extractive QA manifest | Not used in Docker |
| `SCREENER_CE_MANIFEST` | unset | The prototype's and smoke tool's cross-encoder (R2) manifest. R2 checkpoints are not part of the Docker set-up | Not used in Docker |
| `SCREENER_CE_SERVICE_URL` | unset | Their cross-encoder service address | Not used in Docker |

### Fixed in the image

| Name | Value | Effect | Safe to change |
|---|---|---|---|
| `ORT_DISABLE_TELEMETRY` | `1` | onnxruntime, which GLiNER imports in both model runtimes, writes no telemetry files and contacts no Microsoft endpoint (S4: 16 telemetry file operations without it, 0 with it) | No |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `1` | No Hugging Face download at run time: every model is in the image | No |
| `UV_OFFLINE` | `1` | uv fetches no package when it starts the model workers | No |
| `HF_HOME` | `/opt/hf` | Where C2 is found, in the Hugging Face cache layout | No |
| `HOME`, `UV_CACHE_DIR`, `MPLCONFIGDIR` | `/tmp`, `/tmp/uv-cache`, `/tmp/matplotlib` | Writable places for any user id | No |
| `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED` | `1` | No bytecode written at run time (it is compiled at build time); unbuffered output | No |
| `PATH` | `/app/.venv/bin` first | The main environment's `python` and `streamlit` | No |
| `TZ` | not set (UTC) | Times shown by the app and in logs are UTC | Not tested |

The build arguments that pin the base images and the models are listed in
[§13](#13-how-the-image-and-the-folder-are-built); changing one is a release
([§10](#10-releasing-an-update-developer)). Streamlit's command-line flags in the image turn off
usage statistics, the external-IP lookup and the file watcher.

## 9. Backups (operator)

Back up the workspace, `data/datapack/processed/hr-portal/`, regularly and **before every
upgrade**. The rest of the folder can be rebuilt; the workspace cannot. Stop the app first, so
that no review is half-written. Run the commands from the clone; they are the same in bash, zsh
and PowerShell (Windows 10 and later ship `tar`). They write the archive to a `backups` folder
beside the clone, so it can never be committed; create that folder once, and put the date in the
file name yourself:

```
mkdir ../backups
docker compose stop app
tar -czf ../backups/hr-portal-2026-10-28.tar.gz -C data/datapack/processed hr-portal
docker compose start app
```

Keep the archive private: it holds the account hashes, the saved reviews and redacted document
text. To restore, move the current workspace out of the data folder, into `../backups/` beside the
archives, and unpack the archive in its place. (A copy left inside the data folder would be
reported by `verify` as unexpected files and deleted by `purge --yes`.)

```
docker compose stop app
mv data/datapack/processed/hr-portal ../backups/hr-portal-before-restore-2026-10-28
tar -xzf ../backups/hr-portal-2026-10-28.tar.gz -C data/datapack/processed
docker compose start app
```

The `mkdir`, `tar` and `mv` lines were checked in bash on a synthetic folder (28 Oct 2026): the
restored workspace was identical, with its private permissions (0700, 0600), and the replaced one
was in `../backups/`. The PowerShell forms were not run.

## 10. Releasing an update (developer)

**When.** Once a change is settled in native development. Docker and the data folder are never
the place where a change is tried first.

**Breaking or not.** A change is breaking when the old workspace cannot be used by the new code (a
new `accounts.json` format, or a workspace database the app cannot migrate forward), or when you
decide so. Say which in the commit message and the plan record; `verify` reports a workspace format
the code does not read.

**Not breaking:**

1. Back up the workspace ([§9](#9-backups-operator)).
2. Get the release: `git pull` (or check out the release commit).
3. Rebuild the image, recording the commit, in the clone you run the app from:
   - bash, zsh: `GIT_COMMIT=$(git rev-parse HEAD) docker compose build`
   - PowerShell: `$env:GIT_COMMIT = git rev-parse HEAD; docker compose build`

   The image names `candidate-screener/hiring-desk:local` and `:test` are shared by every clone and
   worktree on the machine, so `up` or `tools` in one checkout runs whichever image was built last,
   in any of them. Rebuild in the checkout you run from before `up` or `tools`.
4. Stop the app: `docker compose stop app`. `create` replaces the files the app reads and empties
   its caches, so the app must not be running.
5. Refresh the folder: `uv run python -m candidate_screener.datapack create` (the owner, natively:
   new inputs, empty caches, the Docker workspace kept), or, for an operator, `create` from the new
   copy ([§5](#5-the-data-folder)).
6. `verify` (inside `create` already; run it again after any manual step).
7. `docker compose up -d`, which recreates the container from the new image and starts it.
8. Remove the old image. A rebuild moves the tag to the new image and leaves the old one untagged
   (`<none>`, about 8 GB): `docker image prune` removes untagged images (all of them on the
   machine, not only this project's), or `docker rmi IMAGE_ID` removes one. (`docker compose down
   --rmi local` does not: it skips images with a custom tag, and this one has one.)

**Breaking:** as above, but in step 5, with the app stopped, run `purge` (it lists), then
`purge --yes`, then `create --with-workspace` (the owner, copying the local workspace again, with
the native Hiring Desk stopped) or `create` (to start empty); an operator runs `create` from the new
copy. Replacing only the workspace, `create --with-workspace --replace-workspace`, likewise runs
between `docker compose stop app` and `docker compose up -d`. Accounts must then be created again
unless the workspace was copied.

**Handing the folder over.** `uv run python -m candidate_screener.datapack handover DEST` (DEST: a
new or empty folder outside the clone; inside the clone it accepts only the git-ignored `data/`,
not `data/vocab/`) copies the inputs and the manifest, never the workspace or the caches, and
verifies the copy. Pass it privately ([§14](#14-security-and-privacy)).

**Checklist for each change.**

- [ ] A new Python package: `uv add` (or add it to a runtime project), commit the lock. The image
  re-runs the `deps` stage; the models stage reruns from its cache with nothing downloaded (70.8 s
  for a lock change, 27 Oct 2026).
- [ ] A new setting (`HR_PORTAL_*` or `SCREENER_*`): add it to [§8](#8-configuration-reference)
  (the contract test fails otherwise); add it to `.env.example` and `compose.yaml` if operators
  should set it.
- [ ] A new file the app reads: add it to the folder's list (`expected()` in
  [`datapack.py`](../src/candidate_screener/datapack.py)), run `create`, then `verify`.
- [ ] A new pretrained model: a pinned download in the Dockerfile's `models` stage, at the
  revision the code pins (a contract test for the pin, as for C2).
- [ ] **Review [`docker-optional-parts.md`](docker-optional-parts.md)**: re-decide each part whose
  trigger has fired, and add a row.
- [ ] Remove the old image (step 8 above).

**Checks on the developer's machine**, for the kind of update (the plan's verification rows;
never at each start, and never on an operator's machine):

| Kind of update | Checks |
|---|---|
| Frontend (`src/candidate_screener/app/`, `.streamlit/`) | The tests in the container ([§11](#11-tests)): the same failures as natively. The app smoke (V3) by hand in the browser, with invented documents only: sign in; open a CSV review (top 10 and a source preview); publish an invented JD and rank it (D63 `ready`); refine it with D64; ask a v1 question with an extractable answer (for example "What degree does this role require?" on a JD that names a degree); compare two candidates (entity mentions). A frontend-only rebuild took 0.70 s (V7) |
| A lock or a pin | The above, plus a cold build in a throw-away builder, so the default build cache survives: `docker buildx create --name hd-cold --driver docker-container --bootstrap`, then `docker buildx build --builder hd-cold --no-cache --progress=plain --target app .`, then `docker buildx rm hd-cold` (and `docker rmi moby/buildkit:buildx-stable-1`). Record its time, download and peak memory (V8: 202.6 s, 4.60 GB) |
| A data refresh | `verify`, and the parity check (V2b): rank the same invented JDs natively on `data/` and in Docker on `data/datapack/`, D63 and D64. Pass: the same top 10 in the same order, every score within 1e-5 |

The S6 scripts that drove the smoke and parity checks lived in a session scratch folder and are not
in the repository; until they are, these checks are by hand.

**Trying a frontend change in Docker before an update.**
`docker compose -f compose.yaml -f compose.dev.yaml up` mounts the host's `src/` and `.streamlit/`
read-only over the image's copies and reruns the page on save (Streamlit polls for changes); stop
it with Ctrl+C. A lock, pin or environment change still needs `docker compose build`. Not run
during verification.

## 11. Tests

```
docker compose --profile test build test
docker compose --profile test run --rm test
```

The test image is the `deps` stage plus the `dev` group and the repository as `.dockerignore`
filters it: of `data/`, only `README.md` and `vocab/`. It runs with no network. It reports; nothing
depends on it, and the `app` image is built without it. Its name,
`candidate-screener/hiring-desk:test`, is shared by every clone on the machine, so build it in the
checkout you test.

At `f8dc77b` (27 Oct 2026) it ended with `17 failed, 2207 passed, 51 skipped, 1 warning in
98.03s`; natively at `4e8fb6d` the same code gave `17 failed, 2220 passed, 38 skipped`. The
differences are explained: 12 tests skip in the container because they need `data/`, and 1 because
it needs the host's Hugging Face cache. The 17 failures are the known frontend failures (the
frontend is work in progress), first measured at `e530659`, the branch's base (`17 failed, 2105
passed, 38 skipped`), with the same ids in every run since:

<details>
<summary>The 17 known failures</summary>

```
tests/test_portal_extractive.py::test_focused_answer_is_actual_span_with_source_context_and_natural_question
tests/test_portal_extractive.py::test_qa_abstention_is_explicit_source_only_never_a_model_span[not_found]
tests/test_portal_extractive.py::test_narrow_spans_never_hide_negation_expiry_or_ongoing_study  (3 cases)
tests/test_portal_extractive.py::test_long_complete_context_is_available_to_qa_without_clipping_status
tests/test_portal_extractive.py::test_reviewer_jd_requirement_keeps_negation_when_qa_returns_only_degree
tests/test_portal_extractive.py::test_reviewer_jd_framing_does_not_become_a_literal_retrieval_constraint  (2 cases)
tests/test_portal_extractive.py::test_reviewer_adjacent_qualifiers_remain_visible_in_fact_text_and_jd_requirement  (2 cases)
tests/test_portal_ui.py::test_chat_and_dropdown_share_saved_review_context
tests/test_portal_ui.py::test_dropdown_selection_is_used_by_chat_followup
tests/test_portal_ui.py::test_pending_followup_is_not_rebound_after_dropdown_change
tests/test_portal_v1_ui.py::test_compare_action_uses_extractive_table_and_never_qwen
tests/test_portal_v1_ui.py::test_comparison_table_keeps_qualifying_context_visible  (2 cases)
```

</details>

Any other failure is new. To compare with a native run, keep both logs and compare their
`FAILED` lines (AGENTS testing rule 1): `uv run pytest > logs/pytest.log 2>&1`, then
`grep -n FAILED logs/pytest.log` (PowerShell: `Select-String FAILED logs/pytest.log`).

The suite must run on a fresh clone with no `data/`: tests that need data skip with a stated
reason. The long `--check` runs are not part of this image; they run natively on the owner's
machine (AGENTS testing rules 3-6).

## 12. Troubleshooting

**Memory: the container exits with code 137, or restarts during D64.** It ran out of memory.
`docker compose ps -a` shows `Exited (137)` (the restart policy may already have restarted it, and
a D64 in progress is lost). Give Docker more memory ([§3](#3-requirements)): 15 GB for the default
(chat v3 with Qwen3.5-4B), 10 GB for D64 with chat in the v1 setup. D64 is the heaviest step; run
one at a time. If Docker cannot be given 15 GB, use the v1 setup ([§4](#4-quick-start-operator)).

**Disk full.** `docker system df` shows what uses the space. `docker builder prune` removes the
build cache (up to 14.7 GB after a cold build; the next build downloads again);
`docker builder prune --filter type=exec.cachemount` removes only the cache mounts (the uv wheels
and the second copy of the model downloads). `docker image prune` removes untagged old images
(about 8 GB each).

**Something in the app is "unavailable", or a page shows an error after an update.** Run `verify`
([§5](#what-verify-reports)). Typical causes: a missing, changed or stale input; on Windows, a
folder copied with a tool that converted line endings; on Linux, `data/datapack/` or `logs/`
owned by root because Docker created them (fix: `sudo chown -R "$(id -u):$(id -g)" data/datapack
logs`, bash or zsh); on macOS, a clone outside Docker Desktop's shared folders.

**`verify` cannot read a workspace file** (`workspace: cannot read …: fix it or restore it from a
backup`). The workspace is the one part of the folder that nothing can rebuild, so this is never a
reason to `purge`. The reason in brackets says what is wrong:

- `accounts.json`, a JSON error with its line and column: usually a hand edit ([§7](#7-accounts)),
  for example a comma after the last entry. Correct it in a text editor, or put back the copy you
  made before editing.
- `workspace.sqlite`, `database is locked`: the app is writing it. Stop the app
  (`docker compose stop app`) and run `verify` again.
- `workspace.sqlite`, `an interrupted write left workspace.sqlite-journal beside it`: a write was cut
  short (a crash, or a copy taken while the app was writing). SQLite undoes it the next time a
  program opens the database for writing, which the app does when a reviewer signs in; `verify`
  only reads. Start the app, sign in once, stop it, and run `verify` again. Never delete the
  `-journal` file: the undo needs it.
- Anything else, or if that does not help: restore the workspace from a backup
  ([§9](#9-backups-operator)).

**The build fails while downloading.** The output names the step. `[models …]` downloads the
weights from Hugging Face; finished downloads stay in the cache mount, so a second
`docker compose build` does not download them again. Anonymous pulls from Docker Hub and requests
to Hugging Face are rate-limited; `docker login` helps for the former. For Hugging Face, use a
read token as a **build secret**, never a build argument or an environment variable. Save the
token in a file
**outside the clone** (inside it, the file would enter the build context, and the test image
copies the whole context), then build with plain `docker build`, which accepts the secret:

```
docker build --secret "id=hf_token,src=../hf_token.txt" --target app --tag candidate-screener/hiring-desk:local .
```

(The quotes keep PowerShell from reading the comma as a list separator.)

The secret is mounted only into the download step and is in no layer. This path was not exercised
during verification.

**Behind a proxy, or a TLS-inspecting firewall.** Docker passes `HTTP_PROXY`, `HTTPS_PROXY` and
`NO_PROXY` to builds from the Docker client's proxy configuration (`~/.docker/config.json`, or
Docker Desktop's proxy settings), and `docker compose build --build-arg HTTPS_PROXY=…` sets them
for one build. A firewall that re-signs TLS with its own certificate authority needs that
certificate inside the image for apt, uv and the Hugging Face downloads: a Dockerfile change, not
made. The app itself needs no network. Not tested.

**Port already in use.** `docker compose up -d` fails with `… listen tcp4 127.0.0.1:8503: bind:
address already in use` when another program, such as a native Hiring Desk, holds the port (seen
28 Oct 2026), or with `Bind for 127.0.0.1:8503 failed: port is already allocated` when another
container publishes it. Set `HIRING_DESK_PORT=8504` (or another free port) in `.env`, run
`docker compose up -d` and open `http://localhost:8504`.

**The page does not open.** `docker compose ps` should show `(healthy)`. The port is published on
127.0.0.1 only, so another computer cannot reach it, by design.

**"Ranking unavailable", or a ranking that fails.** Read `logs/hiring-desk-adapter.log` (failures
are logged without document text) and run `verify`: ranking reads the A2 parquets, the role model,
the inference parts and SBERT weak from the folder. `HR_PORTAL_RANKING_ADAPTER` and
`HR_PORTAL_CANDIDATE_POOL` should be unset; the app reports an invalid value as a configuration
error. After a restart, **Refresh ranking status** reruns a job that was in progress.

**A slow first ranking.** Expected. The first D63 in a fresh or refreshed folder embeds the pool
and the background CVs once and caches them: 35-43 s with empty caches, 14 s afterwards. D64 scores
50 new candidate pairs with C2: 94-121 s (27 Oct 2026). `create` empties the caches, so the first
ranking after an update is slow again.

**A model worker is reported unavailable.** v1 answers (RoBERTa-SQuAD2) and entity comparisons
(GLiNER-medium) run in separate worker processes, started on first use from the models in
`/opt/models`. Look at `docker compose logs app` and `logs/medium-worker-*.log`; the usual cause
is memory. `SCREENER_MODELS_ROOT` must stay `/opt/models`.

**A chat question gets no answer.** The extractive model abstains on some wordings: "What degree
does this role require?" gets an answer where "What degree is required in the JD?" does not (S6;
the same natively). Rephrase the question.

**`stat …/compose.llm4b.yaml: no such file or directory`.** The override file was removed on
31 Oct 2026: its service is now part of `compose.yaml`. Your `.env` still names it. Delete the
`COMPOSE_FILE=compose.yaml:compose.llm4b.yaml` line (`;` as the separator on Windows) from `.env`,
then run the command again without `-f compose.llm4b.yaml`.

**Every v3 answer is a fallback** (`mode` `qwen_fallback` under **Answer details**, with
`provenance.fallback_reason` `provider_unavailable`). The 4B service is not answering. Check that
`docker compose ps` lists `llm4b` as `(healthy)`; if it is missing, start it with
`docker compose up -d` (the v1 setup's `docker compose up -d app` does not start it). After `up`,
wait for `(healthy)` (or use `--wait`) before signing in: a session started earlier keeps the
fallback until its next session ([§6](#chat-v3-with-qwen35-4b-the-default)). If `llm4b` exits or
restarts, it is usually memory: give Docker at least 15 GB ([§3](#3-requirements)). Fallbacks with
`deadline_exceeded` while `llm4b` is healthy mean the 60 s budget ran out, typically while a
ranking runs; `unsupported_paraphrase` or `qualification_omission` mean the application's checks
rejected the wording. Both are expected, not a fault.

**Qwen does not answer (chat v2).** v2 needs the `qwen` service: `docker compose --profile
llm up -d`, then `docker compose --profile llm ps` should show `qwen` `(healthy)`. It shares the app
container's network namespace (both insist on loopback), so after the app restarts other than by
`docker compose restart app`, restart it too: `docker compose --profile llm restart qwen`. A
fallback answer while `qwen` is healthy usually means Qwen's output failed the application's checks
([§6](#chat-v2-the-llm-profile), "Qwen or fallback?").

**The account command cannot hide the token** (Git Bash on Windows, or no terminal). With Docker
Engine inside WSL2, run it in a WSL2 terminal (for example in Windows Terminal); with Docker Desktop,
in PowerShell or a WSL2 terminal ([§7](#7-accounts)).

**A report or the department mail fails for some CSV-backed JDs.** Preparing a report or a mail
fails with a `ValueError` for 53 of the 600 CSV candidates, in 38 of the 60 JDs (the first one
included): their ids fail the workspace store's identity check after redaction
(`portal_store._opaque_identity`). This is the app's behaviour, natively too, not Docker's; a JD
whose ten candidates pass (the second, for example) saves its report and captures the mail. Not
fixed.

**Scores of a new JD differ in the 6th decimal between runs.** C2's loader calls
`torch.set_num_threads(6)` for the whole process. After the first D64 that scores new pairs, every
later SBERT embedding in that app process runs on 6 threads, so a JD embedded afterwards can get
last-bit different embeddings: scores differed by up to 4e-6, with the same top 10 in the same order
(V2b, 27 Oct 2026). The embedding cache keeps whichever value came first. Native runs behave the
same. Not changed; it is the owner's decision (restore the thread count after C2, or fix SBERT's).

**onnxruntime telemetry in native runs.** In Docker, `ORT_DISABLE_TELEMETRY=1` stops it. A native
run (outside Docker) does not set it, so onnxruntime writes telemetry files under the home folder
and contacts `mobile.events.data.microsoft.com`. The model workers inherit the app's environment,
so exporting `ORT_DISABLE_TELEMETRY=1` before a native start should stop it; that was not tested
natively.

**Native runs on Apple Silicon.** uv's default macOS target is 13.0, and the model runtimes'
locked `onnxruntime` has macOS 14 wheels only, so the runtimes cannot be installed natively on
Apple Silicon (S2, unchanged by the Docker work). Use Docker there: the image is Linux.

## 13. How the image and the folder are built

### Stages

```
ghcr.io/astral-sh/uv:0.12.7 ──► uv ─────────┐
python:3.12-slim-bookworm ──────────────────► base ──► deps ──┬──► test     (dev group + repository; pytest)
                                                              └──► models ──► app   (the default target)
ubuntu:24.04 ───────────────────────────────► llm4b   (Qwen3.5-4B for chat v3; borrows uv and base's CA bundle)
```

| Stage | Adds | Rebuilt when |
|---|---|---|
| `uv` | the uv binary | the uv pin changes |
| `base` | Debian's `ca-certificates` and `libgomp1` (OpenMP for torch), uv, the user `screener` (1000:1000), fixed uv settings | a base image pin changes |
| `deps` | the three locked environments: `/app/.venv` (groups `app` and `ml`), `runtime/transformer/.venv`, `runtime/gliner-medium/.venv` (`uv sync --frozen`; wheels kept in a cache mount) | a lock or a `pyproject.toml` changes (24.2 s with cached wheels, V7) |
| `test` | the `dev` group and the repository | any source or test change |
| `models` | the four pretrained models, downloaded into a cache mount (one folder per pin, so a rerun copies instead of downloading), copied to `/opt/models` and `/opt/hf`, then verified offline | a pin, a lock, or one of the five files it copies changes (`__init__.py`, `config.py`, `llm/qa_assets.py`, `llm/qwen_assets.py`, `extraction/medium_assets.py`); **never a frontend change** |
| `llm4b` (its own image, built by `docker compose build` for the `llm4b` service since 31 Oct 2026; image `candidate-screener/hiring-desk-llm4b:local`, 2.94 GB) | Ubuntu packages pinned by version (`python3`, `ca-certificates`, `libgomp1`; on amd64 from a dated snapshot); the facade's venv `/opt/gguf-venv`; the llama.cpp runtime and the Qwen3.5-4B GGUF, fetched, verified and start-checked by `gguf_assets.py --platform` (`TARGETARCH` amd64 → `linux-x64`, arm64 → `linux-arm64`, anything else fails the build) into `/opt/models/chatbot/qwen35`, with the manifest copy in `/opt/qwen35-manifest/`; last, `gguf_server.py` and `gguf_probe.py` | a pin in `gguf_assets.py`, the snapshot or a package pin, or one of its copied files changes (`__init__.py`, `config.py`, `llm/__init__.py`, `llm/gguf_assets.py`; the last layer for `llm/gguf_server.py` and `llm/gguf_probe.py`). It uses `uv` and `base` only through build-time mounts and never rebuilds `deps`, `models` or `app` |
| `app` | `output/` and `docs/data/` (committed records), the backend `src/` without `app/`, the editable install, GLiNER-medium's `runtime-versions.json`, compiled bytecode; last, `src/candidate_screener/app/` and `.streamlit/` | a backend change reruns from the `src/` copy; a frontend change only the last two layers (0.70 s, V7) |

`/app` equals the code's `config.PROJECT_ROOT` (an editable install), and everything under `/app`
and `/opt` is world-readable, so any host user id can run it. The image has no `.git`, so the
commit is recorded only through `GIT_COMMIT`.

### Pinned inputs

The model licences were read on 28 Oct 2026 from each model's card metadata on Hugging Face at the
pinned revision, and for RoBERTa-SQuAD2, GLiNER-medium and C2 also from the card files in the image
(the image holds no card for the other two).

| Input | Source | Pin | Licence |
|---|---|---|---|
| Dockerfile syntax | `docker/dockerfile:1` | `sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e` | Apache-2.0 |
| Python base image (`PYTHON_IMAGE`) | `python:3.12-slim-bookworm` (Python 3.12.16, Debian bookworm) | index digest `sha256:fc169880a4d941f3142c0e2e623f05f246150add19dc36fc9224df20fb19e5fd` (serves amd64 and arm64) | Python: PSF License; Debian packages: their own |
| uv (`UV_IMAGE`) | `ghcr.io/astral-sh/uv:0.12.7`, the version that wrote the locks | index digest `sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945` | MIT or Apache-2.0 |
| Python packages | PyPI and the PyTorch CPU index | `uv.lock`, `runtime/transformer/uv.lock`, `runtime/gliner-medium/uv.lock`, installed `--frozen` | per package |
| RoBERTa-SQuAD2 (`QA_REVISION`) | `deepset/roberta-base-squad2` | `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4` | CC-BY-4.0 |
| GLiNER-medium v2.5 | `gliner-community/gliner_medium-v2.5` | `88c3b98b57ad5e7d66fb209ed61c53f4b1fd05da` (`MODEL_REVISION` in `extraction/medium_assets.py`) | Apache-2.0 |
| GLiNER-medium's encoder configuration (`GLINER_ENCODER_REVISION`) | `microsoft/deberta-v3-base`, `config.json` only | `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` | MIT |
| Qwen2.5-0.5B (`QWEN_REVISION`) | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | Apache-2.0 |
| C2 (`C2_MODEL`, `C2_REVISION`) | `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473`, equal to `rerank/cross.py`'s pin (a contract test checks it) | Apache-2.0 |

Sizes: RoBERTa-SQuAD2 497.6 MB, GLiNER-medium 845.4 MB and C2 1.21 GB, as read by the app at run
time (S4); Qwen2.5-0.5B 954 MB. The model runtimes take torch's CPU build from the PyTorch CPU index
on Linux (`2.2.2+cpu` on x86_64), without the CUDA packages; the macOS and Windows lock entries are
unchanged.

**The `llm4b` image** (plan
[`2026-10-28-gguf-cross-platform`](../plan/2026-10-28-gguf-cross-platform/README.md), 02 §3 and
G2b). Its licences were not read for this guide.

| Input | Source | Pin |
|---|---|---|
| Ubuntu base image (`UBUNTU_IMAGE`) | `ubuntu:24.04` (glibc 2.39; the `linux-arm64` runtime needs glibc 2.38, which bookworm lacks) | index digest `sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c` |
| Ubuntu packages (`UBUNTU_SNAPSHOT`) | `main` of noble, noble-updates and noble-security (backports dropped). amd64: `snapshot.ubuntu.com` (owner, Q6). arm64: the live ports archive, because `snapshot.ubuntu.com` serves no `ubuntu-ports` (owner, 28 Oct 2026, E5) | snapshot `20261027T000000Z` (amd64); on both architectures the versions `python3=3.12.3-0ubuntu2.1`, `ca-certificates=20260816ubuntu0.24.04.1`, `libgomp1=14.2.0-4ubuntu2~24.04.1`. On arm64 the build fails loudly once Ubuntu replaces one of them, and the pins are then bumped. Their dependencies (python3.12 and its libraries, about 10 packages) are not pinned there, and follow the live archive |
| The facade's Python packages | PyPI, installed by uv (mounted, not left in the image) into `/opt/gguf-venv` with `--require-hashes --no-deps --only-binary :all:` | `uv.lock`'s pure-Python wheels and hashes: requests 2.34.2, certifi 2026.7.22, charset-normalizer 3.5.1, idna 3.19, urllib3 2.7.0 (a contract test keeps them equal to the lock) |
| Qwen3.5-4B GGUF | `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-Q4_K_M.gguf` (model `Qwen/Qwen3.5-4B`) | revision `e87f176479d0855a907a41277aca2f8ee7a09523`, 2,740,937,888 bytes, SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` (`gguf_assets.py`) |
| llama.cpp runtime | GitHub release `b11896` of `ggml-org/llama.cpp` (commit `9125a2ebe8e5a38f0f4232170c134a29ed730b66`), the CPU build for the key below | one pinned asset per key, below (`RUNTIMES` in `gguf_assets.py`; the Dockerfile repeats none of them) |

The runtime table, measured on 28 Oct 2026 from our own download (plan 01 §1). Docker uses the two
Linux keys; the macOS keys serve native runs.

| Key | Asset | Size (bytes) | SHA-256 |
|---|---|---:|---|
| `macos-x64` | `llama-b11896-bin-macos-x64.tar.gz` | 12,038,557 | `2ff5c5f3b5ba8a12a4a868c290510799416ad78c61528e9d7754404aa6fd6fb5` |
| `macos-arm64` | `llama-b11896-bin-macos-arm64.tar.gz` | 16,297,217 | `f7fdf25367d2a403a91ec91729322a3c6a3a80e95da15844aa2c7be9dab3e11b` |
| `linux-x64` | `llama-b11896-bin-ubuntu-x64.tar.gz` | 22,497,065 | `b02ed815efe497c0e818aadaae1b2538309f25424034555dcc1b7211c783e25c` |
| `linux-arm64` | `llama-b11896-bin-ubuntu-arm64.tar.gz` | 18,221,582 | `68942cd6067dd632831f12e0bcdb4f8edbd91882e633a4e8593e0690f0624261` |

**The CA bundle is borrowed.** `snapshot.ubuntu.com` is HTTPS only and the Ubuntu base has no CA
bundle yet, so on amd64 the stage's two apt commands read the `base` stage's
`/etc/ssl/certs/ca-certificates.crt` through a bind mount (arm64, reading the ports archive, does
not use it). The packages themselves are verified by
the archive's signatures as always, and the bundle is not left in the `llm4b` image (its own
`ca-certificates` package is installed at its pinned version).

### Where each folder file comes from

`create` copies four inputs from the repository's committed copies (the three review tables from
`output/matching/`, the vocabulary from `data/vocab/`; in Docker the vocabulary comes from the
source instead, because the folder's mount hides the repository's copy) and the other twenty from
the source. In the owner's `data/`, those twenty were produced by the native research runs: the
A2 text caches by the data and matching code, the role-family model and SBERT weak by training, and
the inference parts by `python -m candidate_screener.inference --cache-parts`
([`README.md`](../README.md), [`data/README.md`](../data/README.md)). Every name comes from the
code that reads the file (`expected()` in
[`datapack.py`](../src/candidate_screener/datapack.py)), so `verify` names an input the code now
expects and the folder lacks.

## 14. Security and privacy

- **Loopback only.** The port is published on 127.0.0.1. Local token sign-in is not single
  sign-on, and there is no TLS: exposing the port on a network needs both, and an authentication
  decision, which are out of scope.
- **Local tokens.** Only salted PBKDF2-SHA256 hashes (310,000 iterations) are stored, in a 0600
  file in a 0700 folder. Tokens are never printed, logged or saved in reviews.
- **No secrets in the image or in commits.** `.dockerignore` keeps `.env`, `compose.override.yaml`
  (and `.yml`), `.streamlit/secrets.toml`, `.git`, `logs/` and all of `data/` except `README.md` and
  `vocab/` out of the build context; `.gitignore` keeps the same settings and secrets files out of
  commits. A Hugging Face token enters a build only as a build secret, from a file kept outside the
  clone. The image holds code, environments and public pretrained weights; no Djinni text.
- **The data folder is private.** It holds Djinni-derived CV and JD text (MIT-licensed, anonymised
  by the publishers), the project's trained models and this deployment's workspace. Never commit
  it (it is git-ignored) or share it publicly. Pass it on only with `handover`, which leaves the
  workspace and caches out and writes its copy only outside the clone or in the git-ignored
  `data/`, and only privately.
- **Redaction is unchanged.** Displayed text still passes `annotation.redact`, inside the
  container as natively. Mechanical redaction does not guarantee anonymity.
- **No network at run time.** The app ran its whole smoke test with the container's network
  disabled (V6). Streamlit's usage statistics, its external-IP lookup and onnxruntime's telemetry
  are switched off in the image.
- **Nothing is trained and no label is read** in the container; nothing is downloaded at start-up.

## 15. Known limitations

- **The frontend is work in progress**: 17 frontend tests fail ([§11](#11-tests)), and two app
  behaviours stand (the report and mail failure for 53 CSV candidates, and C2's thread count;
  [§12](#12-troubleshooting)).
- **Chat v3 is experimental**, though it is the default since 31 Oct 2026 (D69): being the default
  is not quality acceptance, and v1 is the rollback. Chat v3 with Qwen3.5-4B and chat v2 with the
  `llm` profile were verified by scripted sessions, not yet by hand in a browser
  ([§6](#chat-v3-with-qwen35-4b-the-default), [§6](#chat-v2-the-llm-profile)).
- **Only the first account can be created in the browser**; further accounts use the command, and
  there is no renew, disable or reset command and no administrator role ([§7](#7-accounts)).
- **Parts not built**: a research workbench image and the ONNX embedding back end; see
  [`docker-optional-parts.md`](docker-optional-parts.md). Qwen3.5-4B is built by default since
  31 Oct 2026, as its own image for the `llm4b` service ([§6](#chat-v3-with-qwen35-4b-the-default)).
- **Qwen3.5-4B (chat v3), its limits** (plan
  [`2026-10-28-gguf-cross-platform`](../plan/2026-10-28-gguf-cross-platform/README.md)):
  - *The 60 s budget under load.* On the owner's machine a 4B turn fits the 60 s budget only when
    the CPU is otherwise idle: with rankings running in `app`, three of four turns hit it and fell
    back (G2c). The budget stays 60 s (D67); this is recorded, not tuned.
  - *Equivalence across builds (E4) is pending, and not a gate.* The teammate's Intel Mac
    (`macos-x64` against `linux-x64`, the near-tie rule of
    [`03-equivalence.md`](../plan/2026-10-28-gguf-cross-platform/03-equivalence.md)) has not run yet;
    its result will be recorded, not required (owner, 28 Oct 2026, Q3). E6 (`macos-arm64`) waits for
    an Apple Silicon Mac. E1-E3 passed on the owner's machine (native against Docker, `linux-x64`).
  - *linux-arm64 has been checked only under emulation (E5).* The arm64 `llm4b` image built under
    QEMU on the owner's machine (445 s). Its probe passed against native `linux-x64`: E1, E2 and
    the near-tie generation rule, with identical tokens. That shows the arm64 build is correct, not
    how fast it runs. Emulated generation took about 6 s per token. Speed on Apple Silicon waits
    for a real Mac (E6).
  - *arm64 packages are not snapshot-pinned.* `snapshot.ubuntu.com` serves no `ubuntu-ports`
    (401/404), so an arm64 build of `llm4b` reads the live `ports.ubuntu.com` archive with the same
    version pins as amd64 (owner, 28 Oct 2026). It fails loudly once Ubuntu replaces one of those
    versions; the pins are then bumped ([§13](#pinned-inputs)).
  - *A deprecation warning.* `llama-server` warns that `--chat-template-kwargs` for
    `enable_thinking` is deprecated in favour of `--reasoning`. The flag comes from
    `gguf_server.backend_command`, which this work left unchanged; the warning is recorded, not
    acted on.
- **arm64 numbers are unchecked.** Under emulation (28 Oct 2026) the arm64 image built, started and
  passed its tests, but the models ran about 300 times slower, so no ranking finished; the owner
  chose to wait for a real Apple Silicon Mac. The first run there is checked against the bar
  registered before it (plan, V10): every ranking score within 1e-3 of amd64; the same top 10 in
  the same order, except candidates whose amd64 scores are within 1e-3 of each other; the same
  extractive answer text; the same entity mentions, except those scored within 0.01 of GLiNER's 0.5
  threshold.
- **Only Linux (Docker Engine in WSL2) has been verified**; Docker Desktop on Windows and macOS has
  not. A build or a run under a Docker Desktop memory limit was not measured.
- **One image name per machine.** Every clone and worktree builds and runs
  `candidate-screener/hiring-desk:local` (and `:test`), so the last build wins; rebuild in the
  checkout you run from ([§10](#10-releasing-an-update-developer)).
- **The data sources C1 (Kaggle PDFs) and ESCO (D1) are not included**; the app does not read them.
- **No CI** (Q22). The tests and checks run when someone runs them.
- **The checks are native.** `checks --affected` and `inference --freeze` need `.git`, which the
  image does not have; the long checks run on the owner's machine, never in the container.
- **The release checks are by hand**: the scripts that drove the S6 smoke and parity runs are not
  in the repository ([§10](#10-releasing-an-update-developer)).
- **Times are UTC** in the container.

## 16. First day for a new maintainer

- [ ] Run the quick start ([§4](#4-quick-start-operator)) on your machine and sign in.
- [ ] Run the tests in the container ([§11](#11-tests)) and check that only the 17 known failures
  appear.
- [ ] Read [`project-overview.md`](project-overview.md) and [`AGENTS.md`](../AGENTS.md).
- [ ] Take a backup of `data/datapack/processed/hr-portal/` ([§9](#9-backups-operator)), and decide
  where backups live.
- [ ] Know who holds the full `data/` and the data folder, and who releases updates
  ([§10](#10-releasing-an-update-developer)). Today both are the owner.
