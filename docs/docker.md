# Running the Hiring Desk with Docker

This guide covers running and maintaining the containerised Hiring Desk on any operating system.
It does not describe the system itself (what it ranks, how, and how well): for that, read
[`project-overview.md`](project-overview.md), [`ownership/`](ownership/01_project_map.md) and
[`AGENTS.md`](../AGENTS.md). The plan behind this set-up is
[`plan/2026-10-27-docker/`](../plan/2026-10-27-docker/README.md), decision D68 in
[`decisions.md`](decisions.md).

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
written or in its fresh-clone walkthrough (S8). Your machine and network will give other times. GB
means 10^9 bytes.

**Not verified yet:** Docker Desktop on Windows or macOS; any arm64 machine (Apple Silicon); the
PowerShell forms of the commands (every command here was run in bash). **Chat v2/v3 with the `llm`
profile:** the `qwen` service runs and Qwen composes v2 and v3 answers in the container (S9, 28 Oct
2026), but the Hiring Desk cannot yet save a Qwen-composed answer (an app issue, not Docker;
[§6](#chat-v2v3-the-llm-profile)).

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
  chat answers), GLiNER-medium (entity comparisons), Qwen2.5-0.5B (chat v2/v3) and
  Qwen3-Reranker-0.6B, called C2 (used by D64).
- **The data folder**, `data/datapack/`, is mounted into the container at `/app/data`. It holds
  the project's own files that the app reads (the A2 texts, the CSV-backed reviews, the
  role-family model and SBERT weak that the project trained, the fitted inference parts, the
  skill vocabulary), the **workspace** people create while using the app (accounts, saved
  reviews, JD drafts, mail captures) and caches.

```
host browser ── 127.0.0.1:8503 ──► app container (runs as the host user; /app is the project root)
                                     ├─ Streamlit Hiring Desk, v1 chat by default
                                     ├─ ranking adapter thread ─► inference.Screener (D63; D64 adds C2)
                                     ├─ QA worker      uv run --project runtime/transformer   (RoBERTa-SQuAD2)
                                     └─ entity worker  uv run --project runtime/gliner-medium (GLiNER-medium)

  /app/data      ◄── ./data/datapack, read-write: project files, workspace, caches
  /app/logs      ◄── ./logs
  /opt/models    ◄── in the image: RoBERTa-SQuAD2, GLiNER-medium, Qwen2.5-0.5B (SCREENER_MODELS_ROOT)
  /opt/hf        ◄── in the image: C2, in the Hugging Face cache layout (HF_HOME)

[profile llm]    qwen container in app's network namespace ─► 127.0.0.1:8000 (Qwen2.5-0.5B; phase 2)
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
| macOS, Apple Silicon; Linux arm64 | Docker Desktop; Docker Engine | expected to work, not verified. `docker compose build` builds an arm64 image natively and every lock resolves for arm64 (checked without building, 27 Oct 2026), but no arm64 image has been built or run yet ([§15](#15-known-limitations)) |

**Windows.** Use Docker Desktop with the WSL2 back end, or Docker Engine inside a WSL2
distribution (the owner's set-up). Keep the clone inside the WSL2 file system (for example under
your Linux home folder) and run the commands from a WSL2 terminal: the data folder is
bind-mounted, and bind mounts from a `C:\` path are slow (a known Docker Desktop property; not
measured here). With WSL2, Docker's memory is the WSL2 virtual machine's. Set it in
`%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=10GB
```

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
| Memory, all of phase 1 (D63, D64, v1 chat with both model workers) | A scripted session in the container (V3): the app process 5.26 GB, the whole container 7.56 GB including page cache. One native process ranking two JDs with D63 and D64: 7.38 GB | Give Docker **at least 10 GB**: the container's 7.6 GB peak plus about a third, because one process alone reached 7.4 GB without the chat workers |
| Memory, chat v2/v3 (Qwen, `llm` profile) | The `qwen` container (S9): the model server peaked at 3.26-3.36 GB resident; the container at 2.66-3.82 GB including page cache. The app container in chat-only sessions (no ranking): 2.66-4.26 GB | Give Docker **about 3.5 GB more** than for phase 1 when the `llm` profile runs: the server's measured peak, rounded up. A session with D64 and Qwen together was not measured |
| Disk | Image 7.98 GB. Build cache after a cold build 14.74 GB (the layers, plus a second copy of the weights and wheels in the cache mounts). Test image 4.59 GB. Data folder: 256.8 MB of inputs, plus the workspace (2.6 MB on the owner's machine) and caches (1.1 MB after the owner's browser session), both measured 28 Oct 2026 | **About 25 GB free** for the first build, **30 GB** if you also build the test image: the sum of the measured sizes, rounded up. It is an upper bound, because the image's layers can also appear in the build cache. Afterwards `docker builder prune` returns most of the build cache, at the price of downloading again at the next build |
| Network | The cold build downloaded 4,382 MiB (4.60 GB): base images, Debian packages, Python wheels and model weights | Needed **only to build**. The app ran every step of the smoke test with the container's network disabled (V6) |
| Time | Cold build: 3 min 23 s (V8, the measured run), and 5 min 28 s in an earlier run. The model downloads took 156 s of the measured run and 280 s of the earlier one | Your download speed decides it |

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
   apply.

4. **Build the image.**

   ```
   docker compose build
   ```

   The output ends with `✔ app  Built`. From an empty cache this took 3 min 23 s (5 min 28 s in an
   earlier run) and downloaded 4.60 GB; the image is 7.98 GB. With nothing changed, a rebuild takes
   a few seconds (5.5 s). Build in the clone you run the app from: the image's name is shared by
   every clone on the machine ([§10](#10-releasing-an-update-developer)).

5. **Check the folder.**

   ```
   docker compose --profile tools run --rm tools verify
   ```

   The last line should read `OK: 24 inputs (256.8 MB) match the manifest and what the code
   expects` (about 2 s, 28 Oct 2026). A received copy also prints
   `workspace processed/hr-portal/: none`. Anything else: [§5](#what-verify-reports).

6. **Create your reviewer account** ([§7](#7-accounts) explains accounts).

   ```
   docker compose run --rm app python -m candidate_screener.app.portal_auth --create-local-user reviewer --department benchmark
   ```

   Replace `reviewer` with your login ID. The command asks twice for an access token of at least
   24 characters, first `Choose an access token (…; not displayed):`, then `Confirm access token:`.
   Nothing appears as you type; press Enter after each. It then prints `Local reviewer account
   created. This is local token authentication, not company SSO.` (about 2 s, 28 Oct 2026). If the
   two entries differ (`error: Tokens differ`) or the token is too short (a traceback ending in
   `ValueError: Use a bounded login ID, token of …`), nothing is created: run the command again.
   On Windows, run it where the token can be hidden: with Docker Engine inside WSL2 (the verified
   set-up), a WSL2 terminal, for example in Windows Terminal; with Docker Desktop, any terminal
   where `docker` works (PowerShell or a WSL2 terminal). Not Git Bash.

7. **Start the app.**

   ```
   docker compose up -d
   docker compose ps
   ```

   Right after `up -d`, `docker compose ps` shows the app as `Up … (health: starting)`. Run it again
   after a few seconds: it shows `Up … (healthy)` after about 7 s (7.6 s on 27 Oct 2026, 6.8 s on
   28 Oct 2026). Alternatively, `docker compose up -d --wait` returns only once the app is healthy
   (6.2 s, 28 Oct 2026). If `up` fails with `address already in use` or `port is already
   allocated`, something else holds the port: see [§12](#12-troubleshooting).

8. **Open http://localhost:8503 and sign in** with your login ID and token, on the page headed
   "Local reviewer sign-in". If you changed `HIRING_DESK_PORT` in step 3, open
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
| Start (in the background) | `docker compose up -d` |
| Stop, keeping the container | `docker compose stop` |
| Stop and remove the container and its network (the data folder and the image stay) | `docker compose down` |
| Restart | `docker compose restart app` |
| Status and health | `docker compose ps` (`Up … (healthy)` when Streamlit answers, `(health: starting)` for the first seconds after a start). It lists running containers only: a stopped app appears in `docker compose ps -a` |
| The app's output | `docker compose logs app`, or `docker compose logs -f --tail 100 app` to follow it (Ctrl+C stops following; the app keeps running) |

The app's healthcheck asks Streamlit's `/_stcore/health` every 30 s and verifies nothing else. The
`qwen` service has its own (below); `tools`, which runs no server, disables it. Nothing is
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

### Chat v2/v3: the `llm` profile

Chat v2 (Qwen orders checked comparison relations, D65) and v3 (experimental Qwen prose, D66) need
the `qwen` service, Qwen2.5-0.5B on the CPU, whose weights are already in the image. **Status
(S9, 28 Oct 2026): blocked by an app issue.** Qwen composes the answers in the container, but the
Hiring Desk refuses to save any answer Qwen composed: the review store treats the provenance field
`output_token_budget` as a credential ("Credentials cannot be persisted"), so the page shows an
error instead of the answer. Answers through the fallback save and show normally. Until the app is
fixed, keep `HR_PORTAL_CHAT_VERSION=v1` (the default).

| Task | Command |
|---|---|
| Start (set `HR_PORTAL_CHAT_VERSION=v2` or `v3` in `.env` first) | `docker compose --profile llm up -d` (`--wait` to return once both are healthy) |
| Status | `docker compose --profile llm ps`: `qwen` shows `(healthy)` once the model has loaded |
| Stop only Qwen | `docker compose --profile llm stop qwen` |
| Stop everything | `docker compose --profile llm down` (name the profile so `qwen` is included) |

- **Healthcheck.** `qwen` asks its own server, `GET http://127.0.0.1:8000/v1/models`, every 30 s,
  with 120 s allowed for the model to load. The server answers only after loading, so `(healthy)`
  means Qwen can answer. Measured: the model loaded in 3-4 s and `qwen` was healthy 5.7 s after
  `up` (S9).
- **Memory.** The model server peaked at 3.3 GB resident (S9); see [§3](#3-requirements).
- **Speed.** A turn composed by Qwen took 2-5 s (v2) and 3.4-5.3 s (v3) on the owner's machine;
  the model's share was 1.5-3.5 s. Both versions give up after 30 s and answer through the fallback.
- **The restart caveat.** `qwen` uses the app container's network (both insist on loopback).
  `docker compose restart app` restarts `qwen` with it (S9). After the app restarts any other way
  (its restart policy, Docker restarting, a recreate; not tested), restart Qwen:
  `docker compose --profile llm restart qwen`. `qwen` has no restart policy of its own.
- **The app does not wait for Qwen.** It starts alone; while `qwen` is stopped or loading, v2 and
  v3 answer through their fallback (S9: every turn answered, in under 4 s).
- **Qwen or fallback?** Open **Answer details** under a v3 answer: `mode` is `qwen_prose` and
  `model` is `Qwen/Qwen2.5-0.5B-Instruct` when Qwen wrote it; `qwen_fallback`, with
  `provenance.fallback_reason` (`provider_unavailable` when the service is down), when the
  application did. A v2 answer's caption reads "Qwen selected validated comparison points" when
  Qwen composed it and "Qwen unavailable or rejected · application comparison" for the fallback.
  The fallback is also used when Qwen answers but its output fails the application's checks (S9
  saw `unsupported_paraphrase` and `qualification_omission` in v3): that is expected, not a fault.

## 7. Accounts

- **One account per reviewer**, created with the command in [§4](#4-quick-start-operator):

  ```
  docker compose run --rm app python -m candidate_screener.app.portal_auth --create-local-user LOGIN --department benchmark
  ```

  It runs in a one-off container (`run` publishes no port, so it works while the app is running)
  and writes `data/datapack/processed/hr-portal/accounts.json`. The running app reads that file at
  every sign-in, so there is nothing to restart.
- **The token.** The command asks twice, hidden, for a token of at least 24 characters (at most
  512 at sign-in). It stores only a salted PBKDF2-SHA256 hash (310,000 iterations), the
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
- **The app's own hint.** With no accounts, the sign-in page says that accounts are not configured
  and shows the native command (`uv run python -m …`). In Docker, use the `docker compose run`
  form above.
- **Accounts never travel.** The image contains none, and `handover` copies none: each operator
  creates their own. Rebuilding the image leaves accounts alone; `purge --yes` deletes them.
- **Coming next (S10, on its own branch).** With no accounts, the app will show a "create the
  first account" form (login ID, departments, the token twice, expiry) instead of the hint; further
  accounts keep using the command, and there is no administrator role. While no account exists,
  whoever opens the page first creates one, which the loopback-only port limits to someone on the
  same machine. This section changes when that branch merges.

## 8. Configuration reference

Every setting the code under `src/` reads, plus the compose settings and the image's fixed
settings worth knowing. `tests/test_docker_contract.py` fails when the code reads an
`HR_PORTAL_*` or `SCREENER_*` name that these tables do not list.

**How settings reach the app.** Compose reads `.env` (copy it from
[`.env.example`](../.env.example); it is git-ignored) for the settings in the first table, and
passes only `HR_PORTAL_CHAT_VERSION` into the app container. To pass another setting, add a
`compose.override.yaml` (or `.yml`) beside `compose.yaml`. Compose reads it automatically unless
you name files with `-f`. Like `.env`, it is git-ignored and docker-ignored, so it stays out of
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
| `HR_PORTAL_CHAT_VERSION` | `v1` | Passed to the app. `v1`: extractive answers (RoBERTa-SQuAD2) and entity comparisons (GLiNER-medium). `v2`: Qwen orders checked comparison relations (D65). `v3`: experimental Qwen prose and JD requirement search (D66). Any other value behaves as `v1`. `v2` and `v3` need the `llm` profile (`docker compose --profile llm up -d`) | `v1`: yes. `v2`/`v3`: not yet: Qwen composes in Docker, but the app cannot save its answers ([§6](#chat-v2v3-the-llm-profile)) |
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
| `HR_PORTAL_CHAT_VERSION` | `v1` | See the compose settings above | See above |

### Models and the data folder (`SCREENER_*`)

| Name | Docker default | Effect | Safe to change |
|---|---|---|---|
| `SCREENER_MODELS_ROOT` | `/opt/models` (set in the image) | Where `config.MODELS` points: the folder holding RoBERTa-SQuAD2, GLiNER-medium and Qwen2.5-0.5B. Unset (native runs), it is `data/processed` | No |
| `SCREENER_DATAPACK_SOURCE` | `/source` (`tools` service) | `datapack create`'s default source. Unset (native runs), it is `data/` | No in Docker: it must name the read-only mount (a contract test checks it). Natively, prefer `--source` |
| `SCREENER_DATAPACK_DEST` | `/app/data` (`tools` service) | The data folder every `datapack` command works on, and the only case in which the project's `data/` may be that folder (the container mounts the folder there). Unset (native runs), it is `data/datapack` | No in Docker (a contract test checks it). Natively, prefer `--dest` |

### Qwen for chat v2/v3 (`SCREENER_QWEN_*`)

With none of these set, chat v2/v3 take their settings from the image's Qwen manifest
(`/opt/models/chatbot/qwen/manifest.json`) and talk to the `qwen` service of the `llm` profile at
`http://127.0.0.1:8000/v1`. Setting `SCREENER_QWEN_BASE_URL` or `SCREENER_QWEN_MODEL` switches to
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
| `SCREENER_QWEN_MANIFEST` | unset | Selects the Qwen3.5-4B GGUF service's manifest for v3 (D67). That service is not in the image ([`docker-optional-parts.md`](docker-optional-parts.md)) | No |

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
a D64 in progress is lost). Give Docker more memory ([§3](#3-requirements)): 10 GB for D64 with
chat. D64 is the heaviest step; run one at a time.

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

**Qwen does not answer (chat v2/v3).** v2 and v3 need the `qwen` service: `docker compose --profile
llm up -d`, then `docker compose --profile llm ps` should show `qwen` `(healthy)`. It shares the app
container's network namespace (both insist on loopback), so after the app restarts other than by
`docker compose restart app`, restart it too: `docker compose --profile llm restart qwen`. If the
page shows "Credentials cannot be persisted" after a v2/v3 question, Qwen did answer: that is the
app issue in [§6](#chat-v2v3-the-llm-profile); use v1.

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
```

| Stage | Adds | Rebuilt when |
|---|---|---|
| `uv` | the uv binary | the uv pin changes |
| `base` | Debian's `ca-certificates` and `libgomp1` (OpenMP for torch), uv, the user `screener` (1000:1000), fixed uv settings | a base image pin changes |
| `deps` | the three locked environments: `/app/.venv` (groups `app` and `ml`), `runtime/transformer/.venv`, `runtime/gliner-medium/.venv` (`uv sync --frozen`; wheels kept in a cache mount) | a lock or a `pyproject.toml` changes (24.2 s with cached wheels, V7) |
| `test` | the `dev` group and the repository | any source or test change |
| `models` | the four pretrained models, downloaded into a cache mount (one folder per pin, so a rerun copies instead of downloading), copied to `/opt/models` and `/opt/hf`, then verified offline | a pin, a lock, or one of the five files it copies changes (`__init__.py`, `config.py`, `llm/qa_assets.py`, `llm/qwen_assets.py`, `extraction/medium_assets.py`); **never a frontend change** |
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
- **Chat v3 is experimental**, and chat v2/v3 cannot yet be used in the Hiring Desk: Qwen composes
  in the container, but the app refuses to save its answers (an app issue found in S9;
  [§6](#chat-v2v3-the-llm-profile)).
- **Accounts are managed by command** until the in-app first-account screen ships; there is no
  renew, disable or reset command ([§7](#7-accounts)).
- **Parts not built**: Qwen3.5-4B, a research workbench image and the ONNX embedding back end; see
  [`docker-optional-parts.md`](docker-optional-parts.md).
- **arm64 has not been run.** The first arm64 run (Apple Silicon) is checked against the bar
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
