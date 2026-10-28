# Qwen3.5-4B GGUF trial on macOS and Linux, and in Docker

Status: **approved by the owner (28 Oct 2026), "approve as recommended"**; branch
`feat/gguf-cross-platform` off `feat/docker-v1` (`bccf03b`). The measurements in `01` were taken
before the design, as the request asked: the four runtime archives (about 70 MB) and one 3-minute
source build.

Answers (owner, 28 Oct 2026): Q1 **yes** (pin `macos-arm64`); Q2 **a separate `llm4b` image**, used
through `compose.llm4b.yaml`; Q4 **`feat/gguf-cross-platform`**, rebased onto `release/integration` and **left as its own branch: no merge into `feat/docker-v1`** (owner, 28 Oct 2026);
Q6 **snapshot pinning**. Q3 (owner, 28 Oct 2026): **the teammate runs G4/E4, and it need not
pass** (its result is recorded, not a gate); **E1-E3 and E5 must pass on this machine**. Q5: **no
decision-log amendment**; this plan record is enough.

Request (owner, 28 Oct 2026): the D67 trial (`plan/2026-10-25-hiring-desk-v3/04-qwen35-trial.md`)
must work natively on macOS and Linux, and Docker must use only a Linux llama.cpp pin. Today
`llm/gguf_assets.py`:

- pins one runtime (`llama-b11896-bin-macos-x64.tar.gz`, `2ff5c5f3…`) with no platform check;
- writes to a hard-coded `data/processed/chatbot/qwen35` instead of `config.MODELS`.

Docker does not include the trial (`docs/docker-optional-parts.md`).

| File | What it holds |
|---|---|
| `01-measurements.md` | What b11896 publishes; what the Linux builds need (glibc, libraries, `RUNPATH`); start tests on bookworm and Ubuntu 24.04, arm64 under emulation; the source-build alternative, timed |
| `02-design.md` | The runtime table, platform selection, the start check, the manifest, `config.MODELS`, the Docker image and compose file, costs |
| `03-equivalence.md` | The cross-platform bar, registered before any run; what it can and cannot prove |

## Findings that shaped the design (01)

- b11896 publishes CPU builds for `ubuntu-x64`, `ubuntu-arm64` and `macos-arm64` besides
  `macos-x64`. Each size and SHA-256 in the plan was measured from our own download.
- `linux-x64` needs glibc ≥ 2.34 and `libgomp1`. It starts on the app's bookworm base once
  `libgomp1` is installed, and natively on this Ubuntu 24.04 host.
- **`linux-arm64` needs glibc 2.38 and does not start on bookworm (2.36).** It starts on Ubuntu
  24.04, including under the emulation still installed here.
- Libraries are found through `RUNPATH $ORIGIN`; no `LD_LIBRARY_PATH` is needed.
- A source build at commit `9125a2ebe` (amd64) took 158 s to compile on 22 cores and gives 42 MB of
  binaries. arm64 was not compiled.

Hence the recommendation: a **separate `llm4b` image on Ubuntu 24.04** with the published
binaries for both architectures, rather than compiling or growing the default image (02 §3).

## Stages

Every stage ends with `uv run pytest` (where code changed), a commit and a push on the branch from
Q4. Long jobs run once in the background with an exit marker
(`( echo start…; /usr/bin/time -v CMD; echo "exit $?" ) > logs/gguf/<job>.log 2>&1`) and the turn
ends (AGENTS Testing 9).

| Stage | What | Who | Estimate (agent / machine) | Long job? |
|---|---|---|---|---|
| G0 | Commit the approved plan on the branch | parent | minutes | no |
| G1 | `gguf_assets`: the runtime table, `platform_key`, `--platform`, the start check, manifest fields, `DESTINATION` from `MODELS`; `gguf_server` accepts the new fields; tests shown failing first (below) | sub-agent | 0.5 day / minutes | no |
| G2a | Native Linux provisioning on this machine (`linux-x64`): the 2.74 GB model download and the start check; one facade start and D67's four walkthrough prompts timed | parent | 0.25 day / download about 3-10 min, walkthrough about 5 min | **yes** (download) |
| G2b | The `llm4b` Dockerfile target, `compose.llm4b.yaml`, contract tests; build amd64 | sub-agent writes; parent builds | 0.5 day / build about 5-10 min | **yes** (build) |
| G2c | Measure the Docker service: start and healthy time, memory (idle, in a turn), the four prompts idle and during a D63 ranking | parent | 0.25 day / about 30 min | **yes** |
| G3 | Equivalence: the `gguf_probe` script with its tests; E1-E3 here (native and Docker, linux-x64); E5 (linux-arm64 image built and probed under emulation) | sub-agent writes the probe; parent runs | 0.5 day / E1-E3 about 15 min; arm64 build about 10-20 min; E5 generation up to 2 h | **yes** (E5, arm64 build) |
| G4 | E4 on the teammate's Intel Mac (and E6 if Q1 = yes): they run `gguf_assets` (reuses their files) and `gguf_probe`, then send the JSON; the parent compares | teammate (Q3); parent | teammate about 30 min; parent minutes | teammate's machine |
| G5 | Independent review; fixes; full `uv run pytest`; `checks --explain` (no registered record reads these modules today: D67's `{}`); docs | reviewer sub-agent; parent | 0.25-0.5 day | no |

Total: about 2.5-3 agent days, plus the teammate's time for G4. G4 can wait without blocking G5;
the plan record then says E4 is pending.

## Tests (each shown failing against the current code first; synthetic, no network)

`tests/test_gguf_assets.py`:

- `platform_key` for each pinned (system, machine) pair, including `amd64`/`aarch64` spellings;
- an unknown pair, and a known but unpinned key, raise the clear error;
- `--platform` overrides the detection and skips the start check, with the skip recorded;
- the macos-x64 entry is **byte-for-byte unchanged**: URL, SHA-256 and size equal to the literals
  in today's file, the local names `llama-b11896-macos-x64.tar.gz` and `runtime/`;
- each key fetches its own URL with its own size and digest (a fake `urlopen`);
- the manifest records `runtime_platform`, `runtime_archive` and that runtime's
  `runtime_archive_sha256`;
- `DESTINATION` follows `SCREENER_MODELS_ROOT`, with the native default unchanged;
- a failed start check publishes no manifest and names glibc and `libgomp1`.

`tests/test_gguf_server.py`:

- `load_manifest` accepts manifests with and without the new fields;
- `--launch-backend`'s help no longer says "Intel".

`tests/test_docker_contract.py`:

- the `llm4b` stage's base is pinned by digest;
- the `TARGETARCH` → key mapping covers exactly `amd64` and `arm64`;
- downloads happen only in `RUN` at build time;
- the model and runtime pins in the Dockerfile equal `gguf_assets`'s;
- the facade's dependency versions equal `uv.lock`'s;
- `compose.llm4b.yaml` uses `network_mode: service:app`, publishes no port, has its own healthcheck
  on 8001, and sets `SCREENER_QWEN_MANIFEST` on `app` only there;
- `compose.yaml` is unchanged by the 4B.

`tests/test_gguf_probe.py`: the probe's request shapes, the E4 near-tie rule on fixtures, and the
refusal to write CV-like text (invented prompts only).

## Documentation (in the same change, G5)

- `docs/docker.md`:
  - the configuration table (`SCREENER_QWEN_MANIFEST` in Docker; `COMPOSE_FILE`);
  - a new section "Chat v3 with Qwen3.5-4B: `compose.llm4b.yaml`";
  - §13 pins (the Ubuntu base digest, the runtime table, the snapshot date);
  - §15 (E4/E5 status).
- `docs/docker-optional-parts.md`: the D67 row and a dated entry.
- `README.md` and `AGENTS.md` (Qwen3.5-4B paragraph): mention `--platform`, and that the commands
  are unchanged natively.
- `docs/decisions.md`: a D67 amendment if the scope changes (macOS-only → macOS and Linux, plus
  Docker); the owner decides (Q5).
- The D67 plan file (`04-qwen35-trial.md`): one dated line pointing here.

## Open questions for the owner

1. **macos-arm64 pin?** Measured and available (`01` §1). Yes means a fourth entry plus E6 on an
   Apple Silicon Mac. No means Apple Silicon users run the 4B in Docker (`linux-arm64`) or under
   Rosetta (`macos-x64`). *Recommendation: yes. It costs one table row and one test; E6 waits for a
   Mac.*
2. **The 4B in the default image, or a separate one?** *Recommendation: separate*
   (`hiring-desk-llm4b`, target `llm4b`, on Ubuntu 24.04): the default image does not grow by
   2.75 GB, the arm64 glibc problem goes away without compiling, and `compose.yaml` is unchanged.
   Also: is `compose.llm4b.yaml` (recommended) acceptable instead of a profile inside
   `compose.yaml`? Both are described in 02 §3.
3. **Who runs the teammate-side check (G4, E4)** on the Intel Mac, and is E4 required before
   `llm4b` is called usable in Docker, or recorded as pending?
4. **Branch:** *recommendation:* `feat/gguf-cross-platform` off `feat/docker-v1` (it needs
   `config.MODELS` and the Docker files, which exist only there), merged back into `feat/docker-v1`
   when done. Or directly on `feat/docker-v1`?
   *Answered:* own branch, rebased onto `release/integration`, never merged into `feat/docker-v1`.
5. **D67 scope:** does extending the trial from Intel macOS to macOS and Linux and to Docker need a
   decision-log amendment, or is this plan record enough?
6. **Ubuntu packages:** pin by version against a dated `snapshot.ubuntu.com` mirror (recommended;
   reproducible, one more URL in the build), or accept the current archive's versions?

## Implementation record

### Rebase onto `release/integration` (owner, 28 Oct 2026)

The branch was rebased from `feat/docker-v1` (`bccf03b`) onto `origin/release/integration` at
`d27e931`, which already contains `bccf03b` plus 10 commits from the teammate's
`fix/jd-quotation-fallback`: six v3 plans, new v3 chat modules and a new v3 prompt
(`hiring-desk-v3-summary-conclusion-7`, which asks for `[Reasoning Steps]` before a `[Conclusion]`
and keeps only the conclusion). The rebase was clean.

Impact on this plan:

- **Nothing to redesign.** The incoming commits change no file this plan touches (`gguf_assets`,
  `gguf_server`, the Dockerfile, compose, `config.py`, `datapack.py`, `docs/docker.md`) and add no
  `HR_PORTAL_*`/`SCREENER_*` setting. The provenance key `output_max_new_tokens`, the 256-token
  output limit and the 60 s budget for the 4B are unchanged.
- **Timings.** G2a/G2c measure v3 with the new prompt, which asks for reasoning before the answer.
  The 4B's turns may be longer, so the D67 timings of 27 Oct are not directly comparable; the
  record will say which prompt version was measured.
- **Equivalence (03): unaffected.** It uses its own invented prompts against the model server, not
  the v3 prompt.

Full suite after the rebase (`logs/pytest-g1-rebased.log`): 17 failed (the known ids), 2467 passed.

### G1 (28 Oct 2026; sub-agent, reviewed by the parent; `0a07153`)

- **`gguf_assets.py`:**
  - `RELEASE`, `RUNTIMES` (four keys, digests from 01 §1), `HOSTS`, `platform_key`, `local_names`
    (macos-x64 keeps `llama-b11896-macos-x64.tar.gz` and `runtime/`);
  - `start_check` (`--version`, 120 s, requires `build 11896`, names glibc and `libgomp1`);
  - `provision_runtime`, and `provision(destination, platform=None)` with `--platform`;
  - the runtime and its start check come before the 2.74 GB model download, so a failed check
    downloads nothing more and writes no manifest;
  - manifest fields `runtime_platform`, `runtime_archive` and `start_check` (`passed` or `skipped
    (platform override)`);
  - `DESTINATION = MODELS / "chatbot/qwen35"`.
- **`gguf_server.py`:** only the `--launch-backend` help text changed. `load_manifest` already
  accepted the new fields.
- **Tests:**
  - Before the change, 42 failed and 6 errored (`logs/g1-tests-before.log`); after, 101 passed
    (`logs/g1-tests-after.log`).
  - Three tests passed before by design, since they guard behaviour that was already right: the
    native default destination (2) and `load_manifest` with the new fields.
  - The test refusing unknown `--platform` values is not a true fail-first test: the old parser
    rejected any unknown flag.
- **Checks:** `checks --explain` selects none.
- **Native, this host:**
  - `provision_runtime` for `linux-x64` into a scratch `SCREENER_MODELS_ROOT`: fetched, verified and
    started (`build 11896, commit 9125a2ebe`, GNU 11.4.0) in 1.5 s, reused in 0.5 s
    (`logs/g1-native-runtime.log`).
  - The 2.74 GB model was not downloaded (G2a).
- **Review note:** `provision`'s parameter `platform` shadows the `platform` module inside that
  function only; detection happens in `platform_key`, so the behaviour is right. Left as is.

### G2a: native Linux, this machine (28 Oct 2026; parent)

- **Provisioning** (`uv run python -m candidate_screener.llm.gguf_assets`, no flags;
  `logs/gguf/g2a-provision.log`):
  - The `linux-x64` runtime started first, then the 2.74 GB model was downloaded and verified:
    36 s in all, exit 0.
  - The manifest records `runtime_platform: linux-x64`, the runtime archive's digest
    `b02ed815…` and `start_check: passed`.
  - Destination: the native default, `data/processed/chatbot/qwen35`. In this worktree `data/`
    points to the main checkout's `data/`, so the files live there (git-ignored).
- **Service and walkthrough** (`logs/gguf/g2a-walkthrough.log`, `g2a-walkthrough-{1,2}.json`):
  - `gguf_server --launch-backend`: the facade was ready 4.0 s after start, and `/v1/models`
    returned the pinned model, revision and SHA-256.
  - The D67 walkthrough (`conversation_v3_smoke --real-model`, invented sources) was run twice with
    the v3 prompt `hiring-desk-v3-summary-conclusion-7`. Both runs were identical: Qwen's answer
    was accepted for 1 of 4 factual turns (the degree question). The other three fell back with
    `unsupported_paraphrase`: Qwen answered within the time limit and the application's checks
    rejected the wording.
  - Turn times 33.9-36.1 s, 49.8-50.4 s, 52.8-52.9 s and 42.8-43.1 s: all under the 60 s budget
    on an idle 22-core machine (the backend runs with `--threads 4`). About 181 s per walkthrough.
  - `llama-server` peak resident memory: 4.55 GB.
  - This is a development smoke on one seen invented source family, not a quality result.
- **Finding:**
  - Stopping the `uv run … gguf_server` process with SIGTERM left `llama-server` running, still
    listening on 8002; it was stopped by hand. The facade handles only `KeyboardInterrupt`, so on
    SIGTERM its `finally` (which terminates the child) never runs.
  - In Docker, `docker compose stop` sends SIGTERM, so the service would wait out its grace
    period.
  - Fixed in G2b, with a test that fails first: SIGTERM is handled the same way as Ctrl+C.

### G2b: the `llm4b` image and `compose.llm4b.yaml` (28 Oct 2026; sub-agent, reviewed by the parent)

- **SIGTERM fix (the G2a finding).** `gguf_server.main` turns SIGTERM into the Ctrl+C path, so its
  `finally` stops `llama-server`. The new test (a fake backend, then SIGTERM) failed before the fix
  (`logs/g2b-sigterm-before.log`) and passes after it. In Docker, `stop llm4b` took 0.31 s and
  exited 0, with no `llama-server` left.
- **Dockerfile, target `llm4b`** (between `test` and `models`, so `app` stays the last stage and the
  default):
  - base: `ubuntu:24.04@sha256:24737cef…`;
  - apt from snapshot `20261027T000000Z`, `main` of noble and noble-updates only: `python3=3.12.3-0ubuntu2.1`,
    `ca-certificates=20260816ubuntu0.24.04.1`, `libgomp1=14.2.0-4ubuntu2~24.04.1`. The snapshot
    is HTTPS-only and the base has no CA bundle, so the two apt commands borrow the bundle from the
    `base` stage through a bind mount;
  - the facade's dependencies are installed by uv (mounted, not left in the image) with
    `--require-hashes --no-deps --only-binary :all:` into `/opt/gguf-venv`: requests 2.34.2,
    certifi 2026.7.22, charset-normalizer 3.5.1, idna 3.19 and urllib3 2.7.0, the lock's
    pure-Python wheels and hashes;
  - `TARGETARCH` amd64 → `linux-x64`, arm64 → `linux-arm64`, anything else fails the build.
    `gguf_assets.py --platform` fetches, verifies and runs the start check; no pin is repeated in
    the Dockerfile. A cache mount (`screener-gguf-downloads`) holds one folder per pin;
  - run time: uid 1000, `HF_HUB_OFFLINE`, `HOME=/tmp`; healthcheck `127.0.0.1:8001/v1/models` (no
    proxy, start period 300 s); CMD the facade with `--launch-backend`.
- **`compose.llm4b.yaml`:** `llm4b` (network of `app`, no ports, no data, `init`), plus for `app`
  `SCREENER_QWEN_MANIFEST` and the read-only volume `qwen35-manifest-e87f176-b11896`. `compose.yaml`
  is unchanged. `.env.example` gains a commented `COMPOSE_FILE` line.
- **Contract tests:** 3 failed before (no stage, no file; `logs/g2b-contract-before.log`), 177 passed
  after. 33 broken Dockerfile texts and 16 broken compose texts were added. Full suite: the 17
  known failures, 2520 passed.
- **Build (amd64, `logs/gguf/g2b-build.log`):** 3 min 30 s, of which apt took 157 s (the snapshot
  index at about 450 kB/s) and the download step 39 s. Image 2.94 GB. The default `app` image is
  unchanged (`sha256:b01e70b6…`, not rebuilt).
- **Scratch run (`logs/gguf/g2b-run.log`):**
  - Both services were healthy 8.3 s after `up -d`, and `/v1/models` gave the pinned identity from
    inside `app`.
  - `app` reads the manifest copy read-only, and its `configured_prose_composer()` gave Qwen3.5-4B
    with a 60 s limit. `llama-server` used 4.3 GB of memory at rest.
- **Notes:**
  - **No `hf_token` secret:** `gguf_assets.fetch` uses plain urllib and reads no token, so the
    secret would do nothing.
  - **Manifest timing:** Docker fills the manifest volume when `llm4b` is created, seconds after
    `app`. A v3 session started in that gap uses the marked fallback until its next session.
  - **llama-server warning:** it says `--chat-template-kwargs` for `enable_thinking` is deprecated
    in favour of `--reasoning`. That comes from `backend_command`, unchanged here; it is recorded,
    not changed.
