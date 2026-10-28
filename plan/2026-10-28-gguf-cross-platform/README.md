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
