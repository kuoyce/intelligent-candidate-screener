# 05 — Execution in stages

Owner, 27 Oct 2026: plan approved; execute in stages because of usage limits; commit and push
**only on `feat/docker-v1`** (the one exception: the account-creation screen gets its own branch);
no scheduled resume job: Claude resumes on its own, so the plan only has to survive a long job
ending early when a limit is hit.

The stages follow the README's steps, re-sequenced by dependency: the run-time audit (step 1a)
needs the image, so it runs after the Dockerfile; the cheap measurements (1b-1c) come first.

## Rules

1. **Branch.** All Docker work on `feat/docker-v1`, branched off `e530659`. Every stage ends with a
   commit and a push; long stages also commit work in progress. No frontend file (`app/`,
   `.streamlit/`) changes on this branch.
2. **Status lives in the table below**, updated in each stage's final commit, so a resumed session
   (or a person) knows where things stand from the last pushed commit.
3. **Long jobs** (image builds, the cold build, `inference --check`, the warm check run, `create`):
   one background command per job, never a foreground wait or a polling loop (AGENTS testing 9):

   ```bash
   mkdir -p logs/docker
   ( echo "start $(date -Is) pid $$"; /usr/bin/time -v <command>; echo "exit $?" ) > logs/docker/<job>.log 2>&1
   ```

   The stage table names the running job and its log. Then the turn ends; the job's exit wakes
   the session.
4. **Resuming after a limit** (or any interruption). Read `git log -3 origin/feat/docker-v1`, this
   table, and the tail of each named job log:
   - `exit 0` at the end: the job finished; continue the stage;
   - `exit N` (N ≠ 0): it failed; diagnose from the log, never retry blindly;
   - no `exit` line and the process is alive (`ps -p <pid>`, `docker ps` for containers): still
     running; end the turn and wait;
   - no `exit` line and no process: **it ended early**. Restart the same command (rule 5),
     note "interrupted, restarted" in the implementation record, and never record numbers from
     the interrupted run.
5. **Restarting is cheap by design:**

   | Job | On restart |
   |---|---|
   | Image build | BuildKit's layer cache and cache mounts keep finished layers and downloads; only unfinished steps re-run |
   | Cold build (V8) | its time and download figures are only valid uninterrupted: clear the build cache again and rerun |
   | `inference --check` (V2), test image run (V1), smoke (V3-V6) | rerun whole (minutes) |
   | Warm affected check run (S8) | the runner keeps one log per check (`logs/checks/<name>.log`); rerun only the checks without a pass result, by name |
   | `datapack create` | rerun; it rewrites inputs and verifies at the end |

6. **Checks.** `uv run pytest` at the end of every stage that changed code; the owner-approved warm
   affected run (~70 min) once, in S8; no cold run (Q17, owner: no model weights change). Long
   runs only on the owner's machine.
7. **Sub-agents** (AGENTS workflow 3) implement S2, S3, S5 and S7 and review in S8, each with a
   self-contained brief pointing at these plan files; the parent runs the measurements,
   verification and every commit and push.

## Stages

| Stage | What | Who | Estimate (agent / machine) | Done when | Status |
|---|---|---|---|---|---|
| S0 | Commit the approved plan on `feat/docker-v1`; push | parent | minutes | plan pushed | **done** `4a00c14` |
| S1 | Cheap measurements: aarch64 resolution of the three locks (dry run, no emulation); `torch==2.2.2` from the CPU index for linux x86_64/aarch64 (trial relock in a scratch copy); `COPY --exclude` on this engine; every place a runtime version string is compared | parent | 0.25 day / minutes | findings recorded below; design amended if needed | **done** (record below) |
| S2 | Repository changes: `.gitattributes`, `.dockerignore`, `.env` ignored; `config.MODELS` and the three asset modules; the two download pins; runtime locks on the CPU index (Linux only); tests shown failing first | sub-agent | 0.25-0.5 day / minutes | `uv run pytest` = the base failures only; pushed | **done** `907c61c`, D68 `75ef929` |
| S3 | `Dockerfile`, `compose.yaml`, `compose.dev.yaml`, `.env.example`, `tests/test_docker_contract.py`; build `app` and `test` | sub-agent writes; parent builds (long job) | 0.5 day / build est. 20-40 min | both images build; pushed | **done** `9c7b5e3` (built, record below) |
| S4 | Run-time audit: the image with the owner's `data/` mounted read-only and scratch copies for write paths; a scripted session with an `open` audit hook | parent | 0.25 day / minutes | the folder's file list and write paths recorded; `04` amended | **done** (record below) |
| S5 | `candidate_screener.datapack` (`create`, `--with-workspace`, `--source`, `verify`, `handover`, `purge`) and tests; then the owner's `create --with-workspace` | sub-agent; parent runs `create` | 0.5 day / minutes | tests pass; `data/datapack/` verified; pushed | **done** `4e8fb6d` (record below) |
| S6 | Verification V1-V9, V11 (V10 deferred) | parent | 0.5 day / V8 cold build est. 20-40 min, V2 ~15 min | every row recorded; pushed | **done** (record below; V2b and V4 settled by the owner, 28 Oct 2026) |
| S7 | Docs: `docs/docker.md`, README, AGENTS.md, `data/README.md`, the workspace card, `docs/project-overview.md` §7, D68, `docs/docker-optional-parts.md` | sub-agent | 0.25-0.5 day | contract test passes; pushed | **done** (record below) |
| S8 | Independent review; fixes; `uv run pytest`; the warm affected check run (long job); phase 1 marked complete | reviewer sub-agent; parent | 0.25 day / ~70 min | review findings closed; checks pass; pushed | **in progress** |
| S9 | Phase 2: the `llm` profile (Qwen2.5-0.5B service) and V3's v2/v3 rows | parent or sub-agent | 0.25 day | v2 and v3 answers composed by Qwen in the container; pushed | pending |
| S10 | Account-creation screen, on its **own branch** (below) | sub-agent | 0.5 day | tests pass; branch pushed; merge left to the owner | pending |
| S11 | Apple Silicon (arm64) test under emulation (below) | parent | 0.25 day / several hours of emulated runs | V10 recorded against its registered bar; emulation removed; pushed | pending |

## After v1: continue with v2 without waiting (owner, 27 Oct 2026)

"If I do not respond after v1 is completed, continue with v2 (test the Apple Silicon files) unless
there are blocking items that need my response."

- **v1** = S0-S8 (Docker phase 1). **v2** = S9, S10 and S11. When S8 is pushed and the owner has
  not replied, v2 starts on its own.
- **Blocking** (stop and wait for the owner): a registered stop condition (a failing V2 item; the
  V10 bar exceeded), a decision this plan does not cover, a build failure that needs a design
  change. Anything else: record it and continue.
- Branches: v2's Docker work stays on `feat/docker-v1`; S10 uses `feat/account-create-screen`.

### S11: the Apple Silicon test

- This machine has no arm64 CPU. **The owner installed the QEMU emulation themselves on
  27 Oct 2026** (`tonistiigi/binfmt --install arm64`). The agent never runs the install or
  uninstall commands. S11 first confirms that `docker buildx ls` lists `linux/arm64`. The
  registration does not survive a WSL restart; if it is gone, that is a blocking item: ask the
  owner to reinstall it.
- **When everything is done** (S11 and the rest of v2), remind the owner to uninstall it
  (`docker run --privileged --rm tonistiigi/binfmt --uninstall qemu-aarch64`).
- Build the arm64 image (`--platform linux/arm64`, its own tag, never replacing the amd64 image),
  then V1 (tests), V3 (smoke) and V2b (parity) against the amd64 results with the bar registered
  in the README's V10 row. Emulation is slow: D64 runs on a handful of C2 pairs rather than 50, and
  the record says so. Every emulated run is a long job under rule 3.
- What it can and cannot prove: emulation executes arm64 code, so it catches missing arm64
  packages and real numerical differences of arm64 builds; it does not measure Apple Silicon
  speed or Docker Desktop's macOS behaviour (file sharing, memory), which wait for a real Mac.

## S10: the account-creation screen (owner, 27 Oct 2026)

- **No accounts → a "create the first account" screen; otherwise the sign-in screen.** No
  administrator role yet: further accounts keep using the command.
- Branch `feat/account-create-screen`, off the frontend branch head at the time (today
  `feat/hiring-desk-v3-inference-prompt` at `e530659`), so it merges into the frontend line.
- Backend: reuse `portal_auth.create_account` (validation, hashing, locking, no overwrite) through
  a new `create_first_account` that checks, **under the same file lock**, that no account exists,
  so two people submitting the form at once cannot both succeed.
- The form asks for a login ID, departments, the token twice (at least 24 characters, hidden) and
  the expiry (default and maximum 40 days).
- Accepted risk (owner's choice, no setup code): while no account exists, whoever opens the page
  first creates the first account. The port is published on host loopback only, so that means
  someone on the same machine. If the port is ever exposed on a network, add a one-time setup code
  printed in the container log.
- Tests (synthetic, shown failing first): the screen shows only with no accounts; the second
  concurrent creation is refused; a created account signs in. The Docker guide's account section
  changes when this merges.

## Implementation record

(Each stage appends its results here: commits, measurements, interruptions and restarts,
deviations.)

### S0 (27 Oct 2026)

Plan committed and pushed as `4a00c14` on `feat/docker-v1`, off `e530659`.

### S1: cheap measurements (27 Oct 2026, uv 0.12.7, Docker Engine 27.5.1, BuildKit 0.18.2)

1. **arm64 without emulation.** `uv sync --frozen --dry-run --python-platform …` resolves every
   lock for `aarch64-unknown-linux-gnu` and `x86_64-unknown-linux-gnu`: the main lock with all
   groups (torch `2.14.0+cpu` on both), `runtime/transformer` and `runtime/gliner-medium` (torch
   `2.2.2`). On x86_64 the two runtimes still pull the CUDA build plus 12 `nvidia-*` packages; on
   aarch64 they get a CPU wheel.
2. **CPU torch for the runtimes (trial relock of a scratch copy of `runtime/transformer`).**
   Adding `[tool.uv.sources] torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]`
   with the explicit PyTorch CPU index, then `uv lock`: linux x86_64 gets `torch 2.2.2+cpu`
   (`linux_x86_64` wheel), linux aarch64 `torch 2.2.2` from the CPU index; the macOS and Windows
   entries stay on PyPI's `2.2.2` (the teammate's native runtime is unchanged); every `nvidia-*`
   package and triton leave the lock; no other package changes version. S2 applies this to both
   runtimes.
3. **`COPY --exclude` works** with `# syntax=docker/dockerfile:1`, which resolved to
   `docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e`
   (S3 pins that digest). Patterns are **relative to the source path**: with source `pkg/`,
   `--exclude=app` drops `pkg/app`, while `--exclude=pkg/app/**` drops nothing; from the context
   root, `--exclude=src/candidate_screener/app` works. The explicit-list fallback is not needed.
4. **Runtime version strings.** The portal's GLiNER-medium path stores the worker's versions but
   compares them with nothing; `llm/qa_assets.py` records `torch 2.2.2` as a literal and compares
   nothing. Only research campaign modules (`extraction/live.py`, `medium_run.py`, `fit_run.py`,
   `transformer_run.py`) compare against recorded files, and those run natively. A `2.2.2+cpu`
   torch therefore changes nothing in the app.

### S2: repository changes (27 Oct 2026, `907c61c`; sub-agent, reviewed by the parent)

- As planned: `.gitattributes`, `.dockerignore`, `.env` ignored; `config.MODELS`
  (`SCREENER_MODELS_ROOT`, resolved against `PROJECT_ROOT` when relative; unset = `PROCESSED`);
  `qa_assets`, `medium_assets` and `qwen_assets` build their folders from it; `--revision` and
  `--encoder-revision`; both runtime locks on the CPU index for Linux.
- Tests: `tests/test_config_models_root.py`, 13 tests, failing 13/13 against the unchanged code
  (`logs/s2-tests-before.log`), passing after (`logs/s2-tests-after.log`); two mutants caught
  (`logs/s2-tests-mutations.log`). Full suite (`logs/pytest-s2.log`): 2117 passed, 18 failed: the 17
  known frontend failures plus `test_decisions.py::test_every_cited_decision_is_in_the_log`, caused
  by this plan citing D68 before the log had it (since `4a00c14`, not by S2).
- **Deviation:** D68 was added to `docs/decisions.md` now (`75ef929`) rather than in S7, which
  fixes that test; the owner approved D68 on 27 Oct 2026.
- Locks (both runtimes alike): torch `2.2.2+cpu` (linux x86_64, CPU index), `2.2.2` (linux aarch64,
  CPU index), PyPI's `2.2.2` with unchanged hashes on macOS and Windows; 12 `nvidia-*` packages
  removed; no other version changed. Dry runs for x86_64 and aarch64 exit 0 with identical package
  lists apart from torch; macOS and Windows dry runs give the same lists as the old locks.
- `.gitattributes` rewrites nothing: `git ls-files --eol` identical before and after for all 846
  tracked files; `git add --renormalize .` stages nothing.
- Accepted choices: `.dockerignore` keeps `.claude/agents/a1-judge.md` (four tests read it);
  `**/` prefixes for nested patterns; `qa_assets --revision` refuses an existing checkpoint at
  another revision; `--verify` ignores the revision flags; `qwen_assets` now imports
  `candidate_screener.config` (still standard library only).
- For the guide (S7): on native Apple Silicon, uv's default macOS 13.0 target cannot install the
  runtime locks' `onnxruntime` (macOS 14 wheels only), before and after this change.

### S3: Dockerfile and compose (27 Oct 2026, files `9c7b5e3`; sub-agent, reviewed by the parent)

- Files: `Dockerfile` (stages uv, base, deps, test, models, app; `app` last, so it is the
  default), `compose.yaml` (app, qwen/`llm`, tools/`tools`, test/`test`), `compose.dev.yaml`,
  `.env.example`, `tests/test_docker_contract.py` (66 tests: 8 on the files, 57 negative cases,
  1 sanity check; 21 deliberate breakages of copies of the real files each caught,
  `logs/s3-contract-tests.log`). `docker build --check` and `docker compose config` clean. Full
  suite: the 17 known frontend failures (`logs/pytest-s3.log`).
- Pins: `python:3.12-slim-bookworm@sha256:fc169880…` (Python 3.12.16; native is 3.12.3),
  `ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1f…` (both multi-architecture); RoBERTa-SQuAD2
  `adc3b06…` (the revision earlier plans record for the teammate's checkpoint); GLiNER-medium's
  encoder `microsoft/deberta-v3-base` `8ccc9b6…`; Qwen2.5-0.5B `7ae5576…`; C2 `e61197e…`.
- The models stage downloads into a cache mount, one folder per pin, copies into `/opt/models`
  and `/opt/hf`, then verifies offline (`qa_assets --verify`, `medium_assets --verify`, Qwen's
  digest, C2 with `local_files_only`). Everything under `/app` and `/opt` is world-readable.
- **First build** (job `logs/docker/s3-build.log`, from `9c7b5e3`): `app` built in 2 min 59 s
  (packages 90 + 29 + 60, models: RoBERTa 26 s, GLiNER-medium 30 s, Qwen 28 s, C2 21 s), `test`
  in 7 s after it; exit 0 for both. Image sizes: `app` 7.98 GB, `test` 4.59 GB. Ten base steps
  came from an earlier cache, so these are not cold figures (V8 measures those).
- Deviations, accepted: `init: true` and `stop_grace_period` on every service; `tools` also mounts
  `logs/`; the healthcheck has a 10 s timeout, 3 retries and no proxy; the optional `hf_token`
  build secret is wired in the Dockerfile only (the guide shows the `buildx` form).
- For the guide (S7): on Linux run `mkdir -p data/datapack logs` before the first `up` (compose
  would create them as root); the build needs `COPY --exclude`, symbolic `--chmod` and optional
  secrets (works on Engine 27.5.1, BuildKit 0.18.2); the cache mounts hold a second copy of the
  weights (`docker builder prune --filter type=exec.cachemount`).

### S4: run-time audit (27 Oct 2026; sub-agent, reviewed by the parent)

- Method: the `app` image plus strace (`-f`, file syscalls and `connect`), as uid 1000,
  `--network none`, the owner's data read-only, an empty scratch `hr-portal/` and scratch copies of
  the cache folders. Three runs: the full session; report and mail; D63/D64 from empty caches.
  Traces and summaries: `scratchpad/s4/out/` (session-local).
- Every step ran: the entry script via `AppTest` (sign-in, a CSV review, the CV dialog; no
  exception), an account created and authenticated, 60 CSV jobs, D63 ready in 14.4 s (37.1 s from
  empty caches), D64 ready in 120.5 s (50 uncached C2 pairs), the RoBERTa worker answered (verdict
  "not found": V3 needs a question with an extractable answer), GLiNER-medium found 12/62/57
  mentions. Models were read only from `/opt` (RoBERTa 497.6 MB, GLiNER-medium 845.4 MB, C2 1.21 GB).
- Inputs: exactly the list now in `04` §1 (24 files, 256.8 MB), the same in every run; nothing
  missing. `concept-cache/` and `jd-title/` are unused (dropped from the list).
- Writes: `processed/hr-portal/`, `processed/embeddings/`, `processed/rerank/cross/`, `logs/`,
  `/tmp`, `/dev/shm`. No write was refused by the read-only data; nothing changed under `/app` or
  `/opt`.
- **Findings and what was done** (Docker settings only; no app code changed):
  1. transformers 4.48 in the model runtimes tried to write `/opt/hf/hub/version.txt` (refused,
     harmless warning): the image now writes that marker at build time.
  2. onnxruntime, imported by GLiNER in both runtimes, writes telemetry files under `$HOME` and
     looks up `mobile.events.data.microsoft.com`; offline it fails harmlessly, online it would
     send telemetry. `ORT_DISABLE_TELEMETRY=1` (now in the image) removes both (measured: 16
     telemetry file operations without it, 0 with it). **For the owner:** native runs outside
     Docker still send it; disabling it there is a code or environment change, not made.
  3. Streamlit looked up the host's external IP at start (an HTTP request to
     `checkip.amazonaws.com`): the image now passes `--browser.serverAddress=localhost`, which
     skips it (and prints `http://localhost:8503`). The origin check still allows `localhost` and
     `127.0.0.1` without any network call.
  4. **App behaviour, not Docker (for the owner):** preparing a report or mail fails with a
     `ValueError` in `portal_store._opaque_identity` for 53 of the 600 CSV candidates (38 of 60
     JDs, including the first): their IDs fail the store's check after redaction. A JD whose
     candidates pass (the second) saves its report and captures the mail.
  Items 1-3 are verified in S6 (V5, V6).

### S5: the data folder commands (27 Oct 2026, `4e8fb6d`; sub-agent, reviewed by the parent)

- `src/candidate_screener/datapack.py` (`create`, `verify`, `handover`, `purge`) and
  `tests/test_datapack.py` (30 tests; 20 deliberate breakages caught,
  `logs/s5-tests-evidence.log`); compose's `tools` service sets `SCREENER_DATAPACK_SOURCE=/source`
  and `SCREENER_DATAPACK_DEST=/app/data`; the contract tests grew to 72. Full suite: the 17 known
  frontend failures (`logs/pytest-s5.log`). Importing the module loads no torch.
- Names come from the code that reads each file (`corpus`, `SbertWeak.cache_key("v2")`, the role
  and parts `LATEST` pointers, `rules.VOCAB_CSV`), so `verify` reports when the code expects a
  different file. It also reports the workspace's format (accounts 1, sqlite 2 today).
- Accepted choices beyond the brief: `config.DATA` is allowed as the destination only when
  `SCREENER_DATAPACK_DEST` names it (the `tools` container); `create` refuses unknown non-empty
  folders, a source inside the destination and symbolic links inside it; in Docker the vocabulary
  comes from the source (the repository copy is hidden by the mount), so a Docker `create` needs a
  source with `vocab/skills.csv` (a handover copy has it; the main checkout's `data/` does not);
  `handover` is best run natively (in Docker its destination must be on a mount).
- **The owner's folder** (27 Oct 2026, native, no native app running): `create --with-workspace`
  copied 24 inputs, 256.8 MB (4 from committed copies, 20 from `data/`), emptied the caches and
  copied the workspace (3 files, 0600 in a 0700 folder; inputs 0644/0755); its closing full verify
  said OK at commit `4e8fb6d` (`logs/docker/s5-create.log`). `purge` without `--yes` lists 28 files
  and deletes nothing.

### S6: one-time verification (27 Oct 2026, images rebuilt at `f8dc77b`; sub-agent, reviewed by the parent)

Results: `logs/docker/s6-results.md` and the `logs/docker/s6-*.log` files (git-ignored).

| Row | Result | Key numbers |
|---|---|---|
| V9 | pass | 34/34 digests identical in a `core.autocrlf=true` clone; at `e530659` (no `.gitattributes`) all 34 differ |
| V1 | pass | the same 17 failing test ids as native; 2207 passed / 51 skipped (native 2220 / 38): 12 tests skip without `data/`, 1 without the host's Hugging Face cache (`test_extraction_ner.py`); root user and no `.git` changed nothing |
| V2 | pass | `inference --check` 14/14 in the container, 614 s (native 600 s), peak 2.36 GB |
| V3 | pass | healthy in 7.6 s; host health `ok`; sign-in, CSV review (top 10, source preview); D63 35.2 s and D64 102.5 s from empty caches; an extractive answer with its source; entities 13/62/57 |
| V5+V6 | pass | read-only image, tmpfs `/tmp`, no network: every step; D63 43.2 s, D64 93.9 s; S4's three fixes confirmed |
| V2b | **failed the registered bar; owner decides** | same top-10 ids and order in 4/4 rankings; one JD's D63 had 2 of 10 scores differing at the 6th decimal (4e-6) |
| V7 | pass | (a) frontend change: 0.70 s, models stage cached; (b) lock change: 70.8 s, the models stage re-ran from its cache in 16.5 s with nothing downloaded |
| V8 | recorded | cold build 202.6 s; 4,382 MiB downloaded; peak anonymous memory 1.26 GB (cgroup 10.3 GB, mostly page cache); builder cache 14.7 GB; image 7.98 GB |
| V11 | pass | `verify --full` OK; `purge` without `--yes` deleted nothing; 4 handover copies verified, none with a workspace or caches |
| V4 | owner | browser check by hand |
| V10 | deferred | S11 |

**V2b cause (app behaviour, also native):** `rerank/cross.py`'s C2 loader calls
`torch.set_num_threads(6)` for the whole process. Natively, C2 had loaded before the second JD's
D63, so its embedding ran on 6 threads; in the container C2's scores were already cached, C2 never
loaded, and the embedding ran on 11. Embeddings at a given thread count are identical in both
environments, and a container rerun with the native order of operations matched bitwise in all
four rankings. The embedding cache keeps whichever value came first. Not patched.

Also: the native V2b run added the two invented JDs to the owner's `processed/embeddings/` npz
(+3.6 kB) and 20 C2 score files to `processed/rerank/cross/873b24509614/`; nothing else in `data/`
changed. The default builder's cache grew from 6.35 to 14.26 GB (V7b; not pruned). A Windows
checkout gets CRLF in `uv.lock`, `runtime/*/uv.lock` and `.python-version` (outside V9's set; uv
accepts them). The WSL VM was suspended for about 5 hours during V5+V6 (host sleep); timings use
the monotonic clock.

**Owner, 28 Oct 2026.**
- **V2b passes** under the owner's amended bar, "5 dp", recorded as every score within 1e-5:
  the largest gap is 4e-6 (JD-B D63); the other three rankings are identical. (Rounding both runs
  to 5 dp would still differ once, because a 4e-6 gap can straddle a rounding boundary; a
  tolerance avoids that.) Same top 10, same order, in all four rankings.
- **Known app issue (not Docker):** C2's loader sets `torch.set_num_threads(6)` for the whole
  process, so scores of a JD embedded after C2 first loads can differ in the 6th decimal; the
  embedding cache keeps the first value. Recorded for the guide's known limitations; no change.
- **V4 passes:** the owner ran `docker compose up`, signed in through the frontend, opened a JD,
  re-ranked with C2 and asked chat questions; all worked.

### S7: documentation (28 Oct 2026; sub-agent, reviewed by the parent)

- New: `docs/docker.md` (the maintainer guide, sections 1-16 of `02`), `docs/docker-optional-parts.md`
  (the living log, moved from `03`, which now links to it). Updated: `README.md` ("Run with Docker
  (any OS)"), `AGENTS.md` (commands, "Where code goes", the D68 rules paragraph), `data/README.md`,
  `docs/data/cards/HR-operational-workspace.md`, `docs/project-overview.md` §7, both runtime READMEs.
- `tests/test_docker_contract.py`: every `HR_PORTAL_*`/`SCREENER_*` name read under `src/` (29 today)
  must appear in the guide's configuration reference; 13 tests including 11 broken cases; removing
  a row from the real guide fails it. Full suite (`logs/pytest-s7.log`): 17 failed (the known ids),
  2233 passed. All 133 relative links resolve.
- Figures the guide cites that exist only in git-ignored logs, recorded here: V1 ran in 98 s; in
  V3 the app process peaked at 5.26 GB and the whole container at 7.56 GB including page cache
  (hence the guide's "at least 10 GB of Docker memory for all of phase 1; 4 GB for D63 only");
  `tools verify` takes about 2 s. V8's download is 4,382 MiB = 4.60 GB (4.28 GiB), not "4.4 GB".
- Not exercised, and said so in the guide: a Docker `tools create`, typing the hidden token prompt
  in a container, `compose.dev.yaml`, the `hf_token` secret, proxies, PowerShell forms.
- Parent fixes: `04` §3 now removes old images with `docker image prune` (`docker compose down
  --rmi local` skips custom tags); `.env.example` gives the PowerShell form of `GIT_COMMIT`.
