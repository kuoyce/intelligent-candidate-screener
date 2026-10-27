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
| S3 | `Dockerfile`, `compose.yaml`, `compose.dev.yaml`, `.env.example`, `tests/test_docker_contract.py`; build `app` and `test` | sub-agent writes; parent builds (long job) | 0.5 day / build est. 20-40 min | both images build; pushed | **in progress** (sub-agent writes the files; then the build job) |
| S4 | Run-time audit: the image with the owner's `data/` mounted read-only and scratch copies for write paths; a scripted session with an `open` audit hook | parent | 0.25 day / minutes | the folder's file list and write paths recorded; `04` amended | pending |
| S5 | `candidate_screener.datapack` (`create`, `--with-workspace`, `--source`, `verify`, `handover`, `purge`) and tests; then the owner's `create --with-workspace` | sub-agent; parent runs `create` | 0.5 day / minutes | tests pass; `data/datapack/` verified; pushed | pending |
| S6 | Verification V1-V9, V11 (V10 deferred) | parent | 0.5 day / V8 cold build est. 20-40 min, V2 ~15 min | every row recorded; pushed | pending |
| S7 | Docs: `docs/docker.md`, README, AGENTS.md, `data/README.md`, the workspace card, `docs/project-overview.md` §7, D68, `docs/docker-optional-parts.md` | sub-agent | 0.25-0.5 day | contract test passes; pushed | pending |
| S8 | Independent review; fixes; `uv run pytest`; the warm affected check run (long job); phase 1 marked complete | reviewer sub-agent; parent | 0.25 day / ~70 min | review findings closed; checks pass; pushed | pending |
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
