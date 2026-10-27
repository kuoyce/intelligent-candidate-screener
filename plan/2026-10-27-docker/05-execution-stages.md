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
| S0 | Commit the approved plan on `feat/docker-v1`; push | parent | minutes | plan pushed | **in progress** |
| S1 | Cheap measurements: aarch64 resolution of the three locks (dry run, no emulation); `torch==2.2.2` from the CPU index for linux x86_64/aarch64 (trial relock in a scratch copy); `COPY --exclude` on this engine; every place a runtime version string is compared | parent | 0.25 day / minutes | findings recorded below; design amended if needed | pending |
| S2 | Repository changes: `.gitattributes`, `.dockerignore`, `.env` ignored; `config.MODELS` and the three asset modules; the two download pins; runtime locks on the CPU index (Linux only); tests shown failing first | sub-agent | 0.25-0.5 day / minutes | `uv run pytest` = the base failures only; pushed | pending |
| S3 | `Dockerfile`, `compose.yaml`, `compose.dev.yaml`, `.env.example`, `tests/test_docker_contract.py`; build `app` and `test` | sub-agent writes; parent builds (long job) | 0.5 day / build est. 20-40 min | both images build; pushed | pending |
| S4 | Run-time audit: the image with the owner's `data/` mounted read-only and scratch copies for write paths; a scripted session with an `open` audit hook | parent | 0.25 day / minutes | the folder's file list and write paths recorded; `04` amended | pending |
| S5 | `candidate_screener.datapack` (`create`, `--with-workspace`, `--source`, `verify`, `handover`, `purge`) and tests; then the owner's `create --with-workspace` | sub-agent; parent runs `create` | 0.5 day / minutes | tests pass; `data/datapack/` verified; pushed | pending |
| S6 | Verification V1-V9, V11 (V10 deferred) | parent | 0.5 day / V8 cold build est. 20-40 min, V2 ~15 min | every row recorded; pushed | pending |
| S7 | Docs: `docs/docker.md`, README, AGENTS.md, `data/README.md`, the workspace card, `docs/project-overview.md` §7, D68, `docs/docker-optional-parts.md` | sub-agent | 0.25-0.5 day | contract test passes; pushed | pending |
| S8 | Independent review; fixes; `uv run pytest`; the warm affected check run (long job); phase 1 marked complete | reviewer sub-agent; parent | 0.25 day / ~70 min | review findings closed; checks pass; pushed | pending |
| S9 | Phase 2: the `llm` profile (Qwen2.5-0.5B service) and V3's v2/v3 rows | parent or sub-agent | 0.25 day | v2 and v3 answers composed by Qwen in the container; pushed | pending |
| S10 | Account-creation screen, on its **own branch** (below) | sub-agent | 0.5 day | tests pass; branch pushed; merge left to the owner | pending |

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
