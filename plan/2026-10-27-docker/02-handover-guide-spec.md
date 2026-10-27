# 02 — Maintainer guide: `docs/docker.md`

Written in step 6, from what steps 1-5 measured: no command, size or time in it may be
unmeasured. It covers running and maintaining the containerised Hiring Desk; for the system itself
it links to `docs/project-overview.md`, `docs/ownership/` and `AGENTS.md` instead of repeating them.

It serves two readers, and says at the top which sections are whose:

- **the operator**, who receives the repository and a data folder and runs the app, on an unknown
  OS: build, start, accounts, backups, troubleshooting. Runs no long checks (owner, 27 Oct 2026);
  at most `verify`, which takes seconds;
- **the developer** (today the owner), who develops natively, decides when a change is settled,
  refreshes the image and the data folder, runs the checks on their own machine, and hands over.

## Rules for the guide

- Every command works unchanged in bash, zsh and PowerShell, or is given in both forms. Settings
  go in `.env`, not inline `VAR=value` (PowerShell does not accept it).
- Each figure (time, size, memory) carries its date and the machine it was measured on.
- No real CV or JD text, no tokens, no reviewer names in examples.
- It says what is *not* verified (e.g. "macOS Apple Silicon: expected to work, not verified").

## Sections

1. **What this is.** One paragraph and the diagram of `01-design-spec.md` §1: the image (code,
   environments, pretrained model weights) plus the data folder `data/datapack/` (the project's
   own files, and the workspace people create). What it is not: a public deployment (no TLS, SSO
   or real mail), the development setup, the research reproduction.
2. **Why Docker, in one paragraph.** Weights are portable; per-OS packages are not (the Intel-Mac
   SBERT failure); the container gives every OS the same Linux packages.
3. **Requirements, per OS.** Windows 10/11 (Docker Desktop on WSL2, where to set its memory, keep
   the clone inside WSL2), macOS Intel and Apple Silicon (Docker Desktop, Settings → Resources,
   file sharing), Linux (Docker Engine and the compose plugin; `HOST_UID`/`HOST_GID`). Measured
   minimums: Docker memory for D63 only and with D64, the chat workers and Qwen; disk for the
   image, the folder and build cache; network for the build only.
4. **Quick start (operator).** Clone (Windows: the line-ending note), build, put the received
   folder at `data/datapack/`, create a reviewer, start, open http://localhost:8503, sign in. The
   expected output and duration of each step.
5. **The data folder.** What it contains and why (`04-app-data-folder.md` §1); what is replaceable
   (inputs, caches) and what is not (`processed/hr-portal/`); the five commands (`create`,
   `create --with-workspace`, `verify`, `handover`, `purge`); what each `verify` finding means.
6. **Daily operation (operator).** Start, stop, restart, status and health, logs
   (`docker compose logs`; `logs/` on the host), what survives a restart, an image rebuild and a
   purge.
7. **Accounts.** How they work (README "Accounts in a new Docker build"): one per reviewer with
   the `docker compose run --rm app …` command, departments, expiry (default and maximum 40 days),
   a lost or expired token (a new login, or the documented edit of `accounts.json`, until the
   in-app flow ships), the hidden prompt on Windows, accounts never travelling in a handover.
8. **Configuration reference.** Every setting: name, Docker default, effect, safe to change. The
   contract test keeps it complete. Covers D63/D64, the chat versions (v1 default; v2/v3 with the
   `llm` profile), the embedding backend (`torch` in Docker; `onnx` is not used),
   `SCREENER_MODELS_ROOT`, `HIRING_DESK_PORT`, `HOST_UID`/`HOST_GID`, `SCREENER_DATA_SOURCE`.
9. **Backups (operator).** Backing up and restoring `processed/hr-portal/` with commands that work
   on every OS; doing it before every upgrade.
10. **Releasing an update (developer).**
    - When: a change is settled in native development.
    - The flows of `04-app-data-folder.md` §3: not breaking (rebuild, `create`, workspace kept)
      and breaking (`purge --yes`, then `create` or `create --with-workspace`); stating which in
      the commit; removing the old image.
    - Checklist: a new package (`uv add`, commit the lock); a new setting (add it to §8, or the
      contract test fails); a new file the app reads (add it to the folder's list, `create`,
      `verify`); a new pretrained model (a pinned step in the `models` stage); **review the
      optional-parts log** (`docs/docker-optional-parts.md`) and add a row.
    - Checks on the developer's machine for that kind of update (the plan's verification rows:
      frontend, tests and smoke; lock or pin, also the cold build; data, `verify` and the parity
      run). Never at each start.
    - Trying a frontend change in Docker first: `docker compose -f compose.yaml -f compose.dev.yaml up`.
    - `handover` and passing the folder privately.
11. **Tests.** `docker compose run --rm test`; the known failures with the commit they were
    measured at; comparing with a native run; the no-data rule of the suite.
12. **Troubleshooting.** Exit 137 (memory); disk full and `docker builder prune`; something
    "unavailable" in the app → run `verify` (missing, changed or stale file; CRLF on Windows; file
    owners on Linux; file sharing on macOS); build download failures, Hugging Face rate limits and
    the token secret; proxies and CA certificates; port in use; "ranking unavailable" (the adapter
    log); a worker reported unavailable; Qwen not answering (the `llm` profile, restarting it after
    `app`); a slow first ranking (model loading, empty caches); the account prompt in Git Bash.
13. **How the image and the folder are built.** The stage diagram; every pinned input with its
    source, revision and licence as read from its model card at implementation time; where each
    folder file comes from (copied by `create`, or committed).
14. **Security and privacy.** Loopback-only port, local tokens, secrets never in the image, the
    folder holds anonymised real text, the project's models and the workspace (keep it private;
    hand over with `handover`), redaction unchanged.
15. **Known limitations.** Frontend in progress (failing tests); v3 experimental; accounts by
    command until the in-app flow ships; Qwen3.5-4B and other parts not built (link to the
    optional-parts log); arm64 not yet run; no ONNX backend; C1/ESCO not included; no CI.
16. **First day for a new maintainer.** Run the quick start, run the tests, read
    `docs/project-overview.md` and `AGENTS.md`, take a backup of `processed/hr-portal/`, know who
    holds the data folder and who releases updates.

## Acceptance

- The contract test passes (every setting documented).
- A reader who has not used the project follows §4 on a clean machine and reaches the sign-in
  page: the owner on Linux/WSL2 now; Windows and macOS once someone runs them (after the Docker
  build is complete). Every place a reader stumbles is fixed in the guide and noted in the
  implementation record.
- README gets a short "Run with Docker" section linking here; AGENTS.md gets the Docker commands
  and the rule that a new run-time file, setting, pretrained model or optional part updates the
  folder list, the `models` stage, the guide or the log in the same change.
