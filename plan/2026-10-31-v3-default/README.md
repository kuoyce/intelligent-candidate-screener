# Chat v3 with Qwen3.5-4B as the default, natively and in Docker

Status: **approved by the owner (31 Oct 2026), "approve as recommended"**: Q1-Q5 take the
recommended answers. Branch `feat/v3-default` off `release/integration` (`0f83030`).

## Request (owner, 31 Oct 2026)

- "change v3 to be the default now".
- `docker compose up -d` did not work for the owner; `docker compose --profile llm up -d` did.
  Cause (verified in the files): the owner's `.env` sets `HR_PORTAL_CHAT_VERSION=v3`, the
  value `.env.example` carries since `c52a06f`, while every Qwen service sits behind a profile or an
  override file. Plain `up -d` therefore starts only `app`, and every v3 turn falls back.
- The teammate runs the Docker application on an Intel Mac.

Answers (owner, 31 Oct 2026):

| Question | Answer |
|---|---|
| Model behind default v3 | **Qwen3.5-4B** (the D67 GGUF, `llm4b` image), not Qwen2.5-0.5B |
| Should plain `docker compose up -d` start the model too? | **yes**: one command brings everything up |
| Where is v3 the default? | **Docker and native** (the code default too) |
| Branch | **a new branch off `release/integration`** |

## Measured before the design (31 Oct 2026, this machine)

- **Tests** (`logs/v3-default/`): on `release/integration` the suite has 18 failures:
  - the 17 known ones (11 in `test_portal_extractive`, 3 in `test_portal_ui`, 3 in `test_portal_v1_ui`);
  - 1 in `test_conversation_summary`.

  With `HR_PORTAL_CHAT_VERSION=v3` set, there are 21 failures. The 3 new ones all assume v1 without
  setting it:
  - `test_portal_ui.py::test_primary_extractive_chat_does_not_instantiate_configured_qwen`;
  - `::test_local_qwen_settings_do_not_control_extractive_chat`;
  - `::test_chat_answers_question_before_collapsed_supporting_sources`.
- **Why a 4B turn takes 35-57 s** (from `logs/gguf/g2a-walkthrough-2.json`, `g2c-idle.json` and
  the probe's timings):
  - each v3 prompt is 1,285-1,720 tokens;
  - on 4 CPU threads, llama.cpp reads about 40-45 tokens/s and writes about 11 tokens/s;
  - so reading the prompt is about 85-90% of a turn, and turn time grows with the prompt.
  - A ranking that runs at the same time pushes most turns past the 60 s budget (G2c: three of four).
- **Memory** (G2b-G2c): `llm4b` 4.42 GB at rest, 4.66 GB at peak; with D64 running, `app` and
  `llm4b` together about 11.5 GB. The guide asks for at least 15 GB when the 4B runs. A machine
  with 16 GB in total cannot give Docker that (see Q3).

## Design

### 1. The code default (native and Docker)

- One reader, `chat_version()`, in `app/` (next to its current readers). It returns
  `HR_PORTAL_CHAT_VERSION` lower-cased, **default `v3`**, and any unknown value still behaves as
  `v1`, as today.
- It replaces the four inline reads: `portal_workspace.py:61` and `portal_ui.py:319, 559, 733`.
- `v1` stays the rollback (`HR_PORTAL_CHAT_VERSION=v1`), and `v2` is unchanged.

### 2. Which Qwen v3 uses when nothing is set (native)

`configured_prose_composer` today, when no `SCREENER_QWEN_*` is set, uses the 0.5B settings (an
explicit `SCREENER_QWEN_MANIFEST` selects the 4B). The new order:

1. explicit `SCREENER_QWEN_BASE_URL` or `SCREENER_QWEN_MODEL`: as today;
2. explicit `SCREENER_QWEN_MANIFEST`: as today;
3. **new:** the 4B manifest at `gguf_assets.DESTINATION / "manifest.json"`
   (`config.MODELS/chatbot/qwen35/`), if it exists;
4. otherwise, as today: the 0.5B settings (Q1).

The composer still loads configuration only. It never downloads, starts a server or trains. If
the 4B facade is not running, v3 gives its marked fallback, as today.

Native start (documented, unchanged commands), in two terminals:

- `uv run python -m candidate_screener.llm.gguf_assets` once (the 2.74 GB download);
- `uv run python -m candidate_screener.llm.gguf_server --manifest … --launch-backend`, then the
  Hiring Desk.

The app does not start the model service itself (Q2).

### 3. Docker: one command

- **`compose.yaml` takes in `compose.llm4b.yaml`'s contents**, unchanged:
  - the `llm4b` service: its own image, target `llm4b`, `network_mode: service:app`, user, init,
    the manifest volume, `stop_grace_period`;
  - on `app`, `SCREENER_QWEN_MANIFEST` and the read-only manifest volume;
  - the pinned volume name `qwen35-manifest-e87f176-b11896`.

  `llm4b` has **no profile**. `compose.llm4b.yaml` is deleted (Q4), along with `.env.example`'s
  `COMPOSE_FILE` line.
- `app` defaults to `HR_PORTAL_CHAT_VERSION: "${HR_PORTAL_CHAT_VERSION:-v3}"`. `.env.example`
  shows `v3` with a comment that it is the default.
- The result:
  - `docker compose build` builds `app` and `llm4b`; the first build downloads the model;
  - `docker compose up -d` starts both;
  - the 0.5B `qwen` service stays in the `llm` profile, for v2 only;
  - `tools` and `test` are unchanged.
- **The lighter setup** for machines that cannot give Docker 15 GB:
  - set `HR_PORTAL_CHAT_VERSION=v1` in `.env`;
  - then run `docker compose up -d app`, which starts only `app`; `llm4b` is never built or started.
  - Documented in §3/§4 of the guide (Q3).
- The 60 s turn budget (D67) and `--threads 4` are unchanged. Speed is out of scope (§ Not in scope).

### 4. Rules and records

- **`AGENTS.md`:**
  - "Keep v3 opt-in until quality acceptance" becomes "v3 with the 4B is the default (owner, 31
    Oct 2026); v1 is the rollback".
  - The Docker commands block loses the `-f compose.llm4b.yaml` lines.
  - The quality caveat stays: v3 is experimental and its answers fall back often. Being the default
    is not quality acceptance.
- **`docs/decisions.md`:** a new **D69**: the owner made v3 with Qwen3.5-4B the default on 31 Oct
  2026. Not quality acceptance; v1 is the rollback; no training, no labels, no ranking change
  (Q5). D66's and D67's status lines point to it.
- **`docs/docker.md`:**
  - §1 diagram;
  - §3 memory: 15 GB for the default, about 10 GB for the v1 setup;
  - §4 quick start: one command, plus the first build's download;
  - §6: the `llm4b` section becomes the default, and the `llm` profile is for v2 only;
  - §8: the `HR_PORTAL_CHAT_VERSION` row's default is `v3`, `SCREENER_QWEN_MANIFEST` is set by
    `compose.yaml`, and the `COMPOSE_FILE` note is removed;
  - §12 troubleshooting: "every v3 answer is a fallback";
  - §15.
- **Also updated:**
  - `docs/docker-optional-parts.md`: the D67 row becomes "default";
  - `README.md`;
  - `plan/2026-10-25-hiring-desk-v3/04-qwen35-trial.md` and the GGUF plan README: a pointer line.

## Tests (each new or changed test is shown failing on the old code first)

- `chat_version()` returns `v3` when unset, `v1` for `v1` or an unknown value, and `v2` for `v2`.
  Each of the four call sites uses it.
- The three `test_portal_ui` tests above set `HR_PORTAL_CHAT_VERSION=v1` explicitly. They test v1
  behaviour, which stays the rollback.
- The composer, with no `SCREENER_QWEN_*` set:
  - if the 4B manifest exists under a temporary `MODELS`, the GGUF settings are used;
  - if it is absent, the 0.5B settings are used;
  - an explicit `SCREENER_QWEN_MANIFEST` and `SCREENER_QWEN_DISABLED` still win;
  - nothing is fetched or started.
- `test_docker_contract.py`:
  - `llm4b_compose_problems` reads `compose.yaml`;
  - `llm4b` has no profile, and `qwen` keeps `profiles: [llm]`;
  - `app`'s chat-version default is `v3`;
  - `compose.llm4b.yaml` no longer exists;
  - the guide's configuration table shows `v3` as the default.
- The full suite: the 18 current failures, no new ones.

## Stages

| Stage | What | Who | Estimate | Long job? |
|---|---|---|---|---|
| V0 | Commit this plan on `feat/v3-default` | parent | minutes | no |
| V1 | §1-§2 code and their tests | sub-agent; parent reviews | 1-2 h | no |
| V2 | §3 compose, `.env.example`, contract tests | sub-agent; parent reviews | 1 h | no |
| V3 | Docker check on this machine. Build with the image cache (`llm4b` was built on 28 Oct, so mostly cached; cold build 3 min 30 s plus the download). Then from a stopped state: plain `docker compose up -d --wait`, both healthy. One scripted v3 turn with invented documents through the app (`mode: qwen_prose` or a marked fallback, under 60 s). Then `docker compose up -d app` with `v1` (only `app` starts). | parent | about 15 min (more on a cold cache) | build only, in the background |
| V4 | Native check: provisioned 4B, `gguf_server --launch-backend`, Hiring Desk with no chat settings, one v3 turn uses the 4B | parent | about 10 min | no |
| V5 | §4 docs | sub-agent; parent reviews | 1-2 h | no |
| V6 | Independent review, fixes, full `uv run pytest` (about 4 min), `checks --explain` (no record is built from this code, so no warm/cold run is expected; D58 (research)), mark complete, push | parent + review sub-agent | 1 h | no |

The teammate's Intel Mac is not re-tested here. The guide states its memory need (Q3).

## Not in scope

- **Making turns faster:** more llama.cpp threads, shorter v3 prompts, or a larger turn budget.
  Each would change the pinned settings or D67's budget, and the equivalence records. Possible
  follow-up, not part of this change.
- **v3 quality acceptance**, held-out evaluation, or any training, labels or change to the ranking
  or the research winner.
- **The app starting the model service by itself** (Q2).

## Open questions

1. **Native, when the 4B is not provisioned:** fall back to the 0.5B settings (recommended: no
   behaviour lost for anyone without the 2.74 GB download), or to v3's marked fallback only?
2. **Native start:** keep the model service a separate command (recommended: today's rule that the
   app never starts a server), or have the Hiring Desk start `gguf_server` on its own?
3. **Memory, high impact.** The default needs about 15 GB for Docker (11.5 GB measured with D64
   running). The D67 Intel Mac has 16 GB in total, so it cannot give Docker that much. How much
   memory does the teammate's Docker Desktop have? Options:
   - keep 15 GB as the requirement, with the v1 setup (`docker compose up -d app`) for smaller
     machines (recommended);
   - measure a lower figure for "chat while no ranking runs" in V3 and document both.
4. **`compose.llm4b.yaml`:** delete it (recommended: one way to run), or keep an empty file so
   that old `-f compose.llm4b.yaml` commands still work?
5. **Decision log:** add D69 (recommended: this reverses an AGENTS.md rule, and D45 keeps every
   rule's reason in the log), or record it only in this plan?

## Implementation record

- **V0** `3d5dba5`: the plan, on `feat/v3-default`.
- **V1** `18af9bb` (sub-agent; reviewed by the parent):
  - `chat_version()` is in `app/portal_workspace.py` (`portal_ui` already imported that module,
    so there is no cycle).
  - `configured_prose_composer` looks the default 4B manifest up at call time.
  - 16 new tests failed on the old code (`logs/v3-default/v1-tests-before.log`); 57 pass after.
  - Edge cases, recorded:
    - an empty `HR_PORTAL_CHAT_VERSION=""` stays v1 as before; only a truly unset variable gives
      v3 (in Docker, compose's `:-v3` turns an empty value into v3);
    - a malformed default 4B manifest gives v3's marked fallback, as a bad explicit one does; it
      does not drop back to the 0.5B.
  - `app/conversation_v3_smoke.py --real-model` now picks the 4B natively when it is provisioned.
- **V2** `7e4ca4e` (sub-agent; reviewed):
  - `app` repeats the anchor's two mounts, because its own `volumes:` list replaces the anchor's.
  - `docker compose config --services`:
    - plain: `app llm4b`;
    - `--profile llm`: adds `qwen`;
    - `run --rm tools` and `run --rm test` start only their own service.
  - The real-file contract test failed before (`v2-tests-before.log`); 188 pass after.
  - **Found:** an `.env` with `COMPOSE_FILE=compose.yaml:compose.llm4b.yaml` (the old optional
    line, which the owner's `.env` has) makes every `docker compose` command fail with
    `stat …/compose.llm4b.yaml: no such file or directory`. The guide's §12 covers it. The
    owner's `.env` was not edited.
- **V3** (`logs/v3-default/v3-docker.log`; project `hd-v3`, scratch image tags and data, the
  owner's stack untouched; `--env-file` without a chat setting, so the true default was tested):
  - **Build:** plain `docker compose build` built `app` and `llm4b` in 81 s with a warm cache.
  - **The v1 setup:** with the manifest volume removed first, `up -d app` started `app` alone
    (`chat_version v1`), and the volume stayed empty.
  - **The default:** plain `up -d --wait` started `app` and `llm4b`, both healthy after 18.8 s.
    - `HR_PORTAL_CHAT_VERSION=v3`, and `chat_version v3`;
    - `llm4b` filled the manifest volume after the app-only run.
  - **Smoke** (`conversation_v3_smoke --real-model`, invented documents):
    - turn 1 `qwen_prose` in 58.0 s;
    - turns 2-4 `deadline_exceeded` at 60.1 s, marked fallbacks;
    - peaks: `app` 0.22 GB, `llm4b` 2.44 GB.
- **V4** (`logs/v3-default/v4-native.log`, `v4-native.json`): native, with every chat and
  `SCREENER_QWEN_*` setting unset.
  - `gguf_server --launch-backend` was ready, and `chat_version v3`. The composer is
    `IsolatedProseComposer` for `Qwen/Qwen3.5-4B` at `127.0.0.1:8001/v1`.
  - Smoke: turn 1 `qwen_prose` in 51.3 s; turns 2-4 `deadline_exceeded`.
- **Slower turns than on 28 Oct (observed, not investigated):**
  - Both V3 and V4 were slower: idle native turns took 34-53 s on 28 Oct.
  - Turn 1's prompt was the same 1,285 tokens.
  - During these runs, the WSL VM had all of its 4 GB of swap in use and 2.1 GB of memory free; the
    owner's 0.5B `qwen` container was running.
  - This change does not touch generation, prompts or the budget, so the machine's state is the
    likely factor. It is not tuned here. On a loaded 15.7 GB machine, most 4B turns fall back.
- **V5** (sub-agent; reviewed): AGENTS.md, D69 (D66 and D67 point to it), `docs/docker.md`,
  `docs/docker-optional-parts.md`, README and the two plan pointers.
  - **Deviation:** `docker-optional-parts.md` keeps its old dated rows; that file says never edit
    an old row, so a new 31 Oct row supersedes them.
  - The V3 figures above replace the guide's placeholders. The free-disk guidance rises to about
    30 GB, or 35 GB with the test image; the `llm4b` download cache was not measured.
  - Renamed anchors: `#chat-v3-with-qwen35-4b-the-default` and `#chat-v2-the-llm-profile`.
