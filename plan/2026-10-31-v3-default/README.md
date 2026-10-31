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

(Filled in as stages finish.)
