# The 18 failing tests, and a fresh-clone setup test

Status: **requested by the owner on 31 Oct 2026, with its scope set by four answers** (below). Branch:
`feat/v3-default`.

## Request (owner, 31 Oct 2026)

1. "fix the failing test".
2. "clone and copy only the datapack to a new directory. test out how to setup everything (uv,
   cache files, docker compose, .env file) to see if the applications works". Use the new directory
   `intelligent-candidate-screener-clone`, note the steps, and update the README afterwards.

Answers (owner, 31 Oct 2026):

| Question | Answer |
|---|---|
| Which failing tests to fix | **all 18**, deciding case by case whether the code or the expectation is wrong |
| How the datapack reaches the clone | **`datapack handover`**: the inputs and manifest only, never the owner's workspace or caches |
| What to test | **Docker and native** |
| What the clone checks out | **`feat/v3-default`**, from GitHub |

## Part 1: the 18 tests

The baseline on `release/integration` and on `feat/v3-default` (`logs/v3-default/pytest-baseline.log`,
`v6-pytest-final.log`) has 18 failures:

- `test_conversation_summary`: 1;
- `test_portal_extractive`: 11;
- `test_portal_ui`: 3;
- `test_portal_v1_ui`: 3.

Steps:

1. **Diagnose first** (a sub-agent, read-only). For each test: the root cause, the commit that
   introduced the failure, and whether the code or the test expectation is wrong, judged against
   the plan record that defined the behaviour.
2. **Bring the decisions to the owner** where a fix changes behaviour that a person sees: answer
   text, qualification context, or which model answers.
3. **Then fix them.** Each fix keeps the test that guards it, and the test fails before the fix.
4. Finally, run the full suite. Run `checks --explain`, and if any record is selected, ask the
   owner before a long run.

First finding: `test_manual_qa_summary_without_entity_mentions_has_grounded_closing` (added in
`168ce9b`, the design in `plan/2026-10-25-hiring-desk-v3/06-summary-closing.md`) has failed since
the commit that added it. The bundle passed to `summary_conclusion` holds only the JD sentences
that the answer cited ("QA Manual tester required."), so the literal overlap the design relies on
("bug reports using Jira") is never seen.

### Diagnosis and the owner's decisions (31 Oct 2026)

The diagnosis (a read-only sub-agent; logs in `logs/tests-diagnosis/`) ran the four test files at
three commits: `d35d632` 0 failures, `7468a0b` ("feat: aadd v3 chatbot", 26 Oct) 17, and `168ce9b`
(29 Oct) 18. The failures have three causes:

| Cause | Tests | The owner's decision |
|---|---|---|
| A. `7468a0b` commented out the qualifying source context in v1 answers (`portal_extractive._visible`), the "QA found no answer. Source excerpt:" label, and the context in comparison-table cells (`portal_ui._comparison_table`). Plan `2026-10-24-grounded-qwen-comparisons/05`, AGENTS.md and D65 require them | 11 `test_portal_extractive`, 2 `test_portal_v1_ui` | **restore**; tests unchanged |
| B. `7468a0b` removed "Requirement evidence", "Compare candidates", "Prepare top10 package" and "Save session" from the Review action dropdown | 3 `test_portal_ui`, 1 `test_portal_v1_ui` | **keep them removed**. The three dropdown-context tests drive the context change through "Open CV" instead, keeping their stale-context checks. The compare test checks the chat path only. The dead branch is deleted, and README is corrected |
| C. The v3 fit-summary closing (plan `2026-10-25-hiring-desk-v3/06`) is never reached, because the summary sees only the JD sentence it cited. The test failed from the commit that added it | 1 `test_conversation_summary` | **drop the test**; record that 06's closing does not appear through the service path |
| (outside the 18) `168ce9b` appends "This shows that candidate's CV contains relevant experience and qualifications." to every short summary, without any check | none | **remove the sentence** |

## Part 2: a fresh clone

- **Clone:** `/home/yuchen/project/intelligent-candidate-screener-clone`, from `origin`, branch
  `feat/v3-default`.
- **Data:** `datapack handover` into the clone's git-ignored `data/datapack/`.
- **Docker, as a teammate would follow `docs/docker.md`:**
  - `.env` from `.env.example`;
  - `docker compose build`, then `docker compose up -d --wait`;
  - the first-account screen, sign-in, a review, and a v3 answer.
- **Native:** `uv sync --all-groups`, the model downloads (the asset modules), and any cache step the
  app needs (`inference --cache-parts`, if the datapack lacks its parts). Then the 4B service and the
  Hiring Desk, which must not be run at the same time as Docker, because memory is limited.
- **Notes:** every step and its time is written to the implementation record. README gets a
  "Set up a fresh clone" section.
- The owner's checkout, `.env`, workspace, images and running containers are not touched: the
  clone gets its own compose project and port.

## Implementation record

### Part 1: the fixes (31 Oct 2026; a sub-agent, reviewed by the parent; logs in `logs/tests-fix/`)

- **A:** the three lines are restored exactly as before `7468a0b`, with the tests unchanged. The
  failures went from 14 to 0 (`a-before.log`, `a-after.log`). `7468a0b` changed no existing v2/v3
  test.
- **B:** the removed actions' unreachable branches are deleted from `portal_ui.py`, as are the
  stale comment and the unused `CATEGORIES` import.
  - The three dropdown-context tests change the context through "Open CV". They were shown to
    still fail when the stale-context check, or the dropdown context, is removed.
  - The v1 compare test runs through chat.
  - README's dropdown paragraph is rewritten, and its "Refresh extraction" sentence removed (that
    button lived only in the deleted branch).
  - The failures went from 6 to 0 (`b-before.log`, `b-after.log`).
- **C:** the one test is deleted.
- **D:** `body = " ".join(parts)` is restored, and a new regression test failed before the change
  (`d-before.log`, `d-after.log`).
- **Full suite:** 2708 passed, 37 skipped, **0 failed** (`pytest-full.log`).
- **Checks:** `checks --affected HEAD --explain` selects no records.
- **Left as they are, for the owner:**
  - "Save session" is still offered when a snapshot has no candidates.
  - `portal_qa.py:318-322` still suggests "Prepare top10 package" in chat wording, although the UI
    displays nothing from `suggested_actions`.

### Part 2: the fresh clone (31 Oct 2026; logs in `logs/fresh-setup/` and the clone's `logs/`)

**Get the code and the data.**

- `git clone --branch feat/v3-default <origin>` into `intelligent-candidate-screener-clone`: 3 s,
  76 MB.
- From the owner's checkout: `uv run python -m candidate_screener.datapack handover
  ../intelligent-candidate-screener-clone/data/datapack`. It took 1 s and verified 24 inputs
  (256.8 MB), with no workspace and no caches.

**Docker, following `docs/docker.md` §4 as written.**

- Setup: `mkdir logs`, `cp .env.example .env` (uid and gid are already 1000).
- `docker compose build`: 8 s. Every layer was cached on this machine, so this is not a
  first-machine time.
  - **Deviation:** the image names are shared by every clone on a machine (guide §10), so this
    build replaced the owner's `:local` tags with the same commit's build. The owner's previous
    images are kept as `candidate-screener/hiring-desk:owner-before-clone-test` and
    `candidate-screener/hiring-desk-llm4b:owner-before-clone-test`.
- `docker compose --profile tools run --rm tools verify`: OK, 4 s.
- `docker compose up -d --wait`: `app` and `llm4b` were healthy after 20.0 s.
- In the browser:
  - the **Create the first account** form appeared (a fresh datapack has no accounts);
  - it created a test account, whose token is kept only in the clone's git-ignored `logs/`;
  - sign-in worked; the 60 benchmark jobs and the v3 chat caption appeared.
  - On "QA Manual Engineer", "Does candidate 1 have experience with Jira?" gave the marked
    fallback `deadline_exceeded`. The real JD makes a longer prompt than the smoke test's, so it
    needs more than 60 s on this loaded machine.
- `docker compose down`: 4 s.

**Native**, with `SCREENER_MODELS_ROOT=data/models` (the models stay out of the datapack) and a
clone-local `HF_HOME` (to see what a machine without this one's Hugging Face cache must
download). Script: the session scratchpad's `fresh/native-setup.sh`, times in
`logs/fresh-setup/native-setup.log`.

- **Environments:** `uv sync --all-groups` took 2 s, `--project runtime/transformer` 8 s and
  `--project runtime/gliner-medium` 2 s, with a warm uv cache.
- **Data:** link the datapack's folders into `data/`: `data/processed/<each>` points to
  `../datapack/processed/<each>`, and `data/vocab` to `datapack/vocab`.
  - The `hr-portal` link also existed because Docker had run first. Native and Docker then
    share one workspace, which is why native showed sign-in rather than the first-account form.
- **Models**, at the Dockerfile's pins: 4.9 GB under `data/models`, 1.2 GB under `data/hf`.

  | Model | Command | Time |
  |---|---|---|
  | RoBERTa-SQuAD2 | `qa_assets --revision adc3b06…` | 10 s |
  | GLiNER-medium | `extraction.medium_assets --encoder-revision 8ccc9b6…` | 19 s |
  | GLiNER-medium | the `runtime-versions.json` one-liner from `data/README.md` | 0 s |
  | Qwen2.5-0.5B | `qwen_assets --revision 7ae5576…` | 13 s |
  | C2 | `huggingface_hub.snapshot_download("Qwen/Qwen3-Reranker-0.6B", revision="e61197e…")` | 16 s |
  | Qwen3.5-4B | `gguf_assets` | 44 s |

  `rerank/cross.py` loads C2 with `local_files_only`, so it has to be downloaded in advance.
- **No cache step:** `inference --cache-parts` was not needed, because the datapack already holds
  `processed/inference/parts/`.
- **Run:**
  - `gguf_server --manifest data/models/chatbot/qwen35/manifest.json --launch-backend` was ready
    in about 20 s;
  - `streamlit run src/candidate_screener/app/streamlit_app.py --server.address 127.0.0.1
    --server.port 8503`;
  - sign-in worked;
  - "What degree does candidate 1 have?" reached the 4B, whose output failed the app's checks:
    the marked fallback, `unsupported_paraphrase`.
- **Ranking**, natively and offline (`HF_HUB_OFFLINE=1`): `Screener.load()` took 22.9 s on the
  600-CV pool. An invented JD ranked with D63 in 47.2 s, and with D64 in 149.4 s (C2: 50 pairs,
  2.9 s per pair).
- **Not tested:** a first-machine cold Docker build (the cache was warm; the guide's 3.5-5.5 min
  measurements stand), and D63/D64 through the Docker UI in the clone (measured on 27-28 Oct).
