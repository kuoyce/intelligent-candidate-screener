# 15 — P6 record: the D50 scoring batch (tasks 6.1–6.2; code for 6.3–6.4)

6 Oct 2026. Spec: [04](04-llm-scoring-batch.md). Status: **built and tested; nothing
dispatched.** No `claude` subprocess was started and no LLM was called while doing this
work. The user authorises the dispatch separately.

## 1. Where the code lives (answers Q-A3)

`src/candidate_screener/rerank/scoring_batch.py` (new package `rerank/`), tests in
`tests/test_scoring_batch.py`. **`annotation/` is not edited**, and
`llm_recheck.DISPATCH_CLOSED` stays `True`.

What was read: `llm_recheck.dispatch_a2` (how runs 9/10 were dispatched), `judge`,
`judge_one`, `run_coverage`, `collect`, the `DispatchClosed` gate, and `sample.py`.

Why a separate module with its own loop, and not a change to `llm_recheck`:

| Option | Why not |
|---|---|
| Flip `DISPATCH_CLOSED` (even in-process, scoped) | It is one switch for every path behind it: `dispatch`, `dispatch_all_a1`, `dispatch_a2`, `judge`, `judge_one`. Opening it for one batch opens all of them for as long as it is open |
| Bank into `llm-recheck-prompts.jsonl` / `-raw.jsonl` and reuse `llm.judge()` | `llm.collect()` raises `BadReturn` on any record whose `draw` is not in `DRAW_FILES`, so one `a2_train_batch` record in the frozen bank would break `llm_recheck --collect` for the whole repository. Adding the draw there is an edit to frozen code |
| Add a D50 path inside `llm_recheck` | An edit to the frozen module for something the import route already does |

So the new module **imports the instrument unchanged**: `render_prompt`,
`assert_not_anchoring`, `guide_example_pairs`, `assert_agent_definition_current` (the
committed `.claude/agents/a1-judge.md` and its hash), `judge_sandbox` (the tool-free,
context-free working directory), `judge_argv` (which passes `--model`), `classify_failure`,
`JudgeUnavailable` / `JudgeFailures`, `parse_return` (strict; score 0–100), `read_jsonl`,
`subsample_ids`, `redact`. It re-implements two things: `spawn_one`, a copy of
`judge_one`'s ~20 lines behind the D50 guard instead of the D42 gate, and a `judge` loop
with the same semantics (idempotent per `(pair_id, run)`, append-only, a failed spawn
quarantined and never written as a return, `JudgeFailures` raised and exit 1 on any
loss). It banks into its own append-only files:

```
data/processed/rerank/llm-train-prompts.jsonl   # dispatch
data/processed/rerank/llm-train-raw.jsonl       # answers, appended as they arrive
data/processed/rerank/llm-train-failed.jsonl    # quarantined spawns (retried on re-run)
```

## 2. The guard (D50 in code)

`OutOfScope` is a subclass of `llm.DispatchClosed`. `assert_in_scope` runs at dispatch, again
over the whole run before judging, and **again inside every spawn**, so no record reaches
`subprocess.run` without being checked. A record must pass all of these:

| Clause | Check | Test (each fails with the check removed) |
|---|---|---|
| Instrument `5d986c82` | `check_instrument`: hash starts `5d986c82` **and** equals the committed agent's | `..._a_different_instrument`, `..._another_judge` |
| Committed judge is that instrument | `current_instrument` | `test_the_committed_judge_must_be_the_d50_instrument` |
| Model `claude-haiku-4-5` | `check_model` (the spawn's `--model` comes from the record) | `..._a_different_model`, `test_judge_refuses_a_tampered_bank_before_spawning` |
| Draw `a2_train_batch`, run 11 | `check_run` | `test_the_guard_refuses_any_other_run_or_draw` |
| Replicate (run 12) only behind the explicit flag | `check_run(allow_replicate)`; default `False`; CLI `--replicate` | `test_the_guard_refuses_the_replicate_without_its_flag` |
| Every pair in the committed manifest, same ids | `check_manifest`; run 12 also needs the pair in the `replicate` subset | `..._a_pair_not_in_the_manifest`, `..._replicate_subset` |
| No final-set document | `check_documents` vs `a2-final-labels.csv` ∪ `indomain-pairs.csv` | `test_the_guard_refuses_a_final_set_document` |
| No eval-region document (D19) | `check_documents` vs `a2-partition.csv` | `test_the_guard_refuses_an_eval_region_document` |
| Both documents in the train region | `check_documents` | `..._outside_the_train_region` |
| Cap 1,000 (+100 replicate) | `assert_manifest_in_scope`, `check_cap` per run | `..._manifest_over_the_cap`, `..._run_over_its_cap` |
| Runs 11/12 unused by the frozen bank and every committed `llm-*.csv` | `assert_runs_unused_elsewhere` | `test_dispatch_refuses_a_run_number_the_frozen_bank_holds` |
| The guard runs inside the spawn | `spawn_one` | `test_the_spawn_checks_the_scope_itself` |
| Collect refuses a short run | `collect` → `JudgeFailures` | `test_collect_refuses_a_short_run` |
| Never `judgements.csv` | `collect` refuses that target | `test_collect_never_writes_the_human_judgement_file` |
| Validation CV-disjoint | `split_by_jd` drops overlaps | `test_the_split_is_by_jd_and_drops_validation_cvs_seen_in_train` |

The mutation runs (15 mutants, one per check) were done by editing the source, running the
named test, and restoring: **all 15 tests fail against their mutant**. Every other entry
point still refuses: `test_the_frozen_entry_points_still_refuse_after_this_module_loads`
(`llm.dispatch`, `dispatch_a2`, `dispatch_all_a1`, `judge`, `judge_one`) beside the
unchanged `tests/test_dispatch_closed.py` (which also covers `prompt_calibration`).

The CLI has no `--run`, `--model` or `--draw` flag: run, draw and model are constants.

## 3. The batch (task 6.2)

`--build` (seed 0) mirrors `sample.load_english` on the **train** region: English, non-empty
JD `Long Description`; CV text = `sample.cv_text`; experience bands as before. Eval-region
and final-set documents are also removed explicitly, so the exclusion does not rest on
one file. Then `sample.draw_batch` is called with the final set's `generic` spec (Q-A2):
`n_jds=100, per_jd=10, same_keyword_min=8, keywords=None`, i.e. JDs spread evenly over
experience bands, 8 CVs from the JD's `Primary Keyword` (TF-IDF bands
high/high/high/mid/mid/mid/low/low over a 200-CV pool) and 2 from other keywords
(high/mid over a 100-CV pool), each CV drawn at most once.

| | Count |
|---|---|
| Eligible JDs (train region, English, non-empty, not final set) | **106,423** (0-1: 21,046; 2-3: 64,618; 4-6: 20,759) |
| Eligible CVs | **157,688** |
| Batch | **100 JDs, 1,000 CVs, 1,000 pairs** (10 per JD for every JD) |
| JDs by experience band | 0-1: 33, 2-3: 33, 4-6: 34 |
| Train split | 80 JDs, 800 CVs, 800 pairs (82 replicate pairs) |
| Validation split | 20 JDs, 200 CVs, 200 pairs (18 replicate pairs) |
| CVs in both splits | **0**; validation pairs dropped for CV overlap: **0** |
| Selection reason | same_keyword: high 300, mid 300, low 200; other_keyword: high 100, mid 100 (on-category 80%) |
| Replicate subset | 100 pairs (`llm_recheck.subsample_ids`, seed 0), dispatched only with `--replicate` |
| Overlap with final set / eval region | **0 / 0** (tested on the committed file, no `data/` needed) |

The English filter is vacuous on this release: every A2 row is tagged `en`, on both
regions. The split is 80/20 by JD (seeded permutation of sorted JD ids). CV-disjointness
holds by construction (a CV is drawn once), and `split_by_jd` would **drop** (not
reassign) any validation pair whose CV appears under a train JD. It dropped none.

Files: `docs/data/manifests/llm-train-batch.csv` (`pair_id, jd_id, cv_id, split,
selection_reason, replicate`, ids only) and `llm-train-batch.json` (spec, counts,
`manifest_sha256 = a21c6e25586c727e7a42bc7ab74a35c61c6cc009fb63ca2ddaea6e77dc55871c`).
`--check` rebuilds the batch in memory (about 90 s), compares bytes and summary, and
re-runs the scope check over the committed manifest: **ok**.

## 4. The text sent

The prompt is exactly what runs 9/10 sent, so the instrument is unchanged:
`llm_recheck.render_prompt(redact.redact_text(JD), redact.redact_text(CV))` with JD =
`Long Description`, stripped, and CV = `Position + CV + Highlights + Moreinfo + Looking For`
(`sample.cv_text`). This is what `dispatch_a2` sent through `queue.indomain_rows`.
`--verify-text-form` rebuilds **all 600** run-9/10 prompts this way: **600 / 600
`prompt_sha256` identical**. (Also a test, skipped without `data/`.)

**Note for P8:** the instrument's CV text **includes `Position`**. This is the instrument's
form, not the matching rungs' `text_v2` (D53). The scores were therefore produced by a judge
that saw the title, and a reranker trained on them inherits whatever that carries.

## 5. Commands for the user (none has been run)

```bash
# 0. what would happen (spawns nothing, writes nothing)
uv run python -m candidate_screener.rerank.scoring_batch --dry-run
# 1. write the 1,000 prompts to the bank (idempotent)
uv run python -m candidate_screener.rerank.scoring_batch --dispatch
# 2. each sitting: judge what is unanswered; exits 1 when the session limit hits.
#    Wait for the stated reset, then re-run the same command; it retries exactly the
#    quarantined pairs. /low-priority does not help (memory note).
uv run python -m candidate_screener.rerank.scoring_batch --judge
# 3. progress (also after a lost tool result: answers are banked as they arrive)
uv run python -m candidate_screener.rerank.scoring_batch --coverage
# 4. only when coverage says complete
uv run python -m candidate_screener.rerank.scoring_batch --collect
# optional, only if Q-A1b is answered yes: the 100-pair replicate as run 12
uv run python -m candidate_screener.rerank.scoring_batch --dispatch --replicate
uv run python -m candidate_screener.rerank.scoring_batch --judge --replicate
uv run python -m candidate_screener.rerank.scoring_batch --coverage --replicate
```

`--judge --limit N` caps a sitting; `--concurrency` defaults to 10, as before.
`--collect` writes `docs/data/manifests/llm-train-scores.csv`, sorted by run and pair, with
columns `agent_sha256, model, run, draw, pair_id, jd_id, cv_id, split, label, score,
parsed_ok, prompt_sha256, chars_sent, judged_at`. It holds ids and scores only: no document
text, and not the judge's free-text reason, which can quote the CV. It refuses to run while
any dispatched run is short. Its summary reports `all_parsed` and `scores_in_range`
(acceptance for 6.4).

Dry-run output (6 Oct 2026):

```
run 11, draw a2_train_batch, claude-haiku-4-5, agent 5d986c8232ea…
cap 1000, pairs_in_scope 1000, already_dispatched 0, already_answered 0, owed 1000
~67 pairs/sitting -> sittings_estimate 15
chars_sent median 3,465, mean 3,781, max 11,926
spawned 0, written 0
(--replicate: run 12, cap 100, 100 pairs, ~2 sittings, median 3,366 chars)
```

## 6. Verification

- `uv run pytest`: **492 passed**. This includes other agents' in-progress tests. My file
  adds 44 tests (one is skipped without `data/`); with the D42 and decision tests: 53 passed.
- `rerank.scoring_batch --check`: ok. `--verify-text-form`: 600/600.
- Mutation runs: 15/15 guard tests fail against their mutant.

## 7. Deviations and incidents

1. **Separate bank, not `llm-recheck-raw.jsonl`** (§1). The plan left this open ("how is not
   assumed here"). The frozen bank's `collect()` would refuse the new draw.
2. **`judge_one` is copied, not called.** It is gated by D42, so calling it would require
   opening the frozen gate.
3. **The replicate subset is fixed in the manifest now** (`replicate` column), so the
   pairs are committed before any score exists. Dispatching it stays off until Q-A1b is
   answered.
4. **Unparsed returns are kept** with `parsed_ok=False` and counted (`all_parsed`), not
   refused at collect. The bank is idempotent per `(pair_id, run)`, so a junk answer has no
   automatic re-judge path. If any occur, how to handle them is a question for the user
   (open item below). Runs 9/10 had zero unparsed returns.
5. **Incident, fixed:** during the mutation runs, the mutant that removes `collect`'s
   `judgements.csv` refusal let the test write synthetic rows over the committed
   `docs/data/manifests/judgements.csv`. I restored it at once with
   `git checkout -- docs/data/manifests/judgements.csv`: 1,035 lines, identical to HEAD.
   The test now points the guard at a tmp stand-in, and a re-run of all 15 mutants left
   every `docs/data/manifests/*.csv` byte-identical (checked with sha256).

## 8. Open items

- **Q-A1b** (the replicate): still open. The code is ready behind `--replicate`.
- **Unparsed returns** (deviation 4): drop them from training (default), or allow a scoped
  re-judge under a new run number? A re-judge needs the user's decision.
- `rerank/__init__.py` was created here. If P8 puts the reranker elsewhere, this module
  can move with it.
