# Running the preliminary CUI evaluation

The [approved plan](../plan/2026-10-29-cui-evaluation/README.md) specifies a preliminary synthetic comparison of controlled answers and v3. The [implementation record](../plan/2026-10-29-cui-evaluation/01-implementation.md) records scope and remaining gates. No held-out model quality results are available yet.

## Offline verification

```bash
uv run python -m candidate_screener.evaluation.cui validate
uv run python -m candidate_screener.evaluation.cui run --split dev --provider stub --run-id dev-001 --output logs/cui-evaluation/dev-001.json
uv run python -m candidate_screener.evaluation.cui recover --provider stub --output logs/cui-evaluation/recovery-001.json
uv run python -m candidate_screener.evaluation.cui report --run logs/cui-evaluation/dev-001.json --output logs/cui-evaluation/dev-001-summary.json
```

Use new artifact names for every run. Existing outputs are never overwritten. Stub execution is restricted to development fixtures and must not be reported as model accuracy. The fixture validator checks structure/source coordinates, not whether expected answers are semantically correct.

## Prepare the frozen study

1. Independently review the draft synthetic fixtures in `src/candidate_screener/evaluation/cui/fixtures.json`. Preserve source/template families and do not use the evaluation cases for prompt tuning. Revise the draft before freezing if reviewers find coupled families or incorrect expectations.
2. Assign two named reviewers. Add top-level `review: {"author": "name", "reviewers": ["name1", "name2"]}` and change `review_status` to `reviewed` only after actual review. Record who authored scenarios and any involvement in system development.
3. Confirm the existing local Qwen3.5 and RoBERTa manifests, target hardware and runtime availability. The tooling downloads/trains nothing and does not start the model server.
4. Obtain permission for live development timing. Use `run --split dev --provider live --pilot` with an output path and run ID to time only the first question pair before attempting the larger workload. The live command also requires `--approved-live-run`, explicit `--model-manifest` and `--qa-manifest`. This flag records a real prior authorization; it does not grant it.
5. Freeze the stable protocol and code before evaluation. The full evaluation contains 70 turns per arm (140 records). Its live runtime requires approval after development timing.

```bash
uv run python -m candidate_screener.evaluation.cui freeze --fixtures /path/to/reviewed-fixtures.json --protocol plan/2026-10-29-cui-evaluation/README.md --model-manifest /path/to/qwen35/manifest.json --qa-manifest /path/to/qa/manifest.json --output data/processed/cui-evaluation/study-001.json
uv run python -m candidate_screener.evaluation.cui run --split eval --provider live --mode paired --fixtures /path/to/reviewed-fixtures.json --manifest data/processed/cui-evaluation/study-001.json --protocol plan/2026-10-29-cui-evaluation/README.md --approved-live-run --run-id eval-001 --output data/processed/cui-evaluation/eval-001.json
```

The paths above are placeholders, not provisioned assets. Changed fixtures, protocol, package, runtime or manifests invalidate the freeze. Keep interrupted or failed runs; do not selectively replace bad turns.

## Independent review

```bash
uv run python -m candidate_screener.evaluation.cui export-review --run data/processed/cui-evaluation/eval-001.json --output-dir data/processed/cui-evaluation/review-001
uv run python -m candidate_screener.evaluation.cui report --run data/processed/cui-evaluation/eval-001.json --reviews /path/to/completed-ratings.json --mapping data/processed/cui-evaluation/review-001/private-mapping.json --output output/cui-evaluation/eval-001-summary.json
```

Give reviewers `packet.json` and their own rows from `ratings-template.json`; keep `private-mapping.json` private until ratings are complete. Each item includes the question, displayed answer, source documents, expected outcomes and relevant displayed dialogue history. Replace placeholder reviewer IDs with actual identities. Reviewers rate independently, including empty/error answers rather than dropping them.

| Field | Instruction |
|---|---|
| `total_claims`, `supported_claims` | Count independently checkable factual clauses and those supported by the provided sources. Preserve candidate ownership and qualifications. Zero claims is allowed and is not perfect factual accuracy. |
| `complete_answer` | All required facts/caveats or the appropriate uncertainty outcome are present. |
| `abstained` | The reply withheld a substantive answer because evidence or service capability was insufficient. |
| `correct_abstention` | Assess correctness on predeclared uncertainty cases; otherwise use null. |
| `reference_correct` | Correct participant/job/criterion resolution or appropriate clarification on annotated opportunities; otherwise null. |
| `unnecessary_clarification` | The reply asks for information already sufficiently specified in the current context. |
| `critical_errors` | List invented credentials/tenure/ranks, incorrect ownership or access failures; empty list only when none were observed. |
| `directness` | 1: does not answer; 3: partly answers with distraction; 5: directly answers the current question. |
| `clarity` | 1: confusing; 3: understandable with effort; 5: concise and readily understandable. |
| `usefulness` | 1: cannot support the review task; 3: useful but incomplete; 5: completes the intended evidence-review task with necessary caveats. |

Retain the two original rating sets. If adjudicating, add explicit `adjudicated` rating rows with `adjudication_reason` and reviewer identity; never replace the independent ratings. Reports separate reviewer results and report agreement. Style ratings cannot establish factual correctness.

## Interpretation limits

- This is a service-level study; asynchronous GLiNER preparation and browser/UI timings are excluded.
- Standalone paired QA outputs are cached to isolate generation. First-arm/replay timings are not comparable production latencies.
- Dialogue arms retain separate histories. Related turns remain together in family-level bootstrap samples. The harness resets history on a job switch; this tests the service after a correct caller reset, not whether the UI/controller initiates that reset. Separate recovery probes exercise controller authorization and source freshness.
- Reports without reviews contain operational metrics only, not factual correctness or human usefulness.
- Accepted generation, fallback and execution failures remain separate. Synthetic provider acceptance is never live-model acceptance.
- Ten evaluation families are preliminary evidence. They do not satisfy the earlier v3 promotion study or establish real recruiter productivity.
