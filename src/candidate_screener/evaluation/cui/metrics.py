"""Auditable paired CUI metrics, with families as bootstrap sampling units."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
import statistics

ARMS = ("controlled", "v3")


def validate_run(run):
    if run.get("schema_version") != 1 or not isinstance(run.get("run_id"), str) or not run["run_id"]:
        raise ValueError("Invalid run identity/schema")
    turns = run.get("expected_turn_ids")
    if not isinstance(turns, list) or not turns or any(not isinstance(t, str) or not t for t in turns) or len(set(turns)) != len(turns):
        raise ValueError("Expected turn IDs must be nonempty and unique")
    records = run.get("records")
    if not isinstance(records, list):
        raise ValueError("Missing run records")
    observed = Counter((r.get("turn_id"), r.get("arm")) for r in records)
    required = Counter((turn, arm) for turn in turns for arm in ARMS)
    if observed != required:
        raise ValueError("Run must contain every expected turn exactly once per arm")
    paired = defaultdict(list)
    for record in records:
        for key in ("family_id", "task_id", "question"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Invalid {key}")
        if record.get("task_type") not in {"standalone", "dialogue"}:
            raise ValueError("Unknown task type")
        for key in ("eligible_generation", "completed"):
            if not isinstance(record.get(key), bool):
                raise ValueError(f"{key} must be boolean")
        elapsed = record.get("elapsed_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid elapsed time")
        if not isinstance(record.get("answer"), dict) or not isinstance(record["answer"].get("text", ""), str):
            raise ValueError("Invalid displayed answer")
        if not isinstance(record.get("expected"), dict) or not isinstance(record.get("source_documents"), dict):
            raise ValueError("Missing expected outcomes/source documents")
        paired[record["turn_id"]].append(record)
    for pair in paired.values():
        for key in ("family_id", "task_id", "task_type", "question", "eligible_generation", "expected", "source_documents"):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f"Paired arms differ in {key}")
    return {"turns_per_arm": len(turns), "records": len(records),
            "families": len({r["family_id"] for r in records})}


def _quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def family_ratio(values, *, replicates=10000, seed=0):
    """values maps family ID to (numerator, denominator); resample whole families."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("bootstrap_replicates must be a positive integer")
    pairs = list(values.values())
    numerator, denominator = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
    rate = numerator / denominator if denominator else None
    result = {"numerator": numerator, "denominator": denominator, "value": rate,
              "families": len(pairs), "ci95": None}
    if len(pairs) >= 2 and denominator:
        rng, samples = random.Random(seed), []
        for _ in range(replicates):
            selected = rng.choices(pairs, k=len(pairs))
            d = sum(p[1] for p in selected)
            if d:
                samples.append(sum(p[0] for p in selected) / d)
        result["ci95"] = [_quantile(samples, .025), _quantile(samples, .975)]
    return result


def paired_family_delta(controlled, v3, *, replicates=10000, seed=0):
    if set(controlled) != set(v3):
        raise ValueError("Paired bootstrap requires the same families")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    families = sorted(controlled)
    def delta(selected):
        cn, cd = map(sum, zip(*(controlled[f] for f in selected)))
        vn, vd = map(sum, zip(*(v3[f] for f in selected)))
        return vn / vd - cn / cd if cd and vd else None
    result = {"value": delta(families) if families else None, "families": len(families), "ci95": None,
              "direction": "v3_minus_controlled"}
    if len(families) >= 2:
        rng = random.Random(seed)
        samples = [delta(rng.choices(families, k=len(families))) for _ in range(replicates)]
        samples = [value for value in samples if value is not None]
        if samples:
            result["ci95"] = [_quantile(samples, .025), _quantile(samples, .975)]
    return result


def _aggregate(records, numerator, denominator=lambda r: 1):
    values = defaultdict(lambda: [0, 0])
    for record in records:
        values[record["family_id"]][0] += numerator(record)
        values[record["family_id"]][1] += denominator(record)
    return dict(values)


def _latency(records):
    seconds = [r["elapsed_seconds"] for r in records]
    return {"n": len(seconds), "median_seconds": statistics.median(seconds) if seconds else None,
            "maximum_seconds": max(seconds) if seconds else None, "p95_seconds": _quantile(seconds, .95),
            "caveat": "Observed sample percentile; small samples do not establish population p95 or an SLA."}


def summarize_run(run, reviews=None, bootstrap_replicates=10000, seed=0):
    coverage = validate_run(run)
    # Also validate replicate count for empty metric subsets.
    family_ratio({}, replicates=bootstrap_replicates, seed=seed)
    result = {"schema_version": 1, "run_id": run["run_id"], "kind": run.get("kind"),
              "coverage": coverage, "bootstrap": {"unit": "family", "replicates": bootstrap_replicates, "seed": seed},
              "groups": {}, "human_review": None,
              "limitations": ["Synthetic preliminary study; no recruiter productivity or general quality claim.",
                              "Successful fallback is not accepted model generation.",
                              "No factual correctness is inferred from lexical matching."]}
    for task_type in ("standalone", "dialogue"):
        groups = result["groups"][task_type] = {}
        for arm in ARMS:
            rows = [r for r in run["records"] if r["task_type"] == task_type and r["arm"] == arm]
            eligible = [r for r in rows if r["eligible_generation"]]
            metric = lambda n, d=lambda r: 1: family_ratio(_aggregate(eligible, n, d), replicates=bootstrap_replicates, seed=seed)
            modes = Counter(r.get("generation_mode") or "unrecorded" for r in eligible)
            groups[arm] = {
                "turns": len(rows), "eligible_turns": len(eligible),
                "execution_completed": sum(r["completed"] for r in rows),
                "accepted_generation": metric(lambda r: int(r.get("generation_mode") == "qwen_prose")),
                "fallback": metric(lambda r: int(r.get("generation_mode") == "qwen_fallback")),
                "eligible_mode_counts": dict(sorted(modes.items())),
                "fallback_reasons": dict(Counter(r.get("fallback_reason") or "unspecified" for r in eligible if r.get("generation_mode") == "qwen_fallback")),
                "execution_errors": sum(bool(r.get("error")) for r in rows),
                "timeout_count": sum("timeout" in str(r.get("error") or r.get("fallback_reason") or "").lower() or "deadline" in str(r.get("error") or r.get("fallback_reason") or "").lower() for r in rows),
                "latency": _latency(rows),
                "latency_by_mode": {mode: _latency([r for r in rows if (r.get("generation_mode") or "unrecorded") == mode]) for mode in sorted({r.get("generation_mode") or "unrecorded" for r in rows})},
                "latency_by_cache_state": {state: _latency([r for r in rows if r.get("cache_state", "unspecified") == state]) for state in sorted({r.get("cache_state", "unspecified") for r in rows})}}
    if reviews is not None:
        result["human_review"] = _human_summary(run, reviews, bootstrap_replicates, seed)
    return result


def _human_summary(run, reviews, replicates, seed):
    from .review import validate_rating, validate_reviews
    mapping = reviews["mapping"]
    records = {(r["turn_id"], r["arm"]): r for r in run["records"]}
    identities = mapping["items"]
    if mapping.get("run_id") != run["run_id"] or {(i["turn_id"], i["arm"]) for i in identities} != set(records):
        raise ValueError("Review mapping does not cover this run")
    for item in identities:
        record = records[item["turn_id"], item["arm"]]
        if any(item[key] != record[key] for key in ("family_id", "task_id", "task_type")):
            raise ValueError("Review mapping metadata mismatch")
    packet = {"run_id": run["run_id"], "items": [{"item_id": i["item_id"], "expected": records[i["turn_id"], i["arm"]]["expected"]} for i in identities]}
    ratings_data = {"run_id": reviews.get("run_id"), "ratings": reviews["ratings"]}
    coverage = validate_reviews(ratings_data, packet, mapping)
    by_item = {i["item_id"]: records[i["turn_id"], i["arm"]] for i in identities}
    summary = {"coverage": coverage, "by_reviewer": {}, "adjudication": "not_supplied",
               "agreement": {}, "note": "Independent ratings are preserved; no automatic consensus is inferred."}
    rating_sets = [(reviewer, [r for r in reviews["ratings"] if r["reviewer_id"] == reviewer])
                   for reviewer in coverage["reviewers"]]
    if reviews.get("adjudicated") is not None:
        adjudicated = reviews["adjudicated"]
        if not isinstance(adjudicated, list) or Counter(r.get("item_id") for r in adjudicated) != Counter(by_item.keys()):
            raise ValueError("Adjudication must cover each item exactly once")
        for row in adjudicated:
            validate_rating(row)
            if not isinstance(row.get("adjudication_reason"), str) or not row["adjudication_reason"].strip():
                raise ValueError("Adjudication requires a documented reason")
        # Apply the same opportunity/nullability checks without changing original reviews.
        duplicate = [{**r, "reviewer_id": reviewer} for r in adjudicated for reviewer in ("audit-a", "audit-b")]
        validate_reviews({"run_id": run["run_id"], "ratings": duplicate}, packet, mapping)
        summary["adjudication"] = "explicit_human_adjudication"
        rating_sets.append((None, adjudicated))
    for reviewer, ratings in rating_sets:
        rows = [{**by_item[r["item_id"]], "rating": r} for r in ratings]
        if reviewer is None:
            output = summary["adjudicated"] = {}
        else:
            output = summary["by_reviewer"][reviewer] = {}
        for task_type in ("standalone", "dialogue"):
            task = output[task_type] = {}
            arm_values = {}
            for arm in ARMS:
                selected = [r for r in rows if r["task_type"] == task_type and r["arm"] == arm]
                def ratio(n, d=lambda r: 1):
                    return family_ratio(_aggregate(selected, n, d), replicates=replicates, seed=seed)
                uncertainty = [r for r in selected if r["expected"].get("uncertainty")]
                answerable = [r for r in selected if not r["expected"].get("uncertainty")]
                refs = [r for r in selected if r["expected"].get("participants") or r["expected"].get("clarification")]
                accepted = [r for r in selected if r.get("generation_mode") == "qwen_prose"]
                task[arm] = {
                    "supported_claims": ratio(lambda r: r["rating"]["supported_claims"], lambda r: r["rating"]["total_claims"]),
                    "claim_free_answers": sum(r["rating"]["total_claims"] == 0 for r in selected),
                    "complete_answers": ratio(lambda r: int(r["rating"]["complete_answer"])),
                    "correct_abstention": family_ratio(_aggregate(uncertainty, lambda r: int(r["rating"]["correct_abstention"])), replicates=replicates, seed=seed),
                    "unnecessary_abstention": family_ratio(_aggregate(answerable, lambda r: int(r["rating"]["abstained"])), replicates=replicates, seed=seed),
                    "reference_resolution": family_ratio(_aggregate(refs, lambda r: int(r["rating"]["reference_correct"])), replicates=replicates, seed=seed),
                    "accepted_generation_supported_claims": family_ratio(_aggregate(accepted, lambda r: r["rating"]["supported_claims"], lambda r: r["rating"]["total_claims"]), replicates=replicates, seed=seed),
                    "accepted_generation_complete_answers": family_ratio(_aggregate(accepted, lambda r: int(r["rating"]["complete_answer"])), replicates=replicates, seed=seed),
                    "unnecessary_clarifications": ratio(lambda r: int(r["rating"]["unnecessary_clarification"])),
                    "critical_errors": sum(len(r["rating"]["critical_errors"]) for r in selected),
                    "critical_error_families": sorted({r["family_id"] for r in selected if r["rating"]["critical_errors"]})}
                arm_values[arm] = {}
                for dimension in ("directness", "clarity", "usefulness"):
                    values = _aggregate(selected, lambda r: r["rating"][dimension])
                    arm_values[arm][dimension] = values
                    task[arm][dimension] = family_ratio(values, replicates=replicates, seed=seed)
                if task_type == "dialogue":
                    dialogues = defaultdict(list)
                    for record in selected:
                        dialogues[(record["family_id"], record["task_id"])].append(record)
                    completion = defaultdict(lambda: [0, 0])
                    for (family, _), turns in dialogues.items():
                        completion[family][0] += int(all(r["completed"] and r["rating"]["complete_answer"] and not r["rating"]["critical_errors"] for r in turns))
                        completion[family][1] += 1
                    task[arm]["whole_dialogue_completion"] = family_ratio(completion, replicates=replicates, seed=seed)
            task["paired_deltas"] = {dimension: paired_family_delta(arm_values["controlled"][dimension], arm_values["v3"][dimension], replicates=replicates, seed=seed) for dimension in ("directness", "clarity", "usefulness")}
    by_rating = defaultdict(dict)
    for row in reviews["ratings"]:
        by_rating[row["item_id"]][row["reviewer_id"]] = row
    a, b = coverage["reviewers"]
    for key in ("complete_answer", "correct_abstention", "reference_correct", "directness", "clarity", "usefulness"):
        pairs = [(rows[a][key], rows[b][key]) for rows in by_rating.values() if rows[a][key] is not None and rows[b][key] is not None]
        summary["agreement"][key] = {"agreements": sum(x == y for x, y in pairs), "items": len(pairs), "rate": sum(x == y for x, y in pairs) / len(pairs) if pairs else None}
    return summary
