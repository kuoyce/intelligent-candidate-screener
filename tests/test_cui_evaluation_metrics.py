from copy import deepcopy

import pytest

from candidate_screener.evaluation.cui.metrics import family_ratio, paired_family_delta, summarize_run, validate_run


def run_fixture():
    records = []
    for family in ("f1", "f2"):
        for arm in ("controlled", "v3"):
            records.append({"family_id": family, "task_id": family + "-q", "turn_id": family + "-t",
                "task_type": "standalone", "arm": arm, "question": "What degree?",
                "answer": {"text": "An ongoing degree.", "model": "private-model-id"},
                "eligible_generation": True, "expected": {"participants": ["c1"], "uncertainty": False},
                "source_documents": {"c1": "Degree in progress."}, "elapsed_seconds": 1.0,
                "generation_mode": "qwen_prose" if arm == "v3" and family == "f1" else "qwen_fallback",
                "fallback_reason": "timeout" if family == "f2" else None,
                "error": None, "completed": True})
    return {"schema_version": 1, "run_id": "test", "kind": "stub_development",
            "metadata": {}, "expected_turn_ids": ["f1-t", "f2-t"], "records": records}


def test_missing_or_duplicate_turn_cannot_shrink_denominator():
    run = run_fixture()
    run["records"].pop()
    with pytest.raises(ValueError, match="every expected turn"):
        summarize_run(run, bootstrap_replicates=20)
    run = run_fixture()
    run["records"].append(deepcopy(run["records"][0]))
    with pytest.raises(ValueError, match="every expected turn"):
        validate_run(run)


def test_successful_fallback_is_not_accepted_generation():
    report = summarize_run(run_fixture(), bootstrap_replicates=20)
    arm = report["groups"]["standalone"]["v3"]
    assert arm["execution_completed"] == 2
    assert arm["accepted_generation"]["numerator"] == 1
    assert arm["accepted_generation"]["denominator"] == 2
    assert arm["fallback"]["value"] == .5
    assert report["human_review"] is None


def test_duplicate_record_cannot_inflate_generation_count():
    run = run_fixture()
    run["records"].append(deepcopy(run["records"][1]))
    with pytest.raises(ValueError, match="every expected turn"):
        summarize_run(run, bootstrap_replicates=20)


def test_ineligible_turn_is_not_in_generation_denominator():
    run = run_fixture()
    for record in run["records"]:
        if record["family_id"] == "f2":
            record["eligible_generation"] = False
    report = summarize_run(run, bootstrap_replicates=20)
    assert report["groups"]["standalone"]["v3"]["accepted_generation"]["value"] == 1


def test_paired_arms_must_use_same_sources_and_expectations():
    run = run_fixture()
    run["records"][0]["source_documents"]["c1"] = "Different degree"
    with pytest.raises(ValueError, match="source_documents"):
        validate_run(run)


def test_cluster_resampling_preserves_pairing():
    values = {"a": (0, 20), "b": (1, 1)}
    result = paired_family_delta(values, values, replicates=100)
    assert result["ci95"] == [0, 0]
    assert result["value"] == 0
    assert family_ratio(values, replicates=100)["families"] == 2
    assert family_ratio({"a": (1, 1)}, replicates=100)["ci95"] is None
    with pytest.raises(ValueError, match="same families"):
        paired_family_delta(values, {"a": (0, 1)}, replicates=10)


def test_invalid_timings_and_replication_rejected():
    run = run_fixture()
    run["records"][0]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError, match="elapsed"):
        validate_run(run)
    with pytest.raises(ValueError, match="positive"):
        summarize_run(run_fixture(), bootstrap_replicates=0)
