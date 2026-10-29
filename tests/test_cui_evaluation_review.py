from copy import deepcopy
import json

import pytest

from candidate_screener.evaluation.cui.review import create_review_packet, validate_reviews
from candidate_screener.evaluation.cui.metrics import summarize_run


def fixture():
    records = []
    for arm in ("controlled", "v3"):
        records.append({"family_id": "family", "task_id": "dialogue", "turn_id": "turn",
            "task_type": "dialogue", "arm": arm, "question": "What qualification?",
            "answer": {"text": "Degree ongoing.", "model": "secret-model", "generation_provenance": {"fallback_reason": "secret"}},
            "eligible_generation": True, "expected": {"participants": ["c1"], "uncertainty": True},
            "source_documents": {"c1": "Degree ongoing."}, "elapsed_seconds": .1,
            "generation_mode": "qwen_prose" if arm == "v3" else "qwen_fallback",
            "fallback_reason": None, "error": None, "completed": True})
    run = {"schema_version": 1, "run_id": "test", "kind": "stub_development",
           "expected_turn_ids": ["turn"], "records": records}
    export = create_review_packet(run, seed=7)
    reviews = {"run_id": "test", "ratings": [{"item_id": item["item_id"], "reviewer_id": reviewer,
        "supported_claims": 1, "total_claims": 1, "complete_answer": True,
        "abstained": True, "correct_abstention": True, "reference_correct": True, "unnecessary_clarification": False,
        "critical_errors": [], "directness": 4, "clarity": 4, "usefulness": 4}
        for item in export["packet"]["items"] for reviewer in ("r1", "r2")]}
    return run, export, reviews


def test_packet_does_not_expose_system_metadata():
    _, export, _ = fixture()
    text = json.dumps(export["packet"])
    assert "secret" not in text
    assert "generation_mode" not in text
    assert '"arm"' not in text
    assert {item["arm"] for item in export["mapping"]["items"]} == {"v3", "controlled"}


def test_two_complete_reviewers_are_required():
    _, export, reviews = fixture()
    assert validate_reviews(reviews, **export)["ratings"] == 4
    reviews["ratings"].pop()
    with pytest.raises(ValueError, match="Missing, duplicate"):
        validate_reviews(reviews, **export)


def test_duplicate_rating_cannot_replace_missing_review():
    _, export, reviews = fixture()
    reviews["ratings"][-1] = deepcopy(reviews["ratings"][0])
    with pytest.raises(ValueError, match="Missing, duplicate"):
        validate_reviews(reviews, **export)


def test_invalid_claim_counts_cannot_overstate_support():
    _, export, reviews = fixture()
    reviews["ratings"][0]["supported_claims"] = 2
    with pytest.raises(ValueError, match="claim counts"):
        validate_reviews(reviews, **export)


def test_human_results_preserve_disagreement_without_consensus():
    run, export, reviews = fixture()
    reviews["ratings"][0]["complete_answer"] = False
    bundle = {**reviews, "mapping": export["mapping"]}
    report = summarize_run(run, bundle, bootstrap_replicates=20)["human_review"]
    assert report["agreement"]["complete_answer"]["rate"] == .5
    assert set(report["by_reviewer"]) == {"r1", "r2"}
    assert report["adjudication"] == "not_supplied"


def test_mapping_cannot_silently_reassign_family():
    run, export, reviews = fixture()
    export["mapping"]["items"][0]["family_id"] = "other"
    with pytest.raises(ValueError, match="metadata mismatch"):
        summarize_run(run, {**reviews, "mapping": export["mapping"]}, bootstrap_replicates=20)


def test_reference_opportunities_cannot_be_removed_by_reviewer():
    _, export, reviews = fixture()
    reviews["ratings"][0]["reference_correct"] = None
    with pytest.raises(ValueError, match="predefined opportunities"):
        validate_reviews(reviews, **export)


def test_history_is_allowlisted_as_displayed_text():
    run, _, _ = fixture()
    run["records"][0]["history"] = [{"question": "Hi", "answer": {"text": "Hello", "model": "hidden"}}]
    packet = create_review_packet(run)["packet"]
    assert "hidden" not in json.dumps(packet)
    assert any(item["history"] == [{"question": "Hi", "answer": "Hello"}] for item in packet["items"])


def test_explicit_adjudication_preserves_original_disagreement():
    run, export, reviews = fixture()
    adjudicated = [{**r, "reviewer_id": "adjudicator", "adjudication_reason": "Reviewed the complete source context"}
                  for r in reviews["ratings"] if r["reviewer_id"] == "r1"]
    reviews["ratings"][0]["complete_answer"] = False
    report = summarize_run(run, {**reviews, "mapping": export["mapping"], "adjudicated": adjudicated}, bootstrap_replicates=20)["human_review"]
    assert report["agreement"]["complete_answer"]["rate"] == .5
    assert report["adjudication"] == "explicit_human_adjudication"
    assert report["adjudicated"]["dialogue"]["v3"]["complete_answers"]["value"] == 1
