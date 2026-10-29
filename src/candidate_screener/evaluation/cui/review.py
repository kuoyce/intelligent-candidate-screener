"""Blinded synthetic CUI review packets; system identities remain in a private map."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import random


def create_review_packet(run, seed=0):
    from .metrics import validate_run
    validate_run(run)
    records = list(run["records"])
    random.Random(seed).shuffle(records)
    items, identities = [], []
    for number, record in enumerate(records, 1):
        item_id = f"item-{number:04d}"
        # Explicit allowlist: response records contain model/fallback diagnostics.
        items.append({"item_id": item_id, "question": record["question"],
                      "answer": record["answer"].get("text", ""),
                      "source_documents": deepcopy(record["source_documents"]),
                      "expected": deepcopy(record["expected"]),
                      "history": [{"question": h["question"],
                                   "answer": h["answer"].get("text", "") if isinstance(h["answer"], dict) else h["answer"]}
                                  for h in record.get("displayed_history", record.get("history", []))]})
        identities.append({"item_id": item_id, **{key: record[key] for key in
            ("turn_id", "arm", "family_id", "task_id", "task_type")}})
    return {"packet": {"schema_version": 1, "run_id": run["run_id"], "items": items},
            "mapping": {"schema_version": 1, "run_id": run["run_id"], "items": identities}}


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_rating(row):
    for key in ("item_id", "reviewer_id"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Missing {key}")
    total, supported = row.get("total_claims"), row.get("supported_claims")
    if not _integer(total) or not _integer(supported) or not 0 <= supported <= total:
        raise ValueError("Invalid supported/total claim counts")
    for key in ("complete_answer", "unnecessary_clarification", "abstained"):
        if not isinstance(row.get(key), bool):
            raise ValueError(f"{key} must be boolean")
    for key in ("correct_abstention", "reference_correct"):
        if key not in row or row[key] is not None and not isinstance(row[key], bool):
            raise ValueError(f"{key} must be boolean or null")
    if not isinstance(row.get("critical_errors"), list) or any(
            not isinstance(error, str) or not error.strip() for error in row["critical_errors"]):
        raise ValueError("critical_errors must be a list of descriptions")
    for key in ("directness", "clarity", "usefulness"):
        if not _integer(row.get(key)) or not 1 <= row[key] <= 5:
            raise ValueError(f"{key} must be an integer from 1 to 5")


def validate_reviews(reviews, packet, mapping):
    if not reviews.get("run_id") or reviews["run_id"] != packet.get("run_id") or reviews["run_id"] != mapping.get("run_id"):
        raise ValueError("Review run identity mismatch")
    packet_ids = [item["item_id"] for item in packet["items"]]
    map_ids = [item["item_id"] for item in mapping["items"]]
    if len(set(packet_ids)) != len(packet_ids) or len(set(map_ids)) != len(map_ids) or set(packet_ids) != set(map_ids):
        raise ValueError("Review mapping is not one-to-one")
    keys = [(item["turn_id"], item["arm"]) for item in mapping["items"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate mapped run record")
    rows = reviews.get("ratings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing ratings")
    for row in rows:
        validate_rating(row)
    reviewers = sorted({row["reviewer_id"] for row in rows})
    if len(reviewers) != 2:
        raise ValueError("Exactly two independent reviewer identities required")
    observed = Counter((row["item_id"], row["reviewer_id"]) for row in rows)
    required = Counter((item, reviewer) for item in packet_ids for reviewer in reviewers)
    if observed != required:
        raise ValueError("Missing, duplicate or unexpected review ratings")
    expected = {item["item_id"]: item["expected"] for item in packet["items"]}
    for row in rows:
        outcome = expected[row["item_id"]]
        if outcome.get("uncertainty") and row["correct_abstention"] is None:
            raise ValueError("Uncertainty cases require an abstention assessment")
        if not outcome.get("uncertainty") and row["correct_abstention"] is not None:
            raise ValueError("Answerable cases must use null for correct_abstention")
        opportunity = bool(outcome.get("participants") or outcome.get("clarification"))
        if opportunity != (row["reference_correct"] is not None):
            raise ValueError("Reference assessment must match predefined opportunities")
    return {"reviewers": reviewers, "items": len(packet_ids), "ratings": len(rows)}


def review_template(packet, reviewer_ids=("reviewer-1", "reviewer-2")):
    """Blank responses, deliberately invalid until each human completes them."""
    return {"run_id": packet["run_id"], "ratings": [
        {"item_id": item["item_id"], "reviewer_id": reviewer,
         "supported_claims": None, "total_claims": None, "complete_answer": None,
         "abstained": None, "correct_abstention": None, "reference_correct": None,
         "unnecessary_clarification": None, "critical_errors": [],
         "directness": None, "clarity": None, "usefulness": None}
        for item in packet["items"] for reviewer in reviewer_ids]}
