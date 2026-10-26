"""Versioned, source-bound dialogue state; prose is never candidate evidence."""
from __future__ import annotations

import copy
from collections.abc import Mapping

FRAME_VERSION = 2


def frame_for(snapshot, context=None):
    """Read legacy frames conservatively and reject foreign snapshot scope."""
    available = {row["cv_id"] for row in snapshot.candidates}
    previous = context if isinstance(context, dict) else {}
    if previous.get("snapshot_id") != snapshot.snapshot_id or previous.get("jd_id") != snapshot.jd_id:
        previous = {}
    selected = previous.get("selected_candidates", [])
    if not isinstance(selected, list) or any(not isinstance(owner, str) or owner not in available for owner in selected):
        selected = []
    category = previous.get("category", "GENERAL")
    if not isinstance(category, str) or category not in {"GENERAL", "KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS", "CERTIFICATION"}:
        category = "GENERAL"
    criteria = previous.get("criteria", [])
    criteria = [item[:200] for item in criteria[:6] if isinstance(item, str)] if isinstance(criteria, list) else []
    pending = previous.get("pending_task")
    if not (isinstance(pending, Mapping) and pending.get("slot") == "candidates"
            and isinstance(pending.get("question"), str) and len(pending["question"]) <= 2000):
        pending = None
    result = {"schema_version": FRAME_VERSION, "snapshot_id": snapshot.snapshot_id,
              "jd_id": snapshot.jd_id, "selected_candidates": list(dict.fromkeys(selected))[:10],
              "category": category, "intent": previous.get("intent", "unknown")
              if previous.get("intent") in ("comparison", "comparison_explanation", "candidate_qa", "requirements", "rank_explanation") else "unknown",
              "criteria": criteria,
              "pending_clarification": previous.get("pending_clarification")[:2000]
              if isinstance(previous.get("pending_clarification"), str) else None,
              "pending_task": copy.deepcopy(pending),
              "last_question": previous.get("last_question", "")[:2000]
              if isinstance(previous.get("last_question", ""), str) else "",
              "last_evidence": evidence_descriptors(previous.get("last_evidence", []))
              if isinstance(previous.get("last_evidence"), list) else []}
    options = previous.get("candidate_options")
    if isinstance(options, list):
        result["candidate_options"] = [item for item in options[:10] if isinstance(item, str) and item in available]
    if isinstance(previous.get("focal_candidate_id"), str) and previous["focal_candidate_id"] in available:
        result["focal_candidate_id"] = previous["focal_candidate_id"]
    if (isinstance(previous.get("explanation_subject"), str) and previous["explanation_subject"] in available
            and isinstance(previous.get("explanation_reference"), str) and previous["explanation_reference"] in available):
        result.update(explanation_subject=previous["explanation_subject"], explanation_reference=previous["explanation_reference"],
                      explanation_relation="ranked_above" if previous.get("explanation_relation") == "ranked_above" else "better")
    return result


def evidence_descriptors(citations):
    """Persist just source coordinates; full excerpts already belong to the turn."""
    keys = ("evidence_id", "owner", "owner_kind", "snapshot_id", "document_digest",
            "start", "end", "coordinate_space")
    records = []
    for item in citations[:24]:
        if not isinstance(item, Mapping):
            continue
        value = {key: item[key] for key in keys if key in item}
        if any(not isinstance(value.get(key), str) or len(value[key]) > 128
               for key in keys if key not in {"start", "end"}):
            continue
        if type(value.get("start")) is not int or type(value.get("end")) is not int:
            continue
        records.append(value)
    return records
