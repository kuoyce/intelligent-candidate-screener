"""Exact JD/CV entity mention overlap; no generation, ratings or inferred tenure."""
from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
import json
import re
from types import SimpleNamespace

from .conversation_schema import evidence_descriptors, frame_for
from .portal_extractive import PortalExtractiveService, _citation, _documents, _selection
from .portal_store import AuthorizationError

_INCOMPLETE = {"queued", "running", "partial", "failed", "unavailable", "invalid_evidence"}
_QUALIFIER = re.compile(
    r"\b(?:no|not|never|without|lacks?|learning|studying|training|pursuing|expired|expires|"
    r"pending|conditional|provisional|incomplete|in progress|hopes?|plans?|planned|planning|"
    r"wants?|wishes?|aspir\w*|would like|expected|unless)\b", re.I)
_ADJACENT_STATUS = re.compile(
    r"^\s*(?:status\s*:\s*)?(?:learning|studying|training|expired|expires|valid until|valid through|"
    r"pending|conditional|provisional|incomplete|in progress|not\b|no\b|without\b)", re.I)


def _key(text):
    return " ".join(text.casefold().split())


def _mentions(snapshot, document, entities):
    """Keep complete qualifier-bearing units for every individually anchored span."""
    mentions = {}
    for entity in entities:
        if entity["label"] not in {"SKILL", "KNOWLEDGE"}:
            continue
        source, a, b = document["source"], entity["start"], entity["end"]
        if ((a and re.match(r"\w", source[a - 1]) and re.match(r"\w", source[a]))
                or (b < len(source) and re.match(r"\w", source[b - 1]) and re.match(r"\w", source[b]))):
            continue
        units = document["units"]
        index = next((index for index, unit in enumerate(units)
                      if unit[0] <= entity["start"] < entity["end"] <= unit[1]), None)
        if index is None:
            continue
        unit = units[index]
        start, end = unit[:2]
        if index and units[index - 1][2].rstrip().endswith(":") and _QUALIFIER.search(units[index - 1][2]):
            start = units[index - 1][0]
        if index + 1 < len(units) and _ADJACENT_STATUS.match(units[index + 1][2]):
            end = units[index + 1][1]
        unit = (start, end, document["source"][start:end])
        record = _citation(snapshot, document, unit)
        mentions.setdefault(_key(entity["text"]), []).append({
            "term": entity["text"], "qualified": bool(_QUALIFIER.search(unit[2])), "citation": record})
    return mentions


def skill_overlap(snapshot, candidate_ids, base, context=None):
    """Read one extraction snapshot and report source-owned overlap for up to ten CVs."""
    selected = _selection(snapshot, candidate_ids)
    if not selected:
        raise ValueError("Select at least one candidate for relevant skill overlap")
    documents = _documents(snapshot, selected)
    try:
        raw = base.entity_service.prepare(snapshot, selected) if base.entity_service is not None else {}
    except (AuthorizationError, PermissionError):
        raise
    except Exception:
        raw = {}
    raw = raw if isinstance(raw, Mapping) else {}
    raw_documents = raw.get("documents", {})
    raw_documents = raw_documents if isinstance(raw_documents, Mapping) else {}
    # Reuse existing exact span/hash/owner validation without scheduling another
    # extraction job. Validate separately so one stale CV cannot taint its peers.
    reader = PortalExtractiveService(SimpleNamespace(prepare=lambda *_: raw))
    states, mentions = {}, {}
    for owner, document in documents.items():
        row = raw_documents.get(owner, {})
        row = row if isinstance(row, Mapping) else {}
        state = row.get("status", raw.get("status", "unavailable"))
        state = state if isinstance(state, str) and state in {"complete", *_INCOMPLETE} else "unavailable"
        valid, checked = reader._entities(snapshot, selected, {owner: document})
        if document["status"] == "unavailable":
            state = "unavailable"
        elif checked == "invalid_evidence":
            state = "invalid_evidence"
        elif (document["status"] == "partial" or state == "partial"
              or isinstance(row.get("entities"), (list, tuple)) and len(row["entities"]) > 512):
            state = "partial"
        states[owner] = state
        mentions[owner] = (_mentions(snapshot, document, valid.get(owner, []))
                           if state in {"complete", "partial"} else {})
    labels = {row["cv_id"]: f"Candidate {row['rank']}" for row in snapshot.candidates if row["cv_id"] in selected}
    matches = []
    jd_mentions = mentions[snapshot.jd_id]
    for owner in selected:
        for key, jd_records in jd_mentions.items():
            cv_records = mentions[owner].get(key)
            if not cv_records:
                continue
            records = [*jd_records, *cv_records]
            matches.append({"owner": owner, "term": jd_records[0]["term"],
                            "qualified": any(record["qualified"] for record in records),
                            "citations": list({record["citation"]["evidence_id"]: record["citation"]
                                               for record in records}.values())})
    state = frame_for(snapshot, context)
    state.update(selected_candidates=selected, category="SKILL", criteria=[],
                 intent="comparison" if len(selected) > 1 else "candidate_qa",
                 pending_task=None, pending_clarification=None)
    omitted = False
    original_counts = Counter(record["owner"] for record in matches)

    def result_for(records):
        citations, overlaps, lines = {}, [], []
        jd_state = states[snapshot.jd_id]
        if jd_state != "complete":
            lines.append(f"JD entity extraction is {jd_state.replace('_', ' ')}; the overlap review is incomplete.")
        for owner in selected:
            owned = [record for record in records if record["owner"] == owner]
            owner_omitted = len(owned) < original_counts[owner]
            positive = list(dict.fromkeys(record["term"] for record in owned if not record["qualified"]))
            qualified = list(dict.fromkeys(record["term"] for record in owned if record["qualified"]))
            ids = []
            for record in owned:
                for citation in record["citations"]:
                    citations[citation["evidence_id"]] = citation
                    if citation["evidence_id"] not in ids:
                        ids.append(citation["evidence_id"])
            if positive:
                lines.append(f"The JD and {labels[owner]}'s CV both mention {', '.join(positive)}.")
            if qualified:
                lines.append(f"{labels[owner]} has matching mentions of {', '.join(qualified)} in qualified or negative JD/CV context; review the supporting sources.")
            if states[owner] != "complete":
                lines.append(f"{labels[owner]}'s entity extraction is {states[owner].replace('_', ' ')}; this review is incomplete.")
            if owner_omitted:
                lines.append(f"Some matching mentions for {labels[owner]} were omitted by the saved-reply budget.")
            elif not owned and states[owner] == "complete" and jd_state != "complete":
                lines.append(f"{labels[owner]}'s JD/CV overlap cannot yet be established while JD extraction is incomplete.")
            elif not owned and states[owner] == "complete":
                lines.append(f"I did not establish matching JD skill mentions for {labels[owner]} in the reviewed evidence.")
            overlaps.append({"owner": owner, "status": states[owner], "terms": positive,
                             "qualified_terms": qualified, "evidence_ids": ids,
                             "overlap_status": "partial" if owner_omitted else states[owner]})
        lines.append("Mention overlap does not establish proficiency or skill-specific tenure; missing evidence does not establish absence.")
        if omitted:
            lines.append("Some whole skill matches were omitted to fit the saved reply; review the supporting sources.")
        partial = omitted or any(value != "complete" for value in states.values())
        provenance = {"composer": "application", "method": "source_owned_skill_overlap"}
        if omitted:
            provenance["partial_reason"] = "answer_storage_budget"
        result = {"text": " ".join(lines), "citations": list(citations.values()), "claims": [],
                  "comparison": [], "context": {**state, "last_evidence": evidence_descriptors(list(citations.values()))},
                  "mode": "conversation_v3", "answer_version": 3, "generation_mode": "controlled",
                  "status": "partial" if partial else "answered", "model": None,
                  "suggested_actions": [], "entity_status": "partial" if partial else "complete",
                  "entity_status_by_owner": states, "qa_status": "not_run",
                  "candidate_labels": {snapshot.jd_id: "The role", **labels},
                  "show_comparison_table": False, "comparison_table_available": False,
                  "skill_overlaps": overlaps, "generation_provenance": provenance}
        return result

    result = result_for(matches)
    while len(json.dumps(result, ensure_ascii=False).encode()) > 23900 and matches:
        matches.pop()
        omitted = True
        result = result_for(matches)
    if len(json.dumps(result, ensure_ascii=False).encode()) > 23900:
        raise ValueError("Skill overlap metadata exceeds the saved reply budget")
    return result
