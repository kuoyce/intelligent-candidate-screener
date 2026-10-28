"""Exact, source-owned entity overlap across the authorized JD and CVs."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from candidate_screener.app.portal_extractive import PortalExtractiveService


def fixture(jd="Ruby and SQL required.", sources=("Ruby and SQL skills.", "SQL skills.")):
    snapshot = SimpleNamespace(snapshot_id="overlap", jd_id="JD-A", jd_text=jd,
        candidates=tuple({"cv_id": f"CV-{index}", "rank": index, "preview_text": source,
                          "preview_status": "available_redacted_text"}
                         for index, source in enumerate(sources, 1)))
    documents = {"JD-A": jd, **{row["cv_id"]: row["preview_text"] for row in snapshot.candidates}}
    records = {}
    for owner, source in documents.items():
        entities = []
        for term in ("Ruby", "SQL", "Java", "JavaScript", "Python"):
            start = source.find(term)
            if start >= 0:
                entities.append({"start": start, "end": start + len(term), "text": term,
                                 "label": "SKILL" if owner == "JD-A" else "KNOWLEDGE"})
        records[owner] = {"status": "complete", "source_digest": hashlib.sha256(source.encode()).hexdigest(),
                          "entities": entities}

    class Entities:
        calls = 0
        def prepare(self, snapshot, selected):
            self.calls += 1
            return {"status": "complete", "documents": records}

    entities = Entities()
    return snapshot, PortalExtractiveService(entities), records, entities


def test_exact_cross_label_overlap_is_controlled_and_cites_both_sources():
    from candidate_screener.app.skill_overlap import skill_overlap
    current, base, _, entities = fixture()
    result = skill_overlap(current, ["CV-1", "CV-2"], base)
    assert entities.calls == 1
    assert result["generation_mode"] == "controlled"
    assert result["model"] is None
    assert result["skill_overlaps"][0]["terms"] == ["Ruby", "SQL"]
    assert result["skill_overlaps"][1]["terms"] == ["SQL"]
    assert "Candidate 1" in result["text"] and "Candidate 2" in result["text"]
    assert {item["owner"] for item in result["citations"]} == {"JD-A", "CV-1", "CV-2"}
    assert all(item["quote"] for item in result["citations"])
    assert len({item["evidence_id"] for item in result["citations"]}) == len(result["citations"])


@pytest.mark.parametrize("source", ["No Ruby experience.", "Learning Ruby development skills.",
                                     "Ruby certification expired in 2021."])
def test_qualified_or_negative_mentions_never_become_positive_overlap(source):
    from candidate_screener.app.skill_overlap import skill_overlap
    current, base, _, _ = fixture(jd="Ruby required.", sources=(source,))
    result = skill_overlap(current, ["CV-1"], base)
    assert result["skill_overlaps"][0]["terms"] == []
    assert result["skill_overlaps"][0]["qualified_terms"] == ["Ruby"]
    assert "both mention Ruby" not in result["text"]
    assert any(item["quote"] == source for item in result["citations"])


@pytest.mark.parametrize("corruption", ["hash", "span", "owner"])
def test_tampered_entity_cannot_authorize_overlap(corruption):
    from candidate_screener.app.skill_overlap import skill_overlap
    current, base, records, _ = fixture(jd="Ruby required.", sources=("Ruby skills.",))
    if corruption == "hash":
        records["CV-1"]["source_digest"] = "stale"
    elif corruption == "span":
        records["CV-1"]["entities"][0]["start"] = 1
    else:
        records["CV-1"]["entities"][0]["owner"] = "CV-OTHER"
    result = skill_overlap(current, ["CV-1"], base)
    assert not result["skill_overlaps"][0]["terms"]
    assert result["status"] == "partial"


def test_pending_extraction_is_not_reported_as_no_skills_or_completed_search():
    from candidate_screener.app.skill_overlap import skill_overlap
    current, base, records, _ = fixture()
    records["CV-2"].update(status="queued", entities=[])
    result = skill_overlap(current, ["CV-1", "CV-2"], base)
    assert result["entity_status_by_owner"]["CV-2"] == "queued"
    assert result["status"] == "partial"
    assert "Candidate 2" in result["text"] and "queued" in result["text"]
    assert "does not establish absence" in result["text"]


def test_java_is_not_fuzzy_matched_to_javascript():
    from candidate_screener.app.skill_overlap import skill_overlap
    current, base, records, _ = fixture(jd="Java required.", sources=("JavaScript skills.",))
    records["CV-1"]["entities"] = [item for item in records["CV-1"]["entities"] if item["text"] == "JavaScript"]
    result = skill_overlap(current, ["CV-1"], base)
    assert result["skill_overlaps"][0]["terms"] == []


def test_top_ten_overlap_budget_omits_whole_contexts_and_reports_partial():
    from candidate_screener.app.skill_overlap import skill_overlap
    units = [term + " skills " + "building reliable systems " * 28 + "." for term in ("Python", "Ruby", "SQL")]
    source = "\n".join(units)
    jd = "Python, Ruby and SQL required."
    current, base, _, _ = fixture(jd=jd, sources=(source,) * 10)
    result = skill_overlap(current, [row["cv_id"] for row in current.candidates], base)
    assert len(result["skill_overlaps"]) == 10
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 23900
    assert all(item["quote"] in {*units, jd} for item in result["citations"])
    assert result["status"] == "partial"
    assert result["generation_provenance"]["partial_reason"] == "answer_storage_budget"
    assert any(row.get("overlap_status") == "partial" for row in result["skill_overlaps"])


def test_adjacent_learning_status_cannot_be_detached_from_skill_mention():
    from candidate_screener.app.skill_overlap import skill_overlap
    source = "Ruby skills. Learning only."
    current, base, _, _ = fixture(jd="Ruby required.", sources=(source,))
    result = skill_overlap(current, ["CV-1"], base)
    assert result["skill_overlaps"][0]["terms"] == []
    assert result["skill_overlaps"][0]["qualified_terms"] == ["Ruby"]
    assert any(item["quote"] == source for item in result["citations"])
