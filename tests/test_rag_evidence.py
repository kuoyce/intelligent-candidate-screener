"""Question passages use synthetic source text only; no neural dependencies."""
from dataclasses import FrozenInstanceError
import pytest

from candidate_screener.app.rag_evidence import EvidenceIndex
from candidate_screener.entity_annotation.text import prepare_text


def test_late_cv_fact_retrieved_from_complete_source_instead_of_prefix():
    text = ("Worked on general software projects.\n" * 100) + "CERTIFICATIONS:\nAWS Certified Solutions Architect, valid through 2028."
    index = EvidenceIndex.from_sources("snapshot-one", "AWS experience preferred.", {"A": text}, max_chars=180, overlap_chars=60)
    found = index.search("What AWS certifications does A hold?", ["A"], include_jd=False)
    assert any("AWS Certified Solutions Architect" in p.quote for p in found)
    assert any(p.original_start > 3000 for p in found)
    assert len(found) <= 2
    assert all(text[p.original_start:p.original_end] == p.quote for p in found)


def test_redacted_display_coordinates_map_back_to_unchanged_unicode_source():
    text = "😀 Contact long.private@example.com\nCERTIFICATIONS:\nC++ and .NET certificate."
    index = EvidenceIndex.from_sources("snapshot", "Role.", {"A": text}, max_chars=60, overlap_chars=20)
    prepared = prepare_text(text)
    found = index.search(".NET certificate", ["A"], include_jd=False)
    assert found
    for passage in index.passages:
        if passage.candidate_id != "A":
            continue
        assert passage.quote == prepared["display_text"][passage.display_start:passage.display_end]
        assert prepared["boundary_map"][passage.display_start] == passage.original_start
        assert prepared["boundary_map"][passage.display_end] == passage.original_end
        assert "long.private@example.com" not in passage.quote
    assert any(p.original_start != p.display_start or p.original_end != p.display_end for p in index.passages if p.candidate_id == "A")
    assert all(p.page is None for p in index.passages)


def test_requested_owner_only_and_stale_snapshot_evidence_rejected():
    first = EvidenceIndex.from_sources("one", "AWS role", {"A": "AWS certification.", "B": "AWS certification."})
    found = first.search("AWS certification", ["B"], include_jd=False)
    assert found and all(p.candidate_id == "B" for p in found)
    with pytest.raises(ValueError, match="owner"):
        first.get(found[0].passage_id, candidate_id="A")
    second = EvidenceIndex.from_sources("two", "AWS role", {"B": "AWS certification."})
    with pytest.raises(ValueError, match="snapshot"):
        second.get(found[0].passage_id, candidate_id="B")
    with pytest.raises(ValueError, match="outside"):
        first.search("AWS", ["unknown"])


def test_immutable_session_owned_index_and_stable_ids():
    sources = {"A": "SQL engineer."}
    first = EvidenceIndex.from_sources("same", "SQL role.", sources)
    second = EvidenceIndex.from_sources("same", "SQL role.", dict(sources))
    assert first.index_digest == second.index_digest
    assert first.passages == second.passages
    sources["A"] = "Changed source"
    assert any(p.quote == "SQL engineer." for p in first.passages)
    with pytest.raises(FrozenInstanceError):
        first.snapshot_id = "changed"
    with pytest.raises(TypeError):
        first._by_id["new"] = first.passages[0]
    changed = EvidenceIndex.from_sources("same", "SQL role.", sources)
    assert first.index_digest != changed.index_digest


def test_candidate_coverage_precedes_extra_passages_and_question_specific_search():
    index = EvidenceIndex.from_sources("snapshot", "Unrelated role.", {
        "A": "AWS certification.\n" * 20, "B": "AWS certification.\nPython developer."}, max_chars=64, overlap_chars=20)
    found = index.search("AWS certification", ["A", "B"], limit=2, include_jd=False)
    assert {p.candidate_id for p in found} == {"A", "B"}
    python = index.search("Python", ["A", "B"], include_jd=False)
    assert python and all(p.candidate_id == "B" for p in python)
    assert index.search("nevermentionedtechnology", ["A", "B"]) == ()


def test_no_stemming_or_source_cleanup_and_no_invented_pages():
    text = "C++ C# .NET Node.js\nStudying studies."
    index = EvidenceIndex.from_sources("snapshot", "", {"A": text})
    assert any(p.quote == text for p in index.passages)
    assert index.search("study", ["A"]) == ()
    assert index.search("Node.js C++", ["A"])


def test_cross_sentence_context_and_full_source_interval_union():
    text = "Earlier general work.\n" * 15 + "Completed an AWS examination.\nThe certification remains valid in 2028."
    index = EvidenceIndex.from_sources("snapshot", "", {"A": text}, max_chars=160, overlap_chars=80)
    passages = [p for p in index.passages if p.candidate_id == "A"]
    assert all(any(p.original_start <= i < p.original_end for p in passages) for i in range(len(text)))
    assert any("AWS examination." in p.quote and "remains valid" in p.quote for p in passages)
