"""Synthetic source ownership, abstention, comparison and model-budget invariants."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from candidate_screener.app import portal_extractive as module
from candidate_screener.app.portal_extractive import CATEGORIES, PortalExtractiveService
from candidate_screener.llm.qa_assets import MODEL_ID


def snapshot(first="Python and software testing skills. No completed degree; Bachelor studies ongoing.",
             second="SQL skills. AWS certification. Expired in 2021.", identity="snapshot-one",
             jd="Python programming skills. Bachelor degree required. AWS certification required."):
    return SimpleNamespace(snapshot_id=identity, jd_id="JD-A", jd_text=jd,
        candidates=tuple({"cv_id": owner, "rank": rank, "score": .8 / rank,
                          "preview_text": text, "preview_status": "available_redacted_text"}
                         for owner, rank, text in (("CV-A", 1, first), ("CV-B", 2, second))))


class FakeQa:
    """Mirrors QaClient's local-passage offset schema; no neural imports."""
    def __init__(self, status="supported", target="Python", transform=None):
        self.status, self.target, self.transform = status, target, transform
        self.calls = []

    def answer(self, question, passages):
        self.calls.append((question, copy.deepcopy(passages)))
        spans = []
        if self.status == "supported":
            for passage in passages:
                if self.target in passage["quote"]:
                    start = passage["quote"].index(self.target)
                    spans.append({"passage_id": passage["passage_id"],
                                  "candidate_id": passage["candidate_id"], "text": self.target,
                                  "start": start, "end": start + len(self.target)})
        response = {"status": self.status if spans or self.status != "supported" else "not_found",
                    "spans": spans, "model_id": MODEL_ID, "revision": "f" * 40}
        if self.transform:
            response = self.transform(response, passages)
        return response


class Entities:
    def __init__(self, records=(), *, status="complete", mutation=None):
        self.records, self.status, self.mutation = records, status, mutation
        self.calls = []

    def prepare(self, current, ids):
        self.calls.append((current.snapshot_id, list(ids)))
        sources = {current.jd_id: current.jd_text,
                   **{row["cv_id"]: row["preview_text"] for row in current.candidates if row["cv_id"] in ids}}
        documents = {}
        for owner, source in sources.items():
            entities = []
            for target, label in self.records:
                if target in source:
                    start = source.index(target)
                    entities.append({"start": start, "end": start + len(target), "text": target, "label": label})
            documents[owner] = {"status": self.status, "entities": entities,
                                "coverage": {"complete": self.status == "complete"},
                                "source_digest": hashlib.sha256(source.encode()).hexdigest()}
        result = {"status": self.status, "model_id": "synthetic-medium", "revision": "e" * 40,
                  "documents": documents}
        return self.mutation(result) if self.mutation else result


def cell(result, owner="CV-A", row=0):
    return result["comparison"][row]["answers"][owner]


def assert_citations(result, current):
    sources = {current.jd_id: current.jd_text, **{row["cv_id"]: row["preview_text"] for row in current.candidates}}
    ids = {item["evidence_id"] for item in result["citations"]}
    for item in result["citations"]:
        assert item["evidence_id"].startswith("qa:")
        assert item["snapshot_id"] == current.snapshot_id
        assert item["quote"] == sources[item["owner"]][item["start"]:item["end"]]
        assert item["document_digest"] == hashlib.sha256(sources[item["owner"]].encode()).hexdigest()
        assert item["coordinate_space"] == "redacted_preview"
    for row in result.get("comparison", []):
        assert set(row["requirement_evidence_ids"]) <= ids
        for answer in row["answers"].values():
            assert set(answer["evidence_ids"]) <= ids
    for claim in result["claims"]:
        assert claim["evidence_id"] in ids
        if claim["kind"] == "model_span":
            assert sources[claim["owner"]][claim["start"]:claim["end"]] == claim["text"]


def test_focused_answer_is_actual_span_with_source_context_and_natural_question():
    qa = FakeQa(target="Bachelor")
    current = snapshot()
    result = PortalExtractiveService(qa_client=qa).answer(current, "What degree does candidate 1 have?")
    answer = cell(result)
    assert answer["text"] == "Bachelor"
    assert answer["context"] == "No completed degree; Bachelor studies ongoing."
    assert "No completed degree" in result["text"] and "ongoing" in result["text"]
    assert result["mode"] == "extractive_qa" and result["model"] == MODEL_ID
    assert len(qa.calls) == 1 and qa.calls[0][0] == "What degree does the candidate have?"
    assert {p["candidate_id"] for p in qa.calls[0][1]} == {"CV-A"}
    assert {c["owner"] for c in result["citations"]} == {"CV-A"}
    assert_citations(result, current)


def test_focused_comparison_asks_factual_question_separately_for_each_owner():
    qa = FakeQa()
    current = snapshot(second="No commercial Python experience.")
    result = PortalExtractiveService(qa_client=qa).answer(current, "Compare candidates 1 and 2 on Python")
    assert len(qa.calls) == 3  # JD criterion, then each selected CV, independently.
    assert all(question.startswith("What does the ") and "Python" in question.casefold().title()
               for question, _ in qa.calls)
    assert all(question.endswith("?") and "compare" not in question.casefold() for question, _ in qa.calls)
    assert all(len({p["candidate_id"] for p in passages}) == 1 for _, passages in qa.calls)
    assert cell(result)["text"] == "Python"
    assert cell(result, "CV-B")["text"] == "Python"
    assert "No commercial Python experience." in cell(result, "CV-B")["context"]
    assert_citations(result, current)


def test_compare_never_starts_qa_and_has_all_six_categories_and_selected_owners(monkeypatch):
    def forbidden():
        raise AssertionError("Comparison started QA")
    monkeypatch.setattr(module, "shared_qa_client", forbidden)
    entities = Entities((("Python", "SKILL"), ("Bachelor", "DEGREE")))
    current = snapshot()
    result = PortalExtractiveService(entity_service=entities).compare(current, ["CV-B", "CV-A"])
    assert [row["category"] for row in result["comparison"]] == list(CATEGORIES)
    assert all(list(row["answers"]) == ["CV-B", "CV-A"] for row in result["comparison"])
    skills = next(row for row in result["comparison"] if row["category"] == "SKILL")
    assert skills["answers"]["CV-A"]["status"] == "entity_mentions"
    assert skills["requirement_evidence_ids"]
    assert result["qa_status"] == "not_run" and entities.calls == [("snapshot-one", ["CV-B", "CV-A"])]
    assert_citations(result, current)


def test_broad_chat_uses_same_nonblocking_comparison_as_dropdown():
    qa = FakeQa()
    entities = Entities((("Python", "SKILL"),))
    service = PortalExtractiveService(entity_service=entities, qa_client=qa)
    current = snapshot()
    direct = service.compare(current, ["CV-A", "CV-B"])
    chat = service.answer(current, "Compare candidates 1 and 2 overall")
    assert chat["comparison"] == direct["comparison"]
    assert not qa.calls


@pytest.mark.parametrize("status", ["not_found", "unavailable", "failed"])
def test_qa_abstention_is_explicit_source_only_never_a_model_span(status):
    current = snapshot()
    result = PortalExtractiveService(qa_client=FakeQa(status=status)).answer(current, "What skills does candidate 1 have?")
    assert result["qa_status"] == status
    assert cell(result)["status"] == "source_only"
    assert result["mode"] == "source_only" and result["model"] is None
    assert all(claim["kind"] == "source_only" for claim in result["claims"])
    assert "Source excerpt" in result["text"] and "Python and software testing" in result["text"]
    assert_citations(result, current)


@pytest.mark.parametrize("mutation", ["owner", "passage", "text", "start", "end", "bool",
                                    "snapshot", "digest", "model", "missing_owner", "mutate_input"])
def test_foreign_stale_or_altered_model_spans_are_rejected(mutation):
    def transform(response, passages):
        span = response["spans"][0]
        if mutation == "owner":
            span["candidate_id"] = "CV-B"
        elif mutation == "passage":
            span["passage_id"] = "qa:foreign"
        elif mutation == "text":
            span["text"] = "verified Python expert"
        elif mutation == "start":
            span["start"] = -1
        elif mutation == "end":
            span["end"] = 10000
        elif mutation == "bool":
            span["start"] = False
        elif mutation == "snapshot":
            span["snapshot_id"] = "stale-snapshot"
        elif mutation == "digest":
            span["document_digest"] = "foreign"
        elif mutation == "model":
            response["model_id"] = "unapproved/model"
        elif mutation == "missing_owner":
            del span["candidate_id"]
        else:
            passages[0]["quote"] = "verified Python expert"
            span.update(text="verified", start=0, end=8)
        return response
    result = PortalExtractiveService(qa_client=FakeQa(transform=transform)).answer(snapshot(), "What skills does candidate 1 have?")
    assert result["qa_status"] == "invalid_output"
    assert result["mode"] == "source_only" and result["model"] is None
    assert not any(claim["kind"] == "model_span" for claim in result["claims"])
    assert "verified Python expert" not in json.dumps(result)


def test_real_no_answer_cannot_be_overridden_by_a_plausible_span():
    def transform(response, _):
        response["status"] = "not_found"
        return response
    result = PortalExtractiveService(qa_client=FakeQa(transform=transform)).answer(snapshot(), "Candidate 1 Python skills")
    assert result["qa_status"] == "not_found" and result["model"] is None
    assert cell(result)["status"] == "source_only"


@pytest.mark.parametrize("source,target,question,qualifier", [
    ("AWS certification. Expired in 2021.", "AWS", "What certification does candidate 1 have?", "Expired in 2021."),
    ("Bachelor of Science. Studies ongoing.", "Bachelor", "What degree does candidate 1 have?", "Studies ongoing."),
    ("No commercial Python experience.", "Python", "What does candidate 1 state about Python?", "No commercial"),
])
def test_narrow_spans_never_hide_negation_expiry_or_ongoing_study(source, target, question, qualifier):
    result = PortalExtractiveService(qa_client=FakeQa(target=target)).answer(snapshot(first=source), question)
    assert cell(result)["text"] == target
    assert qualifier in cell(result)["context"] and qualifier in result["text"]


def test_late_unicode_span_and_citation_offsets_match_active_preview():
    current = snapshot(first="General introduction.\n" * 300 + "😀 No commercial Python experience.")
    result = PortalExtractiveService(qa_client=FakeQa()).answer(current, "Candidate 1 Python")
    assert cell(result)["text"] == "Python"
    assert result["citations"][0]["start"] > 3000
    assert_citations(result, current)


def test_certifications_use_source_text_even_without_entity_label_or_qa():
    result = PortalExtractiveService(entity_service=Entities((("Python", "SKILL"),))).compare(snapshot(), ["CV-A", "CV-B"])
    row = next(row for row in result["comparison"] if row["category"] == "CERTIFICATION")
    assert row["answers"]["CV-B"]["status"] == "source_only"
    assert row["answers"]["CV-B"]["context"] == "AWS certification. Expired in 2021."
    assert row["answers"]["CV-A"]["status"] == "not_found"
    assert "does not establish absence" in row["answers"]["CV-A"]["text"]


@pytest.mark.parametrize("mutation", ["digest", "owner", "text", "boundary"])
def test_invalid_entity_mentions_do_not_become_supported_mentions(mutation):
    def change(bundle):
        document = bundle["documents"]["CV-A"]
        entity = document["entities"][0]
        if mutation == "digest":
            document["source_digest"] = "stale"
        elif mutation == "owner":
            entity["owner"] = "CV-B"
        elif mutation == "text":
            entity["text"] = "Rust"
        else:
            entity["start"] = 99_999
        return bundle
    entities = Entities((("Python", "SKILL"),), mutation=change)
    result = PortalExtractiveService(entity_service=entities).compare(snapshot(), ["CV-A"])
    assert result["entity_status"] == "invalid_evidence"
    assert not any(claim["kind"] == "entity_mentions" and "software testing" in claim["text"] for claim in result["claims"])


def test_pending_entities_and_missing_sources_remain_explicit_and_do_not_block():
    current = snapshot()
    current.candidates[1]["preview_status"] = "unavailable"
    result = PortalExtractiveService(entity_service=Entities(status="queued")).compare(current, ["CV-A", "CV-B"])
    assert result["entity_status"] == "queued" and result["qa_status"] == "not_run"
    assert all(row["answers"]["CV-B"]["status"] == "source_unavailable" for row in result["comparison"])
    assert result["comparison_status"] == "partial"
    assert not any(item["owner"] == "CV-B" for item in result["citations"])


def test_total_experience_is_not_inferred_as_skill_duration():
    qa = FakeQa(target="7.5 years")
    current = snapshot(first="Worked 7.5 years in software; Python skills.")
    result = PortalExtractiveService(qa_client=qa).answer(current, "How many years of Python experience does candidate 1 have?")
    assert not qa.calls and not result["claims"]
    assert "Total experience does not establish" in result["text"]
    assert cell(result)["status"] == "not_found"


def test_explicit_skill_duration_can_be_extracted_with_same_source_context():
    result = PortalExtractiveService(qa_client=FakeQa(target="3 years")).answer(
        snapshot(first="Worked with Python for 3 years."), "How many years of Python experience does candidate 1 have?")
    assert cell(result)["text"] == "3 years"
    assert "Worked with Python for 3 years." == cell(result)["context"]


def test_action_guidance_and_stale_context_never_reach_models():
    class Never:
        def answer(self, *args):
            raise AssertionError("Action or clarification reached QA")
    service = PortalExtractiveService(qa_client=Never())
    assert service.answer(snapshot(), "Email candidate 1")["status"] == "action_guidance"
    assert service.answer(snapshot(), "Generate report")["status"] == "action_guidance"
    assert service.answer(snapshot(), "Candidate 99 skills")["status"] == "clarification"
    context = {"snapshot_id": "other", "jd_id": "JD-A", "selected_candidates": ["CV-A"], "category": "DEGREE"}
    assert service.answer(snapshot(), "What about their degree?", context)["status"] == "clarification"


def test_followup_keeps_category_but_explicit_reference_replaces_candidate():
    qa = FakeQa(target="Bachelor")
    service = PortalExtractiveService(qa_client=qa)
    previous = service.answer(snapshot(), "What degree does candidate 1 have?")["context"]
    answer = service.answer(snapshot(), "Candidate 2", previous)
    assert answer["context"]["selected_candidates"] == ["CV-B"]
    assert answer["context"]["category"] == "DEGREE"
    assert not cell(answer, "CV-B")["evidence_ids"]


def test_jd_question_only_sends_jd_passages_and_retains_jd_owned_citations():
    qa = FakeQa()
    result = PortalExtractiveService(qa_client=qa).answer(snapshot(), "What does the job description require about Python?")
    assert len(qa.calls) == 1
    assert all(p["candidate_id"] is None for _, passages in qa.calls for p in passages)
    assert {c["owner"] for c in result["citations"]} == {"JD-A"}
    assert result["context"]["selected_candidates"] == []


def test_unredacted_or_foreign_documents_are_never_used_or_submitted_to_qa():
    qa = FakeQa()
    current = snapshot(first="Contact private@example.com for Python skills.")
    result = PortalExtractiveService(qa_client=qa).answer(current, "Candidate 1 Python skills")
    assert not qa.calls and not result["citations"]
    assert "private@example.com" not in json.dumps(result)
    with pytest.raises(PermissionError):
        PortalExtractiveService(qa_client=qa).compare(current, ["CV-FOREIGN"])


def test_large_tables_bound_default_json_bytes_without_cutting_qualifiers_or_owners():
    # Non-ASCII forces the default stored JSON to be larger than UTF-8 source.
    source = "No completed degree; Bachelor studies ongoing. Python programming skills. " + "界" * 290 + "."
    current = snapshot(jd=source)
    current.candidates = tuple({"cv_id": f"CV-{index}", "rank": index, "score": .1,
                               "preview_text": source, "preview_status": "available_redacted_text"}
                              for index in range(1, 11))
    ids = [row["cv_id"] for row in current.candidates]
    result = PortalExtractiveService(entity_service=Entities((("Python", "SKILL"), ("Bachelor", "DEGREE")))).compare(current, ids)
    assert len(json.dumps(result).encode()) < 24_000
    assert len(result["comparison"]) == 6
    assert all(list(row["answers"]) == ids for row in result["comparison"])
    assert len(result["citations"]) <= 66 and len(result["claims"]) <= 88
    assert any(answer["status"] == "budget_limited" for row in result["comparison"] for answer in row["answers"].values())
    assert_citations(result, current)


def test_oversized_source_unit_is_explicitly_partial_not_clipped_to_hide_qualifier():
    text = "Python " + "background " * 150 + "; no commercial experience."
    result = PortalExtractiveService(qa_client=FakeQa()).answer(snapshot(first=text), "Candidate 1 Python")
    assert result["source_status"] == "partial"
    assert result["comparison_status"] == "partial"
    assert not result["citations"]


def test_long_complete_context_is_available_to_qa_without_clipping_status():
    text = "Python " + "background " * 50 + "; no commercial experience."
    result = PortalExtractiveService(qa_client=FakeQa()).answer(snapshot(first=text), "Candidate 1 Python")
    assert cell(result)["text"] == "Python"
    assert cell(result)["context"] == text
    assert "no commercial experience" in result["text"]


def test_natural_programming_language_question_retrieves_skill_context():
    qa = FakeQa()
    result = PortalExtractiveService(qa_client=qa).answer(snapshot(), "What programming language does candidate 1 use?")
    assert cell(result)["text"] == "Python"
    assert qa.calls[0][0] == "What programming language does the candidate use?"


def test_entity_authorization_failure_propagates_instead_of_returning_source_fallback():
    class Revoked:
        def prepare(self, *args):
            raise module.AuthorizationError("Revoked")
    with pytest.raises(module.AuthorizationError):
        PortalExtractiveService(entity_service=Revoked()).compare(snapshot(), ["CV-A"])


def test_incomplete_model_output_is_explicit_even_with_valid_spans():
    def incomplete(response, _):
        return {**response, "incomplete": True}
    result = PortalExtractiveService(qa_client=FakeQa(transform=incomplete)).answer(snapshot(), "Candidate 1 Python skills")
    assert result["qa_status"] == "partial" and result["exhaustive"] is False
    assert cell(result)["status"] == "model_span"


def test_foreign_entity_document_identity_is_rejected_even_when_text_and_digest_match():
    def foreign(bundle):
        bundle["documents"]["CV-A"]["owner_id"] = "CV-B"
        return bundle
    result = PortalExtractiveService(entity_service=Entities((("Python", "SKILL"),), mutation=foreign)).compare(snapshot(), ["CV-A"])
    assert result["entity_status"] == "invalid_evidence"
    assert not any(claim["kind"] == "entity_mentions" and "software testing" in claim["text"] for claim in result["claims"])


def test_storage_budget_removes_whole_model_claim_and_updates_visible_mode(monkeypatch):
    monkeypatch.setattr(module, "MAX_BYTES", 1300)
    result = PortalExtractiveService(qa_client=FakeQa()).answer(snapshot(), "Candidate 1 Python")
    assert len(json.dumps(result).encode()) <= 1300
    assert result["model"] is None and result["mode"] == "source_only"
    assert result["qa_status"] == "budget_limited" and not result["citations"]
    assert "Python" not in result["text"]
    assert cell(result)["status"] == "budget_limited"


def test_shared_factory_is_lazy_and_reuses_only_the_worker(monkeypatch):
    import candidate_screener.llm.extractive_qa as native
    created = []
    monkeypatch.setattr(native, "QaClient", lambda: created.append(object()) or created[-1])
    module.shared_qa_client.cache_clear()
    try:
        assert module.shared_qa_client() is module.shared_qa_client()
        assert len(created) == 1
    finally:
        module.shared_qa_client.cache_clear()


@pytest.mark.parametrize("question", [None, "", "x" * 2001])
def test_question_bounds(question):
    with pytest.raises(ValueError):
        PortalExtractiveService(qa_client=FakeQa()).answer(snapshot(), question)


@pytest.mark.parametrize("ids", ["CV-A", [], ["CV-A"] * 11, [None]])
def test_comparison_selection_bounds(ids):
    with pytest.raises(ValueError):
        PortalExtractiveService(qa_client=FakeQa()).compare(snapshot(), ids)


def test_reviewer_jd_requirement_keeps_negation_when_qa_returns_only_degree():
    current = snapshot(jd="Bachelor degree is not required.", first="Bachelor degree completed.", second="Bachelor degree completed.")
    result = PortalExtractiveService(qa_client=FakeQa(target="Bachelor")).answer(
        current, "Compare candidates 1 and 2 on degree")
    row = result["comparison"][0]
    assert "Bachelor degree is not required." in row["requirement"]
    assert row["requirement_status"] == "model_span"
    assert row["requirement_evidence_ids"]
    assert_citations(result, current)


@pytest.mark.parametrize("source", [
    "Python developer with 8 years of total experience.",
    "Python developer with 8 years of overall experience.",
    "Python developer with 8 years of experience.",
])
def test_reviewer_role_and_total_experience_do_not_establish_skill_tenure(source):
    qa = FakeQa(target="8 years")
    result = PortalExtractiveService(qa_client=qa).answer(
        snapshot(first=source), "How many years of Python experience does candidate 1 have?")
    assert not qa.calls
    assert result["status"] == "insufficient_evidence"
    assert cell(result)["status"] == "not_found"
    assert not any(claim["kind"] == "model_span" for claim in result["claims"])
    assert "8 years" not in result["text"]


def test_reviewer_model_cannot_select_total_duration_beside_explicit_skill_duration():
    source = "8 years of total experience, including 3 years with Python."
    qa = FakeQa(target="8 years")
    result = PortalExtractiveService(qa_client=qa).answer(
        snapshot(first=source), "How many years of Python experience does candidate 1 have?")
    assert qa.calls  # Relevant explicit evidence exists, but the returned span is the total.
    assert result["qa_status"] == "insufficient_evidence"
    assert result["status"] == "insufficient_evidence" and result["model"] is None
    assert cell(result)["status"] == "source_only"
    assert not any(claim["kind"] == "model_span" for claim in result["claims"])


def test_reviewer_explicit_skill_duration_remains_supported_beside_total():
    current = snapshot(first="8 years of total experience, including 3 years with Python.")
    result = PortalExtractiveService(qa_client=FakeQa(target="3 years")).answer(
        current, "How many years of Python experience does candidate 1 have?")
    assert cell(result)["text"] == "3 years" and cell(result)["qa_status"] == "supported"
    assert result["mode"] == "extractive_qa"
    assert_citations(result, current)


def test_reviewer_total_experience_question_remains_answerable():
    current = snapshot(first="Python developer with 8 years of total experience.")
    result = PortalExtractiveService(qa_client=FakeQa(target="8 years")).answer(
        current, "How many years of total experience does candidate 1 have?")
    assert cell(result)["text"] == "8 years" and cell(result)["qa_status"] == "supported"
    assert result["mode"] == "extractive_qa"
    assert_citations(result, current)


@pytest.mark.parametrize("question,target,expected", [
    ("What does this role require?", "Python", "Python programming skills."),
    ("What degree is required in the JD?", "Bachelor", "Bachelor degree required."),
])
def test_reviewer_jd_framing_does_not_become_a_literal_retrieval_constraint(question, target, expected):
    current = snapshot(jd="Python programming skills. Bachelor degree required.")
    qa = FakeQa(target=target)
    result = PortalExtractiveService(qa_client=qa).answer(current, question)
    assert qa.calls
    assert result["mode"] == "extractive_qa" and result["status"] == "answered"
    assert expected in result["text"]
    assert all(passage["candidate_id"] is None for _, passages in qa.calls for passage in passages)
    assert {item["owner"] for item in result["citations"]} == {"JD-A"}
    assert_citations(result, current)


@pytest.mark.parametrize("source,target,category,qualifier", [
    ("AWS certification. Valid until 2021.", "AWS", "certification", "Valid until 2021."),
    ("Bachelor degree. Status: in progress.", "Bachelor", "degree", "Status: in progress."),
])
def test_reviewer_adjacent_qualifiers_remain_visible_in_fact_text_and_jd_requirement(source, target, category, qualifier):
    current = snapshot(first=source, second=source, jd=source)
    service = PortalExtractiveService(qa_client=FakeQa(target=target))
    fact = service.answer(current, f"What {category} does candidate 1 have?")
    assert cell(fact)["text"] == target
    assert qualifier in cell(fact)["context"]
    assert qualifier in fact["text"]
    compared = service.answer(current, f"Compare candidates 1 and 2 on {category}")
    assert qualifier in compared["comparison"][0]["requirement"]
    assert_citations(fact, current)
    assert_citations(compared, current)
