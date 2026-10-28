"""V3 display coverage stays broad while model synthesis remains JD scoped."""
import json

import pytest

from test_conversation_v3 import Composer, service, snapshot


@pytest.mark.parametrize("output", [
    "Candidate 1 has completed a Bachelor degree, while Candidate 9 is pursuing a Master degree.",
    "Both candidates are fully qualified.", None,
])
def test_v3_table_retains_six_categories_alongside_single_jd_criterion(output):
    current = snapshot()
    current.candidates[1]["rank"] = 9
    current.candidates[0]["preview_text"] += " Python development skills. AWS certification expired in 2021."
    model = Composer(output) if output else None
    engine = service(model)
    first = engine.answer(current, "Compare candidates 1 and 9.")
    assert first["generation_mode"] == ("qwen_prose" if output and output.startswith("Candidate 1") else "qwen_fallback")
    assert all(set(item["evidence_ids"]) <= {source["evidence_id"] for source in first["citations"]}
               for item in first["validated_claim_sources"])
    for answer in [first,
                   engine.answer(current, "What about their qualifications?", first["context"]),
                   engine.answer(current, "Why is candidate 1 better than candidate 9?", first["context"])]:
        rows = answer.get("comparison_rows", answer["comparison"])
        assert {row["category"] for row in rows[:6]} == {
            "KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS", "CERTIFICATION"}
        assert all(row["table_scope"] == "Broad evidence" for row in rows[:6])
        assert len(rows) >= 7
        assert any(row["table_scope"] == "JD criterion" for row in rows[6:]) or answer is not first
        assert any("expired in 2021" in row["answers"]["CV-A"]["text"] for row in rows)
        assert all(set(row["answers"]) == {"CV-A", "CV-B"} for row in rows)
        assert len(json.dumps(answer, ensure_ascii=False).encode()) < 24_000
    if model:
        assert all(len(call[2]["cells"]) == 2 for call in model.calls)


def test_added_broad_table_rejects_tampered_source_cells():
    current = snapshot()
    engine = service(None)
    original = engine.reader.base.compare

    def tampered(*args, **kwargs):
        result = original(*args, **kwargs)
        for citation in result["citations"]:
            citation["document_digest"] = "tampered"
        for row in result["comparison"]:
            row["requirement"] = "forged JD statement"
            for cell in row["answers"].values():
                cell.update(text="forged completion", context="forged completion")
        return result

    engine.reader.base.compare = tampered
    result = engine.answer(current, "Compare candidates 1 and 2.")
    assert "forged" not in json.dumps(result.get("comparison_rows", []))


def test_storage_guard_discards_broad_rows_and_dangling_model_claims():
    from candidate_screener.app.conversation_v3 import ConversationV3Service
    result = service(None).answer(snapshot(), "Compare candidates 1 and 2.")
    result["comparison_rows"] = [{"context": "x" * 30_000 + "Studies ongoing."}]
    result["validated_claim_sources"] = [{"evidence_ids": ["stale"]}]
    bounded = ConversationV3Service._bounded(result)
    assert not bounded.get("comparison_rows")
    assert bounded["validated_claim_sources"] == []
    assert bounded["status"] == "partial"
    assert bounded["comparison_table_status"] == "partial"
    assert bounded["generation_provenance"]["fallback_reason"] == "answer_storage_budget"
    assert len(json.dumps(bounded, ensure_ascii=False).encode()) < 24_000


def test_display_reuses_existing_broad_rows_without_duplicate_payload():
    current = snapshot()
    current.jd_text = "Build reliable services for the education team."
    result = service(None).answer(current, "Compare candidates 1 and 2.")
    assert "comparison_rows" not in result
    assert len(result["comparison"]) == 6
    assert all(row["table_scope"] == "Broad evidence" for row in result["comparison"])


def test_focused_table_retains_numeric_criterion_as_requested_text():
    current = snapshot()
    current.candidates[0]["preview_text"] += " Five years of Python experience."
    result = service(None).answer(current, "Compare candidates 1 and 2 on 5 years of Python experience.")
    rows = result["comparison_rows"]
    assert any("5 years of Python experience" in row.get("table_criterion", "") for row in rows[6:])
    for row in rows:
        for owner, cell in row["answers"].items():
            assert all(item["owner"] == owner for item in result["citations"]
                       if item["evidence_id"] in cell["evidence_ids"])
