import json
import pytest
from candidate_screener.entity_annotation.boundary_review import render_review
from candidate_screener.entity_annotation.text import prepare_text


def fixture(tmp_path):
    original = "Contact private@example.com 😀 Python <script>alert(1)</script>"
    prepared = prepare_text(original)
    a = original.index("Python")
    doc = {"doc_id": "synthetic-id", "source_text": original, **prepared}
    span = {"start": a, "end": a + 6, "label": "KNOWLEDGE"}
    campaign = {"campaign_id": "synthetic", "annotators": ["annotator1", "annotator2"], "documents": [doc]}
    record = {"campaign_id": "synthetic", "annotator": "annotator1", "doc_id": doc["doc_id"],
              "revision": 1, "source_sha256": doc["source_sha256"], "spans": [span]}
    case = {"doc_id": doc["doc_id"], "revision": 1, "source_sha256": doc["source_sha256"],
            "label": "KNOWLEDGE", "reason": "inside_word_cut", "gold": span,
            "proposal_for_human_review_only": span}
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"annotator": "annotator1", "issues": [case]}))
    return campaign, [record], path


def test_review_html_is_redacted_escaped_readonly_and_links_document(tmp_path):
    campaign, records, path = fixture(tmp_path)
    result = render_review(campaign, records, path, "annotator1")
    assert "private@example.com" not in result
    assert "[EMAIL]" in result
    assert "<script>alert" not in result
    assert "&lt;script&gt;" in result
    assert "doc_id=synthetic-id" in result
    assert "TOKEN COVERAGE ONLY" in result
    assert "0 stale cases" in result
    assert "<form" not in result and "fetch(" not in result


@pytest.mark.parametrize("change", ["revision", "source", "span"])
def test_stale_queue_never_offers_old_suggestions(tmp_path, change):
    campaign, records, path = fixture(tmp_path)
    if change == "revision": records[0]["revision"] += 1
    elif change == "source": records[0]["source_sha256"] = "changed"
    else: records[0]["spans"] = []
    result = render_review(campaign, records, path, "annotator1")
    assert "STALE" in result and "1 stale cases" in result
    assert "Open document editor" not in result
    assert "TOKEN COVERAGE ONLY" not in result


def test_independent_second_annotator_cannot_see_boundary_suggestions(tmp_path):
    campaign, records, path = fixture(tmp_path)
    with pytest.raises(ValueError, match="snapshot annotator"):
        render_review(campaign, records, path, "annotator2")
