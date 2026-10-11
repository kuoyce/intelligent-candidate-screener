from types import SimpleNamespace

from candidate_screener.extraction.live import GlinerEntityExtractor


def extractor(tmp_path, infer):
    return GlinerEntityExtractor(tmp_path, loader=lambda *a, **k: SimpleNamespace(),
        infer=infer, manifest_loader=lambda p: {"revision": "test"},
        runtime_loader=lambda p: {"versions": {}})


def span():
    return {"label": "KNOWLEDGE", "start": 0, "end": 6, "surface": "Python",
            "confidence": .9, "method": "gliner-zero-shot", "context_status": "unqualified"}


def test_complete_zero_is_distinct_from_missing_model(tmp_path):
    zero = extractor(tmp_path, lambda *a, **k: ([], {"text": {"complete": True}}))
    assert zero("Python", "cv")["status"] == "completed_zero"
    missing = GlinerEntityExtractor(tmp_path)
    result = missing("Python", "cv")
    assert result["status"] == "unavailable"
    assert result["counts"] == {}


def test_partial_coverage_never_publishes_predictions_as_complete(tmp_path):
    live = extractor(tmp_path, lambda *a, **k: ([span()], {"text": {"complete": False}}))
    result = live("Python", "cv")
    assert result["status"] == "failed"
    assert result["entities"] == []


def test_corrupt_surface_is_failed_not_positive_evidence(tmp_path):
    broken = {**span(), "surface": "AWS"}
    live = extractor(tmp_path, lambda *a, **k: ([broken], {"text": {"complete": True}}))
    assert live("Python", "cv")["status"] == "failed"


def test_cache_is_side_specific_and_caller_cannot_mutate_it(tmp_path):
    calls = []
    def infer(document, model, **kwargs):
        calls.append(document["doc_type"])
        return [span()], {"text": {"complete": True}}
    live = extractor(tmp_path, infer)
    first = live("Python", "cv")
    first["entities"][0]["surface"] = "invented"
    assert live("Python", "cv")["entities"][0]["surface"] == "Python"
    live("Python", "jd")
    assert calls == ["cv", "jd"]
    live.clear()
    live("Python", "cv")
    assert calls == ["cv", "jd", "cv"]


def test_failure_messages_do_not_export_private_input(tmp_path):
    def infer(*a, **k):
        raise RuntimeError("private applicant Alice@example.com")
    live = extractor(tmp_path, infer)
    result = live("Python", "cv")
    assert result["status"] == "failed"
    assert "Alice" not in result["reason"]
