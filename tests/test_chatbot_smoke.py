"""Operational smoke reports count source-backed quotes, not model accuracy."""
import pytest

from candidate_screener.app.chat_smoke import trial


def test_synthetic_smoke_cannot_contact_a_model_even_if_environment_is_configured():
    report = trial(values={"SCREENER_CE_MANIFEST": "/must-not-load",
                          "SCREENER_QWEN_BASE_URL": "http://must-not-contact"})
    assert report["mode"] == "synthetic templates"
    assert report["provider"] is None
    assert report["question_count"] == 4
    assert report["fallback_count"] == 4
    assert report["supported_quote_count"] == report["audited_quote_count"] > 0
    assert report["service_peak_rss_bytes"] is None


def test_live_smoke_requires_explicit_configuration():
    with pytest.raises(ValueError, match="explicit Qwen"):
        trial(live=True, values={})
