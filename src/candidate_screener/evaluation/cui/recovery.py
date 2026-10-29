"""Offline failure-injection probes of the actual CUI and authorization boundary.

All sources and providers are invented. These probes measure guard behaviour,
not model accuracy, production latency or independent human acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

from candidate_screener.app.conversation_v3 import ConversationV3Service
from candidate_screener.app.portal_extractive import PortalExtractiveService


class _NoAnswerQA:
    def answer(self, question, passages):
        return {"status": "not_found", "spans": []}


class _InjectedComposer:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def compose_before(self, question, frame, facts, deadline):
        self.calls.append(deepcopy(facts))
        if self.error is not None:
            raise self.error
        return {"text": self.response, "model": "synthetic-injection", "provenance": {}}


def _snapshot():
    return SimpleNamespace(
        snapshot_id="recovery-invented", jd_id="RECOVERY-JD", jd_title="Invented analyst",
        jd_text="Bachelor degree required.",
        candidates=({"cv_id": "RECOVERY-CV-A", "rank": 1,
                     "preview_status": "available_redacted_text",
                     "preview_text": "Completed Bachelor degree. Python testing experience."},
                    {"cv_id": "RECOVERY-CV-B", "rank": 2,
                     "preview_status": "available_redacted_text",
                     "preview_text": "Master degree in progress."}),
    )


def _service(composer=None):
    return ConversationV3Service(PortalExtractiveService(qa_client=_NoAnswerQA()), composer)


def _case(identity, assertions, **details):
    return {"id": identity, "passed": all(assertions.values()), "assertions": assertions, **details}


def _fallback_probe(identity, composer, expected_reason=None):
    answer = _service(composer).answer(_snapshot(), "What degree does candidate 1 have?")
    reason = answer.get("generation_provenance", {}).get("fallback_reason")
    checks = {
        "fallback_reported": answer.get("generation_mode") == "qwen_fallback",
        "not_claimed_as_model_answer": answer.get("model") is None,
        "usable_nonempty_reply": bool(answer.get("text", "").strip()),
        "supporting_sources_retained": bool(answer.get("citations")),
        "no_accepted_generated_claims": answer.get("validated_claim_sources") == [],
    }
    if expected_reason is not None:
        checks["truthful_failure_reason"] = reason == expected_reason
    if composer is not None and composer.response is not None:
        checks["malformed_text_not_displayed"] = composer.response != answer.get("text")
        checks["malformed_payload_not_embedded"] = "fabricated_doctorate" not in answer.get("text", "")
    return _case(identity, checks, fallback_reason=reason)


def _missing_evidence():
    # Inject an unsupported positive claim; the application must not display it.
    composer = _InjectedComposer("Candidate 1 holds an active Kubernetes certification.")
    answer = _service(composer).answer(_snapshot(), "Does candidate 1 have Kubernetes certification?")
    text = answer.get("text", "").casefold()
    return _case("missing_evidence", {
        "unsupported_credential_not_displayed": composer.response != answer.get("text"),
        "not_accepted_as_generated_fact": answer.get("generation_mode") != "qwen_prose",
        "uncertainty_is_explicit": any(marker in text for marker in
                                       ("not found", "not established", "cannot", "can't", "enough", "unavailable")),
        "absence_not_asserted": "has no kubernetes certification" not in text,
    }, observed_status=answer.get("status"))


def _changed_source_history():
    current = _snapshot()
    composer = _InjectedComposer("Candidate 1 has completed a Bachelor degree.")
    engine = _service(composer)
    first = engine.answer(current, "What degree does candidate 1 have?")
    current.candidates[0]["preview_text"] = "Bachelor degree in progress. Python testing experience."
    answer = engine.answer(current, "What degree does candidate 1 have?",
                           history=[{"question": "What degree does candidate 1 have?", "answer": first}])
    return _case("changed_source_history", {
        "initial_answer_was_accepted": first.get("generation_mode") == "qwen_prose",
        "stale_history_removed": bool(composer.calls) and composer.calls[-1].get("history") == [],
        "stale_completion_claim_rejected": answer.get("generation_mode") != "qwen_prose",
        "stale_completion_not_displayed": "has completed a Bachelor degree" not in answer.get("text", ""),
    }, boundary="source digest and history reauthorization; distinct from account revocation")


def _controller_probe(kind):
    from candidate_screener.app.portal_controller import PortalController, StaleReview
    from candidate_screener.app.portal_data import synthetic_demo
    from candidate_screener.app.portal_store import PortalStore, Principal

    with TemporaryDirectory(prefix="cui-recovery-") as directory:
        access = {"allowed": True, "fresh": True}
        controller = PortalController(
            replace(synthetic_demo(), is_demo=False),
            PortalStore(Path(directory) / "workspace.sqlite"),
            Principal("synthetic-reviewer", ("benchmark",), time.time() + 3600, "local_token"),
            auth_guard=lambda: access["allowed"], freshness_guard=lambda: access["fresh"],
        )
        engine = _service()

        class DuringAnswer:
            accepts_history = True

            def answer(self, *args, **kwargs):
                result = engine.answer(*args, **kwargs)
                access["allowed" if kind == "revoked_access" else "fresh"] = False
                return result

        controller.qa_service = DuringAnswer()
        session_id = controller.open_job(controller.jobs()[0]["jd_id"])["session_id"]
        expected = PermissionError if kind == "revoked_access" else StaleReview
        rejected = False
        try:
            controller.chat(session_id, "What degree does candidate 1 have?", "synthetic-turn")
        except expected:
            rejected = True
        persisted = controller.store.load_session(controller.principal, session_id)
        return _case(kind, {
            "post_inference_boundary_rejected": rejected,
            "no_answer_persisted": not persisted["state"].get("qa_turns"),
        }, boundary="PortalController.chat post-inference authorization/freshness check")


def run_recovery():
    """Return assertions for five failure classes, including three source checks.

    Timeout injection raises immediately; it verifies recovery/provenance rather
    than real process cancellation or elapsed deadline enforcement. Those remain
    covered by the runtime's own worker/deadline tests.
    """
    probes = [
        ("unavailable_generator", lambda: _fallback_probe("unavailable_generator", None, "provider_unavailable")),
        ("generation_timeout", lambda: _fallback_probe("generation_timeout", _InjectedComposer(error=TimeoutError("injected")), "deadline_exceeded")),
        ("malformed_output", lambda: _fallback_probe("malformed_output", _InjectedComposer('{"fabricated_doctorate": true}'))),
        ("missing_evidence", _missing_evidence),
        ("changed_source_history", _changed_source_history),
        ("changed_source_controller", lambda: _controller_probe("changed_source_controller")),
        ("revoked_access", lambda: _controller_probe("revoked_access")),
    ]
    cases = []
    for identity, probe in probes:
        try:
            cases.append(probe())
        except Exception as error:
            # Retain a failed case, without logging arbitrary exception/source text.
            cases.append(_case(identity, {"probe_completed": False}, error_type=type(error).__name__))
    return {"kind": "injected_recovery", "synthetic_only": True,
            "passed": all(case["passed"] for case in cases), "cases": cases,
            "limitations": "Offline synthetic providers. No model accuracy, real timeout latency, worker cancellation or production reliability estimate."}
