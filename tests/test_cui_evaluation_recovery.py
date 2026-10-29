"""Injected failures exercise real CUI guards without model assets or data."""
from candidate_screener.evaluation.cui import recovery


def test_recovery_covers_five_failure_classes_and_source_boundaries():
    result = recovery.run_recovery()
    assert result["kind"] == "injected_recovery" and result["synthetic_only"]
    assert result["passed"], result["cases"]
    assert {case["id"] for case in result["cases"]} == {
        "unavailable_generator", "generation_timeout", "malformed_output", "missing_evidence",
        "changed_source_history", "changed_source_controller", "revoked_access",
    }
    assert all(case["assertions"] and all(case["assertions"].values()) for case in result["cases"])


def test_timeout_probe_is_immediate_and_preserves_deadline_reason(monkeypatch):
    def no_sleep(*args):
        raise AssertionError("Recovery probes must not wait for a real timeout")
    monkeypatch.setattr(recovery.time, "sleep", no_sleep)
    result = recovery._fallback_probe("timeout", recovery._InjectedComposer(error=TimeoutError()), "deadline_exceeded")
    assert result["passed"] and result["fallback_reason"] == "deadline_exceeded"


def test_probe_exceptions_are_counted_as_failures_not_dropped(monkeypatch):
    def broken():
        raise ValueError("sensitive diagnostic text must not be copied")
    monkeypatch.setattr(recovery, "_missing_evidence", broken)
    result = recovery.run_recovery()
    assert not result["passed"]
    case = next(row for row in result["cases"] if row["id"] == "missing_evidence")
    assert case == {"id": "missing_evidence", "passed": False,
                    "assertions": {"probe_completed": False}, "error_type": "ValueError"}


def test_recovery_detects_removed_post_inference_authorization(monkeypatch):
    from candidate_screener.app.portal_controller import PortalController
    original = PortalController._guard

    def guard_once(self, **kwargs):
        # Mutation: suppress authorization after the initial controller checks.
        if self.auth_guard and not self.auth_guard():
            return None
        return original(self, **kwargs)

    monkeypatch.setattr(PortalController, "_guard", guard_once)
    result = recovery._controller_probe("revoked_access")
    assert not result["passed"]
    assert result["assertions"]["post_inference_boundary_rejected"] is False
    assert result["assertions"]["no_answer_persisted"] is False
