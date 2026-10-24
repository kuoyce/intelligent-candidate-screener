"""Authorized guided review operations; no LLM-authored actions or outbound email."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import zipfile

from .portal_reports import build_report, build_package
from .portal_store import RevisionConflict
from candidate_screener.annotation.redact import redact_text


class StaleReview(ValueError):
    pass


class PortalController:
    def __init__(self, dataset, store, principal, *, auth_guard=None, freshness_guard=None):
        self.dataset, self.store, self.principal = dataset, store, principal
        self.auth_guard, self.freshness_guard = auth_guard, freshness_guard
        self.qa_service = None
        self.department = "demo" if dataset.is_demo else "benchmark"
        self._guard()

    def _guard(self, *, fresh=False):
        principal = self.principal
        if (principal.auth_mode not in {"demo", "local_token"} or type(principal.expires_at) not in (int, float)
                or not math.isfinite(principal.expires_at) or principal.expires_at <= time.time()
                or self.department not in principal.departments):
            raise PermissionError("Access expired or the workspace is not authorized")
        if (principal.auth_mode == "demo") != self.dataset.is_demo:
            raise PermissionError("Demonstration identity cannot access company data")
        if not self.dataset.is_demo:
            if self.auth_guard is None or not self.auth_guard():
                raise PermissionError("Sign in again; access was revoked or changed")
        if fresh and self.freshness_guard is not None and not self.freshness_guard():
            raise StaleReview("The source files changed. Reload and review the new snapshot.")

    def jobs(self):
        self._guard(fresh=True)
        return self.dataset.jobs()

    def open_job(self, jd_id):
        self._guard(fresh=True)
        if jd_id not in {job["jd_id"] for job in self.dataset.jobs()}:
            raise PermissionError("Job is not available in this workspace")
        snapshot = self.dataset.snapshot(jd_id)
        return self.store.create_session(self.principal, self.department, jd_id, snapshot.digest,
                                         {"menu": "Review", "history": [], "snapshot_id": snapshot.snapshot_id})

    def sessions(self):
        self._guard()
        return self.store.list_sessions(self.principal)

    def load_session(self, session_id):
        self._guard()
        return self.store.load_session(self.principal, session_id)

    def snapshot(self, session_id):
        self._guard(fresh=True)
        record = self.load_session(session_id)
        if record["department"] != self.department:
            raise PermissionError("Session is outside the active workspace")
        if record["jd_id"] not in {job["jd_id"] for job in self.dataset.jobs()}:
            raise StaleReview("This job is no longer in the active export. View its historical report or choose another job.")
        try:
            snapshot = self.dataset.snapshot(record["jd_id"])
        except ValueError as exc:
            raise StaleReview("This JD revision has no current ranking. View its historical report or return to the job browser.") from exc
        if snapshot.digest != record["source_fingerprint"]:
            raise StaleReview("Saved session is historical. Refresh rankings before preparing or sending a package.")
        return snapshot

    def preview(self, session_id, candidate_id):
        snapshot = self.snapshot(session_id)
        hits = [c for c in snapshot.candidates if c["cv_id"] == candidate_id]
        if len(hits) != 1:
            raise PermissionError("Candidate is outside the selected top10")
        return hits[0]

    def remember(self, session_id, question, answer, *, menu="Review", extra=None):
        snapshot = self.snapshot(session_id)
        record = self.load_session(session_id)
        state = dict(record["state"])
        if extra and ("selected_candidates" in extra or "selected_category" in extra):
            selected = extra.get("selected_candidates", [])
            available = {item["cv_id"] for item in snapshot.candidates}
            if not isinstance(selected, list) or any(item not in available for item in selected):
                raise PermissionError("Candidate is outside the selected top10")
            category = extra.get("selected_category", "GENERAL")
            if category not in {"GENERAL", "KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "CERTIFICATION", "EXPERIENCE_YEARS"}:
                raise ValueError("Unknown requirement category")
            state["qa_dropdown_context"] = {
                "snapshot_id": snapshot.snapshot_id, "jd_id": snapshot.jd_id,
                "selected_candidates": list(dict.fromkeys(selected)), "category": category,
            }
            state["qa_dropdown_revision"] = record["revision"] + 1
        state["history"] = (state.get("history", []) + [{"question": question, "answer": answer}])[-20:]
        state["menu"] = menu
        state.update(extra or {})
        return self.store.save_session(self.principal, session_id, record["revision"], state)

    def chat_history(self, session_id):
        """Return only the authorized, current snapshot's saved conversation."""
        self.snapshot(session_id)
        return self.load_session(session_id)["state"].get("qa_turns", [])

    def compare(self, session_id, candidate_ids):
        """Compare only explicitly selected source owners, with current guards."""
        snapshot = self.snapshot(session_id)
        if not isinstance(candidate_ids, (list, tuple)) or not 2 <= len(candidate_ids) <= 10:
            raise ValueError("Select two to ten candidates for comparison")
        allowed = {row["cv_id"] for row in snapshot.candidates}
        if len(set(candidate_ids)) != len(candidate_ids) or any(item not in allowed for item in candidate_ids):
            raise PermissionError("Comparison candidates are outside this snapshot")
        if self.qa_service is None:
            raise ValueError("Extractive comparison service is not configured")
        result = self.qa_service.compare(snapshot, list(candidate_ids))
        if self.snapshot(session_id).digest != snapshot.digest:
            raise StaleReview("Sources changed during comparison")
        return result

    def extractive_evidence(self, session_id, question):
        """Read-only factual QA with access/source revalidation after inference."""
        snapshot = self.snapshot(session_id)
        if self.qa_service is None:
            raise ValueError("Extractive QA service is not configured")
        result = self.qa_service.answer(snapshot, question)
        if self.snapshot(session_id).digest != snapshot.digest:
            raise StaleReview("Sources changed during evidence lookup")
        return result

    def chat(self, session_id, question, turn_id, provider=None):
        """Generate once per UI submission and save using optimistic concurrency."""
        from .portal_qa import answer

        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Enter a question between 1 and 2000 characters")
        if not isinstance(turn_id, str) or not turn_id or len(turn_id) > 128:
            raise ValueError("Invalid chat submission identifier")
        question = redact_text(question.strip())
        # An alphabetic hash survives prose redaction while retaining opaque identity.
        identity = hashlib.sha256(turn_id.encode()).hexdigest().translate(
            str.maketrans("0123456789abcdef", "abcdefghijklmnop"))
        snapshot = self.snapshot(session_id)
        record = self.load_session(session_id)

        def prior(state):
            for turn in state.get("qa_turns", []):
                if turn["turn_id"] == identity:
                    if turn["question"] != question:
                        raise RevisionConflict("Submission identifier was reused for a different question")
                    return turn["answer"]
            return None

        existing = prior(record["state"])
        if existing is not None:
            return existing
        context = record["state"].get("qa_context")
        dropdown_revision = record["state"].get("qa_dropdown_revision")
        if dropdown_revision is not None and dropdown_revision != record["state"].get("qa_dropdown_consumed"):
            context = record["state"].get("qa_dropdown_context")
        if self.qa_service is not None:
            result = self.qa_service.answer(snapshot, question, context=context)
        else:
            # Compatibility for callers of the historical controller API. The
            # primary app attaches its extractive service before any review.
            result = answer(snapshot, question, context=context, provider=provider)
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ValueError("Invalid assistant response")
        # JSON round-trip detaches mutable provider data and rejects non-JSON values.
        encoded = json.dumps(result, allow_nan=False, ensure_ascii=False)
        if len(encoded.encode()) > 24_000:
            raise ValueError("Assistant response exceeds conversation storage limit")
        result = json.loads(encoded)
        current_snapshot = self.snapshot(session_id)  # Revocation/source checks after inference.
        if current_snapshot.digest != snapshot.digest:
            raise StaleReview("Rankings changed while answering. Refresh this review.")
        current = self.load_session(session_id)
        existing = prior(current["state"])
        if existing is not None:
            return existing
        if current["revision"] != record["revision"]:
            raise RevisionConflict("Review changed while answering. Please submit again.")
        state = dict(current["state"])
        turns = (state.get("qa_turns", []) + [{"turn_id": identity, "question": question, "answer": result}])[-12:]
        while len(turns) > 1 and len(json.dumps(turns, ensure_ascii=False).encode()) > 60_000:
            turns.pop(0)
        state.update(qa_turns=turns, qa_context=result.get("context"))
        if dropdown_revision is not None:
            state["qa_dropdown_consumed"] = dropdown_revision
        try:
            saved = self.store.save_session(self.principal, session_id, current["revision"], state)
        except RevisionConflict:
            self.snapshot(session_id)
            existing = prior(self.load_session(session_id)["state"])
            if existing is not None:
                return existing
            raise
        # Return the redacted persisted representation, including on the first turn.
        return saved["state"]["qa_turns"][-1]["answer"]

    def _report_path(self, session_id, digest):
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid report identity")
        # Session IDs are never used as unchecked paths.
        directory = self.store.path.parent / "reports" / hashlib.sha256(session_id.encode()).hexdigest()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        return directory / (digest + ".json")

    def report(self, session_id):
        snapshot = self.snapshot(session_id)
        report = build_report(snapshot)
        path = self._report_path(session_id, report["report_sha256"])
        encoded = json.dumps(report, sort_keys=True, ensure_ascii=False).encode()
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError("Immutable report content changed")
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=".report-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != encoded:
                    raise ValueError("Concurrent report conflict")
            finally:
                os.unlink(temporary)
        record = self.load_session(session_id)
        state = dict(record["state"], report_sha256=report["report_sha256"])
        self.store.save_session(self.principal, session_id, record["revision"], state)
        return report

    def historical_report(self, session_id):
        record = self.load_session(session_id)
        digest = record["state"].get("report_sha256")
        if not digest:
            return None
        report = json.loads(self._report_path(session_id, digest).read_text())
        if report.get("report_sha256") != digest or hashlib.sha256(report["text"].encode()).hexdigest() != digest:
            raise ValueError("Saved report failed its integrity check")
        return report

    def package(self, session_id):
        snapshot = self.snapshot(session_id)
        report = self.report(session_id)
        content = build_package(snapshot, report)
        return report, content

    def events(self):
        self._guard()
        return self.store.list_events(self.principal)

    def read_event(self, event_id):
        self._guard()
        return self.store.mark_event_read(self.principal, event_id)

    def prepare_handoff(self, session_id, recipient_id, directory=None):
        """Directory is trusted configuration, never a user/model email address."""
        from .portal_mail import department_directory, directory_digest

        snapshot = self.snapshot(session_id)
        record = self.load_session(session_id)
        if directory is None:
            directory = department_directory(snapshot.jd_id, record["department"])
        recipient = directory.get(snapshot.jd_id, {}).get(recipient_id)
        if not recipient or recipient.get("department") != record["department"]:
            raise PermissionError("No authorized department recipient is configured")
        report = build_report(snapshot)
        content = build_package(snapshot, report, routing_digest=directory_digest(recipient))
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        attachments = manifest["attachments"]
        package_manifest = {"schema_version": 2, "candidate_ids": report["candidate_ids"], "comparison_ids": report["candidate_ids"],
                            "attachment_digests": {a["candidate_id"]: manifest["artifact_sha256"][a["path"]] for a in attachments},
                            "report_digest": report["report_sha256"], "snapshot_id": snapshot.snapshot_id,
                            "directory_digest": directory_digest(recipient)}
        current = self.snapshot(session_id)
        if current.digest != snapshot.digest:
            raise StaleReview("Rankings changed while preparing the package; review again")
        return self.store.prepare_package(self.principal, session_id, record["revision"],
                    hashlib.sha256(content).hexdigest(), recipient_id, package_manifest)

    def confirm_handoff(self, session_id, package_id, request_id, *, preview_id=None, directory=None):
        if preview_id is not None:
            from .portal_mail import DemoMailer
            return DemoMailer(self, directory).capture(session_id, package_id, request_id, preview_id=preview_id)
        self.snapshot(session_id)
        package = self.store.load_package(self.principal, package_id)
        if package["session_id"] != session_id:
            raise PermissionError("Package belongs to a different review")
        # No gateway has been supplied; a local draft must never become a fake send.
        return self.store.confirm_queue(self.principal, package_id, request_id, gateway_configured=False)
