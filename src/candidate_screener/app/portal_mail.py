"""Local MIME procedure demonstration. This module never contacts a mail service.

Use ``DemoMailer(controller).preview(session_id, package_id)``, show its fields,
then pass its ``preview_id`` to ``capture`` after an explicit reviewer click.
``download(package_id)`` rechecks owner authorization but permits historical reads.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.message import EmailMessage
from email.policy import SMTP
from email.utils import format_datetime, formataddr
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

from candidate_screener.annotation.redact import redact_text
from .portal_reports import build_report, build_package
from .portal_store import RevisionConflict


_ADDRESSES = {"tech": ("Tech Department", "tech@example.invalid"),
              "business": ("Business Department", "business@example.invalid")}


def department_directory(jd_id, authorization_department="benchmark"):
    """Human routing choices are separate from the authorization workspace."""
    return {jd_id: {key: {"department": authorization_department, "label": label, "address": address}
                    for key, (label, address) in _ADDRESSES.items()}}


def _header(value, name, limit=300):
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError(f"Invalid email {name}; header controls are forbidden")
    return value


def directory_digest(recipient):
    return hashlib.sha256(json.dumps(recipient, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_atomic(path, content):
    """Publish one immutable private artifact; recover safely after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".capture-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError("Existing capture failed its integrity check")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)


class DemoMailer:
    def __init__(self, controller, directory=None):
        self.controller = controller
        self.directory = directory

    def _materialize(self, session_id, package_id):
        controller = self.controller
        snapshot = controller.snapshot(session_id)
        package = controller.store.load_package(controller.principal, package_id)
        if package["session_id"] != session_id:
            raise PermissionError("Package belongs to a different review")
        record = controller.load_session(session_id)
        if (package["session_revision"] != record["revision"]
                or package["source_fingerprint"] != record["source_fingerprint"]):
            raise RevisionConflict("Package is stale; prepare and review it again")
        manifest = package["manifest"]
        if manifest.get("schema_version", 1) != 2:
            raise ValueError("Local top-ten capture requires handoff schema version 2")
        directory = self.directory if self.directory is not None else department_directory(snapshot.jd_id, record["department"])
        recipient = directory.get(snapshot.jd_id, {}).get(package["recipient_id"])
        if not recipient or recipient.get("department") != record["department"]:
            raise PermissionError("No authorized department recipient is configured")
        label = _header(recipient.get("label"), "recipient label", 100)
        address = _header(recipient.get("address"), "recipient address", 100)
        if address not in {entry[1] for entry in _ADDRESSES.values()}:
            raise ValueError("Demo capture permits only reserved example.invalid department addresses")
        if manifest.get("directory_digest") != directory_digest(recipient):
            raise RevisionConflict("Recipient configuration changed; prepare and preview again")
        title = redact_text(_header(snapshot.jd_title, "job title", 200))
        report = build_report(snapshot)
        content = build_package(snapshot, report, routing_digest=directory_digest(recipient))
        digest = hashlib.sha256(content).hexdigest()
        if (digest != package["package_digest"] or manifest.get("snapshot_id") != snapshot.snapshot_id
                or manifest.get("candidate_ids") != report["candidate_ids"]
                or manifest.get("report_digest") != report["report_sha256"]):
            raise RevisionConflict("Package no longer matches the selected ranking snapshot")
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            packaged = json.loads(archive.read("manifest.json"))
            attachments = archive.namelist()
        actual = {entry["candidate_id"]: packaged["artifact_sha256"][entry["path"]]
                  for entry in packaged["attachments"]}
        if actual != manifest.get("attachment_digests"):
            raise ValueError("Package attachments failed their integrity check")
        subject = f"Hiring Desk top 10 — {title}"
        body = (f"Email demonstration — captured locally\nDepartment: {label}\nJob: {title}\n"
                "The attached package contains backend ranks 1–10, ten redacted CV text derivatives, "
                "ten profiles, JD context, comparison and digest manifest. These are not original PDFs.\n"
                + report["score_description"] + "\n\nCandidates:\n" +
                "\n".join(f"{row['rank']}. {row['candidate_id']} | backend score {row['score']} | "
                          f"score meaning (backend): {row['score_semantics'] or 'unspecified'}"
                          for row in report["comparison"]) +
                "\n\nThis message is captured in the local demo outbox. No email has been sent.\n")
        message = EmailMessage(policy=SMTP)
        message["From"] = "Hiring Desk Demo <hiring-desk@example.invalid>"
        message["To"] = formataddr((label, address))
        message["Subject"] = subject
        message["Date"] = format_datetime(datetime.fromtimestamp(package["created_at"], timezone.utc))
        message["Message-ID"] = f"<{package['package_id']}@hiring-desk.example.invalid>"
        message["X-Hiring-Desk-Transport"] = "local-demo-capture-v1"
        message.set_content(body)
        message.add_attachment(content, maintype="application", subtype="zip", filename="hiring-desk-top10.zip")
        message.set_boundary("hiring-desk-" + package["package_id"])
        encoded = message.as_bytes()
        preview_id = hashlib.sha256(encoded).hexdigest()
        preview = {"schema_version": 1, "preview_id": preview_id, "session_id": session_id,
                   "package_id": package_id, "recipient_id": package["recipient_id"],
                   "recipient": address, "label": label, "subject": subject, "body": body,
                   "candidate_ids": report["candidate_ids"], "attachments": attachments,
                   "score_description": report["score_description"], "comparison": report["comparison"],
                   "package_filename": "hiring-desk-top10.zip", "package_digest": digest,
                   "transport": "demo_capture", "delivered": False}
        return preview, encoded

    def preview(self, session_id, package_id):
        preview, _ = self._materialize(session_id, package_id)
        return preview

    def _path(self, package_id):
        # The caller must authorize the package before resolving this private path.
        return Path(self.controller.store.path).parent / "mail-captures" / (hashlib.sha256(package_id.encode()).hexdigest() + ".eml")

    def capture(self, session_id, package_id, request_id, *, preview_id):
        preview, content = self._materialize(session_id, package_id)
        if preview_id != preview["preview_id"]:
            raise RevisionConflict("Email preview changed; review the current recipient, package and message")
        controller = self.controller
        package = controller.store.begin_capture(controller.principal, package_id, request_id, preview_id)
        # A repeated click returns the original request and never rewrites its mail.
        if package["status"] == "captured":
            self.download(package_id)
            return package
        controller.snapshot(session_id)  # Recheck current source/access before the side effect.
        try:
            _write_atomic(self._path(package_id), content)
        except (OSError, ValueError) as exc:
            controller.snapshot(session_id)
            return controller.store.record_capture(controller.principal, package_id, package["request_id"], "failed",
                {"retryable": True, "error": "Local message could not be captured", "error_type": type(exc).__name__})
        controller.snapshot(session_id)
        return controller.store.record_capture(controller.principal, package_id, package["request_id"], "captured",
            {"message_sha256": preview_id, "retryable": False})

    def download(self, package_id):
        controller = self.controller
        controller._guard()
        package = controller.store.load_package(controller.principal, package_id)
        receipt = package["receipt"] or {}
        if package["status"] != "captured" or receipt.get("transport") != "demo_capture":
            raise ValueError("No captured message is available for this package")
        content = self._path(package_id).read_bytes()
        if hashlib.sha256(content).hexdigest() != receipt.get("message_sha256"):
            raise ValueError("Captured message failed its integrity check")
        controller._guard()
        return content
