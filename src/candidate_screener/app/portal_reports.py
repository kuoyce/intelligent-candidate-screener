"""Deterministic local review reports and redacted text attachment packages."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import hashlib
import html
import io
import json
import zipfile

from candidate_screener.annotation.redact import redact_text
from .portal_store import _opaque_identity, _digest_identity


def _mapping(value):
    if is_dataclass(value) and hasattr(value, "candidates"):
        # PortalSnapshot stores a private frozen JSON payload; materialize its
        # public views rather than exporting that private implementation field.
        names = ("snapshot_id", "run_id", "jd_id", "jd_title", "digest", "model_id",
                 "candidates", "jd_entities", "jd_text", "source_flags")
        return {name: getattr(value, name) for name in names if hasattr(value, name)}
    return asdict(value) if is_dataclass(value) else dict(value)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _identity(value):
    """Preserve validated generated IDs; ordinary source prose stays redacted."""
    try:
        return _opaque_identity(value)
    except ValueError:
        return redact_text(value)


def _clean(value, field=None):
    if isinstance(value, dict):
        return {redact_text(key): _clean(item, key) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, str):
        if field in {"candidate_id", "cv_id", "snapshot_id", "source_digest", "source_sha256"}:
            return _identity(value)
        return redact_text(value)
    return value


def _rows(snapshot):
    snapshot = _mapping(snapshot)
    candidates = snapshot.get("candidates", snapshot.get("results", []))
    rows = [_mapping(row) for row in candidates]
    if len(rows) > 10:
        rows = rows[:10]
    if [row.get("rank") for row in rows] != list(range(1, len(rows)+1)):
        raise ValueError("Report requires unchanged contiguous backend ranking order")
    identifiers = [str(row.get("candidate_id", row.get("cv_id", ""))) for row in rows]
    if not all(identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError("Report requires unique stable candidate identities")
    return snapshot, rows, identifiers


def _version(value):
    if type(value) is not int or value not in {1, 2}:
        raise ValueError("Unsupported handoff schema version")
    return value


def _score_semantics(snapshot, candidate):
    """Carry backend-supplied prose without inferring a score's scale/direction."""
    value = candidate.get("score_semantics", (snapshot.get("source_flags") or {}).get("score_semantics"))
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ValueError("Invalid backend score semantics")
    return redact_text(value)


def build_report(snapshot, *, schema_version=2) -> dict:
    _version(schema_version)
    original, rows, identifiers = _rows(snapshot)
    identity = str(original.get("snapshot_id", ""))
    if not identity:
        raise ValueError("Report requires snapshot identity")
    snapshot_sha = _hash(_json(original).encode())
    comparison = []
    for row, identifier in zip(rows, identifiers):
        item = {"rank": row["rank"], "candidate_id": identifier,
                "score": row.get("score"), "profile": row.get("profile", row.get("entities", {})),
                "table_aggregates": row.get("table_aggregates", {}),
                "evidence_status": "Model table fields are unverified mentions unless exact source citations are supplied."}
        if schema_version == 2:
            item["score_semantics"] = _score_semantics(original, row)
        comparison.append(_clean(item))
    complete = len(rows) == 10
    score_description = "Ranks and scores are copied from the backend; scores are not Fit probabilities."
    flags = original.get("source_flags") or {}
    if schema_version == 2 and (flags.get("source_type") == "operational_live" or flags.get("operational") is True):
        score_description = "Ranks and scores are copied from the backend; score meaning is supplied by the backend."
    lines = ["Candidate review report", "Snapshot: " + _identity(identity),
             "JD: " + redact_text(original.get("jd_title", original.get("jd_id", "Unavailable"))),
             "Backend: " + redact_text(original.get("model_id", "Unspecified")),
             "Comparison: " + ("complete top 10" if complete else f"partial ({len(rows)} of 10)"),
             score_description,
             "Role family is not a verified job title.",
             "Model table fields without source offsets are unverified mentions, not source citations."]
    count = 8 if schema_version == 1 else 10
    shortlist = ", ".join(_identity(identifier) for identifier in identifiers[:count])
    lines.append(f"Proposed shortlist ranks 1–{count}: " + (shortlist or "Unavailable") +
                 (f" (incomplete: {len(rows)} of {count} candidates available)" if len(rows) < count else ""))
    if len(rows) < 10:
        lines.append("Department handoff remains blocked while the top-10 comparison is partial.")
    if original.get("jd_entities"):
        lines.append("JD model mentions: " + _json(_clean(original["jd_entities"])))
    for row in comparison:
        line = f"Rank {row['rank']} | {row['candidate_id']} | score {row['score']} | {_json(row['profile'])} | table aggregates {_json(row['table_aggregates'])}"
        if schema_version == 2:
            line += " | score meaning (backend): " + (row["score_semantics"] or "unspecified")
        lines.append(line)
    # All source string leaves above were redacted before rendering. Redacting
    # the serialized numeric backend values would mistake long decimals for PII.
    text = "\n".join(lines)
    rendered = "<!doctype html><meta charset='utf-8'><title>Candidate review report</title><pre>" + html.escape(text) + "</pre>"
    return {"schema_version": schema_version, "snapshot_id": identity, "snapshot_sha256": snapshot_sha, "complete": complete,
            "candidate_ids": identifiers, "top8_ids": identifiers[:8], "comparison": comparison,
            "shortlist_ids": identifiers[:count],
            "score_description": score_description,
            "text": text, "html": rendered, "report_sha256": _hash(text.encode()),
            "package_blockers": (["A complete top-10 comparison is required"] if not complete else [])}


def build_package(snapshot, report: dict, *, routing_digest=None) -> bytes:
    # Unversioned historical reports retain the original eight-CV contract.
    version = _version(report.get("schema_version", 1))
    count = 8 if version == 1 else 10
    if routing_digest is not None:
        _digest_identity(routing_digest)
        if version != 2:
            raise ValueError("Routing context requires handoff schema version 2")
    original, rows, identifiers = _rows(snapshot)
    if report.get("snapshot_sha256") != _hash(_json(original).encode()) or report.get("candidate_ids") != identifiers:
        raise ValueError("Report belongs to a different ranking snapshot")
    if len(rows) != 10 or not report.get("complete"):
        raise ValueError(f"Complete package requires ten ranked candidates and {count} source CVs")
    if report.get("report_sha256") != _hash(report["text"].encode()):
        raise ValueError("Report artifact digest mismatch")
    expected = build_report(snapshot, schema_version=version)
    if report.get("text") != expected["text"] or report.get("html") != expected["html"]:
        raise ValueError("Report differs from deterministic redacted snapshot rendering")
    artifacts = {"comparison-report.txt": report["text"].encode(),
                 "comparison-report.html": report["html"].encode()}
    if version == 2:
        artifacts["jd-context.json"] = _json(_clean({
            "jd_id": original.get("jd_id"), "jd_title": original.get("jd_title"),
            "jd_text": original.get("jd_text"), "jd_entities": original.get("jd_entities", {}),
            "source_flags": original.get("source_flags", {}),
        })).encode()
        artifacts["comparison.json"] = _json(expected["comparison"]).encode()
    attachments = []
    for row, identifier in zip(rows[:count], identifiers[:count]):
        text = row.get("cv_text", row.get("text", row.get("source_text")))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Required source CV unavailable at rank {row['rank']}; package is blocked")
        stem = f"rank-{row['rank']:02d}-{_hash(identifier.encode())[:12]}"
        artifacts[f"cvs/{stem}.txt"] = redact_text(text).encode()
        profile = row.get("profile", row.get("entities", {}))
        if version == 2:
            profile = {"rank": row["rank"], "candidate_id": identifier, "score": row.get("score"),
                       "score_semantics": _score_semantics(original, row),
                       "profile": profile, "entities": row.get("entities", {}),
                       "table_aggregates": row.get("table_aggregates", {})}
        artifacts[f"profiles/{stem}.json"] = _json(_clean(profile)).encode()
        attachments.append({"rank": row["rank"], "candidate_id": _identity(identifier), "path": f"cvs/{stem}.txt"})
        if version == 2:
            attachments[-1]["profile_path"] = f"profiles/{stem}.json"
        attachments[-1].update(_clean({"source_digest": row.get("source_digest"),
                                      "coverage": row.get("coverage", row.get("source_flags", [])),
                                      "preview_status": row.get("preview_status", "text derivative")}))
    manifest = {"schema_version": version, "snapshot_id": _identity(report["snapshot_id"]), "snapshot_sha256": report["snapshot_sha256"],
                "report_sha256": report["report_sha256"], "attachments": attachments,
                "format": "redacted text derivative; original PDFs unavailable; page/layout information not retained",
                "redaction": "annotation.redact; mechanically detectable identifiers masked; names may remain",
                "artifact_sha256": {name: _hash(content) for name, content in artifacts.items()}}
    if routing_digest is not None:
        # A changed route creates a new immutable package and preview identity,
        # even when the ranking and its CV bytes have not changed.
        manifest["routing_digest"] = routing_digest
    artifacts["manifest.json"] = _json(manifest).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(artifacts.items()):
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return output.getvalue()
