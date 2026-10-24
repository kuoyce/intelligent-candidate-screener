"""Validated local span editing with immutable sources and revisioned reviews."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from .campaign import load_manifest
from .text import map_selection, python_to_utf16

STATUSES = frozenset({"draft", "reviewed", "skipped", "uncertain"})


class RevisionConflict(ValueError):
    status_code = 409


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _annotator(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value):
        raise ValueError("Use an annotator ID of 1–40 letters, digits, underscores or hyphens")
    return value


class AnnotationService:
    def __init__(self, campaign_path: Path):
        self.path = Path(campaign_path)
        self._campaign = load_manifest(self.path)
        self.campaign_id = self._campaign["campaign_id"]
        self.journal_path = self.path.parent / "reviews.jsonl"
        self.lock_path = self.path.parent / "reviews.lock"

    @contextmanager
    def _locked(self):
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def campaign(self):
        campaign = load_manifest(self.path)
        if campaign["campaign_id"] != self.campaign_id:
            raise ValueError("Campaign changed while the service was running")
        return campaign

    def records(self):
        if not self.journal_path.exists():
            return []
        records, previous, revisions = [], None, {}
        campaign = self.campaign()
        documents = self._documents(campaign)
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            digest = record.pop("record_sha256", None)
            if digest != _digest(record) or record.get("previous_sha256") != previous:
                raise ValueError("Review journal integrity failure")
            if record.get("campaign_id") != self.campaign_id:
                raise ValueError("Review journal campaign mismatch")
            doc = documents.get(record.get("doc_id"))
            if doc is None or record.get("source_sha256") != doc["source_sha256"] or record.get("guideline_version") != campaign["guideline_version"]:
                raise ValueError("Review journal source/guideline mismatch")
            self._check_annotator(campaign, record.get("annotator"))
            if doc["doc_id"] not in self._assigned(campaign, record["annotator"]):
                raise ValueError("Review is outside annotator assignment")
            key = (doc["doc_id"], record["annotator"])
            revision = record.get("revision")
            if isinstance(revision, bool) or revision != revisions.get(key, 0) + 1:
                raise ValueError("Review journal revision sequence mismatch")
            revisions[key] = revision
            if record.get("status") not in STATUSES or not isinstance(record.get("reviewed_all"), bool):
                raise ValueError("Review journal state mismatch")
            if record["status"] == "reviewed" and not record["reviewed_all"]:
                raise ValueError("Review journal has unconfirmed gold")
            assistance = record.get("assistance")
            self._validate_assistance_context(campaign, doc, record["annotator"], assistance)
            if record.get("annotation_mode", "blind") != ("model_assisted" if assistance else "blind"):
                raise ValueError("Review journal annotation-mode mismatch")
            spans = record.get("spans")
            if not isinstance(spans, list):
                raise ValueError("Review journal spans are invalid")
            end = -1
            for span in spans:
                a, b = span.get("start"), span.get("end")
                if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int)
                        or not isinstance(b, int) or not 0 <= a < b <= len(doc["source_text"])
                        or a < end or span.get("label") not in campaign["labels"]
                        or doc["source_text"][a:b] != span.get("surface")):
                    raise ValueError("Review journal entity is invalid")
                da, db = span.get("display_start"), span.get("display_end")
                mapped = map_selection(doc, python_to_utf16(doc["display_text"], da),
                                        python_to_utf16(doc["display_text"], db))
                if mapped != (a, b):
                    raise ValueError("Review journal display/source mismatch")
                end = b
            previous = digest
            records.append({**record, "record_sha256": digest})
        return records

    def _latest(self, annotator):
        return {r["doc_id"]: r for r in self.records() if r["annotator"] == annotator}

    @staticmethod
    def _documents(campaign):
        return {doc["doc_id"]: doc for doc in campaign["documents"]}

    @staticmethod
    def _check_annotator(campaign, annotator):
        _annotator(annotator)
        if "annotators" in campaign and annotator not in campaign["annotators"]:
            raise ValueError("Annotator is not registered for this campaign")

    @staticmethod
    def _assigned(campaign, annotator):
        return set(campaign.get("assignments", {}).get(annotator,
                   [doc["doc_id"] for doc in campaign["documents"]]))

    @staticmethod
    def _assistance_allowed(campaign, doc, annotator):
        primary = campaign.get("annotators", ["annotator1"])[0]
        return doc["split"] == "train" and annotator == primary

    @classmethod
    def _validate_assistance_context(cls, campaign, doc, annotator, assistance):
        if assistance is None:
            return
        if (not cls._assistance_allowed(campaign, doc, annotator)
                or not isinstance(assistance, dict)
                or set(assistance) != {"suggestion_set_id", "model_id"}
                or any(not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", v)
                       for v in assistance.values())):
            raise ValueError("Model assistance is restricted to primary-annotator training documents")

    def suggestions(self, doc_id, annotator):
        _annotator(annotator)
        if not isinstance(doc_id, str):
            raise ValueError("Document ID must be a string")
        with self._locked():
            campaign = self.campaign()
            self._check_annotator(campaign, annotator)
            doc = self._documents(campaign).get(doc_id)
            if doc is None or doc_id not in self._assigned(campaign, annotator):
                raise ValueError("Document is not assigned to this annotator")
            if not self._assistance_allowed(campaign, doc, annotator):
                raise ValueError("Suggestions are hidden for independent annotators and dev/test")
            if self._latest(annotator).get(doc_id, {}).get("revision", 0):
                raise ValueError("Saved drafts and reviews cannot be replaced by model suggestions")
            from .suggestions import load_for_document
            return load_for_document(campaign, doc, self.path.parent / "suggestions")

    def boundary_review(self, annotator):
        with self._locked():
            campaign = self.campaign()
            self._check_annotator(campaign, annotator)
            if annotator != campaign.get("annotators", ["annotator1"])[0]:
                raise ValueError("Training boundary review is hidden from independent annotators")
            from candidate_screener.config import PROCESSED
            queue = (PROCESSED / "entity-annotation/finetune-snapshots" /
                     f"{self.campaign_id}-ten-reviewed-v1" / "boundary-review.json")
            if not queue.is_file():
                raise ValueError("No training boundary-review report is available for this campaign")
            from .boundary_review import render_review
            return render_review(campaign, self.records(), queue, annotator)

    def session(self, annotator):
        _annotator(annotator)
        with self._locked():
            campaign = self.campaign()
            self._check_annotator(campaign, annotator)
            latest = self._latest(annotator)
            ordered = sorted((doc for doc in campaign["documents"]
                              if doc["doc_id"] in self._assigned(campaign, annotator)), key=lambda d: (
                {"train": 0, "dev": 1, "test": 2}[d["split"]], d["doc_id"]))
            pilot = [doc for side in ("cv", "jd")
                     for doc in [d for d in ordered if d["split"] == "train" and d["side"] == side][:4]]
            pilot_ids = {doc["doc_id"] for doc in pilot}
            ordered = pilot + [doc for doc in ordered if doc["doc_id"] not in pilot_ids]
            documents = [{"doc_id": doc["doc_id"], "side": doc["side"], "split": doc["split"],
                          "status": latest.get(doc["doc_id"], {}).get("status", "untouched"),
                          "revision": latest.get(doc["doc_id"], {}).get("revision", 0)}
                         for doc in ordered]
        progress = {state: sum(d["status"] == state for d in documents)
                    for state in ("untouched", "draft", "reviewed", "skipped", "uncertain")}
        progress["total"] = len(documents)
        return {"campaign": {"campaign_id": campaign["campaign_id"], "labels": campaign["labels"],
                             "guideline_version": campaign["guideline_version"],
                             "primary_annotator": campaign.get("annotators", ["annotator1"])[0],
                             "is_demo": all(doc["source"] == "synthetic-entity-demo" for doc in campaign["documents"])},
                "documents": documents, "progress": progress}

    def document(self, doc_id, annotator):
        _annotator(annotator)
        if not isinstance(doc_id, str):
            raise ValueError("Document ID must be a string")
        with self._locked():
            campaign = self.campaign()
            self._check_annotator(campaign, annotator)
            document = self._documents(campaign).get(doc_id)
            if document is None:
                raise ValueError("Unknown document")
            if doc_id not in self._assigned(campaign, annotator):
                raise ValueError("Document is not assigned to this annotator")
            review = self._latest(annotator).get(doc_id, {})
        text = document["display_text"]
        spans = [{"start": python_to_utf16(text, span["display_start"]),
                  "end": python_to_utf16(text, span["display_end"]), "label": span["label"]}
                 for span in review.get("spans", [])]
        return {"doc_id": doc_id, "side": document["side"], "split": document["split"],
                "text": text, "spans": spans, "revision": review.get("revision", 0),
                "status": review.get("status", "untouched"),
                "reviewed_all": review.get("reviewed_all", False), "offset_unit": "utf16",
                "assistance": review.get("assistance"),
                "can_load_suggestions": self._assistance_allowed(campaign, document, annotator)
                    and not review.get("revision", 0)
                    and (self.path.parent / "suggestions/active.json").is_file()}

    def save(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        annotator = _annotator(payload.get("annotator"))
        if payload.get("offset_unit") != "utf16":
            raise ValueError("Specify browser offsets as offset_unit=utf16")
        revision = payload.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("Revision must be a nonnegative integer")
        status = payload.get("status")
        if status not in STATUSES:
            raise ValueError("Unknown review state")
        reviewed_all = payload.get("reviewed_all", False)
        if not isinstance(reviewed_all, bool):
            raise ValueError("Whole-document review confirmation must be boolean")
        if status == "reviewed" and not reviewed_all:
            raise ValueError("Confirm whole-document review before marking reviewed")
        spans = payload.get("spans")
        if not isinstance(spans, list) or len(spans) > 10000:
            raise ValueError("Expected a bounded list of spans")
        with self._locked():
            campaign = self.campaign()
            self._check_annotator(campaign, annotator)
            doc_id = payload.get("doc_id")
            if not isinstance(doc_id, str):
                raise ValueError("Document ID must be a string")
            doc = self._documents(campaign).get(doc_id)
            if doc is None:
                raise ValueError("Unknown document")
            if doc_id not in self._assigned(campaign, annotator):
                raise ValueError("Document is not assigned to this annotator")
            records = self.records()
            previous = next((r for r in reversed(records)
                             if r["doc_id"] == doc_id and r["annotator"] == annotator), {})
            if previous.get("revision", 0) != revision:
                raise RevisionConflict("A newer review was saved; reload before editing")
            assistance = payload.get("assistance")
            if previous.get("assistance"):
                if assistance is not None and assistance != previous["assistance"]:
                    raise ValueError("Assistance provenance cannot be changed or removed")
                assistance = previous["assistance"]
            self._validate_assistance_context(campaign, doc, annotator, assistance)
            if assistance and not previous.get("assistance"):
                if previous.get("revision", 0):
                    raise ValueError("Existing drafts/reviews cannot be replaced by suggestions")
                from .suggestions import load_for_document
                suggestion = load_for_document(campaign, doc, self.path.parent / "suggestions")
                if not suggestion.get("available") or any(suggestion.get(k) != v for k, v in assistance.items()):
                    raise ValueError("Suggestion identity is unavailable or does not match this source")
            accepted = []
            for span in spans:
                if not isinstance(span, dict) or span.get("label") not in campaign["labels"]:
                    raise ValueError("Unknown entity category")
                start, end = map_selection(doc, span.get("start"), span.get("end"))
                from .text import utf16_to_python
                a = utf16_to_python(doc["display_text"], span["start"])
                b = utf16_to_python(doc["display_text"], span["end"])
                if not doc["source_text"][start:end].strip():
                    raise ValueError("Whitespace is not an entity")
                accepted.append({"start": start, "end": end, "label": span["label"],
                                 "surface": doc["source_text"][start:end],
                                 "display_start": a, "display_end": b})
            accepted.sort(key=lambda span: (span["start"], span["end"], span["label"]))
            if any(right["start"] < left["end"] for left, right in zip(accepted, accepted[1:])):
                raise ValueError("Overlapping entity spans must be resolved to one category")
            record = {"campaign_id": self.campaign_id, "guideline_version": campaign["guideline_version"],
                      "doc_id": doc_id, "source_sha256": doc["source_sha256"], "annotator": annotator,
                      "revision": revision + 1, "status": status, "reviewed_all": reviewed_all,
                      "spans": accepted, "saved_at": datetime.now(timezone.utc).isoformat(),
                      "assistance": assistance, "annotation_mode": "model_assisted" if assistance else "blind",
                      "previous_sha256": records[-1]["record_sha256"] if records else None}
            record["record_sha256"] = _digest(record)
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return {"doc_id": doc_id, "revision": record["revision"], "status": status,
                "reviewed_all": reviewed_all}
