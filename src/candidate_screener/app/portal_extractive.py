"""Source-owned extractive QA and deterministic Hiring Desk comparisons.

The controller authorizes snapshots before calling this service. No applicant
data is cached here. All coordinates refer to the supplied redacted preview,
not to an unavailable original document. Broad comparisons never invoke QA;
``prepare`` only schedules the entity adapter's nonblocking work.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import re

from candidate_screener.annotation.redact import redact_text
from candidate_screener.llm.qa_assets import MODEL_ID
from . import portal_qa as guidance
from .portal_store import AuthorizationError


CATEGORIES = ("KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS", "CERTIFICATION")
_PATTERNS = {
    "KNOWLEDGE": r"\b(?:knowledge|domain|expertise|theory|software|database|framework)\w*\b",
    "SKILL": r"\b(?:(?:skill|technolog|programming|testing|competenc|proficien)\w*|jira|test\s+cases?|bug\s+reports?|qa\s+manual)\b",
    "TITLE": r"\b(?:title|position|role|engineer|developer|manager|architect|analyst|scientist|consultant|director|specialist)\w*\b",
    "DEGREE": r"\b(?:degree|bachelor|master|phd|ph\.d|diploma|[bm]\.?sc|doctorate|education|qualification|mba)\b",
    "EXPERIENCE_YEARS": r"\b(?:years?|months?|tenure|experience)\b",
    "CERTIFICATION": r"\b(?:certificat\w*|certified|credential\w*)\b",
}
_CATEGORY_WORDS = set(" ".join(words for _, _, words in guidance._CATEGORIES).split())
_QUESTION_WORDS = {"state", "states", "stated", "say", "says", "hold", "holds", "language", "languages", "possess"}
_MISSING = "Relevant evidence not found; this does not establish absence."
_UNAVAILABLE = "A redacted source is unavailable."
_LIMITED = "Evidence omitted to keep this result within its storage budget."
MAX_BYTES = 23_900
MAX_CONTEXT_CHARS = 1200
_DURATION = (r"(?<!\w)(?:\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?"
             r"|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
             r"\s*\+?\s*(?:years?|months?|yrs?)\b")


@lru_cache(maxsize=1)
def shared_qa_client():
    """Share only the persistent offline model worker, never evidence or answers."""
    from candidate_screener.llm.extractive_qa import QaClient
    return QaClient()


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 128
            or len(json.dumps(value)) > 130):
        raise ValueError("Snapshot and document identities must be bounded strings")
    return value


def _selection(snapshot, candidate_ids):
    _identity(snapshot.snapshot_id)
    _identity(snapshot.jd_id)
    available = {item["cv_id"] for item in snapshot.candidates}
    if (not isinstance(candidate_ids, (list, tuple)) or len(candidate_ids) > 10
            or any(not isinstance(item, str) for item in candidate_ids)):
        raise ValueError("Select at most ten candidate IDs from the active snapshot")
    selected = list(dict.fromkeys(candidate_ids))
    if any(item not in available for item in selected):
        raise PermissionError("Candidate is outside the active snapshot")
    for item in selected:
        _identity(item)
    return selected


def _documents(snapshot, selected):
    rows = {row["cv_id"]: row for row in snapshot.candidates}
    sources = [(snapshot.jd_id, "jd", snapshot.jd_text)]
    sources += [(owner, "cv", rows[owner].get("preview_text")
                 if rows[owner].get("preview_status") == "available_redacted_text" else None)
                for owner in selected]
    documents = {}
    for owner, kind, source in sources:
        safe = (isinstance(source, str) and bool(source.strip()) and len(source) <= 1_000_000
                and redact_text(source) == source)
        document = {"owner": owner, "kind": kind, "source": source if safe else "",
                    "digest": _digest(source) if safe else None, "units": [],
                    "status": "complete" if safe else "unavailable"}
        if safe:
            for start, end, quote in guidance._sentences(source):
                # Keep whole qualifier-bearing sentences, never a clipped assertion.
                if len(document["units"]) >= 4096:
                    document["status"] = "partial"
                    break
                if len(quote) > MAX_CONTEXT_CHARS:
                    document["status"] = "partial"
                    continue
                document["units"].append((start, end, quote))
        documents[owner] = document
    return documents


def _citation(snapshot, document, unit):
    start, end, quote = unit
    identity = f"{snapshot.snapshot_id}:{document['kind']}:{document['owner']}:{document['digest']}:{start}:{end}"
    return {"evidence_id": "qa:" + _digest(identity)[:24], "owner": document["owner"],
            "owner_kind": document["kind"], "snapshot_id": snapshot.snapshot_id,
            "document_digest": document["digest"], "quote": quote, "start": start, "end": end,
            "coordinate_space": "redacted_preview"}


def _cell(status, text=None, *, evidence_ids=(), context="", qa_status="not_run"):
    return {"text": text or (_UNAVAILABLE if status == "source_unavailable" else _MISSING),
            "status": status, "evidence_ids": list(evidence_ids), "context": context,
            "qa_status": qa_status}


def _aggregate(values, default="not_run"):
    values = set(values)
    return next(iter(values)) if len(values) == 1 else "partial" if values else default


def _focus(question, category):
    words = guidance._terms(question, category) - guidance._BROAD_WORDS - _CATEGORY_WORDS - _QUESTION_WORDS
    words -= {"require", "requires", "required", "requirement", "qualifications"}
    return {word for word in words if not any(re.fullmatch(pattern, word, re.I)
                                               for pattern in _PATTERNS.values())}


def _retrieval_question(question, kind):
    if kind == "jd":
        # Source framing describes where to search, not a literal requirement.
        return re.sub(r"\b(?:(?:this|the|current|active)\s+)?(?:job description|jd|role|job)\b",
                      " ", question, flags=re.I)
    return question


def _duration_focus(question):
    return _focus(question, "EXPERIENCE_YEARS") - {
        "total", "combined", "cumulative", "professional", "commercial", "work", "working",
        "duration", "career", "industry",
    }


def _duration_links(quote, focus):
    """Recognize explicit skill-duration wording, never title/number proximity.

    This deliberately abstains on ambiguous constructions. Returned coordinates
    locate only durations tied to the requested skill, so a model cannot select
    a different total from the same otherwise-relevant passage.
    """
    if not focus:
        return set()
    skill = r"(?<!\w)(?:" + "|".join(re.escape(word) for word in sorted(focus)) + r")(?!\w)"
    duration = rf"(?P<duration>{_DURATION})"
    patterns = (
        rf"{duration}\s*(?:of\s+|['’]\s*)?{skill}(?:\s+(?:experience|development|programming|work))?",
        rf"{duration}\s+(?:of\s+)?(?:(?:professional|commercial|hands-on)\s+)?"
        rf"(?:experience\s+)?(?:using|with|in|developing\s+(?:in|with)|programming\s+(?:in|with))\s+{skill}",
        rf"{skill}(?:\s+(?:experience|development|programming|work))?\s+(?:for|over|of)\s+{duration}",
        rf"{skill}(?:\s+(?:experience|development|programming))?\s*(?::|\()\s*{duration}",
    )
    linked = set()
    for pattern in patterns:
        for match in re.finditer(pattern, quote, re.I):
            start, end = match.span("duration")
            total_after = re.match(r"\s*(?:of\s+)?(?:total|overall|combined|cumulative)\b", quote[end:], re.I)
            total_before = re.search(
                r"\b(?:total|overall|combined|cumulative)(?:\s+(?:work|professional|career))?"
                r"(?:\s+experience)?\s*(?:of|:)?\s*$", quote[:start], re.I)
            if not total_before and not total_after:
                linked.add((start, end))
    return linked


def _supported_duration_span(quote, start, end, focus):
    linked = _duration_links(quote, focus)
    if not any(start <= a < end for a, _ in linked):
        return False
    return all(match.span() in linked for match in re.finditer(_DURATION, quote, re.I)
               if start <= match.start() < end)


def _qa_question(snapshot, original, topic, category, kind, multiple):
    """Keep lexical retrieval normalization out of the model's question."""
    natural = original
    for pattern in guidance._REFERENCE_PATTERNS:
        natural = pattern.sub("the candidate", natural)
    for row in snapshot.candidates:
        natural = re.sub(r"(?<![\w-])" + re.escape(row["cv_id"]) + r"(?![\w-])",
                         "the candidate", natural, flags=re.I)
    if not multiple and len(natural.split()) >= 5 and not re.search(guidance._COMPARISON, natural, re.I):
        return natural
    source = "job description" if kind == "jd" else "CV"
    focus = " ".join(sorted(_focus(topic, category)))[:200]
    fact = {"DEGREE": "degrees or education", "CERTIFICATION": "certifications or credentials",
            "TITLE": "job titles or positions", "KNOWLEDGE": "knowledge or domain expertise",
            "SKILL": "skills or technologies", "EXPERIENCE_YEARS": "years or months of experience"}.get(category, "")
    detail = " ".join(part for part in (focus, fact) if part) or "the stated requirements"
    return f"What does the {source} explicitly state about {detail}?"


def _visible(cell):
    text, context = cell["text"], cell["context"]
    # Preserve status wording regardless of how the source expresses it.
    if context and context not in text:
        text += "\nSource context: " + context
    return text


class PortalExtractiveService:
    def __init__(self, entity_service=None, qa_client=None):
        self.entity_service = entity_service
        self.qa_client = qa_client

    def _entities(self, snapshot, selected, documents):
        bundle = {}
        if self.entity_service is not None:
            try:
                bundle = self.entity_service.prepare(snapshot, list(selected))
            except (AuthorizationError, PermissionError):
                raise
            except Exception:
                bundle = {"status": "unavailable"}
        if not isinstance(bundle, Mapping):
            bundle = {}
        statuses = {"queued", "running", "complete", "partial", "failed", "unavailable"}
        status = bundle.get("status", "unavailable")
        status = status if isinstance(status, str) and status in statuses else "unavailable"
        raw_documents = bundle.get("documents", {})
        raw_documents = raw_documents if isinstance(raw_documents, Mapping) else {}
        valid = {owner: [] for owner in documents}
        for owner, document in documents.items():
            row = raw_documents.get(owner, {})
            if not isinstance(row, Mapping) or not row:
                continue
            if (not isinstance(row.get("status"), str) or row.get("status") not in {"complete", "partial"}
                    or document["status"] == "unavailable"):
                continue
            if row.get("status") == "partial" and status != "invalid_evidence":
                status = "partial"
            if (row.get("source_digest") != document["digest"]
                    or any(row.get(key, expected) != expected for key, expected in
                           (("owner_id", owner), ("owner", owner), ("doc_type", document["kind"]),
                            ("coordinate_space", "redacted_preview")))):
                status = "invalid_evidence"
                continue
            entities = row.get("entities", [])
            if not isinstance(entities, (list, tuple)):
                status = "invalid_evidence"
                continue
            if len(entities) > 512:
                status = "partial"
            for entity in entities[:512]:
                if not isinstance(entity, Mapping):
                    status = "invalid_evidence"
                    continue
                start, end = entity.get("start"), entity.get("end")
                text = entity.get("text", entity.get("surface"))
                label = entity.get("label", entity.get("category"))
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(document["source"])
                        or not isinstance(text, str) or len(text) > 160
                        or document["source"][start:end] != text
                        or not isinstance(label, str) or label not in CATEGORIES[:-1]
                        or any(entity.get(key, expected) != expected for key, expected in
                               (("owner", owner), ("owner_id", owner),
                                ("doc_id", owner), ("coordinate_space", "redacted_preview"),
                                ("candidate_id", owner if document["kind"] == "cv" else None),
                                ("snapshot_id", snapshot.snapshot_id),
                                ("source_digest", document["digest"])))):
                    status = "invalid_evidence"
                    continue
                valid[owner].append({"start": start, "end": end, "text": text, "label": label})
        return valid, status

    @staticmethod
    def _relevant(document, category, question, entities, *, broad=False, aliases=()):
        question = _retrieval_question(question, document["kind"])
        terms = guidance._terms(question, category) - guidance._BROAD_WORDS
        focus = _focus(question, category)
        duration_focus = _duration_focus(question) if category == "EXPERIENCE_YEARS" else set()
        jd_requirements = document["kind"] == "jd" and not focus
        relevant = []
        for unit in document["units"]:
            start, end, quote = unit
            words = set(re.findall(r"[\w+#-]+", quote.casefold()))
            words.update(word[:-1] for word in tuple(words) if word.endswith("s"))
            mentions = [entity for entity in entities if entity["label"] == category
                        and start <= entity["start"] < entity["end"] <= end]
            category_hit = category in _PATTERNS and bool(re.search(_PATTERNS[category], quote, re.I))
            alias_hit = any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", quote, re.I)
                            for alias in aliases)
            if not broad and focus and not focus & words:
                continue
            if not broad and duration_focus:
                if not _duration_links(quote, duration_focus):
                    continue
            if category in {"DEGREE", "CERTIFICATION"} and not (category_hit or mentions):
                continue
            if broad and not (category_hit or mentions or alias_hit):
                continue
            if not broad and not (terms & words or mentions or category_hit or jd_requirements):
                continue
            relevant.append((10 * bool(mentions) + len(terms & words) + bool(alias_hit), start, unit, mentions))
        relevant.sort(key=lambda row: (-row[0], row[1]))
        return relevant[:2]

    @staticmethod
    def _source_cell(snapshot, document, relevant, citations, claims):
        if document["status"] == "unavailable":
            return _cell("source_unavailable")
        if not relevant:
            return _cell("not_found")
        # A broad cell uses a complete source context, including surrounding
        # negation/expiry. Entity text alone is never rendered as a qualification.
        _, _, unit, mentions = relevant[0]
        citation = _citation(snapshot, document, unit)
        citations[citation["evidence_id"]] = citation
        kind = "entity_mentions" if mentions else "source_only"
        claims.append({"text": citation["quote"], "evidence_id": citation["evidence_id"], "kind": kind})
        result = _cell(kind, citation["quote"], evidence_ids=[citation["evidence_id"]], context=citation["quote"])
        if mentions:
            result["mentions"] = list(dict.fromkeys(item["text"] for item in mentions))[:4]
        return result

    def _focused_cell(self, snapshot, document, relevant, question, citations, claims, *, duration_focus=()):
        if document["status"] == "unavailable":
            return _cell("source_unavailable", qa_status="unavailable")
        if not relevant:
            return _cell("not_found")
        records = [_citation(snapshot, document, row[2]) for row in relevant]
        # The QaClient contract is local offsets in one supplied quote. Ownership
        # and snapshot identity stay outside model control, including for the JD.
        passages = [{"passage_id": item["evidence_id"], "quote": item["quote"],
                     "candidate_id": document["owner"] if document["kind"] == "cv" else None,
                     "snapshot_id": snapshot.snapshot_id, "document_digest": document["digest"],
                     "document_id": "cv:" + document["owner"] if document["kind"] == "cv" else "jd",
                     "display_start": item["start"], "display_end": item["end"],
                     "coordinate_space": "redacted_preview"} for item in records]
        # Keep an independent active-source copy: an injected/in-process client
        # may mutate its input as well as return foreign or stale output.
        try:
            client = self.qa_client if self.qa_client is not None else shared_qa_client()
            response = client.answer(question[:1000], [dict(item) for item in passages])
        except Exception:
            response = {"status": "unavailable"}
        qa_status = "invalid_output"
        accepted = []
        if isinstance(response, Mapping):
            qa_status = response.get("status", "invalid_output")
            if not isinstance(qa_status, str) or qa_status not in {"supported", "not_found", "unavailable", "failed"}:
                qa_status = "invalid_output"
            spans = response.get("spans", [])
            if qa_status == "supported":
                try:
                    if (response.get("model_id", MODEL_ID) != MODEL_ID
                            or not isinstance(spans, list) or not 1 <= len(spans) <= 32
                            or response.get("snapshot_id", snapshot.snapshot_id) != snapshot.snapshot_id
                            or response.get("document_digest", document["digest"]) != document["digest"]
                            or response.get("source_digest", document["digest"]) != document["digest"]):
                        raise ValueError("Foreign model/source response")
                    by_id = {item["evidence_id"]: item for item in records}
                    for span in spans:
                        if not isinstance(span, Mapping) or not isinstance(span.get("passage_id"), str):
                            raise ValueError("Invalid span identity")
                        record = by_id.get(span["passage_id"])
                        a, b, text = span.get("start"), span.get("end"), span.get("text")
                        if (record is None or "candidate_id" not in span
                                or span["candidate_id"] != passages[0]["candidate_id"]
                                or type(a) is not int or type(b) is not int
                                or not 0 <= a < b <= len(record["quote"])
                                or not isinstance(text, str) or len(text) > 200
                                or text != record["quote"][a:b]
                                or document["source"][record["start"]:record["end"]] != record["quote"]
                                or any(span.get(key, expected) != expected for key, expected in
                                       (("owner", document["owner"]), ("snapshot_id", snapshot.snapshot_id),
                                        ("document_digest", document["digest"]),
                                        ("source_digest", document["digest"]),
                                        ("coordinate_space", "redacted_preview"),
                                        ("document_id", passages[0]["document_id"])))):
                            raise ValueError("Span does not belong to the active source")
                        if duration_focus and not _supported_duration_span(record["quote"], a, b, duration_focus):
                            qa_status, accepted = "insufficient_evidence", []
                            break
                        accepted.append((record, text, record["start"] + a, record["start"] + b))
                except (ValueError, TypeError, KeyError):
                    qa_status, accepted = "invalid_output", []
        if accepted:
            ids = []
            for record, text, a, b in accepted[:4]:
                identifier = record["evidence_id"]
                citations[identifier] = record
                if identifier not in ids:
                    ids.append(identifier)
                claims.append({"text": text, "evidence_id": identifier, "kind": "model_span",
                               "owner": document["owner"], "start": a, "end": b,
                               "coordinate_space": "redacted_preview"})
            context = " ".join(citations[identifier]["quote"] for identifier in ids)
            text = "; ".join(dict.fromkeys(item[1] for item in accepted[:4]))
            return _cell("model_span", text, evidence_ids=ids, context=context,
                         qa_status="partial" if response.get("incomplete") is True or len(accepted) > 4 else "supported")
        fallback = self._source_cell(snapshot, document, relevant, citations, claims)
        fallback.update(status="source_only", qa_status=qa_status)
        label = ("QA found no answer. Source excerpt: " if qa_status == "not_found"
                 else "QA found insufficient evidence for skill-specific tenure. Source excerpt: "
                 if qa_status == "insufficient_evidence" else "QA unavailable or rejected. Source excerpt: ")
        fallback["text"] = label + fallback["context"]
        # Source fallback is never reported as model or entity inference.
        for claim in claims:
            if claim["evidence_id"] in fallback["evidence_ids"]:
                claim["kind"] = "source_only"
        return fallback

    @staticmethod
    def _result(snapshot, selected, category="GENERAL"):
        return {"text": "", "citations": [], "claims": [],
                "context": {"snapshot_id": snapshot.snapshot_id, "jd_id": snapshot.jd_id,
                            "selected_candidates": list(selected), "category": category,
                            "pending_clarification": None},
                "mode": "comparison", "status": "answered", "model": None,
                "suggested_actions": [], "comparison": [],
                "entity_status": "unavailable", "qa_status": "not_run", "exhaustive": False}

    def _table(self, snapshot, selected, categories, *, question=None, original_question=None, context=None):
        documents = _documents(snapshot, selected)
        entities, entity_status = self._entities(snapshot, selected, documents)
        result = self._result(snapshot, selected, categories[0] if len(categories) == 1 else "GENERAL")
        if context is not None:
            result["context"] = context
        result["entity_status"] = entity_status
        citations, claims = {}, []
        qa_states = []
        for category in categories:
            cells = {}
            aliases = [item["text"] for item in entities[snapshot.jd_id] if item["label"] == category][:16]
            for owner, document in documents.items():
                if owner == snapshot.jd_id and len(selected) == 1 and question is not None:
                    continue
                relevant = self._relevant(document, category, question or "", entities[owner],
                                          broad=question is None, aliases=aliases)
                if question is None:
                    cell = self._source_cell(snapshot, document, relevant, citations, claims)
                else:
                    natural = _qa_question(snapshot, original_question or question, question, category,
                                           document["kind"], len(selected) > 1)
                    focus = _duration_focus(_retrieval_question(question, document["kind"])) if category == "EXPERIENCE_YEARS" else ()
                    cell = self._focused_cell(snapshot, document, relevant, natural, citations, claims, duration_focus=focus)
                    qa_states.append(cell["qa_status"])
                cells[owner] = cell
            criterion = cells.pop(snapshot.jd_id, _cell("not_requested", "JD comparison not requested."))
            result["comparison"].append({"category": category, "requirement": _visible(criterion),
                                         "requirement_status": criterion["status"],
                                         "requirement_evidence_ids": criterion["evidence_ids"],
                                         "answers": cells})
        result["citations"] = list(citations.values())
        result["claims"] = claims
        result["qa_status"] = _aggregate(qa_states)
        result["comparison_status"] = "ready" if all(doc["status"] == "complete" for doc in documents.values()) else "partial"
        result["source_status"] = _aggregate(doc["status"] for doc in documents.values())
        if any(doc["status"] == "partial" for doc in documents.values()):
            result["status"] = "partial"
        result["text"] = ("Bot can make mistake. Review source evidence and citations before acting. ")
        if not selected:
            result["mode"] = "extractive_qa" if any(c["kind"] == "model_span" for c in claims) else "source_only"
            result["text"] = "\n\n".join(row["requirement"] for row in result["comparison"])
        elif len(selected) == 1 and question is not None:
            kinds = {claim["kind"] for claim in claims if citations[claim["evidence_id"]]["owner"] in selected}
            result["mode"] = "extractive_qa" if "model_span" in kinds else "source_only"
            result["text"] = "\n\n".join(_visible(row["answers"][selected[0]]) for row in result["comparison"])
            if categories[0] == "EXPERIENCE_YEARS":
                result["text"] += "\nTotal experience does not establish years using a particular skill."
            elif categories[0] in {"DEGREE", "CERTIFICATION"}:
                result["text"] += "\nA mention does not verify completion or current credential validity."
        elif question is not None:
            result["text"] = "\n\n".join(owner + ": " + _visible(row["answers"][owner])
                                             for row in result["comparison"] for owner in selected)
            result["text"] += "\nMissing evidence does not establish absence; total experience does not establish skill-specific tenure."
        if any(claim["kind"] == "model_span" for claim in claims):
            result["model"] = MODEL_ID
        if not claims or result["qa_status"] == "insufficient_evidence":
            result["status"] = "insufficient_evidence"
        return self._bounded(result, focused=question is not None)

    @staticmethod
    def _bounded(result, *, focused=False):
        """Bound the default JSON representation too (Unicode uses escapes)."""
        def prune():
            used = set()
            for row in result.get("comparison", []):
                used.update(row["requirement_evidence_ids"])
                for cell in row["answers"].values():
                    used.update(cell["evidence_ids"])
            result["citations"] = [item for item in result["citations"] if item["evidence_id"] in used][:66]
            result["claims"] = [item for item in result["claims"] if item["evidence_id"] in used][:88]
            if not any(item["kind"] == "model_span" for item in result["claims"]):
                result["model"] = None
                if result["mode"] == "extractive_qa":
                    result["mode"] = "source_only"
                    result["qa_status"] = "budget_limited"
            if focused:
                selected = result["context"]["selected_candidates"]
                if selected:
                    result["text"] = "\n\n".join(
                        (owner + ": " if len(selected) > 1 else "") + _visible(row["answers"][owner])
                        for row in result["comparison"] for owner in selected)
                    result["text"] += "\nMissing evidence does not establish absence; total experience does not establish skill-specific tenure."
                else:
                    result["text"] = "\n\n".join(row["requirement"] for row in result["comparison"])

        while len(json.dumps(result, allow_nan=False).encode("utf-8")) > MAX_BYTES:
            # Drop whole contexts, with an explicit limitation, instead of cutting
            # a negation or status off the end of a sentence. Never omit an owner.
            options = []
            for row in result.get("comparison", []):
                for cell in row["answers"].values():
                    if cell["evidence_ids"]:
                        options.append((len(json.dumps(cell)), row, cell))
                if row["requirement_evidence_ids"]:
                    options.append((len(json.dumps(row["requirement"])), row, None))
            if not options:
                result["context"].pop("last_question", None)
                result["text"] = "Evidence is limited by the answer storage budget."
                if len(json.dumps(result).encode("utf-8")) > MAX_BYTES:
                    raise ValueError("Result metadata exceeds the answer budget")
                break
            _, row, cell = max(options, key=lambda item: item[0])
            if cell is None:
                row.update(requirement=_LIMITED, requirement_evidence_ids=[], requirement_status="budget_limited")
            else:
                cell.clear()
                cell.update(_cell("budget_limited", _LIMITED))
            result["comparison_status"] = "partial"
            result["status"] = "partial"
            prune()
        return result

    def compare(self, snapshot, candidate_ids):
        """Return six deterministic rows; no QA startup or synchronous extraction."""
        selected = _selection(snapshot, candidate_ids)
        if not selected:
            raise ValueError("Choose at least one candidate to compare")
        return self._table(snapshot, selected, CATEGORIES)

    def answer(self, snapshot, question, context=None):
        """Extract facts independently per authorized document; never generate."""
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Ask a question of 1 to 2,000 characters")
        # Reuse established reference, follow-up, category and action handling.
        # No provider is passed: this helper cannot call a generator.
        turn = guidance.answer(snapshot, question, context=context)
        state = turn["context"]
        selected = _selection(snapshot, state["selected_candidates"])
        if turn["status"] in {"action_guidance", "clarification"} or turn["mode"] == "deterministic":
            turn.update(entity_status="not_run", qa_status="not_run")
            return turn
        topic = guidance._topic_question(snapshot, state.get("last_question", redact_text(question)))
        category = state["category"] if state["category"] in CATEGORIES else "GENERAL"
        if category == "GENERAL" and re.search(r"\b(?:programming languages?|languages?|tools?|technologies)\b", topic, re.I):
            category = state["category"] = "SKILL"
        broad = category == "GENERAL" and not (guidance._terms(topic, category) - guidance._BROAD_WORDS)
        result = self._table(snapshot, selected, CATEGORIES if broad else (category,),
                             question=None if broad else topic, original_question=redact_text(question), context=state)
        return result
