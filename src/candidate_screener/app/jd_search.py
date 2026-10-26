"""Literal, read-only search over JD texts already authorized by the caller.

No ranking, CV access, inference startup or global cache is involved. Offsets refer
to redacted source text. A phrase match alone never establishes a requirement.
"""
from __future__ import annotations

import hashlib
import json
import re

from candidate_screener.annotation.redact import redact_text


_QUERY = re.compile(
    r"^\s*(?:(?:which|what)\s+(?:jds?|jobs?|roles?|job descriptions?)\s+"
    r"(?:requires?|needs?|mentions?|has|have|lists?|includes?)|"
    r"(?:find|search|show(?: me)?)\s+(?:jds?|jobs?|roles?|job descriptions?)\s+(?:with|requiring|mentioning))"
    r"\s+(?P<skill>[^?\n]{1,100})\??\s*$", re.I)
_SENSITIVE = re.compile(r"\b(?:nationality|nationalities|race|ethnicity|religion|gender|sex|age|pregnan\w*|disability|citizenship)\b", re.I)
_NEGATED = re.compile(r"\b(?:not required|not necessary|no (?:need|requirement)|do not require|does not require|never required)\b", re.I)
_PREFERRED = re.compile(r"\b(?:preferred|optional|desirable|nice[- ]to[- ]have|a plus|advantageous|bonus)\b", re.I)
_REQUIRED = re.compile(r"\b(?:required|mandatory|essential|must|require|requires|need|needs)\b", re.I)
_SECTIONS = {"skills", "knowledge", "domain knowledge", "qualifications", "responsibilities",
             "experience", "requirements", "education", "benefits", "location", "work arrangement"}


def _query(question):
    if not isinstance(question, str) or len(question) > 2000:
        return None
    match = _QUERY.fullmatch(question)
    if not match:
        return None
    skill = match["skill"].strip().strip("\"'").rstrip(".")
    skill = re.sub(r"^(?:(?:this|these|the)\s+)?(?:specific\s+)?skills?\b\s*(?:(?:of|in)\s+|[-:]\s*)?", "", skill, flags=re.I)
    skill = re.sub(r"^(?:the )?(?:skills? (?:of |in )?|knowledge (?:of |in ))", "", skill, flags=re.I)
    skill = re.sub(r"\s+skills?$", "", skill, flags=re.I).strip()
    if (not skill or len(skill.split()) > 6 or _SENSITIVE.search(skill)
            or not re.fullmatch(r"[\w+#. -]+", skill)
            or re.search(r"\b(?:and|or|ignore|instructions|candidate|applicant)\b", skill, re.I)):
        return None
    return skill


def is_jd_search(question):
    """Route only explicit, supported job-related phrase searches."""
    return _query(question) is not None


def _passages(text, pattern):
    section = "mention"
    for line in re.finditer(r"[^\n]+", text):
        stripped = line[0].strip().strip(":").casefold()
        if stripped in {"required", "mandatory", "essential"}:
            section = "required"
            continue
        if stripped in {"preferred", "optional", "desirable", "nice to have", "nice-to-have"}:
            section = "preferred"
            continue
        if stripped in _SECTIONS:
            section = "mention"
            continue
        # Sentence/clause boundaries prevent a preference for another skill from
        # changing this phrase's status; keep complete local context and offsets.
        for sentence in re.finditer(r".+?(?:[;.!?](?=\s|$)|$)", line[0]):
            quote = sentence[0]
            if not pattern.search(quote):
                continue
            no_required = re.search(r"\bno\b.{0,100}\brequired\b", quote, re.I)
            if _NEGATED.search(quote) or no_required:
                level = "negated"
            elif _PREFERRED.search(quote) and _REQUIRED.search(quote):
                level = "mixed"
            elif _PREFERRED.search(quote):
                level = "preferred"
            elif _REQUIRED.search(quote):
                level = "required"
            else:
                level = section
            left = len(quote) - len(quote.lstrip())
            start = line.start() + sentence.start() + left
            end = line.start() + sentence.end() - (len(quote) - len(quote.rstrip()))
            yield level, start, end, text[start:end]


def search_jobs(documents, question, *, max_documents=200, max_matches=10):
    """Return bounded source-owned findings; caller reauthorizes before display.

    Missing/unavailable source text and bounds are reported as partial coverage.
    Input must contain only authorized JD identities, never an unrestricted corpus.
    """
    if (type(max_documents) is not int or not 1 <= max_documents <= 1000
            or type(max_matches) is not int or not 1 <= max_matches <= 20):
        raise ValueError("Invalid JD search bounds")
    if not isinstance(documents, (list, tuple)):
        raise ValueError("Provide an authorized JD list")
    skill = _query(question)
    result = {"mode": "jd_search", "generation_mode": "controlled", "status": "answered",
              "text": "", "matches": [], "citations": [], "claims": [], "comparison": [],
              "candidate_labels": {}, "context": None, "model": None, "suggested_actions": []}
    coverage = {"available": len(documents), "searched": 0, "unavailable": 0,
                "not_searched": max(0, len(documents) - max_documents), "omitted_matches": 0, "complete": False}
    result["coverage"] = coverage
    if skill is None:
        result.update(status="clarification", text="Which job-related skill should I look for in the available job descriptions?")
        return result
    result["skill"] = redact_text(skill)
    pattern = re.compile(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", re.I)
    seen, lines = set(), []
    for document in documents[:max_documents]:
        if not isinstance(document, dict):
            raise ValueError("Invalid authorized JD document")
        identity, version = document.get("jd_id"), document.get("source_version")
        if (not isinstance(identity, str) or not identity or len(identity) > 160
                or identity in seen or not isinstance(version, str) or not version):
            raise ValueError("JD search requires unique, versioned source identities")
        seen.add(identity)
        raw = document.get("text")
        if (not isinstance(raw, str) or not raw.strip() or len(raw) > 100_000
                or document.get("source_status", "available") not in {"available", "complete"}):
            coverage["unavailable"] += 1
            continue
        source = redact_text(raw)
        coverage["searched"] += 1
        passages = list(_passages(source, pattern))
        if not passages:
            continue
        if len(result["matches"]) >= max_matches:
            coverage["omitted_matches"] += 1
            continue
        # Do not silently discard contrary evidence. Oversized matching source
        # groups are unavailable instead of being shortened into stronger claims.
        if len(passages) > 3 or sum(len(item[3]) for item in passages) > 1500:
            coverage["unavailable"] += 1
            coverage["searched"] -= 1
            continue
        levels = {item[0] for item in passages}
        level = next(iter(levels)) if len(levels) == 1 else "mixed"
        title = redact_text(str(document.get("title") or "Untitled role"))[:200]
        ids = []
        for local_level, start, end, quote in passages:
            digest = hashlib.sha256(f"{identity}\0{version}\0{start}\0{end}\0{quote}".encode()).hexdigest()[:24]
            evidence_id = "qa:" + digest
            ids.append(evidence_id)
            result["citations"].append({"owner": identity, "evidence_id": evidence_id,
                "quote": quote, "start": start, "end": end, "source_version": version, "level": local_level})
        result["candidate_labels"][identity] = title
        result["matches"].append({"jd_id": identity, "title": title, "level": level, "evidence_ids": ids})
        descriptions = {"required": f"{title} requires {skill}", "preferred": f"{title} lists {skill} as preferred",
            "negated": f"{title} explicitly says {skill} is not required",
            "mention": f"{title} mentions {skill}, without establishing it as mandatory",
            "mixed": f"{title} has mixed or conflicting statements about {skill}; review its sources"}
        lines.append(descriptions[level])
    # Preserve whole match/source groups. Leave room for the final prose and the
    # controller's transport metadata rather than cutting a source qualifier.
    while result["matches"] and len(json.dumps(result, ensure_ascii=False).encode()) > 20_000:
        removed = result["matches"].pop()
        result["citations"] = [row for row in result["citations"] if row["owner"] != removed["jd_id"]]
        result["candidate_labels"].pop(removed["jd_id"], None)
        lines.pop()
        coverage["omitted_matches"] += 1
    coverage["complete"] = not any(coverage[key] for key in ("unavailable", "not_searched", "omitted_matches"))
    if not any(row["level"] == "required" for row in result["matches"]):
        lines.insert(0, f"I haven't found a JD that makes {skill} mandatory in the jobs searched")
    if not coverage["complete"]:
        lines.append(f"Search coverage is partial: {coverage['searched']} of {coverage['available']} jobs searched; "
                     f"{coverage['omitted_matches']} matching jobs omitted from this response")
        result["status"] = "partial"
    elif not result["matches"]:
        result["status"] = "no_match"
    result["text"] = redact_text(". ".join(lines) + ".")
    return result
