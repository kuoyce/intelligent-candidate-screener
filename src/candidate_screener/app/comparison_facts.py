"""Conservative source-claim states and application-owned comparison relations.

These registered language patterns are not a general entailment classifier.
Unrecognized qualifications stay unclear; exact spans never establish polarity.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from itertools import combinations

from .portal_extractive import _duration_links

_UNKNOWN = {"not_found", "omitted", "not_requested"}
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "eleven": 11, "twelve": 12}
_STOP = set("what which who why how does do have has their they them compare comparison "
            "candidate candidates applicant applicants experience experiences years year months month "
            "more most less greater better strongest stronger clearer evidence documented qualifications qualification degree degrees "
            "education educational skill skills knowledge title certification certifications "
            "about with using tenure for this that the a an and or in on of to is are "
            "cv cvs source explicitly state states say says role job requirement requirements "
            "first second third one two three focus only show tell me please versus vs "
            "total relevant against both all compare".split())
_DEGREE = r"(?:bachelor(?:'s)?|master(?:'s)?|bsc|msc|b\.sc|m\.sc|phd|ph\.d|mba|degree|diploma)"
_CERT = r"(?:certification|certificate|credential|certified|license|licence)"


def _id(*parts):
    return hashlib.sha256(":".join(str(x) for x in parts).encode()).hexdigest()[:16]


def jd_criteria(snapshot):
    """Find at most three literal JD criteria with a registered noun-phrase form.

    This is deliberately not general requirement extraction. Unsupported prose,
    negated requirements and partially parsed conjunctions remain unparsed so the
    service can keep its category comparison rather than manufacture a criterion.
    """
    found = []
    for unit in _sentences(snapshot.jd_text):
        if re.search(r"\b(?:not|no|never|without)\b", unit, re.I):
            continue
        phrase = unit.strip(" .;:")
        header = re.fullmatch(r"(?:(?:required|essential|mandatory)\s+(?P<label>skills?|knowledge)|(?P<label2>skills?|knowledge)\s+(?:required|essential|mandatory))\s*:\s*(?P<items>.+)", phrase, re.I)
        knowledge = re.fullmatch(r"knowledge\s+of\s+(?P<items>.+?)\s+(?:is\s+)?(?:required|essential|mandatory)", phrase, re.I)
        if header or knowledge:
            listing = header or knowledge
            items = re.split(r"\s+(?:and|or)\s+|[,;]\s*", listing["items"])
            if items and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9+#./-]*(?:\s+[A-Za-z][A-Za-z0-9+#./-]*){0,3}", item)
                             and not re.search(r"\b(?:we|you|they|should|could|would|may|can|will|not|no|optional|preferred|experience)\b", item, re.I) for item in items):
                label = "knowledge" if knowledge else header["label"] or header["label2"]
                category = "KNOWLEDGE" if label.casefold() == "knowledge" else "SKILL"
                for item in items:
                    if item.casefold() not in {row["criterion"].casefold() for row in found}:
                        found.append({"criterion": item, "topic": item, "category": category})
                continue
        phrase = re.sub(r"^(?:(?:we|candidates?|applicants?)\s+)?(?:require[sd]?|must have|needs?)\s+", "", phrase, flags=re.I)
        phrase = re.sub(r"\s+(?:(?:is|are)\s+)?(?:required|preferred|essential|mandatory)$", "", phrase, flags=re.I)
        fragments = re.split(r"\s+(?:and|or)\s+|[,;]\s*", phrase)
        parsed = []
        for fragment in fragments:
            fragment = fragment.strip()
            degree = re.fullmatch(rf"{_DEGREE}(?:\s+degree)?", fragment, re.I)
            match = re.fullmatch(r"(?P<topic>[A-Za-z][A-Za-z0-9+#./-]*(?:\s+[A-Za-z][A-Za-z0-9+#./-]*){0,3})\s+(?P<kind>experience|skills?|knowledge|certification|certificate|license|licence)", fragment, re.I)
            if degree:
                parsed.append({"criterion": fragment, "topic": "qualifications", "category": "DEGREE"})
            elif match and not re.search(r"\b(?:we|our|you|your|they|their|team|offer|offers|provide|provides|have|has|with|of|years?|months?|one|two|three|four|five|six|seven|eight|nine|ten)\b", match["topic"], re.I):
                kind = match["kind"].casefold()
                category = ("CERTIFICATION" if kind in {"certification", "certificate", "license", "licence"}
                            else "KNOWLEDGE" if kind == "knowledge" else "SKILL")
                parsed.append({"criterion": fragment, "topic": match["topic"], "category": category})
            else:
                parsed = []
                break
        for criterion in parsed:
            if criterion["criterion"].casefold() not in {item["criterion"].casefold() for item in found}:
                found.append(criterion)
    return found[:3]


def _topic(result, row):
    context = result.get("context") or {}
    explicit = row.get("criterion") or row.get("topic")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()[:200]
    criteria = context.get("criteria")
    if isinstance(criteria, list) and len(criteria) == 1 and isinstance(criteria[0], str):
        return criteria[0][:200]
    return str(context.get("last_question") or row.get("category") or "Evidence")[:2000]


def _terms(topic, category):
    words = set(re.findall(r"[a-z][a-z0-9+#.-]*", topic.casefold())) - _STOP
    if category == "DEGREE":
        return set()
    return {word for word in words if len(word) > 1 and not word.startswith("cv-")}


def _sentences(text):
    # Keep semicolon clauses together: they can carry a qualification status.
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _matches(term, sentence):
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", sentence, re.I))


def _negative(sentence, term):
    """Negation must directly govern the requested qualification, not its neighbor."""
    target = re.escape(term)
    return bool(re.search(
        rf"\b(?:no|without|lacks?|never(?:\s+used)?|not(?:\s+have)?)\s+"
        rf"(?:(?:commercial|professional|production|documented|any|a|an|completed|current)\s+){{0,3}}{target}\b"
        rf"|\b{target}\b\s+(?:experience\s+)?(?:is\s+)?(?:absent|not\s+(?:held|completed|obtained|available))\b",
        sentence, re.I))


def support_state(context, category, focus=()):
    """Return registered source-claim status; never interpret a naked model span."""
    units = _sentences(context)
    focus = tuple(focus)
    clauses = [clause.strip() for unit in units
               for clause in re.split(r";|,\s*(?:but|however)\s+|\bbut\b", unit) if clause.strip()]
    if category == "DEGREE":
        statuses = []
        for unit in clauses:
            if not re.search(_DEGREE, unit, re.I):
                # A standalone adjacent status applies to the preceding degree,
                # not arbitrary 'employment ongoing' text elsewhere in a CV.
                if statuses and re.match(r"^(?:status\s*:\s*|studies?\s+)(?:in progress|ongoing|incomplete|(?:not|never)\s+(?:been\s+)?(?:completed|awarded|obtained|earned))\b", unit, re.I):
                    statuses[-1] = "qualified"
                continue
            if re.search(r"\b(?:hopes?|plans?|wants?|aims?|aspires?|wishes?)\b", unit, re.I):
                statuses.append("unclear")
            elif re.search(r"\b(?:in progress|ongoing|pursuing|expected|incomplete|pending|"
                           r"conditional(?:ly)?|provisional(?:ly)?|will(?:\s+be)?|about to|planned|planning|"
                           r"(?:not|never)\s+(?:been\s+)?(?:completed|awarded|obtained|earned))\b", unit, re.I):
                statuses.append("qualified")
            elif re.search(rf"\b(?:no|without)\s+(?:completed\s+)?{_DEGREE}\b", unit, re.I):
                statuses.append("explicit_negative")
            elif re.search(r"\b(?:not|never|no|without)\b", unit, re.I):
                statuses.append("unclear")
            elif re.search(rf"\b(?:completed|holds?|earned|awarded|graduated with)\s+(?:a\s+|an\s+)?{_DEGREE}\b"
                           rf"|\b{_DEGREE}\b\s+(?:(?:degree|was|is|has been)\s+)*(?:completed|awarded|earned)\b", unit, re.I):
                statuses.append("explicit_positive")
            else:
                statuses.append("unclear")
        states = set(statuses)
        if "explicit_positive" in states and ("explicit_negative" in states or "qualified" in states):
            return "conflicting"
        return next(iter(states)) if len(states) == 1 else "unclear"
    if category == "CERTIFICATION" and re.search(_CERT, context, re.I):
        if re.search(r"\b(?:expired|expires|valid until|pending|in progress|not current|lapsed)\b", context, re.I):
            return "qualified"
    if not focus:
        # Broad topics do not establish that different CVs discuss the same skill.
        return "unclear"
    states = []
    if len(focus) > 1 and not any(all(_matches(term, unit) for term in focus) for unit in clauses):
        return "unclear"
    for unit in clauses:
        relevant = [term for term in focus if _matches(term, unit)]
        if not relevant:
            continue
        if len(relevant) != len(focus):
            states.append("unclear")
        elif len(focus) > 1 and not _linked_phrase(unit, focus):
            states.append("unclear")
        elif any(_negative(unit, term) for term in relevant):
            repeated = any(len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", unit, re.I)) > 1 for term in relevant)
            states.append("unclear" if repeated else "explicit_negative")
        elif re.search(r"\b(?:hopes?|hoping|plans?|planned|planning|intends?|intended|wants?|aims?|aspires?|wishes?|would like)\b", unit, re.I):
            states.append("unclear")
        elif any(re.search(rf"\b(?:learning|studying|pursuing|training in)\s+(?:the\s+)?{re.escape(term)}\b", unit, re.I) for term in relevant):
            states.append("qualified")
        elif re.search(r"\b(?:not|no|never|without|lacks?)\b", unit, re.I):
            states.append("unclear")
        elif any(_positive(unit, term, category) for term in relevant) or _duration_links(unit, focus):
            states.append("explicit_positive")
        else:
            states.append("unclear")
    if "explicit_positive" in states and "explicit_negative" in states:
        return "conflicting"
    if states and set(states) == {"explicit_negative"}:
        return "explicit_negative"
    if states and set(states) == {"explicit_positive"}:
        return "explicit_positive"
    if states and set(states) == {"qualified"}:
        return "qualified"
    return "unclear"


def _positive(unit, term, category):
    """Require a local predicate for this term, not an unrelated skill nearby."""
    target = re.escape(term)
    patterns = [rf"\b{target}\b\s+(?:(?:programming|development|engineering|commercial|professional)\s+)?(?:skills?|experience|proficiency|knowledge)\b",
                rf"\b(?:uses?|used|using|proficient\s+in|experienced\s+in|certified\s+in)\s+{target}\b",
                rf"\b(?:built|implemented|developed|deployed)\s+(?:(?:a|an|the|production)\s+){{0,2}}{target}\b"]
    if category == "TITLE":
        patterns.append(rf"\b{target}\b\s+(?:developer|engineer|architect|analyst)\b")
    return any(re.search(pattern, unit, re.I) for pattern in patterns)


def _linked_phrase(unit, focus):
    """Multiword criteria must occur together, not as unrelated nearby concepts."""
    tokens = list(re.finditer(r"[a-z][a-z0-9+#.-]*", unit.casefold()))
    for index in range(len(tokens) - len(focus) + 1):
        window = tokens[index:index + len(focus)]
        if (set(token.group() for token in window) == set(focus)
                and all(unit[first.end():second.start()].isspace() for first, second in zip(window, window[1:]))):
            return True
    return False


def _citations(snapshot, result):
    sources = {snapshot.jd_id: snapshot.jd_text}
    sources.update({row["cv_id"]: row.get("preview_text", "") for row in snapshot.candidates
                    if row.get("preview_status") == "available_redacted_text"})
    accepted = {}
    for record in result.get("citations", []):
        if not isinstance(record, Mapping):
            continue
        owner = record.get("owner")
        source, a, b = sources.get(owner), record.get("start"), record.get("end")
        if (not isinstance(source, str) or type(a) is not int or type(b) is not int
                or not 0 <= a < b <= len(source) or record.get("quote") != source[a:b]
                or record.get("snapshot_id") != snapshot.snapshot_id
                or record.get("document_digest") != hashlib.sha256(source.encode()).hexdigest()
                or record.get("coordinate_space") != "redacted_preview"
                or record.get("owner_kind") != ("jd" if owner == snapshot.jd_id else "cv")
                or not isinstance(record.get("evidence_id"), str)):
            continue
        accepted[record["evidence_id"]] = record
    return accepted


def _duration(context, focus):
    if len(focus) != 1:
        return None
    number = r"(?:\d+(?:\.\d+)?|" + "|".join(_WORDS) + r")"
    values = set()
    for unit in _sentences(context):
        linked = _duration_links(unit, focus)
        if not linked:
            continue
        if (re.search(r"\b(?:no|not|never|without|total|overall|combined|cumulative|about|approximately|over|under|nearly|at least|up to|more than|less than)\b", unit, re.I)
                or re.search(rf"\b{number}\s*(?:to|[-–])\s*{number}\s+(?:years?|months?|yrs?)\b", unit, re.I)):
            return None
        for a, b in linked:
            text = unit[a:b]
            match = re.fullmatch(r"(\d+(?:\.\d+)?|" + "|".join(_WORDS) + r")\s+(years?|months?|yrs?)", text, re.I)
            if match:
                amount = _WORDS.get(match[1].casefold())
                amount = float(match[1]) if amount is None else amount
                values.add(float(amount) * (12 if match[2].casefold().startswith(("year", "yr")) else 1))
    return next(iter(values)) if len(values) == 1 else None


def build_facts(snapshot, result):
    """Build facts only from citations that still slice the authorized snapshot."""
    labels = {row["cv_id"]: f"Candidate {row['rank']}" for row in snapshot.candidates}
    context = result.get("context") or {}
    selected = context.get("selected_candidates", [])
    if not isinstance(selected, list):
        raise ValueError("Invalid requested comparison participants")
    if not selected:
        selected = list(dict.fromkeys(owner for row in result.get("comparison", [])
                                     for owner in row.get("answers", {})))
    if len(selected) > 10 or any(owner not in labels for owner in selected):
        raise PermissionError("Comparison participant outside active snapshot")
    selected = list(dict.fromkeys(selected))
    citations = _citations(snapshot, result)
    facts = {"cells": [], "relations": [], "labels": labels, "participants": selected,
             "citation_labels": {identity: f"S{index}" for index, identity in enumerate(sorted(citations), 1)},
             "requested_owners": selected, "criteria": [], "advantages": [], "intent": context.get("intent")}
    for index, row in enumerate(result.get("comparison", [])):
        category = str(row.get("category", "GENERAL"))
        topic = _topic(result, row)
        focus = _terms(topic, category)
        frame_criteria = context.get("criteria")
        focused_label = (frame_criteria[0] if len(result.get("comparison", [])) == 1
                         and isinstance(frame_criteria, list) and len(frame_criteria) == 1
                         and isinstance(frame_criteria[0], str) else None)
        criterion = str(row.get("criterion") or row.get("topic") or focused_label or category.replace("_", " ").title())
        facts["criteria"].append(criterion)
        requirement_ids = _requirement_ids(snapshot, row, citations, criterion)
        cells = []
        for owner in selected:
            raw = row.get("answers", {}).get(owner, {})
            ids = [identity for identity in raw.get("evidence_ids", [])
                   if identity in citations and citations[identity]["owner"] == owner]
            texts = list(dict.fromkeys(citations[identity]["quote"] for identity in ids))
            full_context = " ".join(texts)
            status = raw.get("status", "not_found")
            state = ("unavailable" if status == "source_unavailable" else
                     "partial_source" if status in {"omitted", "budget_limited"} else
                     "not_found_in_retrieved_evidence" if not full_context or status in _UNKNOWN else
                     support_state(full_context, category, focus))
            cell = {"id": "cell:" + _id(snapshot.snapshot_id, index, owner), "owner": owner,
                    "criterion": criterion, "category": category,
                    "topic": row.get("topic") if isinstance(row.get("topic"), str) else " ".join(sorted(focus)),
                    "text": (raw.get("text", "") if isinstance(raw.get("text"), str)
                             and raw.get("text") in full_context else full_context), "context": full_context,
                    "evidence_ids": ids, "support_status": state,
                    "requirement_evidence_ids": requirement_ids,
                    "source_terms": list(dict.fromkeys(match.group() for term in sorted(focus)
                                      for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", full_context, re.I))),
                    "qa_status": raw.get("qa_status", "not_run")}
            cells.append(cell)
            facts["cells"].append(cell)
        _relations(facts, cells, focus, category, index)
        if requirement_ids and 1 < len(cells) <= 3:
            for first, second in combinations(cells, 2):
                _advantages(facts, first, second, focus, category, index, requirement_ids)
    return facts


def _requirement_ids(snapshot, row, citations, criterion):
    """Directional role relevance requires a literal parsed JD criterion/citation."""
    registered = {item["criterion"].casefold(): item for item in jd_criteria(snapshot)}
    if criterion.casefold() not in registered:
        return []
    return [identity for identity in row.get("requirement_evidence_ids", [])
            if identity in citations and citations[identity]["owner"] == snapshot.jd_id
            and citations[identity]["owner_kind"] == "jd"
            and criterion.casefold() in citations[identity]["quote"].casefold()
            and not re.search(r"\b(?:no|not|never|without)\b", citations[identity]["quote"], re.I)]


def _degree_matches_requirement(cell, criterion):
    aliases = ((r"bachelor|bsc|b\.sc", r"\b(?:bachelor(?:'s)?|bsc|b\.sc)\b"),
               (r"master|msc|m\.sc", r"\b(?:master(?:'s)?|msc|m\.sc)\b"),
               (r"phd|ph\.d|doctorate", r"\b(?:phd|ph\.d|doctorate)\b"))
    for requirement, source in aliases:
        if re.search(requirement, criterion, re.I):
            return bool(re.search(source, cell["context"], re.I))
    return bool(re.search(_DEGREE, cell["context"], re.I))


def _advantages(facts, first, second, focus, category, index, requirement_ids):
    """Evidence-direction claims, never a hiring verdict or a new ranking."""
    a, b = first["support_status"], second["support_status"]
    stronger = weaker = basis = None
    if {a, b} == {"explicit_positive", "explicit_negative"}:
        stronger, weaker = (first, second) if a == "explicit_positive" else (second, first)
        basis = "explicit_contrast"
    elif {a, b} == {"explicit_positive", "not_found_in_retrieved_evidence"}:
        stronger, weaker = (first, second) if a == "explicit_positive" else (second, first)
        basis = "coverage"
    elif category == "DEGREE" and {a, b} == {"explicit_positive", "qualified"}:
        stronger, weaker = (first, second) if a == "explicit_positive" else (second, first)
        basis = "completion"
    elif category in {"SKILL", "EXPERIENCE_YEARS"} and a == b == "explicit_positive":
        values = [_duration(cell["context"], focus) for cell in (first, second)]
        if all(value is not None for value in values) and values[0] != values[1]:
            stronger, weaker = (first, second) if values[0] > values[1] else (second, first)
            basis = "duration"
    if stronger is None or category == "DEGREE" and not _degree_matches_requirement(stronger, first["criterion"]):
        return
    if basis == "explicit_contrast":
        # A denial of commercial Python cannot negate all Python experience;
        # the affirmative source must establish the same registered modifier.
        for unit in _sentences(weaker["context"]):
            if not any(_negative(unit, term) for term in focus):
                continue
            for modifier in ("commercial", "professional", "production"):
                if (_matches(modifier, unit) and modifier not in focus
                        and not any(_matches(modifier, positive) and all(_matches(term, positive) for term in focus)
                                    for positive in _sentences(stronger["context"]))):
                    return
    labels = facts["labels"]
    lead, other, criterion = labels[stronger["owner"]], labels[weaker["owner"]], first["criterion"]
    if basis == "coverage":
        text = (f"On the JD's {criterion} criterion, {lead} has stronger documented evidence: "
                f"the CV states “{stronger['context']}”; {other}'s retrieved CV evidence does not establish this criterion. "
                "That is a source-coverage difference, not proof that the candidate lacks the qualification.")
    elif basis == "duration":
        high, low = _duration(stronger["context"], focus), _duration(weaker["context"], focus)
        text = (f"On the JD's {criterion} criterion, {lead} reports more explicitly linked tenure "
                f"({high:g} versus {low:g} months in {other}'s CV). More tenure alone does not establish proficiency or overall suitability.")
    elif basis == "completion":
        text = (f"For the JD's {criterion} criterion, {lead} states “{stronger['context']}”, "
                f"whereas {other} states “{weaker['context']}”. The documented completion status favors {lead} on this criterion; "
                "a higher degree title or enrollment does not establish a completed qualification.")
    else:
        text = (f"On the JD's {criterion} criterion, {lead} has stronger affirmative source evidence: "
                f"“{stronger['context']}”, compared with {other}'s qualified negative statement “{weaker['context']}”.")
    evidence_ids = list(dict.fromkeys(requirement_ids + stronger["evidence_ids"] + weaker["evidence_ids"]))
    advantage = {"id": "advantage:" + _id(index, basis, stronger["id"], weaker["id"]),
                 "kind": "comparison_advantage", "participants": [stronger["owner"], weaker["owner"]],
                 "stronger_owner": stronger["owner"], "weaker_owner": weaker["owner"], "basis": basis,
                 "criterion": criterion, "cell_ids": [stronger["id"], weaker["id"]],
                 "requirement_evidence_ids": list(requirement_ids), "evidence_ids": evidence_ids, "text": text}
    facts["advantages"].append(advantage)
    if facts.get("intent") == "comparison_explanation":
        facts["relations"].append(advantage)


def _relations(facts, cells, focus, category, index):
    labels = facts["labels"]
    def add(kind, selected, text):
        facts["relations"].append({"id": "relation:" + _id(index, kind, *(cell["id"] for cell in selected)),
                                   "kind": kind, "participants": [cell["owner"] for cell in selected],
                                   "cell_ids": [cell["id"] for cell in selected], "text": text})
    for cell in cells:
        state, label = cell["support_status"], labels[cell["owner"]]
        criterion = cell["criterion"]
        if state in {"not_found_in_retrieved_evidence", "partial_source", "unavailable"}:
            message = ("the redacted source is unavailable" if state == "unavailable" else
                       "the evidence is partial" if state == "partial_source" else
                       "I could not establish this from the retrieved CV evidence")
            add("evidence_incomplete", [cell], f"{label} — {criterion}: {message}; this does not establish absence.")
            facts["relations"][-1].update(criterion=criterion, support_status=state)
        elif state == "conflicting":
            add("source_conflict", [cell], f"{label}'s retrieved {criterion.casefold()} statements are inconsistent or describe different qualifications; the source needs review.")
    if len(cells) < 2 or len(cells) > 3:
        return
    for first, second in combinations(cells, 2):
        _pair_relations(facts, [first, second], focus, category, add)


def _pair_relations(facts, cells, focus, category, add):
    labels = facts["labels"]
    first, second = cells
    a, b = first["support_status"], second["support_status"]
    names = f"{labels[first['owner']]} and {labels[second['owner']]}"
    topic = first.get("topic") or " ".join(sorted(focus)) or first["criterion"].casefold()
    if a == b == "explicit_positive" and focus:
        add("both_documented", cells, f"{names} both have CV statements about {topic}.")
    elif a == b == "explicit_positive" and category == "DEGREE":
        add("both_documented", cells, f"{names} both state completed qualifications; this does not establish the same qualification or level.")
    elif {a, b} == {"explicit_positive", "explicit_negative"}:
        positive, negative = (first, second) if a == "explicit_positive" else (second, first)
        add("explicit_contrast", cells,
            f"{labels[positive['owner']]} reports {topic} evidence (“{positive['context']}”); {labels[negative['owner']]}'s CV explicitly negates it (“{negative['context']}”).")
    if category == "DEGREE" and {a, b} == {"explicit_positive", "qualified"}:
        completed, pending = (first, second) if a == "explicit_positive" else (second, first)
        add("completion_status_differs", cells,
            f"{labels[completed['owner']]} states a completed qualification (“{completed['context']}”), while {labels[pending['owner']]} lists a qualification with an incomplete or ongoing status (“{pending['context']}”).")
    if category in {"EXPERIENCE_YEARS", "SKILL"} and a == b == "explicit_positive":
        values = [_duration(cell["context"], focus) for cell in cells]
        if all(value is not None for value in values) and values[0] != values[1]:
            high, low = (0, 1) if values[0] > values[1] else (1, 0)
            add("duration_greater", cells,
                f"{labels[cells[high]['owner']]} reports more {topic} tenure than {labels[cells[low]['owner']]} ({values[high]:g} versus {values[low]:g} months, as stated in the CVs).")
