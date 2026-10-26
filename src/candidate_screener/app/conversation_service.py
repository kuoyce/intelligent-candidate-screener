"""Conversation over the existing source-owned reader and checked comparison facts."""
from __future__ import annotations

import copy
import json
import math
import re
import time

from candidate_screener.annotation.redact import redact_text
from . import portal_qa as guidance
from .conversation_schema import frame_for, evidence_descriptors
from .portal_extractive import _documents, _citation, _visible, PortalExtractiveService, shared_qa_client


_COMPARISON = re.compile(r"\b(?:compar\w*|contrast|versus|stack up|these two|both candidates)\b", re.I)
_LOOKUP = re.compile(r"\b(?:where does|where did|show (?:me )?(?:the )?source|evidence for that|source for that|citation for that)\b", re.I)
_OPERATIONS = re.compile(r"\b(?:email|call|contact|schedule|appointment|interview)\b|\b(?:prepare|generate|send|save|download|open)\b.*\b(?:report|shortlist|package|cv|resume|session)\b", re.I)
_DIRECTED = re.compile(
    r"^\s*(?:explain\s+)?why\s+(?:is\s*)?(?:candidate\s*|applicant\s*|rank\s*)?#?\s*(?P<subject>\d{1,2})(?!\d|\.\d)\s*(?:is\s*)?"
    r"(?P<relation>(?:better|stronger|more\s+qualified|more\s+suitable)\s*than|ranked\s*(?:above|higher\s+than))\s*"
    r"(?:candidate\s*|applicant\s*|rank\s*)?#?\s*(?P<reference>\d{1,2})(?!\d|\.\d)(?P<tail>.*)$", re.I)


def _ranks(snapshot, owners):
    ranks = {row["cv_id"]: row["rank"] for row in snapshot.candidates}
    return " and ".join(str(ranks[owner]) for owner in owners)


def _topic(question):
    """Keep criterion wording while removing only conversation/reference scaffolding."""
    text = question
    text = re.sub(r"\b(?:for|against)\s+(?:(?:this|the|current|active)\s+)?(?:role|job description|job|jd)\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:compare|contrast|add|include|remove|exclude)\s+(?:candidates?\s*)?\d+(?:\s*(?:and|,|with|versus|vs)\s*(?:candidates?\s*)?\d+)*", " ", text, flags=re.I)
    text = re.sub(r"\b(?:candidates?|applicants?)\s*\d+", " ", text, flags=re.I)
    text = re.sub(r"\b(?:what about|focus on|how do|stack up|these two|their|them|both|compare|contrast|now|please)\b", " ", text, flags=re.I)
    return " ".join(text.strip(" .?!,:;").split())


def _directed_question(snapshot, question, previous):
    """Explicit comparative syntax, never free numbers such as years of tenure."""
    match = _DIRECTED.match(question)
    if match is None or re.match(r"\s*(?:(?:years?|yrs?|months?|weeks?|days?)\b|%)", match["tail"], re.I):
        return None
    by_rank = {row["rank"]: row["cv_id"] for row in snapshot.candidates}
    first, second = int(match["subject"]), int(match["reference"])
    if first not in by_rank or second not in by_rank or first == second:
        return {"invalid": True}
    owners = [by_rank[first], by_rank[second]]
    topic = re.sub(r"^\s*(?:on|in|about|regarding)\s+", "", match["tail"].strip(" .?!"), flags=re.I)
    topic = _topic(topic)
    criteria = previous.get("criteria", [])
    if not topic and set(owners) == set(previous.get("selected_candidates", [])) and len(criteria) == 1:
        topic = criteria[0]
    return {"owners": owners, "topic": topic,
            "relation": "ranked_above" if match["relation"].lower().startswith("ranked") else "better"}


def _blank(snapshot, state, text, status="clarification"):
    return {"text": text, "citations": [], "claims": [], "comparison": [],
            "context": state, "mode": "conversation_v2", "generation_mode": "controlled",
            "status": status, "model": None, "suggested_actions": [], "entity_status": "not_run", "qa_status": "not_run"}


class ConversationService:
    """No transcript/global cache: all dialogue is explicit and snapshot scoped."""
    def __init__(self, base, composer=None, *, turn_timeout=30.):
        if isinstance(turn_timeout, bool) or not isinstance(turn_timeout, (int, float)) or not math.isfinite(turn_timeout) or not 0 < turn_timeout <= 30:
            raise ValueError("Conversation turn timeout must be finite and within 30 seconds")
        self.base, self.composer = base, composer
        self.turn_timeout = float(turn_timeout)

    def _reader_for(self, deadline):
        if not isinstance(self.base, PortalExtractiveService):
            return self.base
        client = self.base.qa_client if self.base.qa_client is not None else shared_qa_client()
        return PortalExtractiveService(self.base.entity_service, _DeadlineReader(client, deadline))

    def prepare(self, *args, **kwargs):
        method = getattr(self.base, "prepare", None)
        return method(*args, **kwargs) if method else None

    def _compose(self, snapshot, result, question, deadline=None):
        facts = self._facts(snapshot, result)
        selected = None
        result["generation_mode"] = "controlled"
        result["generation_provenance"] = {"composer": "application", "schema_version": 2}
        requested = set(result["context"]["selected_candidates"])
        relations = facts.get("relations", [])
        covered = {owner for relation in relations for owner in relation["participants"]}
        balanced = bool(relations) and requested <= covered
        if self.composer is not None and 1 < len(requested) <= 3 and not balanced:
            result["generation_provenance"]["composition_skipped"] = "no_balanced_relations"
        if self.composer is not None and 1 < len(requested) <= 3 and balanced:
            try:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Conversation inference budget exhausted")
                bounded = getattr(self.composer, "compose_before", None)
                composed = (bounded(question, result["context"], facts, deadline)
                            if callable(bounded) and deadline is not None else
                            self.composer.compose(question, result["context"], facts))
                selected = composed["relation_ids"]
                allowed = {item["id"] for item in facts.get("relations", [])}
                if not isinstance(selected, list) or any(not isinstance(item, str) or item not in allowed for item in selected):
                    raise ValueError("Composer returned an unknown relation")
                result["generation_mode"] = composed.get("mode", "qwen_answer_plan")
                result["model"] = composed.get("model")
                result["generation_provenance"] = composed.get("provenance", {})
            except Exception as error:
                selected = None
                result["generation_mode"] = "qwen_fallback"
                reason = getattr(error, "reason", None)
                if reason not in {"deadline_exceeded", "worker_unavailable", "worker_failed", "worker_invalid_output", "provider_unavailable",
                                   "invalid_json", "unsupported_fields", "invalid_relation_ids", "participant_omission",
                                   "qualification_omission", "invalid_response_size"}:
                    reason = type(error).__name__
                result["generation_provenance"] = {"composer": "application", "fallback_reason": reason}
        result["text"] = self._render(snapshot, result, facts, selected)
        if len(result["context"]["selected_candidates"]) > 3:
            result["text"] += " Select two or three candidates or a specific criterion for a shorter discussion."
        result["mode"] = "conversation_v2"
        result["answer_version"] = 2
        # Relations contain supported conclusions; original evidence stays in the
        # existing citation/claim/table records rather than duplicating excerpts.
        result["comparison_relations"] = facts.get("relations", [])
        result["candidate_labels"] = facts.get("labels", {})
        result["show_comparison_table"] = (len(result.get("comparison", [])) > 1
                                           or result["context"].get("intent") == "comparison_explanation")
        return result

    @staticmethod
    def _facts(snapshot, result):
        from .comparison_facts import build_facts
        facts = build_facts(snapshot, result)
        if result["context"].get("intent") != "comparison_explanation":
            facts["relations"] = [row for row in facts["relations"] if row["kind"] != "comparison_advantage"]
        return facts

    @staticmethod
    def _render(snapshot, result, facts, selected=None):
        from .conversation_render import render_comparison
        state = result["context"]
        result["citation_labels"] = facts.get("citation_labels", {})
        if state.get("intent") == "comparison_explanation":
            from .conversation_render import render_explanation
            return render_explanation(snapshot, facts, state["explanation_subject"], state["explanation_reference"],
                                      selected_relation_ids=selected, ranked_above=state["explanation_relation"] == "ranked_above")
        return render_comparison(snapshot, facts, selected_relation_ids=selected, include_source_context=False)

    def _broad_criteria(self, snapshot, result, criterion_topic=None):
        """Give explicit JD requirements balanced source rows without QA startup."""
        from .comparison_facts import jd_criteria
        criteria = jd_criteria(snapshot)[:3]
        if criterion_topic:
            stop = {"experience", "years", "year", "tenure", "skills", "skill", "knowledge", "more", "less"}
            terms = guidance._terms(criterion_topic, "GENERAL") - guidance._BROAD_WORDS - stop
            category = guidance._category(criterion_topic, None)
            criteria = [item for item in criteria
                        if terms & guidance._terms(item["topic"], "GENERAL")
                        or category in {"DEGREE", "CERTIFICATION"} and item["category"] == category
                        or not terms and category in {"SKILL", "KNOWLEDGE"} and item["category"] == category]
        if not criteria:
            return result
        owners = result["context"]["selected_candidates"]
        documents = _documents(snapshot, owners)
        # Parsed literal JD criteria replace the six coarse extraction-category
        # rows: a missing generic "Skill" cell must not contradict found Python.
        citations, claims = {}, []
        result["comparison"], result["claims"] = [], claims
        result.update(comparison_scope="recognized_jd_criteria", criteria_exhaustive=False)
        for criterion in criteria:
            cells = {}
            for owner, document in documents.items():
                relevant = PortalExtractiveService._relevant(document, criterion["category"], criterion["topic"], [], broad=False)
                cells[owner] = self._criterion_cell(snapshot, document, relevant, citations, claims)
                cells[owner]["source_status"] = document["status"]
            jd = cells.pop(snapshot.jd_id)
            result.setdefault("comparison", []).append({"category": criterion["category"],
                "criterion": criterion["criterion"], "topic": criterion["topic"],
                "requirement": _visible(jd), "requirement_status": jd["status"],
                "requirement_source_status": documents[snapshot.jd_id]["status"],
                "requirement_evidence_ids": jd["evidence_ids"], "answers": cells})
        result["citations"] = list(citations.values())
        if result.get("status") == "insufficient_evidence" and any(item["owner"] in owners for item in citations.values()):
            result["status"] = "answered"
        return result

    @staticmethod
    def _criterion_cell(snapshot, document, relevant, citations, claims):
        """Retain the reader's two passage budget, including contradictory text."""
        if not relevant:
            return PortalExtractiveService._source_cell(snapshot, document, [], citations, claims)
        merged = []
        for score, start, unit, mentions in sorted(relevant[:2], key=lambda item: item[2][0]):
            a, b, _ = unit
            if merged and a < merged[-1][2][1]:
                previous = merged.pop()
                left, right = previous[2][0], max(previous[2][1], b)
                # An overlapping context is replaced by its complete source
                # union, rather than duplicating text or clipping a qualifier.
                merged.append((max(score, previous[0]), left,
                               (left, right, document["source"][left:right]), previous[3] + mentions))
            else:
                merged.append((score, start, unit, mentions))
        parts = [PortalExtractiveService._source_cell(snapshot, document, [row], citations, claims) for row in merged]
        if len(parts) == 1:
            return parts[0]
        contexts = list(dict.fromkeys(part["context"] for part in parts if part["context"]))
        result = dict(parts[0])
        result.update(context=" ".join(contexts), text=" ".join(contexts),
                      evidence_ids=list(dict.fromkeys(identity for part in parts for identity in part["evidence_ids"])))
        return result

    def _bounded(self, snapshot, result):
        """Drop whole evidence cells, never source qualifiers or participants."""
        while len(json.dumps(result, allow_nan=False).encode()) > 23_900:
            choices = [(len(json.dumps(cell)), cell) for row in result.get("comparison", [])
                       for cell in row["answers"].values() if cell.get("evidence_ids")]
            if not choices:
                raise ValueError("Conversation metadata exceeds its storage budget")
            _, cell = max(choices, key=lambda pair: pair[0])
            cell.clear()
            cell.update(text="Evidence omitted to fit this saved answer.", status="budget_limited",
                        evidence_ids=[], context="", qa_status="budget_limited")
            used = set()
            for row in result["comparison"]:
                used.update(row.get("requirement_evidence_ids", []))
                for value in row["answers"].values():
                    used.update(value.get("evidence_ids", []))
            result["citations"] = [item for item in result.get("citations", []) if item["evidence_id"] in used]
            result["claims"] = [item for item in result.get("claims", []) if item["evidence_id"] in used]
            facts = self._facts(snapshot, result)
            result["text"] = self._render(snapshot, result, facts)
            result["comparison_relations"] = facts.get("relations", [])
            result["status"] = result["comparison_status"] = "partial"
            attempted_mode = result.get("generation_provenance", {}).get("attempted_mode", result.get("generation_mode"))
            result["generation_mode"] = "controlled"
            result["model"] = None
            result["generation_provenance"] = {"composer": "application", "budget_limited": True,
                                                "fallback_reason": "answer_storage_budget",
                                                "attempted_mode": attempted_mode}
            result["context"]["last_evidence"] = evidence_descriptors(result["citations"])
        return result

    def compare(self, snapshot, candidate_ids):
        deadline = time.monotonic() + self.turn_timeout
        result = self.base.compare(snapshot, candidate_ids)
        state = frame_for(snapshot, result.get("context"))
        state.update(intent="comparison", criteria=[row["category"] for row in result.get("comparison", [])])
        result["context"] = state
        result = self._broad_criteria(snapshot, result)
        result = self._compose(snapshot, result, "Compare selected candidates against the job requirements", deadline)
        state["last_evidence"] = evidence_descriptors(result.get("citations", []))
        return self._bounded(snapshot, result)

    def _lookup(self, snapshot, question, state):
        """Reopen original coordinates, never substitute newly retrieved evidence."""
        descriptors = state.get("last_evidence", [])
        owners = list(dict.fromkeys(item.get("owner") for item in descriptors
                                   if item.get("owner_kind") == "cv"))
        if any(owner not in {row["cv_id"] for row in snapshot.candidates} for owner in owners):
            return _blank(snapshot, state, "The earlier source is no longer available in this review.", "source_unavailable")
        documents = _documents(snapshot, owners)
        reference_ranks, unknown, _ = guidance._references(question)
        rank_ids = {row["rank"]: row["cv_id"] for row in snapshot.candidates}
        if unknown or any(rank not in rank_ids for rank in reference_ranks):
            return _blank(snapshot, state, "Please choose a candidate from this job's saved top ten.")
        requested = {rank_ids[rank] for rank in reference_ranks}
        requested.update(row["cv_id"] for row in snapshot.candidates
                         if re.search(r"(?<![\w-])" + re.escape(row["cv_id"]) + r"(?![\w-])", question, re.I))
        if requested and not requested <= set(owners):
            return _blank(snapshot, state, "That candidate was not part of the earlier cited answer. Ask a new question about their CV.")
        valid = []
        for item in descriptors:
            document = documents.get(item.get("owner"))
            a, b = item.get("start"), item.get("end")
            if (document is None or document["status"] == "unavailable"
                    or item.get("snapshot_id") != snapshot.snapshot_id
                    or item.get("document_digest") != document["digest"]
                    or type(a) is not int or type(b) is not int or not 0 <= a < b <= len(document["source"])):
                return _blank(snapshot, state, "The earlier evidence changed or is unavailable. Refresh the review before reopening it.", "source_unavailable")
            citation = _citation(snapshot, document, (a, b, document["source"][a:b]))
            if citation["evidence_id"] != item.get("evidence_id"):
                return _blank(snapshot, state, "The earlier evidence reference could not be validated.", "source_unavailable")
            valid.append(citation)
        if requested:
            valid = [item for item in valid if item["owner"] in requested]
        if not valid:
            return _blank(snapshot, state, "Which earlier statement would you like me to show evidence for?")
        terms = set(re.findall(r"[a-z]+", question.casefold())) - {"where", "does", "did", "it", "say", "that", "the", "is", "source", "show", "me", "evidence", "for"}
        if terms:
            scores = [len(terms & set(re.findall(r"[a-z]+", item["quote"].casefold()))) for item in valid]
            best = max(scores)
            if best:
                valid = [item for item, score in zip(valid, scores) if score == best]
        result = _blank(snapshot, state, "\n\n".join("The earlier source states: “" + item["quote"] + "”" for item in valid), "answered")
        result.update(citations=valid, claims=[{"text": item["quote"], "evidence_id": item["evidence_id"], "kind": "source_only"} for item in valid],
                      generation_mode="citation_reopen")
        if len(json.dumps(result, allow_nan=False).encode()) > 23_900:
            return _blank(snapshot, state, "Which earlier statement should I reopen? Name the candidate and qualification so I can show the complete source context.")
        return result

    def answer(self, snapshot, question, context=None):
        deadline = time.monotonic() + self.turn_timeout
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Ask a question of 1 to 2,000 characters")
        question = redact_text(question.strip())
        state = frame_for(snapshot, context)
        previous = copy.deepcopy(state)
        if _LOOKUP.search(question):
            return self._lookup(snapshot, question, state)
        if _OPERATIONS.search(question) or re.fullmatch(r"(?:hi|hello|help|thanks|thank you)[.! ]*", question, re.I):
            if re.fullmatch(r"(?:thanks|thank you)[.! ]*", question, re.I):
                return _blank(snapshot, state, "You're welcome.", "answered")
            result = self.base.answer(snapshot, question, context=state)
            result["context"] = {**state, **result.get("context", {})}
            return result
        pending = state.get("pending_task")
        current = re.sub(r"\bcompare(?=\d)", "Compare ", question, flags=re.I)
        current = re.sub(r"(?<=\d)and(?=\d)", " and ", current, flags=re.I)
        explanation = _directed_question(snapshot, current, previous)
        if explanation and explanation.get("invalid"):
            return _blank(snapshot, state, "Choose two different candidates from this job's saved top ten.")
        if explanation:
            current = "Compare candidates " + _ranks(snapshot, explanation["owners"])
            if explanation["topic"]:
                current += " on " + explanation["topic"]
        if isinstance(pending, dict) and pending.get("slot") == "candidates" and re.fullmatch(r"(?:candidates?\s*)?\d+(?:\s*(?:and|,)\s*\d+)*[. ]*", question, re.I):
            current = "Compare candidates " + question.strip(". ") + " on " + pending["question"]
        prior = state["selected_candidates"]
        for ordinal, index in (("first", 0), ("second", 1), ("third", 2)):
            if re.search(r"\b(?:the\s+)?" + ordinal + r"\s+one\b", current, re.I):
                if len(prior) <= index:
                    state.update(pending_task={"slot": "candidates", "question": question}, pending_clarification=question)
                    return _blank(snapshot, state, "Which candidate do you mean? That position is not in the active comparison.")
                current = re.sub(r"\b(?:the\s+)?" + ordinal + r"\s+one\b", prior[index], current, flags=re.I)
        owners, clarification = guidance._resolve(snapshot, current, state)
        explicit, unknown, _ = guidance._references(current)
        explicit_ids = [row["cv_id"] for row in snapshot.candidates
                        if re.search(r"(?<![\w-])" + re.escape(row["cv_id"]) + r"(?![\w-])", current, re.I)]
        has_explicit = bool(explicit or explicit_ids)
        modification = re.search(r"\b(add|include|remove|exclude)\b", current, re.I)
        if modification and has_explicit and not unknown and not clarification:
            if modification[1].casefold() in {"add", "include"}:
                owners = list(dict.fromkeys(prior + owners))
            else:
                owners = [owner for owner in prior if owner not in owners]
            clarification = None if owners else "Which candidates should I compare?"
        plural = bool(re.search(r"\b(?:their|them|both|these two)\b", current, re.I))
        if len(prior) > 1 and plural and not has_explicit and not unknown:
            owners, clarification = list(prior), None
        if (len(prior) > 1 and not has_explicit and not unknown
                and re.search(r"\b(?:who|which)\b.*\b(?:more|clearer|stronger|better|strongest)\b", current, re.I)
                and re.search(r"\b(?:on|evidence|skill|degree|qualification|deployment|python|cloud)\b", current, re.I)
                and not re.search(r"\b(?:rank|ranked|ranking|score)\b", current, re.I)):
            # A criterion comparison is distinct from asking for saved rank one.
            owners, clarification = list(prior), None
        if clarification or not owners and not re.search(r"\b(?:requirements|role require|job require|job description|jd)\b", current, re.I):
            state.update(pending_task={"slot": "candidates", "question": question}, pending_clarification=question)
            return _blank(snapshot, state, clarification or "Which candidates should I compare?")
        comparison = len(owners) > 1
        topic_input = current
        for owner in explicit_ids:
            topic_input = re.sub(r"(?<![\w-])" + re.escape(owner) + r"(?![\w-])", " ", topic_input, flags=re.I)
        topic = _topic(topic_input)
        broad_role = bool(re.search(r"\b(?:for|against)\s+(?:(?:this|the|current|active)\s+)?(?:role|job description|job|jd)\b", question, re.I))
        # Bare participant changes retain the current criterion. A real new topic
        # replaces it; do not append unrelated prior criteria to the reader input.
        if modification or (has_explicit and not topic and not broad_role):
            topic = " ".join(str(item) for item in state.get("criteria", []))
        if comparison:
            current = "Compare candidates " + _ranks(snapshot, owners) + (" on " + topic if topic else "")
        elif owners:
            current = re.sub(r"\b(?:their|them|they|his|her|he|she)\b", "the candidate", current, flags=re.I)
        duration_request = bool(re.search(r"\b(?:years?|months?|tenure|how long|how many)\b|\b(?:more|less|longer|shorter)\b.*\bexperience\b", question, re.I))
        if comparison and re.search(r"\bexperience\b", current, re.I) and not duration_request:
            # V1 routes every 'experience' question through its tenure-only
            # retrieval guard. Cloud/deployment evidence is not a duration query.
            # Keep the actual requested criterion in the v2 frame for synthesis.
            current = re.sub(r"\bexperience\b", "work evidence", current, flags=re.I)
        supplied = {**state, "selected_candidates": owners, "pending_clarification": None}
        if (broad_role or explanation) and not topic:
            supplied["category"] = "GENERAL"
        result = self._reader_for(deadline).answer(snapshot, current, context=supplied)
        updated = frame_for(snapshot, result.get("context"))
        updated.update(intent="comparison" if comparison else "candidate_qa" if owners else "requirements",
                       criteria=[topic] if topic else previous.get("criteria", []), pending_task=None)
        if explanation:
            updated.update(intent="comparison_explanation", explanation_subject=explanation["owners"][0],
                           explanation_reference=explanation["owners"][1], explanation_relation=explanation["relation"])
        if broad_role and not topic:
            updated["criteria"] = []
        result["context"] = updated
        if comparison and result.get("comparison") and (not topic or explanation):
            result = self._broad_criteria(snapshot, result, criterion_topic=topic if explanation else None)
        if comparison and result.get("status") not in {"clarification", "action_guidance"} and result.get("comparison"):
            result = self._compose(snapshot, result, question, deadline)
        if (len(owners) == 1 and result.get("status") not in {"clarification", "action_guidance"}
                and re.search(r"\b(?:why|explain)\b.*\b(?:rank\w*|first|highest|top)\b", question, re.I)):
            result = self._broad_criteria(snapshot, result)
            candidate = next(row for row in snapshot.candidates if row["cv_id"] == owners[0])
            quotes = list(dict.fromkeys(item["quote"] for item in result.get("citations", []) if item["owner"] == owners[0]))[:3]
            result["text"] = (f"Candidate {candidate['rank']} is ranked {candidate['rank']} in the saved ranking. "
                              "The source evidence below does not establish the scoring model's causal reasoning.")
            result["text"] += "\n\n" + ("The CV states: " + " ".join("“" + quote + "”" for quote in quotes)
                                           if quotes else "Relevant CV evidence was not found in the reviewed passages.")
            result.update(mode="conversation_v2", generation_mode="controlled", model=None,
                          candidate_labels={owners[0]: f"Candidate {candidate['rank']}"})
            updated["intent"] = "rank_explanation"
        updated["last_evidence"] = evidence_descriptors(result.get("citations", []))
        return self._bounded(snapshot, result)


class _DeadlineReader:
    """Per-turn adapter: no mutation of the shared worker's timeout or ownership."""
    def __init__(self, client, deadline):
        self.client, self.deadline = client, deadline

    def answer(self, question, passages):
        if time.monotonic() >= self.deadline:
            return {"status": "unavailable", "spans": [], "reason": "turn_deadline"}
        bounded = getattr(self.client, "answer_before", None)
        if callable(bounded):
            return bounded(question, passages, self.deadline)
        # Injected synthetic clients keep their two-argument contract. A custom
        # client's unbounded implementation does not gain a hard deadline here.
        return self.client.answer(question, passages)
