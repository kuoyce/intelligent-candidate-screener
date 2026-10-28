"""Opt-in model-authored prose over v2's source-owned evidence and dialogue frame."""
from __future__ import annotations

import json
import copy
import math
import re
import time

from candidate_screener.annotation.redact import redact_text
from candidate_screener.llm.base import ContextOverflow
from .conversation_context import build_context, fallback_text, scope_digest, validate_prose
from .conversation_schema import frame_for, evidence_descriptors
from .conversation_service import ConversationService, _OPERATIONS
from . import portal_qa as guidance


class ConversationV3Service:
    accepts_history = True

    def __init__(self, base, composer=None, *, turn_timeout=30.):
        if (isinstance(turn_timeout, bool) or not isinstance(turn_timeout, (int, float))
                or not math.isfinite(turn_timeout) or not 0 < turn_timeout <= 60):
            raise ValueError("V3 turn timeout must be finite and within 60 seconds")
        self.reader = ConversationService(base, composer=None, turn_timeout=min(turn_timeout, 30.))
        self.composer, self.turn_timeout = composer, float(turn_timeout)

    def prepare(self, *args, **kwargs):
        return self.reader.prepare(*args, **kwargs)

    def _controlled(self, snapshot, question, context):
        greeting = re.fullmatch(r"(?:hi|hello|hey|thanks|thank you|help)[.! ]*", question, re.I)
        unrelated = re.search(r"\b(?:apple pie|recipe|nationality|nationalities)\b", question, re.I)
        if not greeting and not unrelated:
            return None
        text = ("You're welcome." if greeting and greeting[0].strip(".! ").casefold() in {"thanks", "thank you"}
                else "Hi! What would you like to review?" if greeting
                else OUT_OF_SCOPE_REPLY)
        return {"text": text, "citations": [], "claims": [], "comparison": [],
                "context": frame_for(snapshot, context), "mode": "conversation_v3", "answer_version": 3,
                "generation_mode": "controlled", "status": "out_of_scope" if unrelated else "answered", "model": None,
                "suggested_actions": [], "entity_status": "not_run", "qa_status": "not_run",
                "generation_provenance": {"composer": "application", "scope_digest": scope_digest(snapshot)}}

    def answer(self, snapshot, question, context=None, *, history=None):
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Ask a question of 1 to 2,000 characters")
        question = redact_text(question.strip())
        deadline = time.monotonic() + self.turn_timeout
        controlled = self._controlled(snapshot, question, context)
        if controlled is not None:
            return controlled
        alignment = bool(re.search(r"\b(?:good fit|align\w*|suitab\w*)\b", question, re.I))
        relevant_work = bool(re.search(r"\brelevant\b.*\b(?:job\s+|work\s+)?experiences?\b", question, re.I))
        duration = bool(re.search(r"\b(?:years?|months?|tenure|how long|how many)\b", question, re.I))
        if (alignment or relevant_work and not duration) and not _OPERATIONS.search(question):
            state = frame_for(snapshot, context)
            owners, clarification = guidance._resolve(snapshot, question, state)
            explicit, unknown, _ = guidance._references(question)
            if (not explicit and not unknown and state["selected_candidates"]
                    and re.search(r"\b(?:these candidates|their|them|both|these two)\b", question, re.I)):
                owners, clarification = list(state["selected_candidates"]), None
            if owners and not clarification:
                result = self.reader.base.compare(snapshot, owners)
                state.update(selected_candidates=owners, category="GENERAL", criteria=[],
                             intent="comparison" if len(owners) > 1 else "candidate_qa",
                             last_question=question, pending_task=None, pending_clarification=None)
                result["context"] = state
                result = self.reader._broad_criteria(snapshot, result)
                if relevant_work:
                    work_rows = [row for row in result.get("comparison", [])
                                 if row.get("category") in {"SKILL", "KNOWLEDGE", "TITLE"}]
                    if work_rows:
                        result["comparison"] = work_rows
                        used = {identity for row in work_rows for identity in row.get("requirement_evidence_ids", [])}
                        used.update(identity for row in work_rows for cell in row["answers"].values()
                                    for identity in cell.get("evidence_ids", []))
                        result["citations"] = [item for item in result["citations"] if item["evidence_id"] in used]
                        result["claims"] = [item for item in result["claims"] if item["evidence_id"] in used]
                state["last_evidence"] = evidence_descriptors(result.get("citations", []))
                return self._finish(snapshot, question, result, deadline, history)
        retrieval_question = re.sub(r"\bcanddiate(s?)\b", r"candidate\1", question, flags=re.I)
        # A user's confirmation premise is not a retrieval constraint: asking
        # whether a degree is completed must also retrieve its ongoing status.
        # Preserve the original question for synthesis; change only reader intent.
        confirmation = re.match(r"^(?:so\s+)?(?:(?:does|has|did)\s+)?(?:candidate|applicant)\s+(\d{1,2})\b", question, re.I)
        if confirmation and re.search(r"\b(?:degree|master|bachelor|msc|bsc|phd|diploma|qualifications?)\b", question, re.I):
            retrieval_question = f"What qualifications does candidate {confirmation[1]} have?"
        result = self.reader.answer(snapshot, retrieval_question, context=context)
        return self._finish(snapshot, question, result, deadline, history)

    def compare(self, snapshot, candidate_ids):
        deadline = time.monotonic() + self.turn_timeout
        result = self.reader.compare(snapshot, candidate_ids)
        return self._finish(snapshot, "Compare selected candidates", result, deadline)

    def _finish(self, snapshot, question, result, deadline, history=None):
        result.update(mode="conversation_v3", answer_version=3)
        if result.get("generation_mode") == "citation_reopen" or result.get("status") in {
                "clarification", "action_guidance", "source_unavailable", "out_of_scope"}:
            result.setdefault("generation_provenance", {})["scope_digest"] = scope_digest(snapshot)
            return self._bounded(result)
        bundle = build_context(snapshot, result, history)
        result["retrieved_context_ids"] = [item["id"] for item in bundle["evidence"]]
        result["validated_claim_sources"] = []
        result["candidate_labels"] = bundle["labels"]
        explicit_compare = bool(re.search(r"\b(?:compare|comparison|table|contrast)\b", question, re.I))
        explanation = result.get("context", {}).get("intent") == "comparison_explanation"
        comparison = result.get("context", {}).get("intent") == "comparison"
        result["show_comparison_table"] = (explicit_compare or comparison or explanation) and len(bundle["selected_candidates"]) > 1
        result["comparison_table_expandable"] = explanation and not explicit_compare
        result["comparison_table_available"] = bool(result.get("comparison"))
        result["generation_provenance"] = {"composer": "application", "scope_digest": bundle["scope_digest"],
                                            "support_check": "bounded_rule_checked"}
        reason = "provider_unavailable"
        if len(bundle["selected_candidates"]) > 3:
            reason = "participant_budget"
        elif self.composer is not None:
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Turn deadline exceeded")
                bounded = getattr(self.composer, "compose_before", None)
                output = (bounded(question, result["context"], bundle, deadline) if callable(bounded)
                          else self.composer.compose(question, result["context"], bundle))
                if time.monotonic() >= deadline:
                    raise TimeoutError("Turn deadline exceeded")
                mappings, reason = validate_prose(output.get("text"), bundle)
                if reason is None:
                    result.update(text=output["text"], generation_mode="qwen_prose", model=output.get("model"),
                                  validated_claim_sources=mappings)
                    result["generation_provenance"].update(output.get("provenance", {}))
                    result["generation_provenance"].update(composer="qwen", scope_digest=bundle["scope_digest"],
                                                           support_check="bounded_rule_checked")
            except Exception as error:
                reason = getattr(error, "reason", "context_overflow" if isinstance(error, ContextOverflow)
                                 else "deadline_exceeded" if isinstance(error, TimeoutError) else "provider_unavailable")
                if reason not in {"deadline_exceeded", "provider_unavailable", "worker_unavailable", "worker_failed",
                                  "worker_invalid_output", "invalid_response_size", "invalid_prose_format", "context_overflow",
                                  "non_prose_output", "invalid_context"}:
                    reason = "provider_unavailable"
        if reason:
            if len(bundle["selected_candidates"]) > 1:
                # Rebuild source-owned facts rather than reuse the reader's text
                # or the rejected model output. The v2 renderer preserves full
                # comparison caveats and directed counterevidence.
                facts = self.reader._facts(snapshot, result)
                text = self.reader._render(snapshot, result, facts)
            else:
                text = fallback_text(bundle, reason)
            result.update(text=text, generation_mode="qwen_fallback", model=None)
            result["generation_provenance"]["fallback_reason"] = reason
        if len(bundle["selected_candidates"]) > 1 and (comparison or explanation):
            self._display_comparison(snapshot, result)
        return self._bounded(result)

    def _display_comparison(self, snapshot, result):
        """Add broad coverage after synthesis; all displayed quotes stay source-owned."""
        from .comparison_facts import _citations

        selected = result["context"]["selected_candidates"]
        broad = self.reader.base.compare(snapshot, selected)
        owners = {snapshot.jd_id, *selected}
        citations = {identity: item for identity, item in _citations(
            snapshot, {"citations": [*result.get("citations", []), *broad.get("citations", [])]}).items()
            if item["owner"] in owners}

        def authorized_ids(ids, owner):
            return [identity for identity in ids
                    if identity in citations and citations[identity]["owner"] == owner]

        def display_row(original, scope):
            row = copy.deepcopy(original)
            row["table_scope"] = scope
            if scope != "Broad evidence":
                row["table_criterion"] = str(row.get("criterion") or row.get("topic")
                                             or " ".join(result["context"].get("criteria", [])))
            ids = authorized_ids(row.get("requirement_evidence_ids", []), snapshot.jd_id)
            row["requirement_evidence_ids"] = ids
            if ids:
                row["requirement"] = " ".join(dict.fromkeys(citations[key]["quote"] for key in ids))
            else:
                row["requirement"] = "JD evidence not found in reviewed text."
                if original.get("requirement_evidence_ids"):
                    row["requirement_status"] = "source_unavailable"
            for owner in list(row.get("answers", {})):
                if owner not in selected:
                    del row["answers"][owner]
                    continue
                cell = row["answers"][owner]
                ids = authorized_ids(cell.get("evidence_ids", []), owner)
                context = " ".join(dict.fromkeys(citations[key]["quote"] for key in ids))
                if context:
                    # Complete source context is the display value, retaining
                    # qualifiers even when a QA span contains only a noun phrase.
                    cell.update(text=context, context=context, evidence_ids=ids)
                else:
                    status = "source_unavailable" if cell.get("evidence_ids") else cell.get("status", "not_found")
                    text = ("Source evidence unavailable." if status == "source_unavailable" else
                            "Evidence omitted by the storage limit; review supporting sources." if status in {"omitted", "budget_limited"} else
                            "Not found in reviewed text; this does not establish absence.")
                    cell.update(text=text, context="", evidence_ids=[], status=status, qa_status="not_run")
            return row

        broad_rows = [display_row(row, "Broad evidence") for row in broad.get("comparison", [])]
        if result.get("comparison") == broad.get("comparison"):
            # Save just one copy when synthesis already used all broad rows.
            result["comparison"] = broad_rows
        else:
            extra = [display_row(row, "JD criterion" if "criterion" in row else "Focused evidence")
                     for row in result.get("comparison", []) if row not in broad.get("comparison", [])]
            result["comparison_rows"] = [*broad_rows, *extra]
        result["citations"] = list(citations.values())
        result["claims"] = [item for item in result.get("claims", [])
                            if item.get("evidence_id") in citations]
        result["context"]["last_evidence"] = evidence_descriptors(result["citations"])
        result["comparison_table_available"] = bool(broad_rows)
        result["comparison_table_status"] = broad.get("comparison_status", "partial")

    @staticmethod
    def _bounded(result):
        # Avoid copying the prompt/history into persisted turns. Preserve existing
        # table/citation budgets; remove derived duplication before storing.
        if len(json.dumps(result, ensure_ascii=False).encode()) > 23_900:
            result.pop("comparison_relations", None)
        if len(json.dumps(result, ensure_ascii=False).encode()) > 23_900:
            result["validated_claim_sources"] = []
            result.update(text="The evidence is too large for a saved reply. Please narrow the criteria.",
                          generation_mode="qwen_fallback", model=None)
            result["generation_provenance"]["fallback_reason"] = "answer_storage_budget"
            result.pop("retrieved_context_ids", None)
        if len(json.dumps(result, ensure_ascii=False).encode()) > 23_900:
            # Drop complete table/citation records together, never truncate source
            # qualifiers. The message makes this loss explicit to the reviewer.
            result.update(comparison=[], comparison_rows=[], citations=[], claims=[], status="partial",
                          show_comparison_table=False, comparison_table_available=False,
                          comparison_table_status="partial")
            result["validated_claim_sources"] = []
            result["context"]["last_evidence"] = []
        if len(json.dumps(result, ensure_ascii=False).encode()) > 23_900:
            raise ValueError("Conversation metadata exceeds its storage budget")
        return result
OUT_OF_SCOPE_REPLY = "I am not able to answer this question. It is out of my scope."

