"""Bounded request interpretation and canonical evidence rendering.

The model selects backend references; it cannot author applicant facts. Every
failure leaves a deterministic answer available without disclosing provider errors.
"""
from __future__ import annotations

import json
import re

from candidate_screener.annotation.redact import redact_text
from candidate_screener.app import intents
from candidate_screener.app.chat_context import FACT_FIELDS, build_context, messages_with_budget
from candidate_screener.app.chat_prompts import GROUND, INTERPRET, INTERPRET_EXAMPLES
from candidate_screener.app.chat_schema import ChatAnswer, ChatState
from candidate_screener.llm.base import redact_content

ARG_COUNTS = {"top": (0, 1), "explain": (1, 1), "missing": (1, 1), "entities": (1, 1),
              "source": (1, 1), "compare": (2, 2), "why": (2, 2), "skill": (1, 1), "years": (1, 1), "help": (0, 0)}


def _complete(provider, messages):
    output = provider.complete(messages)
    if not isinstance(output, str) or len(output) > 16000:
        raise ValueError("Invalid or oversized output")
    if provider.count_tokens([{"role": "assistant", "content": output}]) > provider.max_output_tokens:
        raise ValueError("Output budget exceeded")
    return json.loads(redact_content(output))


def _intent(value):
    if not isinstance(value, dict) or set(value) != {"kind", "args"}:
        raise ValueError("Invalid request")
    kind, args = value["kind"], value["args"]
    if kind not in ARG_COUNTS or not isinstance(args, list):
        raise ValueError("Unsupported request")
    lo, hi = ARG_COUNTS[kind]
    if not lo <= len(args) <= hi or any(isinstance(a, bool) or not isinstance(a, (str, int)) or len(str(a)) > 100 for a in args):
        raise ValueError("Invalid arguments")
    args = tuple(str(a) for a in args)
    if kind in {"top", "years"} and args and (not args[0].isdigit() or not 0 <= int(args[0]) <= 100):
        raise ValueError("Invalid count")
    return intents.Intent(kind, args)


def _select(intent, results):
    if intent.kind in {"explain", "missing", "entities", "source", "compare", "why"}:
        selected = [intents.find(list(results), a) for a in intent.args]
        if any(r is None for r in selected):
            return None
        return selected
    if intent.kind == "skill":
        skill = intent.args[0].lower()
        return [r for r in results if any(skill in (e.get("requirement", "") + " " + (e.get("quote") or "")).lower()
                                               for e in r.evidence if e.get("status") != "missing")][:10]
    if intent.kind == "years":
        return [r for r in results if getattr(r, "years_stated", None) is not None and r.years_stated >= int(intent.args[0])][:10]
    return list(results[:min(int(intent.args[0]) if intent.args else 5, 10)])


def _display_fact(value):
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _canonical(r):
    return (f"#{r.rank} ({r.cv_id}) · fit call: {r.fit_call or 'not assessed'}; "
            f"classification: {getattr(r, 'classification_status', 'unavailable')}; "
            f"reranking: {getattr(r, 'rerank_status', 'unavailable')}; "
            f"retrieval score: {_display_fact(getattr(r, 'retrieval_score', None))}; "
            f"cross-encoder score: {_display_fact(getattr(r, 'rerank_score', None))}. "
            "Relative strength is a pool percentile, not a fit probability.")


def _render(value, snapshot, selected, citations, intent):
    if not isinstance(value, dict) or set(value) != {"snapshot_id", "blocks"} or value["snapshot_id"] != snapshot.snapshot_id:
        raise ValueError("Stale or invalid answer")
    blocks = value["blocks"]
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 30:
        raise ValueError("Invalid blocks")
    owners = {str(r.cv_id): r for r in selected}
    covered = set()
    evidence_covered = set()
    lines, used = [], []
    for block in blocks:
        if not isinstance(block, dict) or block.get("candidate_id") not in owners:
            raise ValueError("Unknown candidate")
        cid = block["candidate_id"]
        covered.add(cid)
        if block.get("kind") == "fact":
            if set(block) != {"kind", "candidate_id", "field"} or block["field"] not in FACT_FIELDS:
                raise ValueError("Invalid fact")
            lines.append(f"{cid} · {block['field']}: {_display_fact(getattr(owners[cid], block['field'], None))}")
        elif block.get("kind") == "evidence":
            if set(block) not in ({"kind", "candidate_id", "evidence_id"}, {"kind", "candidate_id", "evidence_id", "quote"}):
                raise ValueError("Invalid evidence")
            citation = citations.get(block.get("evidence_id"))
            if citation is None or citation.candidate_id != cid or ("quote" in block and block["quote"] != citation.quote):
                raise ValueError("Invalid citation ownership or quote")
            evidence_covered.add(cid)
            requirement = next(redact_text(e.get("requirement", ""))
                               for index, e in enumerate(owners[cid].evidence)
                               if f"{cid}:e{index}" == citation.evidence_id)
            if citation.quote:
                lines.append(f'{cid} · {requirement}: “{citation.quote}” [{citation.evidence_id}]')
                used.append(citation)
            else:
                lines.append(f"{cid} · {requirement}: no evidence found in the supplied material.")
        else:
            raise ValueError("Unsupported block")
    if intent.kind in {"explain", "missing", "compare", "why", "skill"}:
        required = {c.candidate_id for c in citations.values()}
        if not required.issubset(evidence_covered) or not set(owners).issubset(covered):
            raise ValueError("Answer omits requested candidate evidence")
    return lines, tuple(used)


class ChatService:
    def __init__(self, provider=None):
        self.provider = provider

    def answer(self, snapshot, question: str, state: ChatState | None = None) -> ChatAnswer:
        state = state or ChatState()
        if state.snapshot_id != snapshot.snapshot_id:
            state = ChatState(snapshot.snapshot_id)
        question = redact_text(question)
        if not question.strip() or len(question) > 2000:
            return ChatAnswer(snapshot.snapshot_id, "Enter a question of at most 2,000 characters.", "clarification", state=state)
        intent = intents.parse(question)
        try:
            intent = _intent({"kind": intent.kind, "args": list(intent.args)})
        except ValueError:
            return ChatAnswer(snapshot.snapshot_id, "Use a valid count from 0 to 100 or an applicant reference.", "clarification", state=state)
        unavailable = self.provider is None
        if intent.kind == "help" and state.candidate_ids and re.search(r"\b(it|them|this candidate|that candidate)\b", question, re.I):
            if len(state.candidate_ids) != 1:
                return ChatAnswer(snapshot.snapshot_id, "Which applicant do you mean? Use a current rank or ID.", "clarification", state=state)
            kind = "missing" if "missing" in question.lower() else "explain"
            intent = intents.Intent(kind, (state.candidate_ids[0],))
        if intent.kind == "help" and self.provider is not None:
            try:
                messages = messages_with_budget(self.provider, INTERPRET, {"question": question, "previous_ids": state.candidate_ids,
                                                                           "history": state.questions[-2:]}, examples=INTERPRET_EXAMPLES)
                intent = _intent(_complete(self.provider, messages))
            except Exception:
                unavailable = True
        if intent.kind == "help":
            return ChatAnswer(snapshot.snapshot_id, intents.HELP, "provider_unavailable" if unavailable else "clarification", fallback=unavailable, state=state)
        selected = _select(intent, snapshot.results)
        if selected is None:
            return ChatAnswer(snapshot.snapshot_id, "Use an unambiguous current applicant rank or ID.", "clarification", state=state)
        ids = tuple(str(r.cv_id) for r in selected)
        next_state = ChatState(snapshot.snapshot_id, ids, (state.questions + (question,))[-3:])
        context, citations = build_context(snapshot, selected)
        if intent.kind == "missing":
            for candidate in context["candidates"]:
                candidate["evidence"] = [e for e in candidate["evidence"] if e["status"] == "missing"]
            allowed = {e["evidence_id"] for c in context["candidates"] for e in c["evidence"]}
            citations = {eid: citation for eid, citation in citations.items() if eid in allowed}
        header = [_canonical(r) for r in selected]
        if intent.kind == "top":
            header.insert(0, "Ranking scope: reranked shortlist first; retrieval-only tail retains retrieval order. Showing at most 10 candidates.")
        if intent.kind in {"why", "compare"}:
            header.append("Evidence observations do not establish causal attribution for model scores.")
            for r in selected:
                observed = [redact_text(e["requirement"]) for e in r.evidence[:20] if e["status"] != "missing"]
                header.append(f"{r.cv_id} · observed requirements: " + (", ".join(observed) or "no evidence found in the supplied material."))
        if intent.kind in {"entities", "source", "years"}:
            header.append(intents.answer(intent, list(snapshot.results)))
        fallback = True
        used = ()
        if self.provider is not None:
            try:
                messages = messages_with_budget(self.provider, GROUND, {"request": {"kind": intent.kind, "args": intent.args}, "context": context})
                lines, used = _render(_complete(self.provider, messages), snapshot, selected, citations, intent)
                fallback = False
            except Exception:
                unavailable = True
        if fallback:
            lines = []
            if intent.kind not in {"top", "entities", "source", "years"}:
                for c in context["candidates"]:
                    for ev in c["evidence"]:
                        if intent.kind == "missing" and ev["status"] != "missing":
                            continue
                        citation = citations[ev["evidence_id"]]
                        lines.append(f"{c['candidate_id']} · {ev['requirement']}: " + (f'“{citation.quote}” [{citation.evidence_id}]' if citation.quote else "no evidence found in the supplied material."))
                        if citation.quote:
                            used += (citation,)
        status = "insufficient_evidence" if not selected or (intent.kind == "missing" and not lines) else ("provider_unavailable" if unavailable else "answered")
        text = "\n\n".join(header + lines) or "No evidence found in the supplied material."
        return ChatAnswer(snapshot.snapshot_id, redact_text(text), status, ids, used, fallback, next_state)


def answer_question(snapshot, question, provider=None, state=None):
    return ChatService(provider).answer(snapshot, question, state)
