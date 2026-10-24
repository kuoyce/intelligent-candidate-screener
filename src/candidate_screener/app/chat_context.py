"""Select and redact evidence before provider use, preserving source coordinates."""
from __future__ import annotations

import json

from candidate_screener.annotation.redact import redact_text
from candidate_screener.app.chat_schema import Citation
from candidate_screener.llm.base import redact_content

FACT_FIELDS = ("rank", "retrieval_score", "rerank_score", "fit_call", "rerank_status", "classification_status")


def build_context(snapshot, selected):
    requirements = sorted({redact_text(e.get("requirement", ""))[:200]
                           for r in selected[:10] for e in r.evidence[:20]})[:20]
    context = {"snapshot_id": snapshot.snapshot_id,
               "jd_title": redact_text(getattr(snapshot, "jd_title", ""))[:200],
               "jd_requirements": requirements, "candidates": []}
    citations = {}
    for r in selected[:10]:
        candidate = {"candidate_id": str(r.cv_id), "facts": {k: getattr(r, k, None) for k in FACT_FIELDS}, "evidence": []}
        for index, ev in enumerate(r.evidence[:20]):
            eid = f"{r.cv_id}:e{index}"
            quote = redact_text(ev.get("quote") or "")
            # Display offsets are local to this redacted evidence span, never applied
            # to the original document. Each citation carries both coordinate spaces.
            citation = Citation(snapshot.snapshot_id, str(r.cv_id), eid, quote,
                                ev.get("start"), ev.get("end"), 0, len(quote))
            citations[eid] = citation
            candidate["evidence"].append({"evidence_id": eid, "requirement": redact_text(ev.get("requirement", "")),
                                          "status": ev.get("status"), "quote": quote})
        context["candidates"].append(candidate)
    return context, citations


def messages_with_budget(provider, system, payload, *, examples=()):
    messages = [{"role": "system", "content": system},
                *[dict(example) for example in examples],
                {"role": "user", "content": redact_content(json.dumps(payload, ensure_ascii=False))}]
    if provider.count_tokens(messages) + provider.max_output_tokens > provider.max_context_tokens:
        raise ValueError("Context budget exceeded")
    return messages
