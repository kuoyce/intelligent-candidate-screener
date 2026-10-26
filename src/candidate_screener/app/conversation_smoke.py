"""Bounded invented-source v2 walkthrough; not a held-out quality benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from .conversation_service import ConversationService
from .portal_extractive import PortalExtractiveService


def fixture():
    sources = (
        ("CV-A", "Three years of Python experience. Production deployment experience. Completed Bachelor degree."),
        ("CV-B", "Five years of Python experience. No production deployment experience. Master degree in progress."),
        ("CV-C", "Cloud engineering experience. AWS certification expired in 2021. Completed Bachelor degree."),
    )
    return SimpleNamespace(snapshot_id="v2-invented-walkthrough", jd_id="JD-INVENTED",
                           jd_title="Invented data engineer",
                           jd_text="Python experience and production deployment experience required. Bachelor degree preferred.",
                           candidates=tuple({"cv_id": owner, "rank": rank,
                                             "preview_text": text, "preview_status": "available_redacted_text"}
                                            for rank, (owner, text) in enumerate(sources, 1)))


class SourceOnlyQA:
    def answer(self, question, passages):
        return {"status": "unavailable", "spans": []}


def walkthrough(service, snapshot=None):
    current = snapshot or fixture()
    questions = ["Compare candidates 1 and 2 for this role",
                 "Who has more Python experience?",
                 "What about their qualifications?",
                 "Where does it say the Master degree is in progress?",
                 "Now include candidate 3",
                 "Remove candidate 2",
                 "Compare their cloud experience",
                 "Why is 1 better than 2 for this role?",
                 "Why is 1 better than 2 on Python experience?",
                 "Why is 2 better than 1 on Python experience?"]
    records, context = [], None
    for question in questions:
        started = time.monotonic()
        answer = service.answer(current, question, context=context)
        context = answer["context"]
        # Logs/artifacts contain only invented texts and opaque ownership checks.
        records.append({"question": question, "text": answer["text"],
                        "status": answer["status"], "mode": answer["mode"],
                        "generation_mode": answer.get("generation_mode"),
                        "generation_provenance": answer.get("generation_provenance"),
                        "show_comparison_table": answer.get("show_comparison_table", False),
                        "selected_candidates": context["selected_candidates"],
                        "citation_owners": sorted({c["owner"] for c in answer.get("citations", [])}),
                        "elapsed_seconds": time.monotonic() - started,
                        "storage_bytes": len(json.dumps(answer, ensure_ascii=False).encode())})
    checks = {
        "initial_pair": records[0]["selected_candidates"] == ["CV-A", "CV-B"],
        "initial_comparison_has_both_sources": {"CV-A", "CV-B"} <= set(records[0]["citation_owners"]),
        "initial_comparison_explains_deployment_contrast": "production deployment" in records[0]["text"].casefold() and "negates" in records[0]["text"],
        "duration_pair_retained": records[1]["selected_candidates"] == ["CV-A", "CV-B"],
        "explicit_tenure_comparison": "Candidate 2 reports more python tenure" in records[1]["text"],
        "plural_qualification_pair_retained": records[2]["selected_candidates"] == ["CV-A", "CV-B"] and records[2]["status"] != "clarification",
        "qualification_context_retained": "in progress" in records[2]["text"],
        "exact_source_reopened": records[3]["generation_mode"] == "citation_reopen" and records[3]["citation_owners"] == ["CV-B"],
        "third_candidate_included": records[4]["selected_candidates"] == ["CV-A", "CV-B", "CV-C"],
        "candidate_removed": records[5]["selected_candidates"] == ["CV-A", "CV-C"],
        "why_role_preserves_tradeoff": "trade-off" in records[7]["text"],
        "why_python_corrects_premise": "favor Candidate 2" in records[8]["text"],
        "why_reversed_pair": records[9]["selected_candidates"] == ["CV-B", "CV-A"] and "stronger documented evidence for Candidate 2" in records[9]["text"],
        "why_tables_visible": all(r["show_comparison_table"] for r in records[7:]),
        "storage_bound": all(r["storage_bytes"] <= 24_000 for r in records),
    }
    return {"kind": "exploratory_integration_walkthrough", "authored_sources_only": True,
            "fixture_sha256": hashlib.sha256(json.dumps(vars(current), sort_keys=True).encode()).hexdigest(),
            "checks": checks, "turns": records,
            "limitations": "One invented JD family; no independent human review or quality/latency population estimate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-models", action="store_true", help="Use existing offline QA and configured local Qwen")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qa, composer = SourceOnlyQA(), None
    if args.real_models:
        from candidate_screener.llm.extractive_qa import QaClient
        from candidate_screener.llm.conversation_qwen import configured_composer
        qa, composer = QaClient(timeout=15), configured_composer()
    try:
        report = walkthrough(ConversationService(PortalExtractiveService(qa_client=qa), composer=composer))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"checks": report["checks"], "output": str(args.output)}))
    finally:
        if hasattr(qa, "close"):
            qa.close()
    raise SystemExit(0 if all(report["checks"].values()) else 1)


if __name__ == "__main__":
    main()
