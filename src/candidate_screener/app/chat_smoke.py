"""Synthetic-document smoke trial; --live requires provisioned model configuration.

No benchmark data, labels, training, model fetching or applicant text is written.
The report measures this application trial, not screening model quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
import resource
import sys
from time import perf_counter

import pandas as pd

from candidate_screener.annotation.redact import redact_text
from candidate_screener.app.chat_prompts import GROUND, INTERPRET, INTERPRET_EXAMPLES, PROMPT_VERSION
from candidate_screener.app.chat_service import answer_question
from candidate_screener.app.screening_service import load_reranker, screen
from candidate_screener.app.rerank_client import load_service_reranker
from candidate_screener.app.submissions import DEMO_CVS, DEMO_JD
from candidate_screener.extraction.rules import SkillMatcher
from candidate_screener.llm.qwen import QwenConfig, QwenProvider
from candidate_screener.llm.base import ProviderUnavailable, ContextOverflow
from candidate_screener.llm.qwen_assets import DEFAULT_MANIFEST, local_settings


def trial(*, live=False, values=None, freeform=False):
    values = values if values is not None else os.environ
    provider = None
    started = perf_counter()
    if live:
        if not values.get("SCREENER_QWEN_BASE_URL") and not values.get("SCREENER_QWEN_MODEL") and values is os.environ:
            values = {**local_settings(), **dict(values)}
        config = QwenConfig.from_mapping(values)
        if config is None:
            raise ValueError("Live smoke requires explicit Qwen service and local tokenizer configuration.")
        provider = QwenProvider(config)
    provider_load = perf_counter() - started
    generated_responses = []
    if provider is not None:
        original_complete = provider.complete
        def capture(messages):
            try:
                content = original_complete(messages)
            except (ProviderUnavailable, ContextOverflow) as exc:
                generated_responses.append({"error": str(exc)})
                raise
            # This CLI can only build invented documents, never user submissions.
            generated_responses.append({"text": content[:2000]})
            return content
        provider.complete = capture
    manifest = values.get("SCREENER_CE_MANIFEST") if live else None
    if live and manifest is None and values.get("SCREENER_QWEN_TOKENIZER") == "service":
        default_ce = DEFAULT_MANIFEST.parent.parent / "reranker.json"
        if default_ce.is_file():
            values = {"SCREENER_CE_SERVICE_URL": "http://127.0.0.1:8000/v1", **dict(values)}
            manifest = str(default_ce)
    started = perf_counter()
    service_url = values.get("SCREENER_CE_SERVICE_URL", "")
    reranker = (load_service_reranker(manifest, service_url) if service_url else load_reranker(manifest)) if manifest else None
    backend_load = perf_counter() - started
    terms = ["Python", "SQL", "software testing", "project management"]
    matcher = SkillMatcher(pd.DataFrame({"canonical": terms, "alias": terms}))
    pool = pd.DataFrame({"cv_id": ["Applicant-001", "Applicant-002"], "cv_text": DEMO_CVS})
    snapshot = screen(DEMO_JD, pool, jd_title="Software engineer", reranker=reranker, matcher=matcher)
    answers, elapsed, input_tokens = [], [], []
    # One configured runtime, two synthetic requests; no unbounded latency/model grid.
    questions = ("explain 1", "explain 1") if live else ("top 2", "explain 1", "compare 1 and 2", "source 1")
    if live and freeform:
        questions += ("Who is the strongest applicant here?",)
    for question in questions:
        started = perf_counter()
        answers.append(answer_question(snapshot, question, provider))
        elapsed.append(perf_counter() - started)
        input_tokens.append(provider.last_input_tokens if provider else None)
    citations = [c for answer in answers for c in answer.citations]
    source = {f"{r.cv_id}:e{i}": (r.cv_id, e) for r in snapshot.results for i, e in enumerate(r.evidence)}
    def supported(citation):
        candidate_id, evidence = source.get(citation.evidence_id, (None, {}))
        return (citation.snapshot_id == snapshot.snapshot_id and candidate_id == citation.candidate_id
                and citation.quote == redact_text(evidence.get("quote") or "")
                and citation.original_start == evidence.get("start")
                and citation.original_end == evidence.get("end"))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"mode": "live synthetic documents" if live else "synthetic templates",
            "backend": snapshot.provenance,
            "provider": provider.provenance() if provider else None,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(json.dumps({"ground": GROUND, "interpret": INTERPRET,
                "examples": INTERPRET_EXAMPLES}, sort_keys=True).encode()).hexdigest(),
            "versions": {"python": sys.version.split()[0], "requests": version("requests"),
                         "candidate_screener": version("intelligent-candidate-screener")},
            "provider_load_seconds": provider_load, "backend_load_seconds": backend_load,
            "answer_seconds": elapsed, "question_count": len(answers),
            "input_token_counts": input_tokens,
            "generated_responses": generated_responses,
            "statuses": dict(Counter(a.status for a in answers)),
            "fallback_count": sum(a.fallback for a in answers),
            "audited_quote_count": len(citations),
            "supported_quote_count": sum(supported(c) for c in citations),
            "peak_client_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
            "service_peak_rss_bytes": (provider.provenance().get("served_model") or {}).get("peak_rss_bytes") if provider else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use configured Qwen/CE on synthetic documents only")
    parser.add_argument("--freeform", action="store_true", help="Also test one fixed free-form question with the same configuration")
    args = parser.parse_args()
    try:
        report = trial(live=args.live, freeform=args.freeform)
    except (ValueError, OSError, ImportError, RuntimeError):
        parser.exit(2, "Smoke trial unavailable: check local model artifacts and provider configuration.\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.live and report["fallback_count"]:
        parser.exit(1, "Live answers fell back; inspect the synthetic report before accepting this configuration.\n")


if __name__ == "__main__":
    main()
