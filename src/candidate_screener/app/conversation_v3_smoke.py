"""Invented-source v3 development walkthrough, never a held-out quality score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from .conversation_v3 import ConversationV3Service
from .portal_extractive import PortalExtractiveService
from .conversation_smoke import SourceOnlyQA


def fixture():
    return SimpleNamespace(snapshot_id="v3-invented", jd_id="JD-INVENTED",
        jd_title="Invented developer", jd_text="Bachelor degree required.",
        candidates=tuple({"cv_id": owner, "rank": rank, "preview_text": source,
                          "preview_status": "available_redacted_text"}
                         for owner, rank, source in [("CV-A", 1, "Completed Bachelor degree."),
                                                    ("CV-B", 2, "Master degree in progress.")]))


class RecordingComposer:
    """Capture attempted model wording only for these explicitly invented sources."""
    def __init__(self, composer):
        self.composer, self.last = composer, None

    def compose_before(self, question, frame, facts, deadline):
        self.last = self.composer.compose_before(question, frame, facts, deadline)
        return self.last


def walkthrough(composer):
    recorder = RecordingComposer(composer) if composer else None
    service = ConversationV3Service(PortalExtractiveService(qa_client=SourceOnlyQA()), recorder)
    snapshot, history, context, records = fixture(), [], None, []
    for question in ["Hi", "What degree does candidate 1 have?",
                     "Compare candidates 1 and 2 on qualifications",
                     "Why is 1 better than 2?", "So candidate 2 has completed a Master degree?",
                     "Where does it say the Master degree is in progress?"]:
        if recorder:
            recorder.last = None
        started = time.monotonic()
        answer = service.answer(snapshot, question, context=context, history=history)
        context = answer["context"]
        history.append({"question": question, "answer": answer})
        records.append({"question": question, "answer": answer["text"],
                        "generation_mode": answer.get("generation_mode"),
                        "provenance": answer.get("generation_provenance", {}),
                        "attempted_generation": recorder.last if recorder else None,
                        "seconds": time.monotonic() - started})
    eligible = records[1:5]
    return {"kind": "development_smoke", "invented_sources_only": True,
            "factual_turns": len(eligible),
            "accepted_model_answers": sum(r["generation_mode"] == "qwen_prose" for r in eligible),
            "turns": records,
            "limitations": "One seen source family; no human quality gate, held-out evaluation or latency population estimate. QA uses source retrieval; extraction model accuracy is not measured."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    composer = None
    if args.real_model:
        from candidate_screener.llm.conversation_prose import configured_prose_composer
        composer = configured_prose_composer()
        if composer is None:
            raise SystemExit("Existing Qwen configuration unavailable; no real-model smoke performed")
    report = walkthrough(composer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("kind", "factual_turns", "accepted_model_answers")}))


if __name__ == "__main__":
    main()
