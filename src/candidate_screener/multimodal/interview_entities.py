"""Local GLiNER preview of translated interview text; no scoring or fit calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from candidate_screener.config import OUTPUT, PROCESSED
from . import entity_bridge
from .entity_bridge import translated_segments, extract_translation_preview


def run() -> dict:
    from candidate_screener.extraction.transformer import (
        load_model, extract_transformer_document, MODEL_LABELS, TRANSFORMER_VERSION)
    from candidate_screener.extraction.transformer_run import MODEL_DIR, _model_manifest, _runtime

    record_path = OUTPUT / "multimodal/french-interview-demo.json"
    record = json.loads(record_path.read_text())
    if record["status"] != "prototype_complete" or not record["translation"]["complete"]:
        raise ValueError("Complete local translation required for entity preview")
    private = PROCESSED / "multimodal" / record["media_sha256"]
    transcript_path = private / "transcripts.json"
    raw = json.loads(transcript_path.read_text())
    segments = translated_segments(raw["linked_chunks"], raw["full_beam4_translation"])
    manifest, runtime = _model_manifest(MODEL_DIR), _runtime(MODEL_DIR)
    model = load_model(MODEL_DIR, device="cpu")
    started = time.monotonic()

    def extractor(text, side):
        document = {"doc_id": "interview-preview", "source": "local-interview",
                    "doc_type": side, "fields": {"text": text}}
        rows, coverage = extract_transformer_document(
            document, model, threshold=.5, overlap_tokens=32, deadline=started + 180)
        if set(coverage) != {"text"} or not coverage["text"]["complete"]:
            raise ValueError("Incomplete GLiNER coverage; preview is not usable")
        return rows

    try:
        expected = runtime.get("manifest", runtime.get("versions", {}))
        actual = getattr(model, "runtime_versions", {})
        if actual and any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError("GLiNER runtime differs from recorded environment")
        preview = extract_translation_preview(segments, extractor)
    finally:
        if callable(getattr(model, "close", None)):
            model.close()
    (private / "entity-preview.json").write_text(json.dumps(preview, ensure_ascii=False, indent=2))
    audit = {"status": preview["entity_audit"]["status"],
             "extractor": "GLiNER", "counts": preview["entity_audit"]["counts"],
             "seconds": time.monotonic() - started,
             "complete_field_coverage": True, "threshold": .5, "overlap_tokens": 32,
             "labels": MODEL_LABELS, "adapter_version": TRANSFORMER_VERSION,
             "model": manifest, "runtime": runtime,
             "transcript_sha256": hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
             "code_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in (__file__, entity_bridge.__file__)},
             "input": "Unreviewed beam-4 English translation; source contains ASR/MT errors",
             "applicant_review_required": True, "screening_ready": False,
             "interpretation": "Predicted mentions, not verified qualifications or fit"}
    record["entity_audit"] = audit
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "counts": result["counts"]}))
