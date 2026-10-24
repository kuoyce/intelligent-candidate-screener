"""Time frozen medium inference on deterministic length quartiles, no text output."""
from __future__ import annotations

import json
import time

from candidate_screener.config import PROJECT_ROOT
from .fit_run import inventory
from .medium_assets import MODEL_DIR
from .medium_runtime import MediumRuntimeProxy
from .medium_inference import extract_document


def main():
    payload = inventory()
    selected = []
    for side in ("cv", "jd"):
        docs = sorted((d for d in payload["documents"] if d["doc_type"] == side),
                      key=lambda d: (sum(len(t) for t in d["fields"].values()), d["doc_id"]))
        selected.extend(docs[min(len(docs) - 1, int((len(docs) - 1) * q))] for q in (.25, .75))
    started = time.monotonic()
    model = MediumRuntimeProxy(MODEL_DIR)
    initialization = time.monotonic() - started
    rows = []
    try:
        for doc in selected:
            started = time.monotonic()
            entities, coverage = extract_document(doc, model, deadline=started + 120)
            rows.append({"doc_type": doc["doc_type"],
                         "characters": sum(len(t) for t in doc["fields"].values()),
                         "seconds": time.monotonic() - started, "entities": len(entities),
                         "complete": all(info["complete"] for info in coverage.values()),
                         "chunks": sum(info["chunks"] for info in coverage.values()),
                         "fallback_windows": sum(info["fallback_windows"] for info in coverage.values())})
    finally:
        model.close()
    inference = sum(row["seconds"] for row in rows)
    characters = sum(row["characters"] for row in rows)
    result = {"initialization_seconds": initialization, "documents": rows,
              "length_scaled_estimate_seconds": initialization + inference / characters * payload["summary"]["total"]["characters"],
              "limitation": "Length-scaled pilot estimate, not a bound; atypical documents may hit the frozen soft deadline."}
    (PROJECT_ROOT / "logs/medium-pilot.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
