"""Offline sentence-aware inference worker; never log document contents."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import sys
import traceback

from .runtime_proxy import PREFIX


def emit(payload):
    print(PREFIX + json.dumps(payload), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        import torch
        from .medium_model import load_local_model
        from .medium_inference import extract_document
        from .sentence_chunking import chunk_parameters
        torch.set_num_threads(4)
        model = load_local_model(args.model_path, args.device)
        emit({"status": "ready", "chunk_parameters": chunk_parameters(model),
              "runtime_versions": {name: version(name) for name in
                                   ("torch", "transformers", "gliner", "huggingface-hub",
                                    "tokenizers", "sentencepiece", "numpy", "spacy", "protobuf", "click")}})
    except Exception as exc:
        # No documents have entered the worker yet; initialization diagnostics
        # are safe in the ignored private log. Inference errors stay class-only.
        traceback.print_exc(file=sys.stderr)
        emit({"status": "error", "error": type(exc).__name__})
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            records, coverage = extract_document(request["document"], model, **request["parameters"])
            emit({"status": "ok", "records": records, "coverage": coverage})
        except Exception as exc:
            # Stack locations are diagnostic without exposing exception messages,
            # request text, local variables or model predictions.
            print(json.dumps({"error": type(exc).__name__, "frames": [
                {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(exc.__traceback__)]}), file=sys.stderr, flush=True)
            emit({"status": "error", "error": type(exc).__name__})


if __name__ == "__main__":
    main()
