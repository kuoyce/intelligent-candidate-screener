"""JSON pipe worker; source text and predictions are never logged to consoles."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import sys

from .runtime_proxy import PREFIX


def emit(payload: dict) -> None:
    print(PREFIX + json.dumps(payload), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        import torch
        import gliner  # Fail here rather than recursively launching another worker.
        from .transformer import load_model, chunk_parameters, extract_transformer_document
        torch.set_num_threads(4)
        model = load_model(args.model_path, args.device)
        if hasattr(model, "extract_document"):
            raise RuntimeError("GLiNER is missing inside the isolated runtime")
        emit({"status": "ready", "chunk_parameters": chunk_parameters(model),
              "runtime_versions": {name: version(name) for name in
                                   ("torch", "transformers", "gliner", "huggingface-hub", "tokenizers", "sentencepiece", "numpy")}})
    except Exception as exc:
        # Avoid exception messages: a third-party error might contain source text.
        emit({"status": "error", "error": type(exc).__name__})
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            records, coverage = extract_transformer_document(request["document"], model, **request["parameters"])
            emit({"status": "ok", "records": records, "coverage": coverage})
        except Exception as exc:
            emit({"status": "error", "error": type(exc).__name__})


if __name__ == "__main__":
    main()
