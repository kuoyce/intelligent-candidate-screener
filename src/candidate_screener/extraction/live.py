"""Session-local five-label GLiNER evidence with explicit processing status."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import threading
import time
import weakref

from .schema import LABELS, validate_entity
from .transformer import (load_model, extract_transformer_document, MODEL_LABELS,
                          TRANSFORMER_VERSION, CONTEXT_VERSION)
from .transformer_run import MODEL_DIR, _model_manifest, _runtime


class GlinerEntityExtractor:
    """Lazy offline worker; text and prediction cache live in memory only.

    Unavailable models and failed/incomplete extraction are visible bundle states.
    Neither is represented as a successful document with zero entities.
    """

    def __init__(self, model_dir: Path = MODEL_DIR, *, loader=load_model,
                 infer=extract_transformer_document, manifest_loader=_model_manifest,
                 runtime_loader=_runtime, max_seconds=120):
        self.model_dir = Path(model_dir)
        self.loader, self.infer = loader, infer
        self.manifest_loader, self.runtime_loader = manifest_loader, runtime_loader
        self.max_seconds = max_seconds
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("A finite positive extraction budget is required")
        self._model, self._provenance = None, None
        self._finalizer = None
        self._cache = {}
        self._lock = threading.RLock()

    def availability(self):
        if not (self.model_dir / "model-manifest.json").is_file():
            return {"status": "unavailable", "reason": "Pinned GLiNER checkpoint is not installed"}
        return {"status": "available" if self._model is not None else "not_loaded",
                "reason": "GLiNER loads locally on the first screening request"}

    def _prepare(self):
        if self._model is not None:
            return
        manifest = self.manifest_loader(self.model_dir)
        runtime = self.runtime_loader(self.model_dir)
        model = self.loader(self.model_dir, device="cpu")
        expected = runtime.get("manifest", runtime.get("versions", {}))
        actual = getattr(model, "runtime_versions", {})
        if actual and any(actual.get(k) != v for k, v in expected.items()):
            if callable(getattr(model, "close", None)):
                model.close()
            raise ValueError("GLiNER runtime differs from recorded environment")
        self._model = model
        if callable(getattr(model, "close", None)):
            # The callback retains the worker, not this extractor/session's cache.
            self._finalizer = weakref.finalize(self, model.close)
        self._provenance = {"model": manifest, "runtime": runtime,
                            "adapter_version": TRANSFORMER_VERSION,
                            "context_version": CONTEXT_VERSION, "labels": MODEL_LABELS,
                            "threshold": .5, "overlap_tokens": 32,
                            "cache": "session memory; no transcript persistence"}

    def __call__(self, text: str, side: str):
        if side not in {"cv", "jd"} or not isinstance(text, str) or not text.strip():
            raise ValueError("Nonempty text and CV/JD side required")
        key = (side, hashlib.sha256(text.encode()).hexdigest())
        with self._lock:
            if key in self._cache:
                return deepcopy(self._cache[key])
            try:
                self._prepare()
                doc = {"doc_id": key[1], "source": "session-input", "doc_type": side,
                       "fields": {"text": text}}
                rows, coverage = self.infer(doc, self._model, threshold=.5, overlap_tokens=32,
                                            deadline=time.monotonic() + self.max_seconds)
                if set(coverage) != {"text"} or not coverage["text"]["complete"]:
                    raise ValueError("GLiNER did not process the complete text")
                for row in rows:
                    validate_entity(row, text)
                counts = {label: sum(r["label"] == label for r in rows) for label in sorted(LABELS)}
                result = {"status": "complete" if rows else "completed_zero", "entities": rows,
                          "counts": counts, "provenance": self._provenance}
                self._cache[key] = deepcopy(result)
                return result
            except FileNotFoundError:
                return {"status": "unavailable", "entities": [], "counts": {},
                        "provenance": {}, "reason": "Pinned local GLiNER model/runtime is unavailable"}
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                return {"status": "failed", "entities": [], "counts": {},
                        "provenance": self._provenance or {},
                        "reason": f"GLiNER extraction failed ({type(exc).__name__}); text was not scored as zero evidence"}

    def clear(self):
        with self._lock:
            self._cache.clear()

    def close(self):
        with self._lock:
            self.clear()
            if self._finalizer is not None and self._finalizer.alive:
                self._finalizer()
            self._model = None
