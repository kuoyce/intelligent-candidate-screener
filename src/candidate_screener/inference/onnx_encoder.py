"""Offline CPU SBERT execution without importing Torch or Transformers.

The exported graph returns token embeddings. Masked mean pooling, per-chunk
normalization and normalized document chunk means reproduce matching.sbert.Encoder.
ONNX embeddings use an artifact-specific cache, separate from historical caches.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import threading

import numpy as np

from candidate_screener.config import PROCESSED

DEFAULT_ROOT = PROCESSED / "inference/sbert-onnx"
DEFAULT_CACHE = PROCESSED / "embeddings/onnx"
CONFIG = {"max_length": 256, "dimension": 384, "pooling": "mean",
          "normalize_chunks": True, "normalize_documents": True, "chunk_chars": 800}
_SHA = re.compile(r"[a-f0-9]{64}")


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_manifest(model_dir=DEFAULT_ROOT, *, require_verified=True):
    """Check artifact bytes and semantics before loading executable model data."""
    root = Path(model_dir).resolve()
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("source_checkpoint_sha256"), str)
            or not _SHA.fullmatch(manifest["source_checkpoint_sha256"])
            or manifest.get("source_model_key") != "sbert-weak-3255167debe9"
            or manifest.get("config") != CONFIG
            or any(type(manifest["config"][key]) is not type(value) for key, value in CONFIG.items())):
        raise ValueError("Invalid ONNX encoder identity or unsupported pooling/chunk configuration")
    source_files = manifest.get("source_checkpoint_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("Source checkpoint file provenance is required")
    for name, digest in source_files.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or not isinstance(digest, str) or not _SHA.fullmatch(digest)):
            raise ValueError("Invalid source checkpoint file provenance")
    source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if source_digest != manifest["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint aggregate digest differs from its provenance")
    validation = manifest.get("validation", {})
    if require_verified and (not isinstance(validation, dict) or validation.get("passed") is not True
                             or validation.get("reference") != "pytorch"
                             or validation.get("atol") != 1e-5 or validation.get("rtol") != 1e-4):
        raise ValueError("ONNX encoder has no accepted PyTorch equivalence validation")
    files = manifest.get("files")
    if (not isinstance(files, dict) or not {"model.onnx", "tokenizer.json"} <= files.keys()
            or len(files) > 100):
        raise ValueError("ONNX model and tokenizer digests are required")
    for name, digest in files.items():
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or not isinstance(digest, str) or not _SHA.fullmatch(digest)):
            raise ValueError("Invalid artifact file identity")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != digest:
            raise ValueError("ONNX artifact file does not match its recorded digest")
    # The full manifest binds caches to export semantics, source and artifact bytes.
    identity = hashlib.sha256(raw).hexdigest()
    return manifest, identity


def chunks(text):
    clean = re.sub(r"\s+", " ", str(text)).strip()
    return [clean[i:i + 800] for i in range(0, max(len(clean), 1), 800)] or [""]


def _unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Encoder returned non-finite or zero embeddings")
    return (vectors / norms).astype(np.float32)


def pooled_chunks(hidden, attention_mask):
    """Padding is excluded; special tokens remain as in Sentence Transformers."""
    hidden = np.asarray(hidden, dtype=np.float32)
    mask = np.asarray(attention_mask)
    if (hidden.ndim != 3 or hidden.shape[-1] != 384
            or mask.shape != hidden.shape[:2] or not np.isin(mask, [0, 1]).all()
            or not np.isfinite(hidden).all() or (mask.sum(axis=1) <= 0).any()):
        raise ValueError("Invalid token embeddings or attention mask")
    weights = mask.astype(np.float32)[..., None]
    return _unit((hidden * weights).sum(axis=1) / weights.sum(axis=1))


class OnnxEncoder:
    """A manifest-verified CPU token encoder with independent document caching."""

    def __init__(self, model_dir=DEFAULT_ROOT, cache_dir=DEFAULT_CACHE, *, session=None, tokenizer=None,
                 require_verified=True):
        self.root = Path(model_dir).resolve()
        self.manifest, self.artifact_sha256 = verify_manifest(self.root, require_verified=require_verified)
        self.path = None if cache_dir is None else Path(cache_dir) / f"sbert-{self.artifact_sha256}.npz"
        if session is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            session = ort.InferenceSession(str(self.root / "model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        if tokenizer is None:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(self.root / "tokenizer.json"))
        self.session, self.tokenizer = session, tokenizer
        self.tokenizer.enable_truncation(max_length=256)
        pad = self.tokenizer.token_to_id("[PAD]")
        if type(pad) is not int or pad < 0:
            raise ValueError("The saved tokenizer has no valid padding token")
        self.tokenizer.enable_padding(pad_id=pad, pad_token="[PAD]")
        self.input_names = [item.name for item in session.get_inputs()]
        if (len(set(self.input_names)) != len(self.input_names)
                or not {"input_ids", "attention_mask"} <= set(self.input_names)
                or set(self.input_names) - {"input_ids", "attention_mask", "token_type_ids"}):
            raise ValueError("Unsupported ONNX input signature")
        self.cache, self._lock = {}, threading.RLock()
        self._load_cache()

    @staticmethod
    def _key(text):
        return hashlib.sha256(str(text).encode()).hexdigest()

    def _load_cache(self):
        if self.path is None or not self.path.exists():
            return
        with np.load(self.path, allow_pickle=False) as saved:
            if set(saved.files) != {"keys", "vectors", "artifact_sha256"}:
                raise ValueError("Invalid ONNX embedding cache fields")
            keys, vectors = saved["keys"], saved["vectors"]
            identity = saved["artifact_sha256"]
            if (identity.shape != () or str(identity.item()) != self.artifact_sha256
                    or keys.ndim != 1 or keys.dtype.kind != "U"
                    or vectors.shape != (len(keys), 384) or vectors.dtype != np.float32
                    or len(set(keys.tolist())) != len(keys)
                    or any(not _SHA.fullmatch(key) for key in keys.tolist())
                    or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)):
                raise ValueError("Invalid ONNX embedding cache identity, shape or values")
            self.cache = {key: vector.copy() for key, vector in zip(keys.tolist(), vectors)}

    def _save_cache(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".sbert-", suffix=".npz", dir=self.path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                keys = sorted(self.cache)
                np.savez(stream, keys=np.asarray(keys, dtype="U64"),
                         vectors=np.stack([self.cache[key] for key in keys]).astype(np.float32),
                         artifact_sha256=np.asarray(self.artifact_sha256))
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def _encode_pieces(self, pieces):
        records = self.tokenizer.encode_batch(pieces)
        if len(records) != len(pieces) or not records:
            raise ValueError("Tokenizer omitted an input chunk")
        fields = {"input_ids": [item.ids for item in records],
                  "attention_mask": [item.attention_mask for item in records],
                  "token_type_ids": [item.type_ids for item in records]}
        feed = {name: np.asarray(fields[name], dtype=np.int64) for name in self.input_names}
        if (feed["input_ids"].ndim != 2 or not 1 <= feed["input_ids"].shape[1] <= 256
                or any(value.shape != feed["input_ids"].shape for value in feed.values())):
            raise ValueError("Tokenizer returned invalid or overlong chunk dimensions")
        output = self.session.run(None, feed)
        if not isinstance(output, (list, tuple)) or not output:
            raise ValueError("ONNX returned no token embeddings")
        hidden = np.asarray(output[0])
        if hidden.shape[:2] != feed["input_ids"].shape:
            raise ValueError("ONNX output does not match the input chunks")
        return pooled_chunks(hidden, feed["attention_mask"])

    def encode(self, texts):
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("SBERT inputs must be strings")
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        with self._lock:
            todo = sorted({text for text in texts if self._key(text) not in self.cache})
            if todo:
                pieces, owners = [], []
                for owner, text in enumerate(todo):
                    for chunk in chunks(text):
                        pieces.append(chunk)
                        owners.append(owner)
                vectors = np.concatenate([self._encode_pieces(pieces[start:start + 64])
                                          for start in range(0, len(pieces), 64)])
                owners = np.asarray(owners)
                for owner, text in enumerate(todo):
                    self.cache[self._key(text)] = _unit(vectors[owners == owner].mean(axis=0))
                self._save_cache()
            return np.stack([self.cache[self._key(text)] for text in texts]).astype(np.float32)
