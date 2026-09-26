"""Shared plumbing for the torch rungs: offline model loading, determinism, fingerprints.

Only the locally cached `sentence-transformers/all-MiniLM-L6-v2` is used (no downloads).
Fine-tuned weights are a cache under `data/processed/matching/` (D23 baseline), keyed by a
fingerprint of everything that went into them.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def require_torch():
    try:
        import torch  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError("this rung needs the `ml` group: uv sync --all-groups") from exc


def set_determinism(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def file_digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
