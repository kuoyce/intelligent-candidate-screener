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


def quiet_loading() -> None:
    """Silence the HF hub / transformers progress bars before a model load (plan W9b).

    transformers 5 draws a `Loading weights: 100%|…|` tqdm bar on every load through
    `transformers.utils.logging.tqdm`, which `disable_progress_bar()` turns off; the hub's
    own bars follow `HF_HUB_DISABLE_PROGRESS_BARS`. Output only: no numeric effect, and
    training-loop bars stay as each trainer sets them (`disable_tqdm`).
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import huggingface_hub.utils
        import transformers.utils.logging
    except ImportError:                                          # pragma: no cover
        return
    huggingface_hub.utils.disable_progress_bars()
    transformers.utils.logging.disable_progress_bar()


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
