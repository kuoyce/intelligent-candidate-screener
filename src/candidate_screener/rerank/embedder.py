"""E1: an instruction-tuned embedder as a zero-shot stage-2 signal (plan
2026-10-18-classifier-then-rerank, step 5a).

`Qwen/Qwen3-Embedding-0.6B` encodes the JD and each CV **separately** and whole (the longest
`chat-1` pair is 3,105 tokens, under `max_length`); a pair's score is the cosine of the two
vectors. Nothing is trained. The instruction goes on the JD side only, in the model card's
`Instruct: ...\nQuery:` form; CVs get none, so one set of CV vectors serves every instruction.

CV vectors are cached under `data/processed/rerank/embed/<fingerprint>/`, keyed by CV id and a
digest of the text, so a later full run encodes only the CVs a pilot did not. float32 on purpose:
the checkpoint's bfloat16 default ran 3x slower on this AVX2 CPU (50 vs 156 tok/s, 18 Oct 2026),
and one process saturates it (one to three workers all gave ~180 tok/s combined).

Nothing here reads a human label.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.matching import neural

ROOT = PROCESSED / "rerank" / "embed"
CONFIG = {"model": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
          "dtype": "float32", "max_length": 4096, "batch": 1, "threads": 6, "normalize": True,
          "text_variant": "v2"}
TASK_FIT = "Given a job description, retrieve the CVs of candidates who fit its requirements"
TASK_CRITERIA = ("Given a job description, retrieve the CVs of candidates whose role, skills and "
                 "years of experience match its requirements")
#: Instruction variants, fixed in the plan before any scoring: (task, JD title first).
INSTRUCTIONS = {"I0": (None, False), "I1": (TASK_FIT, False), "I2": (TASK_CRITERIA, False),
                "I3": (TASK_CRITERIA, True)}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- text (pure) ----------------------------------------------------------------------------

def query_text(jd_text: str, variant: str, title: str | None = None) -> str:
    """The JD side as the model reads it under one instruction variant."""
    task, with_title = INSTRUCTIONS[variant]
    body = f"Job title: {title}\n{jd_text}" if with_title and title else jd_text
    return body if task is None else f"Instruct: {task}\nQuery:{body}"


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cosine(jd_vecs: np.ndarray, cv_vecs: np.ndarray) -> np.ndarray:
    """Row-wise cosine of matched rows (vectors need not be normalised)."""
    a = jd_vecs / np.linalg.norm(jd_vecs, axis=1, keepdims=True)
    b = cv_vecs / np.linalg.norm(cv_vecs, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", a, b)


# --- model and cache ------------------------------------------------------------------------

def load(cfg: dict = CONFIG):
    neural.require_torch()
    import torch
    from sentence_transformers import SentenceTransformer
    neural.quiet_loading()
    torch.set_num_threads(cfg["threads"])
    model = SentenceTransformer(cfg["model"], device="cpu", revision=cfg["revision"],
                                local_files_only=True, model_kwargs={"dtype": getattr(torch, cfg["dtype"])})
    model.max_seq_length = cfg["max_length"]
    return model


def _encode(model, texts: list[str], cfg: dict) -> np.ndarray:
    return np.asarray(model.encode(texts, batch_size=cfg["batch"], normalize_embeddings=cfg["normalize"],
                                   show_progress_bar=False), dtype=np.float32)


def cache_dir(root: Path = ROOT, cfg: dict = CONFIG) -> Path:
    return root / neural.fingerprint(cfg)


def cv_vectors(cvs: pd.DataFrame, model=None, root: Path = ROOT, cfg: dict = CONFIG) -> pd.DataFrame:
    """Vectors for `cvs` (`cv_id`, `cv_text`), indexed by CV id; encodes only what the cache lacks.

    The cache is one parquet per call that added rows, so an interrupted run keeps what it did."""
    out = cache_dir(root, cfg)
    want = cvs.drop_duplicates("cv_id").assign(digest=lambda d: d.cv_text.map(text_digest))
    have = (pd.concat([pd.read_parquet(p) for p in sorted(out.glob("cv-*.parquet"))], ignore_index=True)
            if out.exists() and any(out.glob("cv-*.parquet")) else pd.DataFrame(columns=["cv_id", "digest", "vec"]))
    have = have.drop_duplicates(["cv_id", "digest"])
    todo = want.merge(have[["cv_id", "digest"]], on=["cv_id", "digest"], how="left", indicator=True)
    todo = todo[todo._merge == "left_only"]
    if len(todo):
        model = model or load(cfg)
        _log(f"encoding {len(todo)} CVs ({len(want) - len(todo)} cached)")
        t0 = time.time()
        vecs = _encode(model, todo.cv_text.tolist(), cfg)
        _log(f"encoded {len(todo)} CVs in {time.time() - t0:.0f} s")
        new = pd.DataFrame({"cv_id": todo.cv_id.to_numpy(), "digest": todo.digest.to_numpy(), "vec": list(vecs)})
        out.mkdir(parents=True, exist_ok=True)
        new.to_parquet(out / f"cv-{time.strftime('%Y%m%d-%H%M%S')}.parquet", index=False)
        have = pd.concat([have, new], ignore_index=True)
    got = want[["cv_id", "digest"]].merge(have, on=["cv_id", "digest"], how="left", validate="one_to_one")
    return pd.DataFrame(np.stack(got.vec.to_numpy()), index=got.cv_id.to_numpy())


def jd_vectors(jds: pd.DataFrame, variant: str, model, cfg: dict = CONFIG) -> pd.DataFrame:
    """Vectors for `jds` (`jd_id`, `jd_text`, optional `jd_title`) under one variant; not cached
    (60 JDs take seconds)."""
    jds = jds.drop_duplicates("jd_id")
    titles = jds.jd_title if "jd_title" in jds else pd.Series([None] * len(jds), index=jds.index)
    texts = [query_text(t, variant, ti) for t, ti in zip(jds.jd_text, titles)]
    return pd.DataFrame(_encode(model, texts, cfg), index=jds.jd_id.to_numpy())


def score(pairs: pd.DataFrame, jd_vecs: pd.DataFrame, cv_vecs: pd.DataFrame) -> pd.Series:
    """Cosine per pair (`pair_id`, `jd_id`, `cv_id`), indexed by pair id."""
    s = cosine(jd_vecs.loc[pairs.jd_id.to_numpy()].to_numpy(), cv_vecs.loc[pairs.cv_id.to_numpy()].to_numpy())
    return pd.Series(s, index=pairs.pair_id.to_numpy())
