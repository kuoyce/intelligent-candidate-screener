"""Zero-shot cross-encoders as a stage-2 signal (plan 2026-10-18-classifier-then-rerank, step 5a
cross-encoder pilot).

A cross-encoder reads the JD and one CV **together** and gives one score per pair. Nothing is
trained. Every model reads the pair whole (max length 8,192; the longest `chat-1` pair is 3,390
tokens). Scores are the raw logit (Qwen3-Reranker: the "yes" minus the "no" logit), float32 for the
same reason as `embedder` (bfloat16 is slow on this AVX2 CPU).

Scores are cached per system under `data/processed/rerank/cross/<fingerprint>/`, keyed by pair id
and a digest of the pair's text, one file per `CHUNK` pairs so an interrupted run keeps what it
scored (batch 1, so chunking leaves every score unchanged). Nothing here reads a human label.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.matching import neural
from candidate_screener.rerank.embedder import TASK_FIT, text_digest

ROOT = PROCESSED / "rerank" / "cross"
COMMON = {"dtype": "float32", "max_length": 8192, "batch": 1, "threads": 6, "score": "raw logit",
          "text_variant": "v2"}
CHUNK = 100   # pairs per cache file
QWEN ={"model": "Qwen/Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473"}
#: Fixed in the plan before scoring. `instruction` None = the checkpoint's own (or none at all).
SYSTEMS = {
    "C1 gte-reranker": {"model": "Alibaba-NLP/gte-reranker-modernbert-base",
                        "revision": "f7481e6055501a30fb19d090657df9ec1f79ab2c", "instruction": None},
    "C2 Qwen3-Reranker, fit": {**QWEN, "instruction": TASK_FIT},
    "C3 Qwen3-Reranker, default": {**QWEN, "instruction": None},
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def config(name: str) -> dict:
    return {**COMMON, **SYSTEMS[name]}


def load(cfg: dict):
    neural.require_torch()
    import torch
    from sentence_transformers import CrossEncoder
    neural.quiet_loading()
    torch.set_num_threads(cfg["threads"])
    return CrossEncoder(cfg["model"], device="cpu", revision=cfg["revision"], local_files_only=True,
                        max_length=cfg["max_length"], model_kwargs={"dtype": getattr(torch, cfg["dtype"])})


def score_pairs(pairs: pd.DataFrame, name: str, root: Path = ROOT, chunk: int = CHUNK,
                cached_only: bool = False, progress=None) -> pd.Series:
    """Raw scores for `pairs` (`pair_id`, `jd_text`, `cv_text`) by pair id; scores only what the
    cache lacks, writing every `chunk` pairs. `cached_only` refuses to score anything.
    `progress(done, total)`, if given, is called with the pairs ready so far (cached included)."""
    cfg = config(name)
    out = root / neural.fingerprint(cfg)
    want = pairs.drop_duplicates("pair_id").assign(
        digest=lambda d: [text_digest(j + "\x00" + c) for j, c in zip(d.jd_text, d.cv_text)])
    files = sorted(out.glob("scores-*.parquet")) if out.exists() else []
    have = (pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files
            else pd.DataFrame(columns=["pair_id", "digest", "score"])).drop_duplicates(["pair_id", "digest"])
    todo = want.merge(have[["pair_id", "digest"]], on=["pair_id", "digest"], how="left", indicator=True)
    todo = todo[todo._merge == "left_only"]
    if len(todo) and cached_only:
        raise ValueError(f"{name}: {len(todo)} of {len(want)} pairs are not cached")
    if progress is not None:
        progress(len(want) - len(todo), len(want))
    if len(todo):
        import torch
        model = load(cfg)
        _log(f"{name}: scoring {len(todo)} pairs ({len(want) - len(todo)} cached)")
        out.mkdir(parents=True, exist_ok=True)
        stamp, new, t0 = time.strftime("%Y%m%d-%H%M%S"), [], time.time()
        for i, start in enumerate(range(0, len(todo), chunk)):
            part = todo.iloc[start:start + chunk]
            s = model.predict(list(zip(part.jd_text, part.cv_text)), batch_size=cfg["batch"],
                              prompt=cfg["instruction"], activation_fn=torch.nn.Identity(), show_progress_bar=False)
            frame = pd.DataFrame({"pair_id": part.pair_id.to_numpy(), "digest": part.digest.to_numpy(),
                                  "score": np.asarray(s, dtype=float).reshape(-1)})
            frame.to_parquet(out / f"scores-{stamp}-{i:04d}.parquet", index=False)
            new.append(frame)
            done = start + len(part)
            _log(f"{name}: {done}/{len(todo)} pairs, {(time.time() - t0) / done:.1f} s/pair")
            if progress is not None:
                progress(len(want) - len(todo) + done, len(want))
        _log(f"{name}: {len(todo)} pairs in {time.time() - t0:.0f} s")
        have = pd.concat([have, *new], ignore_index=True)
        del model
    got = want[["pair_id", "digest"]].merge(have, on=["pair_id", "digest"], how="left", validate="one_to_one")
    return pd.Series(got.score.to_numpy(dtype=float), index=got.pair_id.to_numpy())
