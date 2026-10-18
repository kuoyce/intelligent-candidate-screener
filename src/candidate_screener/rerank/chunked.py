"""L3: a MiniLM cross-encoder fine-tuned on the `chat-1` score, reading every CV token (plan
2026-10-18-classifier-then-rerank, step 3).

R1/R2 cut a pair at 384 tokens (longest-first), which drops part of 90% of `chat-1` pairs. L3
keeps the JD's first `jd_tokens` (192) and cuts the CV into windows of `window` (317) tokens,
stride `stride` (256), so each sequence is `[CLS] jd [SEP] window [SEP]` <= 512 tokens and every
CV token is in some window. In training every window carries its pair's target (score / 100,
MSE, as R1). A pair's score is the max over its windows (MaxP).

Folds are `chat_targets.folds` (by JD): a fold's model trains on the `x2_top100` pairs of the
other four folds and scores every chat pair of its own JDs (the extension included). Defaults
are fixed (`CONFIG`), nothing is tuned. `threads` is part of the config because float sums
depend on it: 5 per process, two folds at a time, measured faster than one fold on 11 threads
(2.4 vs 1.9 windows/s on this 11-core machine, 18 Oct 2026). Weights and out-of-fold scores live under
`data/processed/rerank/chat-l3/<fingerprint>/`; `root` redirects them (the cold check trains
into a fresh directory).

Nothing here reads a human label.

    uv run python -m candidate_screener.rerank.chunked                 # every missing fold
    uv run python -m candidate_screener.rerank.chunked --folds 0 2 4   # a worker's share
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.matching import neural

ROOT = PROCESSED / "rerank" / "chat-l3"
CONFIG = {"base": neural.MODEL_ID, "max_length": 512, "jd_tokens": 192, "window": 317,
          "stride": 256, "loss": "MSE on score / 100", "aggregate": "max over windows",
          "lr": 2e-5, "epochs": 1, "batch": 16, "micro_batch": 8, "warmup_ratio": 0.1,
          "seed": 0, "predict_batch": 32, "text_variant": "v2", "threads": 5}
NAME = "L3 cross-encoder (chat-1, MaxP windows)"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- windows (pure) -------------------------------------------------------------------------

def window_starts(n: int, window: int = CONFIG["window"], stride: int = CONFIG["stride"]
                  ) -> list[int]:
    """Start offsets that cover tokens 0..n-1: one window when n <= window."""
    starts = [0]
    while starts[-1] + window < n:
        starts.append(starts[-1] + stride)
    return starts


def sequences(jd_ids: list[int], cv_ids: list[int], cls: int, sep: int, cfg: dict = CONFIG
              ) -> list[tuple[list[int], list[int]]]:
    """`(input_ids, token_type_ids)` per CV window: `[CLS] jd [SEP] window [SEP]`."""
    jd = list(jd_ids[:cfg["jd_tokens"]])
    out = []
    for s in window_starts(len(cv_ids), cfg["window"], cfg["stride"]):
        win = list(cv_ids[s:s + cfg["window"]])
        out.append(([cls, *jd, sep, *win, sep], [0] * (len(jd) + 2) + [1] * (len(win) + 1)))
    return out


def pad(batch: list[tuple[list[int], list[int]]], pad_id: int) -> dict:
    import torch
    n = max(len(ids) for ids, _ in batch)
    ids = [x + [pad_id] * (n - len(x)) for x, _ in batch]
    tt = [t + [0] * (n - len(t)) for _, t in batch]
    mask = [[1] * len(x) + [0] * (n - len(x)) for x, _ in batch]
    return {"input_ids": torch.tensor(ids), "token_type_ids": torch.tensor(tt),
            "attention_mask": torch.tensor(mask)}


def _windows(frame: pd.DataFrame, tok, cfg: dict) -> tuple[list, np.ndarray]:
    """Every window of every pair in `frame` and the row it belongs to."""
    cache: dict[str, list[int]] = {}

    def ids(text: str) -> list[int]:
        if text not in cache:
            cache[text] = tok(text, add_special_tokens=False)["input_ids"]
        return cache[text]
    seqs, owner = [], []
    for i, (jd, cv) in enumerate(zip(frame.jd_text, frame.cv_text)):
        for s in sequences(ids(jd), ids(cv), tok.cls_token_id, tok.sep_token_id, cfg):
            seqs.append(s)
            owner.append(i)
    return seqs, np.array(owner)


# --- train and score ------------------------------------------------------------------------

def _load(path: str):
    neural.require_torch()
    from sentence_transformers.cross_encoder import CrossEncoder
    neural.quiet_loading()
    return CrossEncoder(path, device="cpu", local_files_only=True, config_kwargs={"num_labels": 1})


def train(frame: pd.DataFrame, out: Path, cfg: dict = CONFIG) -> dict:
    """Fine-tune on `frame` (`jd_text`, `cv_text`, `score`); save to `out`."""
    import torch
    from candidate_screener.rerank.models import _schedule
    torch.set_num_threads(cfg["threads"])
    neural.set_determinism(cfg["seed"])
    model = _load(cfg["base"])
    hf, tok = model.transformers_model, model.tokenizer
    seqs, owner = _windows(frame, tok, cfg)
    y = torch.tensor(frame.score.to_numpy(dtype=float)[owner] / 100, dtype=torch.float32)
    rng = np.random.default_rng(cfg["seed"])
    bs, micro = cfg["batch"], cfg["micro_batch"]
    n_steps = cfg["epochs"] * math.ceil(len(seqs) / bs)
    opt = torch.optim.AdamW(hf.parameters(), lr=cfg["lr"], weight_decay=0.0)
    sched = _schedule(opt, n_steps, cfg["warmup_ratio"])
    hf.train()
    losses = []
    for _ in range(cfg["epochs"]):
        order = rng.permutation(len(seqs))
        total = 0.0
        for start in range(0, len(order), bs):
            step = order[start:start + bs]
            for m in range(0, len(step), micro):
                rows = step[m:m + micro]
                logits = hf(**pad([seqs[i] for i in rows], tok.pad_token_id)).logits.squeeze(-1)
                loss = ((logits - y[rows]) ** 2).sum() / len(step)
                loss.backward()
                total += float(loss.detach()) * len(step)
            opt.step()
            sched.step()
            opt.zero_grad()
        losses.append(round(total / len(seqs), 6))
    hf.eval()
    model.save(str(out))
    return {"pairs": int(len(frame)), "windows": int(len(seqs)), "epoch_loss": losses}


def score(frame: pd.DataFrame, path: Path, cfg: dict = CONFIG) -> np.ndarray:
    """MaxP: each pair's highest window logit."""
    import torch
    torch.set_num_threads(cfg["threads"])
    model = _load(str(path))
    hf, tok = model.transformers_model, model.tokenizer
    hf.eval()
    seqs, owner = _windows(frame, tok, cfg)
    logits = np.empty(len(seqs))
    with torch.no_grad():
        for start in range(0, len(seqs), cfg["predict_batch"]):
            batch = seqs[start:start + cfg["predict_batch"]]
            logits[start:start + len(batch)] = hf(**pad(batch, tok.pad_token_id)).logits.squeeze(-1).numpy()
    return pd.Series(logits).groupby(owner).max().reindex(range(len(frame))).to_numpy()


def _digest(frame: pd.DataFrame, cols) -> str:
    return hashlib.sha256(frame.sort_values("pair_id")[list(cols)].to_csv(index=False).encode()).hexdigest()[:16]


def fingerprint(chat: pd.DataFrame, cfg: dict = CONFIG) -> str:
    """Config, fold assignment and training targets: what the weights depend on."""
    train = chat[chat.source == "x2_top100"]
    return neural.fingerprint({**cfg, "folds": _digest(chat, ["pair_id", "fold"]),
                               "targets": _digest(train, ["pair_id", "score"])})


def run_folds(chat: pd.DataFrame, folds, root: Path = ROOT, cfg: dict = CONFIG) -> None:
    """Train and score each fold in `folds` that has no out-of-fold file yet."""
    import gc
    base = Path(root) / fingerprint(chat, cfg)
    base.mkdir(parents=True, exist_ok=True)
    for k in folds:
        scored, meta = base / f"fold{k}-oof.parquet", base / f"fold{k}.json"
        if scored.exists():
            continue
        tr = chat[(chat.fold != k) & (chat.source == "x2_top100")].sort_values("pair_id")
        te = chat[chat.fold == k].sort_values("pair_id")
        _log(f"L3 fold {k}: train {len(tr)} pairs, score {len(te)}")
        t0 = time.perf_counter()
        fit = train(tr, base / f"fold{k}", cfg)
        t1 = time.perf_counter()
        s = score(te, base / f"fold{k}", cfg)
        t2 = time.perf_counter()
        meta.write_text(json.dumps({**fit, "train_seconds": round(t1 - t0, 1),
                                    "score_seconds": round(t2 - t1, 1)}, indent=1))
        pd.DataFrame({"pair_id": te.pair_id.to_numpy(), "score": s}).to_parquet(scored, index=False)
        _log(f"L3 fold {k}: {fit['windows']} windows, train {t1 - t0:.0f} s, score {t2 - t1:.0f} s")
        gc.collect()


def oof(chat: pd.DataFrame, root: Path = ROOT, cfg: dict = CONFIG) -> tuple[pd.Series, list[dict]]:
    """Out-of-fold scores for every pair of `chat` (`pair_id`, `jd_text`, `cv_text`, `score`,
    `source`, `fold`), training any missing fold first. Returns (scores by pair id, fold info)."""
    run_folds(chat, sorted(chat.fold.unique()), root, cfg)
    base = Path(root) / fingerprint(chat, cfg)
    folds = sorted(chat.fold.unique())
    s = pd.concat([pd.read_parquet(base / f"fold{k}-oof.parquet") for k in folds]).set_index("pair_id").score
    if set(s.index) != set(chat.pair_id) or s.index.duplicated().any():
        raise ValueError("L3 out-of-fold scores do not cover the chat pairs exactly once")
    info = [{"fold": int(k), **json.loads((base / f"fold{k}.json").read_text())} for k in folds]
    return s.loc[chat.pair_id.to_numpy()], info


def chat_frame() -> pd.DataFrame:
    """`chat_targets.load()` with fold and text: the frame every stage-2 learner uses."""
    from candidate_screener.rerank import chat_rerank as cr, chat_targets as ct
    chat = ct.load()
    strata = chat.drop_duplicates("jd_id").set_index("jd_id").stratum
    chat["fold"] = chat.jd_id.map(ct.folds(strata))
    return cr.texts(chat)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", type=int, nargs="+", help="train only these folds (one worker)")
    a = ap.parse_args(argv)
    chat = chat_frame()
    if a.folds:
        run_folds(chat, a.folds)
        return 0
    _, info = oof(chat)
    for i in info:
        print(f"  fold {i['fold']}: {i['windows']} windows, train {i['train_seconds']} s, "
              f"score {i['score_seconds']} s, loss {i['epoch_loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
