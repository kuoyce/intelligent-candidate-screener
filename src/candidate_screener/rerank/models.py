"""Stage-2 rerankers trained on the run-11 batch (amendment A3, 06 §8).

    R1  cross-encoder, pointwise: one logit, MSE on score / 100
    R2  cross-encoder, pairwise: RankNet logistic -log σ(s_hi - s_lo) on within-JD pairs
        whose score gap is >= δ
    R3  bi-encoder, the better on validation of pointwise (CosineSimilarityLoss on
        score / 100) and pairwise (the same RankNet loss on cosine × 20)

All start from the base `all-MiniLM-L6-v2` (`neural.MODEL_ID`), never from a fine-tuned
checkpoint (A3.2). The grid (A3.3) is `cross_encoder.CE_GRID` over the CE defaults (batch
16, warmup 0.1, seed 0; max length 384 for the CE, 256 for the bi-encoder, as SBERT weak),
crossed with δ ∈ {10, 20} for the pairwise forms. Grid order is hyperparameters, then δ;
R3 lists its pointwise points first. A tie on validation goes to the earlier point.

Selection (A3.6): per-JD pairwise agreement against the run-11 score on the 20 validation
JDs (`ordering.pairwise`: a reference tie is no pair, a score tie counts 0.5), mean over
JDs. The stop rule (A3.7, 06 §6): each selected model's per-JD Δ against random (0.5), with
a JD-resampled CI. The run-11 score is a target on the batch only (D43).

Weights live under `data/processed/rerank/`, keyed by a fingerprint of the configuration
and the scores file. `weights_root` redirects them (the cold check trains into a fresh
directory). The bi-encoder scores like SBERT weak (chunk-mean embeddings, `sbert.Encoder`),
with its embedding cache keyed by the weights' content digest.

Nothing here reads a final-set or human label (D46).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_screener.evaluation import ordering, resample
from candidate_screener.matching import neural
from candidate_screener.matching.base import BaseSystem
from candidate_screener.matching.cross_encoder import CE_CONFIG, CE_GRID, weights_digest
from candidate_screener.rerank import batch

WEIGHTS = batch.CACHE
DELTAS = (10, 20)
PREDICT_BATCH = 32
#: Preference pairs per forward/backward in the pairwise loop. A step still covers `batch`
#: pairs (gradients accumulate), so the registered batch is unchanged; this only bounds
#: the attention activations held at once (16 pairs = 32 CE sequences of 384 tokens ran a
#: 15 GB machine out of memory on 8 Oct 2026).
MICRO_BATCH = 8
COSINE_SCALE = 20.0
BASE = {"base": neural.MODEL_ID, "batch": CE_CONFIG["batch"],
        "warmup_ratio": CE_CONFIG["warmup_ratio"], "seed": CE_CONFIG["seed"],
        "text_variant": batch.TEXT_VARIANT}
MAX_LENGTH = {"ce": CE_CONFIG["max_length"], "bi": 256}
DECIMALS = 6

R1, R2, R3 = "R1", "R2", "R3"
NAMES = {R1: "cross-encoder (LLM batch, pointwise)", R2: "cross-encoder (LLM batch, pairwise)",
         R3: "bi-encoder (LLM batch)"}


def grid(rung: str) -> list[dict]:
    """Every configuration of a rung, in grid order (A3.3, A3.5)."""
    hps = [{"lr": CE_CONFIG["lr"], "epochs": CE_CONFIG["epochs"], **o} for o in CE_GRID]
    point = [{"form": "pointwise", "delta": None, **hp} for hp in hps]
    pair = [{"form": "pairwise", "delta": d, "micro_batch": MICRO_BATCH, **hp}
            for hp in hps for d in DELTAS]
    arch = "bi" if rung == R3 else "ce"
    forms = {R1: point, R2: pair, R3: point + pair}[rung]
    return [{**BASE, "rung": rung, "arch": arch, "max_length": MAX_LENGTH[arch], **f}
            for f in forms]


# --- validation ----------------------------------------------------------------------------

def per_jd_pairwise(frame: pd.DataFrame, scores: np.ndarray) -> pd.Series:
    """Per-JD pairwise agreement of `scores` with the run-11 score; JDs with all-equal
    scores are skipped (no pair)."""
    s = pd.Series(np.asarray(scores, float), index=frame.pair_id.to_numpy())
    out = {}
    for jd, g in frame.groupby("jd_id", sort=True):
        v = ordering.pairwise(g.score.to_numpy(float), s.loc[g.pair_id].to_numpy(float))
        if not math.isnan(v):
            out[jd] = v
    return pd.Series(out, dtype=float)


def pick(points: list[dict], key: str = "validation_pairwise") -> int:
    """Index of the best point; the earlier point wins a tie (A3.6)."""
    best = 0
    for i, p in enumerate(points):
        if p[key] > points[best][key]:
            best = i
    return best


def vs_random(values: pd.Series, seed: int = 0) -> dict:
    """The stop rule's figure (A3.7): mean per-JD Δ against random (0.5), JD-resampled CI."""
    diff = values - 0.5
    low, high = resample.mean_ci(diff, seed=seed)
    return {"delta": _r(diff.mean()), "ci": [_r(low), _r(high)], "n_jd": int(len(diff)),
            "beats_random": bool(low > 0)}


def _r(x: float) -> float:
    return round(float(x), DECIMALS) + 0.0


# --- training ------------------------------------------------------------------------------

def _schedule(opt, n_steps: int, warmup_ratio: float):
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(opt, math.ceil(warmup_ratio * n_steps), n_steps)


def ranknet_loop(score_hi_lo, params, prefs: pd.DataFrame, config: dict) -> list[float]:
    """The pairwise loop shared by both architectures: shuffled (seeded) batches of
    preference pairs, AdamW (no weight decay, as the HF trainer), linear warmup and decay,
    loss mean softplus(-(s_hi - s_lo)) over the step's pairs, its gradient accumulated over
    micro-batches of `micro_batch` pairs. Returns the mean loss per epoch."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(config["seed"])
    bs = config["batch"]
    micro = config.get("micro_batch", bs)
    n_steps = config["epochs"] * math.ceil(len(prefs) / bs)
    opt = torch.optim.AdamW(params, lr=config["lr"], weight_decay=0.0)
    sched = _schedule(opt, n_steps, config["warmup_ratio"])
    losses = []
    for _ in range(config["epochs"]):
        order = rng.permutation(len(prefs))
        total = 0.0
        for start in range(0, len(order), bs):
            step = order[start:start + bs]
            for m in range(0, len(step), micro):
                rows = prefs.iloc[step[m:m + micro]]
                s_hi, s_lo = score_hi_lo(rows)
                loss = F.softplus(-(s_hi - s_lo)).sum() / len(step)
                loss.backward()
                total += float(loss.detach()) * len(step)
            opt.step()
            sched.step()
            opt.zero_grad()
        losses.append(round(total / len(prefs), 6))
    return losses


def train_ce(frame: pd.DataFrame, config: dict, out: Path) -> dict:
    neural.require_torch()
    neural.set_determinism(config["seed"])
    from sentence_transformers.cross_encoder import CrossEncoder
    neural.quiet_loading()
    model = CrossEncoder(neural.MODEL_ID, device="cpu", local_files_only=True,
                         config_kwargs={"num_labels": 1})
    model.max_seq_length = config["max_length"]
    info: dict = {}
    if config["form"] == "pointwise":
        from datasets import Dataset
        from sentence_transformers.cross_encoder import (CrossEncoderTrainer,
                                                         CrossEncoderTrainingArguments, losses)
        data = Dataset.from_dict({"jd": frame.jd_text.tolist(), "cv": frame.cv_text.tolist(),
                                  "label": (frame.score.astype(float) / 100).tolist()})
        args = CrossEncoderTrainingArguments(
            output_dir=str(out) + "-work", num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch"], learning_rate=config["lr"],
            warmup_ratio=config["warmup_ratio"], seed=config["seed"],
            data_seed=config["seed"], report_to=[], save_strategy="no", use_cpu=True,
            disable_tqdm=True, logging_steps=20)
        CrossEncoderTrainer(model=model, args=args, train_dataset=data,
                            loss=losses.MSELoss(model)).train()
    else:
        prefs = batch.preference_pairs(frame, config["delta"])
        hf, tok = model.transformers_model, model.tokenizer
        hf.train()

        def score_hi_lo(rows):
            jd = rows.jd_text.tolist() * 2
            cv = rows.hi_text.tolist() + rows.lo_text.tolist()
            feats = tok(jd, cv, padding=True, truncation=True,
                        max_length=config["max_length"], return_tensors="pt")
            logits = hf(**feats).logits.squeeze(-1)
            return logits[:len(rows)], logits[len(rows):]

        info = {"preference_pairs": int(len(prefs)),
                "epoch_loss": ranknet_loop(score_hi_lo, list(hf.parameters()), prefs, config)}
        hf.eval()
    model.save(str(out))
    return info


def train_bi(frame: pd.DataFrame, config: dict, out: Path) -> dict:
    neural.require_torch()
    neural.set_determinism(config["seed"])
    from sentence_transformers import SentenceTransformer
    neural.quiet_loading()
    model = SentenceTransformer(neural.MODEL_ID, device="cpu", local_files_only=True)
    model.max_seq_length = config["max_length"]
    info: dict = {}
    if config["form"] == "pointwise":
        from datasets import Dataset
        from sentence_transformers import (SentenceTransformerTrainer,
                                           SentenceTransformerTrainingArguments)
        from sentence_transformers.sentence_transformer import losses
        data = Dataset.from_dict({"jd": frame.jd_text.tolist(), "cv": frame.cv_text.tolist(),
                                  "label": (frame.score.astype(float) / 100).tolist()})
        args = SentenceTransformerTrainingArguments(
            output_dir=str(out) + "-work", num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch"], learning_rate=config["lr"],
            warmup_ratio=config["warmup_ratio"], seed=config["seed"],
            data_seed=config["seed"], report_to=[], save_strategy="no", use_cpu=True,
            disable_tqdm=True, logging_steps=20)
        SentenceTransformerTrainer(model=model, args=args, train_dataset=data,
                                   loss=losses.CosineSimilarityLoss(model)).train()
    else:
        import torch.nn.functional as F
        prefs = batch.preference_pairs(frame, config["delta"])
        model.train()

        def embed(texts):
            return F.normalize(model(model.preprocess(texts))["sentence_embedding"], dim=-1)

        def score_hi_lo(rows):
            jd = embed(rows.jd_text.tolist())
            hi, lo = embed(rows.hi_text.tolist()), embed(rows.lo_text.tolist())
            return (COSINE_SCALE * (jd * hi).sum(-1), COSINE_SCALE * (jd * lo).sum(-1))

        info = {"preference_pairs": int(len(prefs)),
                "epoch_loss": ranknet_loop(score_hi_lo, list(model.parameters()), prefs,
                                           config)}
        model.eval()
    model.save(str(out))
    return info


# --- the systems ---------------------------------------------------------------------------

class TrainedCrossEncoder(BaseSystem):
    stage = "Stage 2 — cross-encoder"

    def __init__(self, rung: str, path: Path, config: dict) -> None:
        super().__init__()
        self.rung, self.name, self.config, self.path = rung, NAMES[rung], config, Path(path)
        self.trained_on = (f"run-11 LLM batch, train split, text_v2, {config['form']}"
                           + (f", δ={config['delta']}" if config["delta"] else ""))

    def _fit(self, train):
        neural.require_torch()
        from sentence_transformers.cross_encoder import CrossEncoder
        neural.quiet_loading()
        self.model = CrossEncoder(str(self.path), device="cpu", local_files_only=True)
        self.model.max_seq_length = self.config["max_length"]
        self.weights_digest = weights_digest(self.path)

    def _score(self, pairs):
        texts = list(zip(pairs.jd_text, pairs.cv_text))
        return np.asarray(self.model.predict(texts, batch_size=PREDICT_BATCH,
                                             show_progress_bar=False), dtype=float)

    def params(self):
        return int(sum(p.numel() for p in self.model.parameters()))


class TrainedBiEncoder(BaseSystem):
    stage = "Stage 2 — bi-encoder"

    def __init__(self, rung: str, path: Path, config: dict) -> None:
        super().__init__()
        self.rung, self.name, self.config, self.path = rung, NAMES[rung], config, Path(path)
        self.trained_on = (f"run-11 LLM batch, train split, text_v2, {config['form']}"
                           + (f", δ={config['delta']}" if config["delta"] else ""))

    def _fit(self, train):
        from candidate_screener.matching.sbert import Encoder
        self.weights_digest = weights_digest(self.path)
        self.encoder = Encoder(str(self.path), f"rerank-{self.weights_digest[:16]}")

    def _score(self, pairs):
        jd = self.encoder.encode(pairs.jd_text)
        cv = self.encoder.encode(pairs.cv_text)
        return np.einsum("ij,ij->i", jd, cv)

    def params(self):
        return int(sum(p.numel() for p in self.encoder.model.parameters()))


def system_for(config: dict, path: Path) -> BaseSystem:
    cls = TrainedBiEncoder if config["arch"] == "bi" else TrainedCrossEncoder
    return cls(config["rung"], path, config)


@dataclass
class Fitted:
    """One rung after selection: its grid trail and the selected, loaded system."""
    rung: str
    points: list[dict]
    selected: int
    system: BaseSystem
    validation: pd.Series = field(repr=False)
    costs: list[dict] = field(default_factory=list, repr=False)   # not in the record


def weights_dir(config: dict, data_digest: str, root: Path = WEIGHTS) -> Path:
    return Path(root) / f"{config['rung'].lower()}-{neural.fingerprint({**config, 'data': data_digest})}"


def fit_rung(rung: str, frame: pd.DataFrame, data_digest: str, root: Path = WEIGHTS,
             seed: int = 0) -> Fitted:
    """Train (or load cached) every grid point, score validation, select (A3.6)."""
    import time
    train = frame[frame.split == "train"]
    val = frame[frame.split == "validation"]
    points, values, costs = [], [], []
    for config in grid(rung):
        out = weights_dir(config, data_digest, root)
        info, seconds = {}, None
        if not (out / "config.json").exists():
            start = time.perf_counter()
            info = (train_bi if config["arch"] == "bi" else train_ce)(train, config, out)
            seconds = round(time.perf_counter() - start, 1)
            _release()
        system = system_for(config, out)
        system.fit(None)
        v = per_jd_pairwise(val, system.score(val).to_numpy())
        points.append({"form": config["form"], "delta": config["delta"], "lr": config["lr"],
                       "epochs": config["epochs"], "weights": out.name,
                       "validation_pairwise": _r(v.mean()), "n_jd": int(len(v))})
        costs.append({"weights": out.name, "train_seconds": seconds, **info})
        values.append(v)
        del system
        _release()
    best = pick(points)
    system = system_for(grid(rung)[best], weights_dir(grid(rung)[best], data_digest, root))
    system.fit(None)
    return Fitted(rung, points, best, system, values[best], costs)


def _release() -> None:
    """Hand freed model memory back to the OS between grid points (glibc keeps it)."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:                                              # pragma: no cover
        pass


def best_form(fitted: Fitted) -> str:
    return fitted.points[fitted.selected]["form"]
