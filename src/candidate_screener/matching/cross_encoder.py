"""Stage 4 rungs: cross-encoders that read the JD and CV together (MiniLM, one logit).

4a trains on A1 as the proposal intended, with labels collapsed to Fit / No Fit (D47):
folds 1-4 of the doubly-disjoint CV train, fold 0 held back for selection and threshold.
4b trains on the weak pairs' train split, with weak validation held back.

Hyperparameters are chosen on that held-back split (plan 03 row 4.1), from a two-point
grid, by validation AUC: the conventional setting, and a larger learning rate with more
epochs for a new head on little data. The first grid point with lr 2e-5 and 1 epoch left
4a's loss at ln 2 (it learned nothing). Nothing is fitted or chosen on the final set (D46).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.matching import neural
from candidate_screener.matching.base import BaseSystem

WEIGHTS = PROCESSED / "matching"
CE_CONFIG = {"base": neural.MODEL_ID, "loss": "BinaryCrossEntropyLoss", "epochs": 1,
             "lr": 2e-5, "batch": 16, "warmup_ratio": 0.1, "max_length": 384, "seed": 0}
#: The selection grid. The first point is `CE_CONFIG` unchanged, so its cache is reused.
CE_GRID = ({}, {"lr": 5e-5, "epochs": 3})
PREDICT_BATCH = 32
#: Call-level memo of `_predict` (plan 2026-10-08, W4), in process: `classify` re-predicts
#: the pairs `score` already predicted. A hit needs the same weights, `max_seq_length`,
#: batch size and the same ordered (JD, CV) texts, so it replays only an identical call
#: (outputs can depend on batch padding, so a reordered or partial call is a new key).
_PREDICTIONS: dict[str, np.ndarray] = {}


def _text_digest(t: object) -> bytes:
    raw = t if isinstance(t, str) else f"\x00{type(t).__name__}:{t!r}"
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).digest()


def call_key(weights: str, max_seq_length: int, batch_size: int,
             texts: list[tuple[object, object]]) -> str:
    """SHA-256 over the weights path, `max_seq_length`, batch size and the ordered pairs."""
    h = hashlib.sha256(f"{weights}\x1f{max_seq_length}\x1f{batch_size}\x1e".encode("utf-8"))
    for jd, cv in texts:
        h.update(_text_digest(jd) + _text_digest(cv))
    return h.hexdigest()


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """The probability cut maximising binary macro-F1 on a validation split."""
    from candidate_screener.evaluation.indomain import _cls_stats
    grid = np.unique(np.round(p, 4))
    scores = [_cls_stats(y, (p >= t).astype(int))["macro_f1"] for t in grid]
    return float(grid[int(np.argmax(scores))])


class _CrossEncoder(BaseSystem):
    stage = "Stage 4 — cross-encoder"
    can_explain = False

    def _training_data(self) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        raise NotImplementedError

    def _fit(self, train):
        neural.require_torch()
        from sklearn.metrics import roc_auc_score
        fit_frame, val_frame, data_digest = self._training_data()
        self.selection = []
        best = None
        for override in CE_GRID:
            config = {**CE_CONFIG, **override, "data": data_digest, "rung": self.rung}
            out = WEIGHTS / f"ce-{self.rung}-{neural.fingerprint(config)}"
            if not (out / "config.json").exists():
                self._train(fit_frame, out, config)
            self._load(out)
            p = self._predict(val_frame)
            auc = float(roc_auc_score(val_frame.fit, p))
            self.selection.append({"lr": config["lr"], "epochs": config["epochs"],
                                   "validation_auc": round(auc, 4),
                                   "validation_spread": round(float(p.max() - p.min()), 4)})
            if best is None or auc > best[0]:
                best = (auc, out, p)
        _, out, p = best
        self._load(out)
        self.threshold = best_threshold(val_frame.fit.to_numpy(), p)

    def _load(self, out) -> None:
        from sentence_transformers.cross_encoder import CrossEncoder
        neural.quiet_loading()
        self.model = CrossEncoder(str(out), device="cpu", local_files_only=True)
        self.model.max_seq_length = CE_CONFIG["max_length"]
        self.weights_path = str(out)

    def _call_key(self, texts: list[tuple[object, object]]) -> str:
        return call_key(self.weights_path, self.model.max_seq_length, PREDICT_BATCH, texts)

    def _train(self, frame: pd.DataFrame, out, config: dict = CE_CONFIG) -> None:
        neural.set_determinism(config["seed"])
        from datasets import Dataset
        from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                         CrossEncoderTrainingArguments, losses)
        neural.quiet_loading()
        model = CrossEncoder(neural.MODEL_ID, device="cpu", local_files_only=True,
                             config_kwargs={"num_labels": 1})
        model.max_seq_length = CE_CONFIG["max_length"]
        data = Dataset.from_dict({"jd": frame.jd_text.tolist(), "cv": frame.cv_text.tolist(),
                                  "label": frame.fit.astype(float).tolist()})
        args = CrossEncoderTrainingArguments(
            output_dir=str(out) + "-work", num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch"], learning_rate=config["lr"],
            warmup_ratio=config["warmup_ratio"], seed=config["seed"],
            data_seed=config["seed"], report_to=[], save_strategy="no", use_cpu=True,
            disable_tqdm=True, logging_steps=20)
        CrossEncoderTrainer(model=model, args=args, train_dataset=data,
                            loss=losses.BinaryCrossEntropyLoss(model)).train()
        model.save(str(out))

    def _predict(self, pairs: pd.DataFrame) -> np.ndarray:
        texts = list(zip(pairs.jd_text, pairs.cv_text))
        key = self._call_key(texts)
        if key not in _PREDICTIONS:
            _PREDICTIONS[key] = np.asarray(self.model.predict(
                texts, batch_size=PREDICT_BATCH, show_progress_bar=False), dtype=float)
        return _PREDICTIONS[key].copy()

    def _score(self, pairs):
        return self._predict(pairs)

    def classify(self, pairs: pd.DataFrame) -> pd.Series:
        """Fit / No Fit at the threshold chosen on this rung's own validation split."""
        calls = np.where(self._predict(pairs) >= self.threshold, "Fit", "No Fit")
        return pd.Series(calls, index=pairs.pair_id.to_numpy())

    def params(self):
        return int(sum(p.numel() for p in self.model.model.parameters()))


class CrossEncoderA1(_CrossEncoder):
    name, rung = "cross-encoder (A1)", "4a"
    trained_on = "A1 train folds 1-4 (Fit / No Fit); threshold on fold 0"

    def _training_data(self):
        from candidate_screener.data import fit_split
        from candidate_screener.matching import corpus
        train = corpus.a1_pairs("train")
        fit, held = fit_split.fold_masks(train.rename(columns={"cv_id": "resume_id"}), k=5,
                                         seed=0, fold=0)
        digest = neural.fingerprint({"fit": sorted(train.pair_id[fit]),
                                     "held": sorted(train.pair_id[held])})
        return train[fit], train[held], digest


class CrossEncoderWeak(_CrossEncoder):
    name, rung = "cross-encoder (weak)", "4b"
    trained_on = "weak pairs, A2 train region; threshold on weak validation"

    def _training_data(self):
        from candidate_screener.data import weak_pairs
        frame = weak_pairs.with_text(weak_pairs.load()).rename(columns={"label": "fit"})
        return (frame[frame.split == "train"], frame[frame.split == "validation"],
                neural.file_digest(weak_pairs.MANIFEST))
