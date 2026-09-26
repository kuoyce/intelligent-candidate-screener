"""Stage 4 rungs: cross-encoders that read the JD and CV together (MiniLM, one logit).

4a trains on A1 as the proposal intended, with labels collapsed to Fit / No Fit (D47):
folds 1-4 of the doubly-disjoint CV train, fold 0 held back for the class threshold.
4b trains on the weak pairs' train split, with the threshold set on weak validation.
Hyperparameters are fixed in advance. Nothing is fitted on the final set (D46).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.matching import neural
from candidate_screener.matching.base import BaseSystem

WEIGHTS = PROCESSED / "matching"
CE_CONFIG = {"base": neural.MODEL_ID, "loss": "BinaryCrossEntropyLoss", "epochs": 1,
             "lr": 2e-5, "batch": 16, "warmup_ratio": 0.1, "max_length": 384, "seed": 0}


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
        from sentence_transformers.cross_encoder import CrossEncoder
        fit_frame, val_frame, data_digest = self._training_data()
        config = {**CE_CONFIG, "data": data_digest, "rung": self.rung}
        out = WEIGHTS / f"ce-{self.rung}-{neural.fingerprint(config)}"
        if not (out / "config.json").exists():
            self._train(fit_frame, out)
        self.model = CrossEncoder(str(out), device="cpu", local_files_only=True)
        self.model.max_seq_length = CE_CONFIG["max_length"]
        self.threshold = best_threshold(val_frame.fit.to_numpy(), self._predict(val_frame))

    def _train(self, frame: pd.DataFrame, out) -> None:
        neural.set_determinism(CE_CONFIG["seed"])
        from datasets import Dataset
        from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                         CrossEncoderTrainingArguments, losses)
        model = CrossEncoder(neural.MODEL_ID, device="cpu", local_files_only=True,
                             config_kwargs={"num_labels": 1})
        model.max_seq_length = CE_CONFIG["max_length"]
        data = Dataset.from_dict({"jd": frame.jd_text.tolist(), "cv": frame.cv_text.tolist(),
                                  "label": frame.fit.astype(float).tolist()})
        args = CrossEncoderTrainingArguments(
            output_dir=str(out) + "-work", num_train_epochs=CE_CONFIG["epochs"],
            per_device_train_batch_size=CE_CONFIG["batch"], learning_rate=CE_CONFIG["lr"],
            warmup_ratio=CE_CONFIG["warmup_ratio"], seed=CE_CONFIG["seed"],
            data_seed=CE_CONFIG["seed"], report_to=[], save_strategy="no", use_cpu=True,
            disable_tqdm=True, logging_steps=20)
        CrossEncoderTrainer(model=model, args=args, train_dataset=data,
                            loss=losses.BinaryCrossEntropyLoss(model)).train()
        model.save(str(out))

    def _predict(self, pairs: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(list(zip(pairs.jd_text, pairs.cv_text)),
                                             batch_size=32, show_progress_bar=False), dtype=float)

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
