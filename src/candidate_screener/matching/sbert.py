"""Stage 3 rungs: Sentence-BERT bi-encoders over the cached MiniLM.

A document is embedded as the normalised mean of its 800-character chunks, so a long JD
is not cut at 256 tokens. Embeddings are cached per (model, text) under
`data/processed/embeddings/`. 3a is zero-shot. 3b is fine-tuned with
MultipleNegativesRankingLoss on the weak pairs' train split (JD, positive CV, hard or
easy negative CV), with hyperparameters fixed in advance: nothing is tuned (D46).

`text_variant` (D53): under `v2` the weak pairs' CV text has no `Position`, and the
embedding cache and the fine-tuned weights live under their own keys. The `v1` keys are
the pre-D53 ones unchanged, so the committed `v1` record still reproduces.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from candidate_screener.config import PROCESSED
from candidate_screener.matching import neural
from candidate_screener.matching.base import BaseSystem

CHUNK_CHARS = 800
EMBED_CACHE = PROCESSED / "embeddings"
WEIGHTS = PROCESSED / "matching"
FT_CONFIG = {"base": neural.MODEL_ID, "loss": "MultipleNegativesRankingLoss", "epochs": 1,
             "lr": 2e-5, "batch": 16, "warmup_ratio": 0.1, "max_seq_length": 256, "seed": 0}


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return [text[i:i + size] for i in range(0, max(len(text), 1), size)] or [""]


class Encoder:
    """Chunk-mean document embeddings with a persistent per-text cache."""

    def __init__(self, model_path: str, key: str) -> None:
        neural.require_torch()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
        self.path = EMBED_CACHE / f"{key}.npz"
        self.cache: dict[str, np.ndarray] = {}
        if self.path.exists():
            data = np.load(self.path)
            self.cache = dict(zip(data["keys"].tolist(), data["vectors"]))

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(str(text).encode()).hexdigest()

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        todo = sorted({t for t in texts if self._key(t) not in self.cache})
        if todo:
            pieces, owner = [], []
            for i, t in enumerate(todo):
                for c in chunks(t):
                    pieces.append(c)
                    owner.append(i)
            vectors = self.model.encode(pieces, batch_size=64, normalize_embeddings=True,
                                        show_progress_bar=False, convert_to_numpy=True)
            owner = np.array(owner)
            for i, t in enumerate(todo):
                v = vectors[owner == i].mean(axis=0)
                self.cache[self._key(t)] = (v / np.linalg.norm(v)).astype(np.float32)
            EMBED_CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, keys=np.array(list(self.cache)), vectors=np.stack(list(self.cache.values())))
        return np.stack([self.cache[self._key(t)] for t in texts])


class _BiEncoder(BaseSystem):
    stage = "Stage 3 — semantic"
    can_explain = False

    def _score(self, pairs):
        jd = self.encoder.encode(pairs.jd_text)
        cv = self.encoder.encode(pairs.cv_text)
        return np.einsum("ij,ij->i", jd, cv)

    def params(self):
        return int(sum(p.numel() for p in self.encoder.model.parameters()))


class SbertZeroShot(_BiEncoder):
    name, rung = "SBERT (zero-shot)", "3a"
    trained_on = "nothing (pretrained MiniLM, cached)"

    def __init__(self, text_variant: str = "v1") -> None:
        super().__init__()
        self.text_variant = text_variant

    @staticmethod
    def cache_key(text_variant: str = "v1") -> str:
        from candidate_screener.matching import corpus
        return f"minilm-zero-shot{corpus.variant_tag(text_variant)}"

    def _fit(self, train):
        self.encoder = Encoder(neural.MODEL_ID, self.cache_key(self.text_variant))


def weak_triplets(frame: pd.DataFrame) -> pd.DataFrame:
    """(JD, positive CV, negative CV): the hard negative where one exists, else the easy one."""
    pos = frame[frame.kind == "positive"].set_index("jd_id")
    hard = frame[frame.kind == "hard_negative"].drop_duplicates("jd_id").set_index("jd_id")
    easy = frame[frame.kind == "easy_negative"].drop_duplicates("jd_id").set_index("jd_id")
    neg = hard.cv_text.combine_first(easy.cv_text)
    out = pd.DataFrame({"anchor": pos.jd_text, "positive": pos.cv_text,
                        "negative": neg.reindex(pos.index)}).dropna()
    return out.sort_index().reset_index(drop=True)


class SbertWeak(_BiEncoder):
    name, rung = "SBERT (weak fine-tuned)", "3b"
    trained_on = "weak pairs, A2 train region (metadata labels)"

    def __init__(self, text_variant: str = "v1") -> None:
        super().__init__()
        from candidate_screener.matching import corpus
        self.text_variant = corpus.check_variant(text_variant)
        if text_variant != "v1":
            self.trained_on = f"{SbertWeak.trained_on}, text_{text_variant}"

    @staticmethod
    def config(text_variant: str = "v1") -> dict:
        """Everything the weights depend on. `v1` omits the key so its pre-D53 fingerprint
        (and cached weights) stay valid; any other variant names itself."""
        from candidate_screener.data import weak_pairs
        from candidate_screener.matching import corpus
        config = {**FT_CONFIG, "weak_pairs": neural.file_digest(weak_pairs.MANIFEST)}
        if corpus.check_variant(text_variant) != "v1":
            config["text_variant"] = text_variant
        return config

    @classmethod
    def cache_key(cls, text_variant: str = "v1") -> str:
        return f"sbert-weak-{neural.fingerprint(cls.config(text_variant))}"

    def _fit(self, train):
        from candidate_screener.data import weak_pairs
        key = self.cache_key(self.text_variant)
        out = WEIGHTS / key
        if not (out / "config.json").exists():
            self._train(weak_pairs.with_text(weak_pairs.load().query("split == 'train'"),
                                             self.text_variant), out)
        self.encoder = Encoder(str(out), key)

    @staticmethod
    def _train(frame: pd.DataFrame, out) -> None:
        neural.set_determinism(FT_CONFIG["seed"])
        from datasets import Dataset
        from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                           SentenceTransformerTrainingArguments, losses)
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
        model = SentenceTransformer(neural.MODEL_ID, device="cpu", local_files_only=True)
        model.max_seq_length = FT_CONFIG["max_seq_length"]
        data = Dataset.from_pandas(weak_triplets(frame), preserve_index=False)
        args = SentenceTransformerTrainingArguments(
            output_dir=str(out) + "-work", num_train_epochs=FT_CONFIG["epochs"],
            per_device_train_batch_size=FT_CONFIG["batch"], learning_rate=FT_CONFIG["lr"],
            warmup_ratio=FT_CONFIG["warmup_ratio"], seed=FT_CONFIG["seed"],
            data_seed=FT_CONFIG["seed"], batch_sampler=BatchSamplers.NO_DUPLICATES,
            report_to=[], save_strategy="no", use_cpu=True, disable_tqdm=True, logging_steps=10)
        SentenceTransformerTrainer(model=model, args=args, train_dataset=data,
                                   loss=losses.MultipleNegativesRankingLoss(model)).train()
        model.save(str(out))
