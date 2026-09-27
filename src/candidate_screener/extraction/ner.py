"""Learned skill tagger: MiniLM token classifier, BIO over SKILL = skill ∪ knowledge (5.3).

Initialised from the locally cached `sentence-transformers/all-MiniLM-L6-v2` (no
download), fine-tuned on SkillSpan **train**, early-stopped on SkillSpan **validation**
strict span F1, CPU only. Seeds, thread count and `torch.use_deterministic_algorithms`
are fixed so a retrain can be compared with the committed record. Weights are a cache
under `data/processed/extraction/ner/` and are never committed (D23).

Words longer than one chunk (`max_subwords`) are split into consecutive chunks, in
training and at inference, so no word is truncated away.

    uv run python -m candidate_screener.extraction.ner --train --seed 0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from candidate_screener.config import PROCESSED

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NER_DIR = PROCESSED / "extraction" / "ner"
LABELS = ("O", "B", "I")
_TRAIL = ",;:.!?)]}\"'"
_LEAD = "([{\"'"


@dataclass(frozen=True)
class NerConfig:
    base_model: str = BASE_MODEL
    max_subwords: int = 128
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 6
    patience: int = 2
    seed: int = 0
    threads: int = 4
    selection: str = "SkillSpan validation strict span F1, SKILL union"


def base_model_path(name: str = BASE_MODEL) -> str:
    """The cached snapshot directory; raises rather than downloading."""
    from huggingface_hub import snapshot_download
    return snapshot_download(name, local_files_only=True)


def set_determinism(seed: int, threads: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def state_digest(model) -> str:
    """sha256 over the state dict in key order: two bit-identical trainings agree."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --- chunking and alignment ------------------------------------------------------------

def word_subword_counts(tokenizer, words: list[str]) -> list[int]:
    enc = tokenizer(words, add_special_tokens=False)["input_ids"]
    return [len(x) for x in enc]


def chunk(counts: list[int], limit: int) -> list[tuple[int, int]]:
    """Greedy `(start, end)` word ranges whose subword total fits `limit`."""
    out, start, total = [], 0, 0
    for i, c in enumerate(counts):
        if i > start and total + c > limit:
            out.append((start, i))
            start, total = i, 0
        total += c
    if counts:
        out.append((start, len(counts)))
    return out


def _encode(tokenizer, batch_words: list[list[str]], max_len: int,
            batch_tags: list[list[str]] | None = None):
    import torch
    enc = tokenizer(batch_words, is_split_into_words=True, truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt")
    first = []
    labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
    for b in range(len(batch_words)):
        seen, pos = set(), {}
        for j, w in enumerate(enc.word_ids(b)):
            if w is not None and w not in seen:
                seen.add(w)
                pos[w] = j
                if batch_tags is not None:
                    labels[b, j] = LABELS.index(batch_tags[b][w])
        first.append(pos)
    return enc, labels, first


# --- the tagger ---------------------------------------------------------------------------

class NerTagger:
    def __init__(self, model, tokenizer, config: NerConfig):
        self.model, self.tokenizer, self.config = model, tokenizer, config

    def predict_word_tags(self, sentences: list[list[str]], batch_size: int = 64) -> list[list[str]]:
        """BIO tags per word; a word with no subword (e.g. a zero-width space) is `O`."""
        import torch
        torch.set_num_threads(self.config.threads)
        self.model.eval()
        limit = self.config.max_subwords - 2
        pieces = []  # (sentence index, word start, word end)
        for si, words in enumerate(sentences):
            for s, e in chunk(word_subword_counts(self.tokenizer, words) if words else [], limit):
                pieces.append((si, s, e))
        out = [["O"] * len(w) for w in sentences]
        order = sorted(range(len(pieces)), key=lambda k: pieces[k][2] - pieces[k][1])
        with torch.no_grad():
            for b0 in range(0, len(order), batch_size):
                idx = order[b0:b0 + batch_size]
                words = [sentences[pieces[k][0]][pieces[k][1]:pieces[k][2]] for k in idx]
                enc, _, first = _encode(self.tokenizer, words, self.config.max_subwords)
                pred = self.model(**enc).logits.argmax(-1)
                for row, k in enumerate(idx):
                    si, s, _e = pieces[k]
                    for w, j in first[row].items():
                        out[si][s + w] = LABELS[int(pred[row, j])]
        return out

    def predict_spans(self, text: str) -> list[tuple[int, int]]:
        """Character spans in `text`. Lines are tagged separately; edge punctuation trimmed."""
        from candidate_screener.extraction.skillspan import bio_spans
        lines: list[list[tuple[int, int]]] = []
        for m in re.finditer(r"[^\n\r]+", text):
            words = [(m.start() + w.start(), m.start() + w.end())
                     for w in re.finditer(r"\S+", m.group())]
            if words:
                lines.append(words)
        tags = self.predict_word_tags([[text[s:e] for s, e in ws] for ws in lines])
        out = []
        for ws, tg in zip(lines, tags):
            for a, b in bio_spans(tg):
                s, e = ws[a][0], ws[b - 1][1]
                while e > s and text[e - 1] in _TRAIL:
                    e -= 1
                while s < e and text[s] in _LEAD:
                    s += 1
                if e > s:
                    out.append((s, e))
        return out


def load_ner(path: Path = NER_DIR) -> NerTagger:
    """The cached fine-tuned tagger; raises if it has not been trained."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    cfg_file = Path(path) / "ner-config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(
            f"no trained tagger at {path}; run "
            "`uv run python -m candidate_screener.extraction.ner --train --seed 0`")
    meta = json.loads(cfg_file.read_text())
    config = NerConfig(**meta["config"])
    torch.set_num_threads(config.threads)
    model = AutoModelForTokenClassification.from_pretrained(path, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return NerTagger(model, tok, config)


@lru_cache(maxsize=1)
def _default() -> NerTagger:
    return load_ner()


def predict_spans(text: str, tagger: NerTagger | None = None) -> list[tuple[int, int]]:
    return (tagger or _default()).predict_spans(text)


# --- training -----------------------------------------------------------------------------

def _training_pieces(tokenizer, split: str, limit: int) -> list[tuple[list[str], list[str]]]:
    from candidate_screener.extraction import skillspan as ss
    out = []
    for _, row in ss.load(split).iterrows():
        words, tags = row["tokens"], ss.training_tags(row)
        for s, e in chunk(word_subword_counts(tokenizer, words), limit):
            out.append((words[s:e], tags[s:e]))
    return out


def _batches(lengths: list[int], batch_size: int, rng: np.random.Generator) -> list[list[int]]:
    """Length-bucketed batches: shuffle, sort within pools of 50 batches, shuffle batches."""
    idx = rng.permutation(len(lengths)).tolist()
    pool = batch_size * 50
    batches = []
    for p0 in range(0, len(idx), pool):
        chunk_ = sorted(idx[p0:p0 + pool], key=lambda i: lengths[i])
        batches += [chunk_[b:b + batch_size] for b in range(0, len(chunk_), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def skillspan_tags(tagger: NerTagger, split: str) -> list[list[str]]:
    from candidate_screener.extraction import skillspan as ss
    return tagger.predict_word_tags(list(ss.load(split).tokens))


def counts_from_tags(tags: list[list[str]], split: str, scope: str = "union") -> np.ndarray:
    """Per-sentence count vectors (token grid) of predicted BIO tags against `scope` gold."""
    from candidate_screener.extraction import skillspan as ss
    from candidate_screener.extraction.metrics import unit_counts
    gold = ss.prepared(split, scope)
    return np.array([unit_counts(set(ss.bio_spans(t)), g[2]) for t, g in zip(tags, gold)])


def skillspan_counts(tagger: NerTagger, split: str, scope: str = "union") -> np.ndarray:
    """Per-sentence count vectors of the tagger on a SkillSpan split (token grid)."""
    return counts_from_tags(skillspan_tags(tagger, split), split, scope)


def train(config: NerConfig = NerConfig(), out_dir: Path = NER_DIR, verbose: bool = True) -> dict:
    """Fine-tune, early-stop on validation, save the best epoch; returns the metadata."""
    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    from candidate_screener.extraction.metrics import prf

    set_determinism(config.seed, config.threads)
    base = base_model_path(config.base_model)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}, local_files_only=True)
    tagger = NerTagger(model, tok, config)
    data = _training_pieces(tok, "train", config.max_subwords - 2)
    lengths = [sum(word_subword_counts(tok, w)) for w, _ in data]
    rng = np.random.default_rng(config.seed)
    steps_per_epoch = -(-len(data) // config.batch_size)
    total = steps_per_epoch * config.max_epochs
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=config.lr)
    sched = get_linear_schedule_with_warmup(opt, int(config.warmup_ratio * total), total)
    history, best, best_state, stale = [], -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in _batches(lengths, config.batch_size, rng):
            words = [data[i][0] for i in batch]
            enc, labels, _ = _encode(tok, words, config.max_subwords, [data[i][1] for i in batch])
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            loss_sum += float(loss.detach())
        m = prf(skillspan_counts(tagger, "validation").sum(axis=0))
        row = {"epoch": epoch, "train_loss": round(loss_sum / steps_per_epoch, 4),
               "val_strict_f1": round(float(m["strict"][2]), 4),
               "val_partial_f1": round(float(m["partial"][2]), 4)}
        history.append(row)
        if verbose:
            print(f"  epoch {epoch}: {row}  ({time.time() - t0:.0f}s)", flush=True)
        if row["val_strict_f1"] > best:
            best, best_state, stale = row["val_strict_f1"], copy.deepcopy(model.state_dict()), 0
            best_epoch = epoch
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    meta = {"config": asdict(config), "best_epoch": best_epoch, "history": history,
            "train_pieces": len(data), "weights_sha256": state_digest(model),
            "base_model_snapshot": Path(base).name}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "ner-config.json").write_text(json.dumps(meta, indent=2) + "\n")
    if verbose:
        print(f"  trained in {time.time() - t0:.0f}s; best epoch {best_epoch}; "
              f"weights {meta['weights_sha256'][:12]} -> {out_dir}")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", action="store_true", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=NER_DIR)
    a = ap.parse_args(argv)
    train(NerConfig(seed=a.seed), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
