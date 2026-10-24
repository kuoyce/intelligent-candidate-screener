"""Frozen CE inference in the local neural runtime, without the main pandas stack."""
from __future__ import annotations

import json
import math
from pathlib import Path
from threading import Lock

from candidate_screener.llm.qwen_assets import asset_digest


class LocalRerankRuntime:
    def __init__(self, tokenizer, model, torch, manifest):
        self.tokenizer, self.model, self.torch = tokenizer, model, torch
        self.manifest = dict(manifest)
        self.lock = Lock()

    def metadata(self):
        return {"name": self.manifest["name"], "checkpoint_sha256": self.manifest["sha256"],
                "score_semantics": self.manifest["score_semantics"]}

    def score(self, body):
        if (not isinstance(body, dict) or set(body) != {"checkpoint_sha256", "jd_text", "cv_text"}
                or body["checkpoint_sha256"] != self.manifest["sha256"]):
            raise ValueError("Configured reranker artifact does not match this request.")
        left, right = body["jd_text"], body["cv_text"]
        if (not isinstance(left, list) or not isinstance(right, list)
                or not 1 <= len(left) == len(right) <= 32
                or any(not isinstance(t, str) or not t.strip() or len(t) > 200000 for t in left + right)):
            raise ValueError("A bounded nonempty pair batch is required.")
        values = []
        with self.lock, self.torch.inference_mode():
            for start in range(0, len(left), self.manifest.get("batch_size", 8)):
                stop = start + self.manifest.get("batch_size", 8)
                inputs = self.tokenizer(left[start:stop], right[start:stop], padding=True,
                                        truncation=self.manifest["truncation_policy"],
                                        max_length=self.manifest["max_length"], return_tensors="pt")
                logits = self.model(**inputs).logits
                if logits.ndim != 2 or logits.shape[1] != 1:
                    raise ValueError("Cross-encoder must supply one scalar logit per pair.")
                values.extend(float(x) for x in logits[:, 0].detach().cpu().tolist())
        if len(values) != len(left) or not all(math.isfinite(v) for v in values):
            raise ValueError("Cross-encoder returned invalid scores.")
        return {"checkpoint_sha256": self.manifest["sha256"], "scores": values}


def load_rerank_runtime(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Invalid reranker manifest.")
    for key in ("checkpoint", "tokenizer", "name", "sha256", "selection_record"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError("Incomplete reranker manifest.")
    checkpoint = (path.parent / manifest["checkpoint"]).resolve()
    if (path.parent / manifest["tokenizer"]).resolve() != checkpoint:
        raise ValueError("Tokenizer must belong to the pinned checkpoint.")
    if asset_digest(checkpoint, exclude_download_cache=False) != manifest["sha256"]:
        raise ValueError("Reranker checkpoint digest mismatch.")
    if manifest.get("text_variant") != "text_v2" or manifest.get("score_semantics") not in {"ordering_logit", "binary_logit"}:
        raise ValueError("Reranker input/score semantics are required.")
    if (type(manifest.get("max_length")) is not int or not 1 <= manifest["max_length"] <= 8192
            or manifest.get("truncation_policy") not in {"longest_first", "only_second"}
            or type(manifest.get("batch_size", 8)) is not int or not 1 <= manifest.get("batch_size", 8) <= 32):
        raise ValueError("Bounded reranker tokenization and batch settings required.")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True,
                                                              trust_remote_code=False, torch_dtype=torch.float32)
    if model.config.num_labels != 1:
        raise ValueError("Cross-encoder must have one scalar output.")
    model.to("cpu")
    model.eval()
    return LocalRerankRuntime(tokenizer, model, torch, manifest)
