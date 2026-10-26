"""Context-only overflow QA with exact passage offsets and explicit abstention."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import signal
import atexit
import time

from candidate_screener.config import PROJECT_ROOT
from .qa_assets import ROOT, verify_manifest


def passage_dict(passage):
    return asdict(passage) if is_dataclass(passage) else dict(passage)


def best_context_span(start_logits, end_logits, offsets, sequence_ids, *, max_answer_tokens=30):
    """Exclude question/special/padding tokens before selecting source boundaries."""
    context = [i for i, sequence in enumerate(sequence_ids) if sequence == 1 and offsets[i][1] > offsets[i][0]]
    best = None
    for position, start in enumerate(context):
        for end in context[position:position + max_answer_tokens]:
            if end - start >= max_answer_tokens:
                break
            score = float(start_logits[start] + end_logits[end])
            a, b = int(offsets[start][0]), int(offsets[end][1])
            if a < b and (best is None or score > best[0]):
                best = (score, a, b)
    return best


def answerable(best, null_score, threshold):
    return best is not None and math.isfinite(best[0]) and math.isfinite(null_score) and null_score - best[0] <= threshold


class ExtractiveQA:
    def __init__(self, manifest_path=ROOT / "manifest.json", *, no_answer_threshold=0., max_length=384, stride=128):
        if not math.isfinite(no_answer_threshold) or type(max_length) is not int or not 64 <= max_length <= 512 or type(stride) is not int or not 0 <= stride < max_length - 32:
            raise ValueError("Invalid QA threshold/context/stride configuration")
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        import torch
        self.manifest = verify_manifest(Path(manifest_path))
        checkpoint = Path(manifest_path).parent / "checkpoint"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("QA requires exact fast-tokenizer offset mappings")
        self.model = AutoModelForQuestionAnswering.from_pretrained(checkpoint, local_files_only=True).eval()
        torch.set_num_threads(4)
        self.threshold, self.max_length, self.stride = no_answer_threshold, max_length, stride

    def answer(self, question, passages):
        from .qa_list import list_windows, is_list_question
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError("QA requires a bounded nonempty question")
        passages = list(passages)
        if len(passages) > 32:
            raise ValueError("QA passage bound exceeded")
        spans, seen = [], set()
        remaining, incomplete = 32, False
        listed = is_list_question(question)
        for passage in passages:
            passage = passage_dict(passage)
            text = passage["quote"]
            if not isinstance(text, str) or len(text) > 100_000:
                raise ValueError("QA passage size bound exceeded")
            if listed and remaining == 0:
                incomplete |= bool(text.strip())
                continue
            contexts = list_windows(question, passage, limit=remaining if listed else 32)
            incomplete |= contexts.incomplete
            if listed:
                remaining -= len(contexts)
            for context, origin in contexts:
                best, null_score = self._context_answer(question, context)
                if answerable(best, null_score, self.threshold):
                    start, end = origin + best[1], origin + best[2]
                    key = (passage["passage_id"], start, end)
                    if key in seen:
                        continue
                    seen.add(key)
                    spans.append({"text": text[start:end], "passage_id": passage["passage_id"],
                                  "candidate_id": passage.get("candidate_id"), "start": start, "end": end,
                                  "score": best[0], "null_score_difference": null_score-best[0]})
        return {"status": "supported" if spans else "not_found", "spans": spans,
                "incomplete": incomplete, "exhaustive": False,
                "model_id": self.manifest["model_id"], "revision": self.manifest["revision"]}

    def _context_answer(self, question, text):
            import torch
            encoded = self.tokenizer(question, text, truncation="only_second", max_length=self.max_length,
                                     stride=self.stride, return_overflowing_tokens=True,
                                     return_offsets_mapping=True, padding=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping")
            encoded.pop("overflow_to_sample_mapping", None)
            best, null_score = None, float("inf")
            for index in range(len(encoded["input_ids"])):
                with torch.inference_mode():
                    result = self.model(**{key: value[index:index+1] for key, value in encoded.items()})
                start_logits, end_logits = result.start_logits[0], result.end_logits[0]
                cls = encoded["input_ids"][index].tolist().index(self.tokenizer.cls_token_id)
                null_score = min(null_score, float(start_logits[cls] + end_logits[cls]))
                candidate = best_context_span(start_logits.tolist(), end_logits.tolist(), offsets[index].tolist(), encoded.sequence_ids(index))
                if candidate is not None and (best is None or candidate[0] > best[0]):
                    best = candidate
            return best, null_score


class QaClient:
    """Persistent serialized subprocess; applicant passages never enter disk logs."""
    def __init__(self, manifest_path=ROOT / "manifest.json", *, timeout=120):
        if not isinstance(timeout, (int,float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("QA worker requires a positive finite timeout")
        self.manifest_path, self.timeout = str(manifest_path), timeout
        self._process = None
        self._buffer = b""
        self._lock = threading.Lock()
        atexit.register(self.close)

    def answer(self, question, passages):
        return self.answer_before(question, passages, time.monotonic() + self.timeout)

    def answer_before(self, question, passages, absolute_deadline):
        """Share the caller's deadline across lock contention and worker exchange."""
        if (isinstance(absolute_deadline, bool) or not isinstance(absolute_deadline, (int, float))
                or not math.isfinite(absolute_deadline)):
            raise ValueError("QA requires a finite absolute deadline")
        unavailable = {"status": "unavailable", "spans": [], "exhaustive": False}
        if not Path(self.manifest_path).is_file():
            return unavailable
        deadline = min(absolute_deadline, time.monotonic() + self.timeout)
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=remaining):
            # Another turn owns the worker: a queued timeout must not kill it.
            return unavailable
        try:
            try:
                if time.monotonic() >= deadline:
                    return unavailable
                if self._process is None or self._process.poll() is not None:
                    self.close()
                    environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                    self._process = subprocess.Popen(["uv", "run", "--project", str(PROJECT_ROOT / "runtime/transformer"), "--no-sync", "python", "-m", "candidate_screener.llm.qa_worker", "--manifest", self.manifest_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=environment, start_new_session=True)
                    os.set_blocking(self._process.stdin.fileno(), False)
                    os.set_blocking(self._process.stdout.fileno(), False)
                payload = (json.dumps({"question": question, "passages": [passage_dict(p) for p in passages]}) + "\n").encode("utf-8")
                return self._exchange(payload, deadline)
            except (OSError, ValueError, TimeoutError, BrokenPipeError):
                self.close(force=time.monotonic() >= deadline)
                return unavailable
        finally:
            self._lock.release()

    def _exchange(self, payload, deadline):
        """One newline-delimited response, with a shared write/read deadline."""
        import selectors
        pending = memoryview(payload)
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdin, selectors.EVENT_WRITE)
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("QA worker deadline")
                try:
                    pending = pending[os.write(self._process.stdin.fileno(), pending):]
                except BlockingIOError:
                    continue
            selector.unregister(self._process.stdin)
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("QA worker deadline")
                try:
                    chunk = os.read(self._process.stdout.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise OSError("QA worker stopped before completing a response")
                self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def close(self, *, force=False):
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGKILL if force else signal.SIGTERM)
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait()
            except ProcessLookupError:
                pass
        if self._process is not None:
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                if stream is not None:
                    stream.close()
            self._process = None
        self._buffer = b""
