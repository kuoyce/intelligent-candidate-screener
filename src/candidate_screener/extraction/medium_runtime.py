"""Tracked offline medium worker with private diagnostic log."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
import tempfile

from candidate_screener.config import PROJECT_ROOT
from .runtime_proxy import PREFIX


class MediumRuntimeProxy:
    def __init__(self, model_path: Path, device="cpu"):
        logs = PROJECT_ROOT / "logs"
        logs.mkdir(exist_ok=True)
        self._stderr = tempfile.NamedTemporaryFile(mode="w+", prefix="medium-worker-", suffix=".log",
                                                   dir=logs, delete=False)
        self.diagnostic_log = self._stderr.name
        environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "src"),
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        self.process = subprocess.Popen(
            ["uv", "run", "--project", str(PROJECT_ROOT / "runtime/gliner-medium"),
             "--no-sync", "python", "-m", "candidate_screener.extraction.medium_worker",
             "--model-path", str(model_path), "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, env=environment, cwd=PROJECT_ROOT)
        atexit.register(self.close)
        try:
            ready = self._read()
            if ready.get("status") != "ready":
                raise RuntimeError(f"Medium worker initialization failed ({ready.get('error')}); see {self.diagnostic_log}")
            self._parameters = ready["chunk_parameters"]
            self.runtime_versions = ready["runtime_versions"]
        except Exception:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            if line.startswith(PREFIX):
                return json.loads(line[len(PREFIX):])
        raise RuntimeError(f"Medium worker stopped; see {self.diagnostic_log}")

    def chunk_parameters(self):
        return dict(self._parameters)

    def extract_document(self, doc, **kwargs):
        self.process.stdin.write(json.dumps({"document": doc, "parameters": kwargs}) + "\n")
        self.process.stdin.flush()
        result = self._read()
        if result.get("status") != "ok":
            raise RuntimeError(f"Medium inference failed ({result.get('error')}); see {self.diagnostic_log}")
        return result["records"], result["coverage"]

    def close(self):
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if hasattr(self, "_stderr"):
            self._stderr.close()
