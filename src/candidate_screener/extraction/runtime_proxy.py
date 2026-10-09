"""Persistent local JSON worker for the separately locked neural runtime."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess

PREFIX = "CANDIDATE_SCREENER_WORKER:"


class TransformerRuntimeProxy:
    def __init__(self, model_path: Path, device: str = "cpu"):
        project = Path(__file__).resolve().parents[3]
        runtime = project / "runtime" / "transformer"
        if not (runtime / "pyproject.toml").is_file():
            raise RuntimeError("Isolated transformer runtime is missing")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project / "src")
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        self.process = subprocess.Popen(
            ["uv", "run", "--project", str(runtime), "--no-sync", "python", "-m",
             "candidate_screener.extraction.transformer_worker", "--model-path", str(model_path),
             "--device", device], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, env=environment, cwd=project,
        )
        atexit.register(self.close)
        ready = self._read()
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError("Isolated transformer runtime failed to initialize: " + str(ready.get("error", "unknown")))
        self._parameters = ready["chunk_parameters"]
        self.runtime_versions = ready["runtime_versions"]

    def _read(self) -> dict:
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Isolated transformer worker stopped; check runtime dependencies and local model artifacts")
            if line.startswith(PREFIX):
                return json.loads(line[len(PREFIX):])

    def chunk_parameters(self) -> dict:
        return dict(self._parameters)

    def extract_document(self, doc: dict, **kwargs):
        self.process.stdin.write(json.dumps({"document": doc, "parameters": kwargs}) + "\n")
        self.process.stdin.flush()
        result = self._read()
        if result.get("status") != "ok":
            raise RuntimeError("Isolated transformer inference failed: " + str(result.get("error", "unknown")))
        return result["records"], result["coverage"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
