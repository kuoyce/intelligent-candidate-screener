"""One private stdin/stdout Qwen composition; no documents or secrets in logs."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import sys
from time import monotonic

from .base import ContextOverflow, ProviderUnavailable
from .conversation_qwen import ConversationComposer, CompositionPlanError
from .qwen import QwenConfig, QwenProvider


def main():
    try:
        raw = sys.stdin.buffer.read(128_001)
        if len(raw) > 128_000:
            raise ValueError("Oversized request")
        request = json.loads(raw)
        deadline = request["deadline"]
        if type(deadline) not in {int, float} or not math.isfinite(deadline):
            raise ValueError("Invalid deadline")
        remaining = min(30., deadline - monotonic())
        if remaining <= 0:
            response = {"error": "deadline_exceeded"}
        else:
            config = replace(QwenConfig(**request["config"]), timeout_seconds=remaining)
            composer = ConversationComposer(QwenProvider(config), deadline_seconds=remaining)
            result = composer.compose(request["question"], request["frame"], request["facts"])
            response = {"result": result} if monotonic() <= deadline else {"error": "deadline_exceeded"}
    except ContextOverflow:
        response = {"error": "context_overflow"}
    except ProviderUnavailable:
        response = {"error": "provider_unavailable"}
    except CompositionPlanError as error:
        response = {"error": "invalid_plan", "reason": error.reason}
    except (KeyError, TypeError, ValueError):
        response = {"error": "invalid_plan"}
    except Exception:
        response = {"error": "worker_failed"}
    sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
