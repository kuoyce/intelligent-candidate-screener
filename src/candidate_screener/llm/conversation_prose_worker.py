"""One private prose request; stdout contains only a bounded result or reason."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import sys
from time import monotonic

from .base import ContextOverflow, ProviderUnavailable
from .conversation_prose import ProseComposer, ProseError, ProseUnavailable, deadline_limit
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
        if deadline <= monotonic():
            response = {"error": "deadline_exceeded"}
        else:
            config = QwenConfig(**request["config"])
            remaining = min(deadline_limit(config), config.timeout_seconds, deadline - monotonic())
            if remaining <= 0:
                raise ProseUnavailable("deadline_exceeded")
            config = replace(config, timeout_seconds=remaining, json_mode=False)
            composer = ProseComposer(QwenProvider(config), deadline_seconds=remaining)
            result = composer.compose_before(request["question"], request["frame"], request["facts"], deadline)
            response = {"result": result}
    except ContextOverflow:
        response = {"error": "context_overflow"}
    except ProseUnavailable as error:
        response = {"error": error.reason}
    except ProviderUnavailable:
        response = {"error": "provider_unavailable"}
    except ProseError as error:
        response = {"error": "invalid_prose", "reason": error.reason}
    except (KeyError, TypeError, ValueError):
        response = {"error": "invalid_prose", "reason": "invalid_context"}
    except Exception:
        response = {"error": "worker_failed"}
    sys.stdout.write(json.dumps(response, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
