"""Inference-only plain-prose adapter; claim support is checked by the application.

This module enforces transport, prompt, size and output-format boundaries. Passing
these checks is not proof of entailment and must not be labelled factual validation.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from importlib.resources import files
import json
import math
import os
import re
import signal
import subprocess
import sys
from time import monotonic

from .base import ContextOverflow, ProviderUnavailable, redact_content

SYSTEM = files("candidate_screener.llm").joinpath("hr_system_prompt_v3.txt").read_text(encoding="utf-8").strip()
PROMPT_VERSION = "hiring-desk-v3-summary-conclusion-7"
PROSE_FAILURES = frozenset({"invalid_response_size", "non_prose_output", "invalid_context"})


class ProseError(ValueError):
    def __init__(self, reason):
        self.reason = reason if isinstance(reason, str) and reason in PROSE_FAILURES else "invalid_context"
        super().__init__("Conversational prose rejected: " + self.reason)


class ProseUnavailable(ProviderUnavailable):
    def __init__(self, reason):
        self.reason = reason if isinstance(reason, str) and reason in {"deadline_exceeded", "worker_unavailable", "worker_failed",
                                          "worker_invalid_output", "provider_unavailable"} else "worker_failed"
        super().__init__("Conversational generation unavailable: " + self.reason)


def deadline_limit(config):
    """The approved CPU 4B trial has a larger finite budget; v1/v2 stay unchanged."""
    return 60. if config.model == "Qwen/Qwen3.5-4B" else 30.


def configured_prose_composer(values=None):
    """Load provisioned configuration only; never download, fit or start a server."""
    from .qwen import QwenConfig
    from .qwen_assets import local_settings

    values = dict(os.environ if values is None else values)
    if str(values.get("SCREENER_QWEN_DISABLED", "false")).lower() == "true":
        return None
    try:
        if not values.get("SCREENER_QWEN_BASE_URL") and not values.get("SCREENER_QWEN_MODEL"):
            selected_manifest = values.get("SCREENER_QWEN_MANIFEST")
            if selected_manifest:
                from .gguf_server import local_settings as gguf_settings
                values = {**gguf_settings(selected_manifest), **values}
            else:
                values = {**local_settings(), **values}
        config = QwenConfig.from_mapping(values)
        if config is None:
            return None
        config = replace(config, max_context_tokens=min(config.max_context_tokens, 4096),
                         max_output_tokens=min(config.max_output_tokens, 256),
                         timeout_seconds=min(config.timeout_seconds, deadline_limit(config)), temperature=0., json_mode=False)
        return IsolatedProseComposer(config)
    except (OSError, KeyError, TypeError, ValueError, ProviderUnavailable):
        return None


def checked_prose(raw):
    """Keep only a completed final conclusion; semantics remain the app's job."""
    if not isinstance(raw, str) or not raw.strip() or len(raw.encode("utf-8")) > 8192:
        raise ProseError("invalid_response_size")
    text = raw.strip()
    sections = list(re.finditer(r"(?m)^\s*\[(Reasoning Steps|Conclusion)\]:", text))
    if sections:
        names = [match[1] for match in sections]
        if names not in (["Conclusion"], ["Reasoning Steps", "Conclusion"]):
            raise ProseError("non_prose_output")
        prefix = text[:sections[0].start()].strip()
        if prefix and (names != ["Conclusion"] or not re.match(r"^\d+[.)]\s", prefix)):
            raise ProseError("non_prose_output")
        text = text[sections[-1].end():].strip()
        if not text:
            raise ProseError("invalid_response_size")
    elif re.match(r"^\d+[.)]\s", text):
        # The prompt opens a brief evidence-check section. An unfinished
        # numbered continuation must never be persisted as the final answer.
        raise ProseError("non_prose_output")
    try:
        json.loads(text)
    except (ValueError, TypeError):
        pass
    else:
        raise ProseError("non_prose_output")
    if (text.startswith(("{", "[", "<")) or "```" in text or re.search(r"\[(?:Reasoning Steps|Conclusion)\]", text)
            or re.search(r"\b(?:chunk\s*\d+|evidence[_ -]?id|relation_ids)\b|\[S\d+\]", text, re.I)
            or re.search(r"(?m)^\s*(?:import |from \w+ import |def \w+\(|class \w+[:(]|#include|SELECT .* FROM )", text)):
        raise ProseError("non_prose_output")
    return redact_content(text)


def _context_text(question, frame, facts, history):
    """Compact model-facing context; hashes and claim-check variants stay private.

    Deduplicate identical owner/context pairs, never split or shorten passages.
    Retain cell statuses independently: a missing cell is information, not absence.
    """
    labels = facts.get("labels", {})
    if not isinstance(labels, dict):
        raise ProseError("invalid_context")
    owners = frame.get("selected_candidates", [])
    jd_id = facts.get("jd_id", frame.get("jd_id"))

    def label(owner):
        value = labels.get(owner)
        if isinstance(value, str) and value:
            return value
        if owner == jd_id:
            return "The role"
        return f"Candidate {owners.index(owner) + 1}" if owner in owners else "Source"

    lines = ["### YOUR TASK", "Trusted review labels:", ", ".join(label(owner) for owner in owners) or "Job description review"]
    criteria = frame.get("criteria", [])
    if isinstance(criteria, list) and criteria:
        lines.append("Discussion criteria: " + ", ".join(str(item) for item in criteria))
    if frame.get("intent") == "comparison_explanation":
        subject, reference = frame.get("explanation_subject"), frame.get("explanation_reference")
        if subject in owners and reference in owners:
            lines.append(f"Requested explanation: {label(subject)} compared with {label(reference)}.")
    priority = facts.get("job_priority")
    if priority is not None and not isinstance(priority, str):
        raise ProseError("invalid_context")
    lines.extend(["", "[Job Priority]: " + json.dumps(
        priority or "No explicit job priority supplied; do not invent a priority or overall winner.", ensure_ascii=False)])
    matrix = facts.get("evidence_matrix", [])
    if not isinstance(matrix, list):
        raise ProseError("invalid_context")
    lines.extend(["", "[Evidence Matrix]:"])
    for item in matrix:
        if (not isinstance(item, dict) or not isinstance(item.get("criterion"), str)
                or not isinstance(item.get("statuses"), dict)
                or set(item["statuses"]) != set(owners)
                or any(not isinstance(value, str) or not value.strip() for value in item["statuses"].values())):
            raise ProseError("invalid_context")
        # Escape newlines and delimiter characters so source/criterion strings
        # cannot create additional prompt sections or change another owner's flag.
        criterion = json.dumps(item["criterion"], ensure_ascii=False)
        scope = json.dumps(item.get("scope", "Evidence"), ensure_ascii=False)
        flags = " | ".join(f"{label(owner)} ({item['statuses'][owner]})" for owner in owners)
        if any(re.search(r"[\n\r\[\]|()]", value) for value in item["statuses"].values()):
            raise ProseError("invalid_context")
        lines.append(f"* {criterion} [{scope}]: {flags}")
    if not matrix:
        lines.append("* Matrix unavailable for this turn; infer no ratings or verified qualifications.")
    lines.extend(["", "[Selected Passages]:", "Background Evidence (source data):"])
    seen = set()
    evidence = facts.get("evidence", [])
    cells = facts.get("cells", [])
    if not isinstance(evidence, list) or not isinstance(cells, list):
        raise ProseError("invalid_context")
    for item in [*evidence, *cells]:
        if not isinstance(item, dict):
            raise ProseError("invalid_context")
        owner, context = item.get("owner"), item.get("context")
        if isinstance(context, str) and context.strip() and (owner, context) not in seen:
            lines.append(f"{label(owner)}: {json.dumps(context, ensure_ascii=False)}")
            seen.add((owner, context))
    statuses = set()
    for item in cells:
        status = item.get("support_status")
        if status:
            row = f"{label(item.get('owner'))} — {item.get('criterion') or 'retrieved information'}: {status}."
            if row not in statuses:
                lines.append(row)
                statuses.add(row)
    if not seen:
        lines.append("No source passage was retrieved. Do not invent missing facts.")
    relations = facts.get("checked_relations", facts.get("relations", []))
    matrix_relations = facts.get("matrix_relations", [])
    if not isinstance(relations, list) or not isinstance(matrix_relations, list):
        raise ProseError("invalid_context")
    differences = list(dict.fromkeys(row["text"] for row in [*relations, *matrix_relations]
                                    if isinstance(row, dict) and isinstance(row.get("text"), str)))
    if differences:
        lines.extend(["", "Checked comparisons (preserve limitations and counterpoints):", *differences])
    if history:
        lines.extend(["", "Recent conversation (discourse only, not evidence):"])
        for turn in history:
            lines.append("Previous question: " + json.dumps(turn["question"], ensure_ascii=False))
            lines.append("Previous reply: " + json.dumps(turn["answer"], ensure_ascii=False))
    lines.extend(["", "Align the final conclusion with the Evidence Matrix and Selected Passages above.",
                  "Current user question:", question, "", "[Reasoning Steps]:"])
    return "\n".join(lines)


class ProseComposer:
    def __init__(self, provider, *, deadline_seconds=30.):
        if (isinstance(deadline_seconds, bool) or not isinstance(deadline_seconds, (int, float))
                or not math.isfinite(deadline_seconds) or not 0 < deadline_seconds <= 60):
            raise ValueError("Use a finite generation deadline within 60 seconds")
        self.provider, self.deadline_seconds = provider, float(deadline_seconds)

    def compose(self, question, frame, facts):
        return self.compose_before(question, frame, facts, monotonic() + self.deadline_seconds)

    def compose_before(self, question, frame, facts, absolute_deadline):
        started = monotonic()
        deadline = _deadline(absolute_deadline, self.deadline_seconds)
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ProseError("invalid_context")
        if not isinstance(frame, dict) or not isinstance(facts, dict):
            raise ProseError("invalid_context")
        requested = frame.get("selected_candidates", [])
        if (not isinstance(requested, list) or len(requested) > 3
                or any(not isinstance(owner, str) or not owner for owner in requested)
                or len(set(requested)) != len(requested)):
            raise ProseError("invalid_context")
        history = facts.get("history", [])
        if not isinstance(history, list) or any(
                not isinstance(turn, dict) or not all(isinstance(turn.get(key), str) for key in ("question", "answer"))
                for turn in history):
            raise ProseError("invalid_context")
        history = [{"question": turn["question"], "answer": turn["answer"]} for turn in history[-4:]]
        provider = self.provider
        output_budget, context_budget = provider.max_output_tokens, provider.max_context_tokens
        if (type(output_budget) is not int or type(context_budget) is not int
                or not 0 < output_budget <= 256 or not output_budget < context_budget <= 4096):
            raise ValueError("V3 requires context <=4096 and output <=256 tokens")
        if getattr(getattr(provider, "config", None), "json_mode", False):
            raise ValueError("V3 requires plain-text provider mode")
        while True:
            encoded = _context_text(question, frame, facts, history)
            if len(encoded.encode()) > 128_000:
                raise ContextOverflow("V3 evidence exceeds byte budget")
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": redact_content(encoded)}]
            _remaining(deadline)
            tokens = provider.count_tokens(messages)
            if type(tokens) is not int or tokens < 0:
                raise ValueError("Invalid provider token count")
            if tokens + output_budget <= context_budget:
                break
            if not history:
                raise ContextOverflow("Complete evidence cannot fit the configured context")
            history.pop(0)  # Drop whole old exchanges, never evidence or qualifications.
        _remaining(deadline)
        text = checked_prose(provider.complete(messages))
        _remaining(deadline)
        config = getattr(provider, "config", None)
        model = getattr(config, "model", None) or getattr(provider, "model_id", None) or type(provider).__name__
        provenance = {"prompt_version": PROMPT_VERSION,
                      "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                      "input_tokens": tokens, "output_max_new_tokens": output_budget,
                      "history_exchanges": len(history), "elapsed_seconds": monotonic() - started,
                      "attempts": 1, "semantic_validation": "application_required"}
        if config is not None:
            provenance.update({key: getattr(config, key, None) for key in
                               ("revision", "runtime", "quantization", "temperature")})
        return {"text": text, "mode": "qwen_prose", "model": str(model)[:200], "provenance": provenance}


def _deadline(value, limit=30.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("A finite shared deadline is required")
    deadline = min(float(value), monotonic() + limit)
    _remaining(deadline)
    return deadline


def _remaining(deadline):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise ProseUnavailable("deadline_exceeded")
    return remaining


def _worker_command():
    return [sys.executable, "-m", "candidate_screener.llm.conversation_prose_worker"]


class IsolatedProseComposer(ProseComposer):
    """Kill/reap the client worker at deadline; server work has its own bound."""
    def __init__(self, config):
        super().__init__(None, deadline_seconds=min(config.timeout_seconds, deadline_limit(config)))
        self.config = replace(config, max_context_tokens=min(config.max_context_tokens, 4096),
                              max_output_tokens=min(config.max_output_tokens, 256), json_mode=False)

    def compose_before(self, question, frame, facts, absolute_deadline):
        deadline = _deadline(absolute_deadline, self.deadline_seconds)
        payload = json.dumps({"config": asdict(self.config), "question": question,
                              "frame": frame, "facts": facts, "deadline": deadline},
                             ensure_ascii=False, allow_nan=False).encode()
        if len(payload) > 128_000:
            raise ContextOverflow("Prose request exceeds worker byte budget")
        process = None
        try:
            process = subprocess.Popen(_worker_command(), stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
            output, _ = process.communicate(payload, timeout=_remaining(deadline))
        except subprocess.TimeoutExpired:
            raise ProseUnavailable("deadline_exceeded") from None
        except OSError:
            raise ProseUnavailable("worker_unavailable") from None
        finally:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
        _remaining(deadline)
        if process.returncode != 0 or len(output) > 64_000:
            raise ProseUnavailable("worker_failed")
        try:
            response = json.loads(output)
        except (ValueError, TypeError):
            raise ProseUnavailable("worker_invalid_output") from None
        if not isinstance(response, dict):
            raise ProseUnavailable("worker_invalid_output")
        if response.get("error"):
            code = response["error"]
            if code == "context_overflow":
                raise ContextOverflow("Complete evidence exceeds model context")
            if code == "invalid_prose":
                raise ProseError(response.get("reason"))
            raise ProseUnavailable(code)
        if set(response) != {"result"} or not isinstance(response["result"], dict):
            raise ProseUnavailable("worker_invalid_output")
        result = response["result"]
        if (set(result) != {"text", "mode", "model", "provenance"} or result["mode"] != "qwen_prose"
                or not isinstance(result["model"], str) or not isinstance(result["provenance"], dict)):
            raise ProseUnavailable("worker_invalid_output")
        result["text"] = checked_prose(result["text"])
        return result
