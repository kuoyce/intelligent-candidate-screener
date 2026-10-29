"""Execute existing CUI services on invented sources with explicit provider provenance."""
from __future__ import annotations

from copy import deepcopy
import json
import time
from types import SimpleNamespace

from candidate_screener.app.conversation_v3 import ConversationV3Service
from candidate_screener.app.portal_extractive import PortalExtractiveService
from candidate_screener.app.conversation_smoke import SourceOnlyQA
from .protocol import digest, runtime_identity


class StubComposer:
    """Exercise the real validator; never masquerade as real model inference."""
    deadline_seconds = 30.

    def compose_before(self, question, frame, facts, deadline):
        choices = facts.get("checked_paraphrases", [])
        text = choices[0]["text"] if choices else "Synthetic output without established support."
        return {"text": text, "model": "synthetic-harness-stub", "provenance": {"provider": "stub"}}


class FrozenQA:
    """Freeze exact QA calls inside one standalone pair; dialogues use fresh wrappers."""
    def __init__(self, client):
        self.client, self.outputs = client, {}
        self.read_only = False

    def answer(self, question, passages):
        return self._answer(question, passages, None)

    def answer_before(self, question, passages, absolute_deadline):
        if time.monotonic() >= absolute_deadline:
            raise TimeoutError("Extraction deadline exhausted")
        return self._answer(question, passages, absolute_deadline)

    def _answer(self, question, passages, deadline):
        key = digest([question, passages])
        if key not in self.outputs:
            if self.read_only:
                raise ValueError("Paired extraction request differs from frozen baseline")
            bounded = getattr(self.client, "answer_before", None)
            result = (bounded(question, passages, deadline) if deadline is not None and callable(bounded)
                      else self.client.answer(question, passages))
            self.outputs[key] = deepcopy(result)
        return deepcopy(self.outputs[key])


class RecordingComposer:
    """Synthetic-only diagnostics of attempted output, without touching application logging."""
    def __init__(self, composer):
        self.composer = composer
        self.deadline_seconds = getattr(composer, "deadline_seconds", 30.)
        self.config = getattr(composer, "config", None)
        self.reset()

    def reset(self):
        self.attempted, self.seconds, self.error = None, 0., None

    def compose_before(self, question, frame, facts, deadline):
        started = time.monotonic()
        try:
            output = self.composer.compose_before(question, frame, facts, deadline)
            self.attempted = deepcopy(output)
            return output
        except Exception as exc:
            self.error = type(exc).__name__
            raise
        finally:
            self.seconds += time.monotonic() - started


class ObservedService(ConversationV3Service):
    """Instrument the existing service without reimplementing its routing."""
    def _finish(self, snapshot, question, result, deadline, history=None):
        self.prepared_sha256 = digest(result)
        self.preparation_seconds = time.monotonic() - self.started
        return super()._finish(snapshot, question, result, deadline, history)

    def answer(self, *args, **kwargs):
        self.started = time.monotonic()
        self.prepared_sha256, self.preparation_seconds = None, None
        return super().answer(*args, **kwargs)


def snapshot_for(family, scope=None):
    source = scope or family
    return SimpleNamespace(snapshot_id=f"cui-{family['id']}-{digest(source)[:16]}",
                           jd_id=source["jd"]["id"], jd_title=source["jd"]["title"],
                           jd_text=source["jd"]["text"],
                           candidates=tuple({"cv_id": c["id"], "rank": i,
                                             "preview_text": c["text"],
                                             "preview_status": "available_redacted_text"}
                                            for i, c in enumerate(source["candidates"], 1)))


def source_documents(snapshot):
    return {snapshot.jd_id: snapshot.jd_text,
            **{c["cv_id"]: c["preview_text"] for c in snapshot.candidates}}


def _service(qa, composer):
    return ObservedService(PortalExtractiveService(qa_client=qa), composer=composer,
                           turn_timeout=getattr(composer, "deadline_seconds", 30.))


def _turn(service, snapshot, turn, family, task_id, task_type, arm, context, history):
    recorder = service.composer
    if isinstance(recorder, RecordingComposer):
        recorder.reset()
    started = time.monotonic()
    answer, error = {}, None
    try:
        answer = service.answer(snapshot, turn["question"], context=context, history=history)
    except Exception as exc:
        # Record failure without exception text (may contain prompts or sensitive provider fields).
        error = type(exc).__name__
    elapsed = time.monotonic() - started
    provenance = answer.get("generation_provenance", {})
    record = {"family_id": family["id"], "task_id": task_id, "turn_id": turn["id"],
              "task_type": task_type, "arm": arm, "question": turn["question"], "answer": answer,
              "eligible_generation": turn["eligible_generation"], "expected": deepcopy(turn["expected"]),
              "source_documents": source_documents(snapshot), "history": deepcopy(history),
              "elapsed_seconds": elapsed, "preparation_seconds": service.preparation_seconds,
              "prepared_sha256": service.prepared_sha256,
              "generation_mode": answer.get("generation_mode"),
              "fallback_reason": provenance.get("fallback_reason"),
              "error": error, "completed": error is None and bool(answer.get("text")),
              "scope_digest": digest(source_documents(snapshot))}
    record["preparation_timing_scope"] = "reader before context construction; not total retrieval/context time"
    record["generation_seconds"] = recorder.seconds if isinstance(recorder, RecordingComposer) else 0.
    record["attempted_generation"] = recorder.attempted if isinstance(recorder, RecordingComposer) else None
    record["generation_error"] = recorder.error if isinstance(recorder, RecordingComposer) else None
    return record


def run(fixtures, *, split="dev", provider="stub", run_id, composer=None, qa=None, manifest=None, protocol_path=None):
    from .fixtures import validate_fixtures
    validate_fixtures(fixtures)
    if split not in {"dev", "eval"} or provider not in {"stub", "live"}:
        raise ValueError("Unknown split/provider")
    if provider == "stub" and split != "dev":
        raise ValueError("Do not expose evaluation outputs while debugging; use development fixtures")
    if provider == "live" and (composer is None or qa is None):
        raise ValueError("Live runs require both an explicit generator and QA client")
    if split == "eval" and manifest is None:
        raise ValueError("Evaluation requires a verified frozen manifest")
    if split == "eval":
        from .protocol import verify_freeze, verify_selected_runtime
        if protocol_path is None:
            raise ValueError("Evaluation requires the frozen protocol path")
        verify_freeze(manifest, fixtures, protocol_path)
        verify_selected_runtime(manifest, composer, qa)
    composer = composer if provider == "live" else StubComposer()
    composer = RecordingComposer(composer)
    qa = qa if provider == "live" else SourceOnlyQA()
    selected = [f for f in fixtures["families"] if f["split"] == split]
    records, expected_ids = [], []
    pair_index = 0
    for family in selected:
        initial = snapshot_for(family)
        for question in family["questions"]:
            expected_ids.append(question["id"])
            frozen = FrozenQA(qa)
            # Alternate arm order; neither generator is involved in extraction.
            arms = ("controlled", "v3") if pair_index % 2 == 0 else ("v3", "controlled")
            pair_index += 1
            paired = []
            for arm in arms:
                service = _service(frozen, composer if arm == "v3" else None)
                record = _turn(service, initial, question, family, question["id"], "standalone", arm, None, [])
                record["qa_cache"] = "paired_replay" if frozen.read_only else "first_arm"
                record["cache_state"] = record["qa_cache"]
                record["timing_scope"] = "service_call_with_paired_QA_cache; not portal end-to-end"
                records.append(record)
                paired.append(record)
                frozen.read_only = True
            if paired[0]["prepared_sha256"] != paired[1]["prepared_sha256"]:
                # Never silently count differing evidence as a fair generator comparison.
                for record in paired:
                    record.update(error="PairedEvidenceMismatch", completed=False)
        for dialogue in family["dialogues"]:
            expected_ids.extend(turn["id"] for turn in dialogue["turns"])
            arms = ("controlled", "v3") if pair_index % 2 == 0 else ("v3", "controlled")
            pair_index += 1
            for arm in arms:
                service = _service(qa, composer if arm == "v3" else None)
                snapshot, context, history = initial, None, []
                for turn in dialogue["turns"]:
                    event = turn.get("scope_event")
                    if event:
                        if event.get("kind") != "switch_scope":
                            raise ValueError("Unsupported scope event")
                        scope = next(s for s in family["scopes"] if s["id"] == event["scope_id"])
                        snapshot, context, history = snapshot_for(family, scope), None, []
                    record = _turn(service, snapshot, turn, family, dialogue["id"], "dialogue", arm, context, history)
                    record["qa_cache"] = "no_harness_cache"
                    record["cache_state"] = "QA_worker_reuse; generator_coldness_unknown"
                    record["timing_scope"] = "service_call; excludes portal UI and entity-cache preparation"
                    records.append(record)
                    if record["completed"]:
                        answer = record["answer"]
                        context = answer.get("context")
                        history.append({"question": turn["question"], "answer": deepcopy(answer)})
    if not expected_ids:
        raise ValueError("No tasks selected")
    result = {"schema_version": 1, "run_id": run_id,
              "kind": "stub_development" if provider == "stub" else ("live_evaluation" if split == "eval" else "live_development"),
              "metadata": {"split": split, "provider": provider, "synthetic_only": True,
                           "fixtures_sha256": digest(fixtures), "runtime": runtime_identity(composer),
                           "study_id": manifest.get("study_id") if manifest else None,
                           "limits": ["Service-level study; asynchronous GLiNER and portal UI not exercised",
                                      "Stub results never establish model quality",
                                      "Standalone paired QA cache timings are not production latency",
                                      "First use may include cold QA load; shared generator coldness not asserted"],
                           "extraction": "RoBERTa local QA" if provider == "live" else "SourceOnlyQA unavailable-model stub"},
              "expected_turn_ids": expected_ids, "records": records}
    # Ensure the persisted artifact can be reproduced as standard JSON.
    json.dumps(result, allow_nan=False)
    return result
