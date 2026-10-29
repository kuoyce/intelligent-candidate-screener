"""Draft synthetic CUI scenarios and strict source/partition validation.

Validation establishes structural consistency, not independent human review or
semantic correctness of an expected answer. Evaluation fixtures remain draft
until the study's human review and freeze gates have been completed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_FIXTURES = Path(__file__).with_name("fixtures.json")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value: Any, name: str) -> None:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")


def _strings(value: Any, name: str, *, nonempty: bool = False) -> None:
    _require(isinstance(value, list), f"{name} must be a list")
    _require(not nonempty or bool(value), f"{name} must not be empty")
    for item in value:
        _text(item, name)


def _source_span(fact: dict, sources: dict[str, str]) -> None:
    _require(isinstance(fact, dict), "required fact must be an object")
    owner = fact.get("owner")
    _require(isinstance(owner, str) and owner in sources, "fact owner is outside active family")
    quote, start, end = fact.get("quote"), fact.get("start"), fact.get("end")
    _text(quote, "fact quote")
    _require(type(start) is int and type(end) is int, "span offsets must be integers")
    _require(0 <= start < end <= len(sources[owner]), "span offsets outside source")
    _require(sources[owner][start:end] == quote, "fact quote does not match source span")


def validate_fixtures(data: dict) -> dict:
    """Raise ValueError on malformed or cross-scope fixtures; return counts.

    The optional ``scope_event`` switches to an alternate scope inside the same
    family, preserving independence of family-level evaluation clusters.
    """
    _require(isinstance(data, dict), "fixtures must be an object")
    _require(type(data.get("schema_version")) is int and data["schema_version"] == 1,
             "unsupported fixture schema_version")
    _require(data.get("synthetic_only") is True, "fixtures must declare synthetic_only")
    _require(isinstance(data.get("review_status"), str)
             and data["review_status"] in {"draft", "reviewed"}, "invalid review_status")
    families = data.get("families")
    _require(isinstance(families, list) and bool(families), "families must be a nonempty list")
    ids: set[str] = set()
    lineage_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    sources_by_family: dict[str, dict[str, str]] = {}
    scopes_by_family: dict[str, dict[str, dict]] = {}

    def identity(value: Any) -> None:
        _text(value, "id")
        _require(value not in ids, f"duplicate id: {value}")
        ids.add(value)

    for family in families:
        _require(isinstance(family, dict), "family must be an object")
        identity(family.get("id"))
        split = family.get("split")
        _require(isinstance(split, str) and split in {"dev", "eval"}, "invalid family split")
        lineage = family.get("lineage")
        _text(lineage, "lineage")
        _require(lineage not in lineage_splits or lineage_splits[lineage] == split,
                 "source/template lineage crosses dev/eval split")
        lineage_splits[lineage] = split
        jd = family.get("jd")
        _require(isinstance(jd, dict), "jd must be an object")
        identity(jd.get("id"))
        _text(jd.get("title"), "JD title")
        _text(jd.get("text"), "JD text")
        candidates = family.get("candidates")
        _require(isinstance(candidates, list) and 2 <= len(candidates) <= 3,
                 "each family needs two or three candidates")
        sources = {jd["id"]: jd["text"]}
        for candidate in candidates:
            _require(isinstance(candidate, dict), "candidate must be an object")
            identity(candidate.get("id"))
            _text(candidate.get("text"), "candidate text")
            sources[candidate["id"]] = candidate["text"]
        for source in sources.values():
            normalized = " ".join(source.casefold().split())
            _require(normalized not in source_splits or source_splits[normalized] == split,
                     "identical source text crosses dev/eval split")
            source_splits[normalized] = split
        sources_by_family[family["id"]] = sources
        scopes_by_family[family["id"]] = {}
        scopes = family.get("scopes", [])
        _require(isinstance(scopes, list), "scopes must be a list")
        for scope in scopes:
            _require(isinstance(scope, dict), "scope must be an object")
            identity(scope.get("id"))
            jd = scope.get("jd")
            _require(isinstance(jd, dict), "scope jd must be an object")
            identity(jd.get("id"))
            _text(jd.get("title"), "scope JD title")
            _text(jd.get("text"), "scope JD text")
            members = scope.get("candidates")
            _require(isinstance(members, list) and 2 <= len(members) <= 3,
                     "scope needs two or three candidates")
            scope_sources = {jd["id"]: jd["text"]}
            for member in members:
                _require(isinstance(member, dict), "scope candidate must be an object")
                identity(member.get("id"))
                _text(member.get("text"), "scope candidate text")
                scope_sources[member["id"]] = member["text"]
            sources_by_family[scope["id"]] = scope_sources
            scopes_by_family[family["id"]][scope["id"]] = scope
            for source in scope_sources.values():
                normalized = " ".join(source.casefold().split())
                _require(normalized not in source_splits or source_splits[normalized] == split,
                         "identical source text crosses dev/eval split")
                source_splits[normalized] = split

    counts = {split: {"families": 0, "questions": 0, "dialogues": 0,
                      "dialogue_turns": 0, "eligible_turns": 0} for split in ("dev", "eval")}

    def turn(value: Any, active: dict, family: dict, *, dialogue: bool) -> dict:
        _require(isinstance(value, dict), "turn must be an object")
        identity(value.get("id"))
        _text(value.get("question"), "question")
        _require(len(value["question"]) <= 2000, "question exceeds application bound")
        _require(type(value.get("eligible_generation")) is bool, "eligible_generation must be boolean")
        if "scope_event" in value:
            event = value["scope_event"]
            _require(dialogue and isinstance(event, dict), "scope_event requires a dialogue")
            _require(event.get("kind") == "switch_scope", "unsupported scope_event kind")
            target = event.get("scope_id")
            _require(isinstance(target, str) and target in scopes_by_family[family["id"]],
                     "scope target must belong to same family")
            active = scopes_by_family[family["id"]][target]
        expected = value.get("expected")
        _require(isinstance(expected, dict), "expected must be an object")
        participants = expected.get("participants")
        _strings(participants, "participants")
        owners = {c["id"] for c in active["candidates"]}
        _require(len(participants) == len(set(participants)) and set(participants) <= owners,
                 "participants must be unique active-family candidate IDs")
        for field in ("criterion", "intent"):
            _text(expected.get(field), field)
        for field in ("clarification", "uncertainty"):
            _require(type(expected.get(field)) is bool, f"{field} must be boolean")
        for field in ("mandatory_caveats", "prohibited_inferences"):
            _strings(expected.get(field), field)
        _strings(expected.get("completion_criteria"), "completion_criteria", nonempty=True)
        facts = expected.get("required_facts")
        _require(isinstance(facts, list), "required_facts must be a list")
        for fact in facts:
            _source_span(fact, sources_by_family[active["id"]])
        _require(bool(facts) or expected["uncertainty"] or expected["clarification"]
                 or not value["eligible_generation"],
                 "answerable factual turn must identify supporting evidence")
        counts[family["split"]]["eligible_turns"] += int(value["eligible_generation"])
        return active

    for family in families:
        summary = counts[family["split"]]
        summary["families"] += 1
        questions, dialogues = family.get("questions"), family.get("dialogues")
        _require(isinstance(questions, list) and bool(questions), "questions must be nonempty")
        _require(isinstance(dialogues, list), "dialogues must be a list")
        for question in questions:
            turn(question, family, family, dialogue=False)
            summary["questions"] += 1
        for dialogue in dialogues:
            _require(isinstance(dialogue, dict), "dialogue must be an object")
            identity(dialogue.get("id"))
            turns = dialogue.get("turns")
            _require(isinstance(turns, list) and 4 <= len(turns) <= 6,
                     "dialogue must have four to six turns")
            active = family
            for item in turns:
                active = turn(item, active, family, dialogue=True)
            summary["dialogues"] += 1
            summary["dialogue_turns"] += len(turns)
    return {"schema_version": 1, "synthetic_only": True,
            "review_status": data["review_status"], "splits": counts}


def load_fixtures(path: str | Path | None = None) -> dict:
    """Load and structurally validate a fixture file without changing it."""
    data = json.loads(Path(path or DEFAULT_FIXTURES).read_text(encoding="utf-8"))
    validate_fixtures(data)
    return data
