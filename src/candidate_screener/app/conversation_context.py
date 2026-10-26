"""Source-owned v3 context and conservative, explicitly bounded prose checks.

This is not a general semantic entailment model. Only registered paraphrases of
complete source predicates and checked comparison relations pass. Novel wording
falls back and remains measurable rather than being labelled verified.
"""
from __future__ import annotations

import hashlib
import json
import re

from candidate_screener.annotation.redact import redact_text
from .comparison_facts import _citations, _degree_matches_requirement, build_facts


def scope_digest(snapshot):
    payload = {"snapshot": snapshot.snapshot_id, "jd": snapshot.jd_id,
               "jd_text": snapshot.jd_text,
               "candidates": [{key: row.get(key) for key in
                               ("cv_id", "rank", "preview_text", "preview_status")}
                              for row in snapshot.candidates]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def scoped_history(snapshot, history):
    """Only previously accepted, unchanged-scope v3 exchanges are discourse data."""
    digest, selected = scope_digest(snapshot), []
    for turn in history if isinstance(history, (list, tuple)) else []:
        if not isinstance(turn, dict) or not isinstance(turn.get("question"), str):
            continue
        answer = turn.get("answer")
        if not isinstance(answer, dict):
            continue
        context, provenance = answer.get("context", {}), answer.get("generation_provenance", {})
        if (not isinstance(context, dict) or not isinstance(provenance, dict)
                or answer.get("mode") != "conversation_v3"
                or context.get("snapshot_id") != snapshot.snapshot_id or context.get("jd_id") != snapshot.jd_id
                or provenance.get("scope_digest") != digest
                or answer.get("generation_mode") not in {"qwen_prose", "controlled", "citation_reopen", "qwen_fallback"}
                or not isinstance(answer.get("text"), str)):
            continue
        if answer.get("citations") and len(_citations(snapshot, answer)) != len(answer["citations"]):
            continue
        # Never clip an answer through a qualification; drop oversized turns.
        if len(answer["text"]) <= 2400 and len(turn["question"]) <= 2000:
            selected.append({"question": redact_text(turn["question"]), "answer": redact_text(answer["text"])})
    return selected[-4:]


def _key(text):
    text = text.casefold().replace("’", "'").strip(" .!;:\n")
    text = re.sub(r"\b(bachelor|master)'s\b", r"\1", text)
    text = re.sub(r"\b(?:a|an|the)\s+", "", text)
    text = re.sub(r"\b(?:still|currently)\s+(?=in progress|pursuing)", "", text)
    return " ".join(text.split())


def _assertions(facts, evidence, labels):
    """Enumerate narrow grammatical alternatives without discarding qualifiers."""
    assertions = []

    def add(text, owners, ids, cell_ids=(), *, detail=True):
        if not ids and not cell_ids:
            return
        item = {"text": text.rstrip(" .") + ".", "owners": list(owners),
                "evidence_ids": list(ids), "cell_ids": list(cell_ids), "detail": detail}
        if not any(_key(row["text"]) == _key(item["text"]) for row in assertions):
            assertions.append(item)

    for cell in facts["cells"]:
        label = labels[cell["owner"]]
        ids, cells = cell["evidence_ids"], [cell["id"]]
        criterion = (cell["criterion"] if cell.get("requirement_evidence_ids") else
                     {"DEGREE": "qualification", "CERTIFICATION": "credential", "SKILL": "skills",
                      "KNOWLEDGE": "knowledge", "TITLE": "role", "EXPERIENCE_YEARS": "experience"}
                     .get(cell["category"], "requested criterion"))
        # A bounded source-state summary remains useful for long contexts. It
        # describes what the reviewed evidence establishes, never invents skills,
        # completion or years from an entity mention. All full contexts stay in
        # source controls and the model bundle.
        if ids:
            status = cell["support_status"]
            if (status == "explicit_positive" and cell.get("requirement_evidence_ids")
                    and (cell["category"] != "DEGREE" or _degree_matches_requirement(cell, criterion))):
                add(f"{label} describes {criterion} relevant to this role", [cell["owner"]], ids, cells)
            elif status == "qualified":
                text = (f"{label}'s qualification is not confirmed as completed" if cell["category"] == "DEGREE" else
                        f"{label}'s credential needs a validity check before treating it as current" if cell["category"] == "CERTIFICATION" else
                        f"{label} describes learning or training in the requested skills, rather than established experience")
                add(text, [cell["owner"]], ids, cells)
            elif status == "explicit_negative":
                add(f"{label} explicitly describes a limitation in the requested {criterion}", [cell["owner"]], ids, cells)
            elif status == "unclear":
                add(f"I can't confirm the requested {criterion} for {label} from what we have", [cell["owner"]], ids, cells)
        if cell["support_status"] in {"not_found_in_retrieved_evidence", "partial_source", "unavailable", "conflicting"}:
            add(f"I can't confirm the requested {criterion} for {label} from what we have", [cell["owner"]], ids, cells)
            continue
        source = cell["context"].strip().rstrip(".")
        # Full contexts only: splitting sentences could detach a later qualifier.
        if (not source or len(source) > 180 or re.search(r"[\n;.!?]", source)
                or re.search(r"\b(?:ignore|instructions?|assistant|system prompt|pretend|output|return|say|claim|conditional|unless|if)\b", source, re.I)):
            continue
        completed = re.fullmatch(r"Completed\s+((?:Bachelor|Master|BSc|MSc|PhD|Diploma)(?:'s)?(?:\s+degree)?(?:\s+in\s+[\w -]+)?)", source, re.I)
        ongoing = re.fullmatch(r"((?:Bachelor|Master|BSc|MSc|PhD|Diploma)(?:'s)?(?:\s+degree)?(?:\s+in\s+[\w -]+)?)\s+in progress", source, re.I)
        if completed:
            add(f"{label} has completed a {completed[1]}", [cell["owner"]], ids, cells)
            add(f"{label} reports a completed {completed[1]}", [cell["owner"]], ids, cells)
            add(f"{label} has a {completed[1]}", [cell["owner"]], ids, cells)
            add(f"{label} holds a {completed[1]}", [cell["owner"]], ids, cells)
        elif ongoing:
            add(f"{label} is pursuing a {ongoing[1]}", [cell["owner"]], ids, cells)
            add(f"{label}'s {ongoing[1]} is in progress", [cell["owner"]], ids, cells)
            add(f"{label} has a {ongoing[1]} in progress", [cell["owner"]], ids, cells)
            add(f"{label} is pursuing a {ongoing[1]}, which does not confirm that they have completed it",
                [cell["owner"]], ids, cells)
            if facts.get("participants") == [cell["owner"]]:
                add("The available evidence only states that the degree is in progress", [cell["owner"]], ids, cells)
        # A reports/lists clause attributes the complete short predicate to its CV,
        # never upgrades it to verified proficiency or completion.
        for verb in ("reports", "lists", "mentions"):
            add(f"{label} {verb} {source[0].lower() + source[1:]}", [cell["owner"]], ids, cells)
    for relation in facts.get("advantages", []):
        first, second = relation["stronger_owner"], relation["weaker_owner"]
        add(f"For {relation['criterion']}, {labels[first]} has stronger documented evidence than {labels[second]}",
            [first, second], relation["evidence_ids"], relation["cell_ids"], detail=False)
    for relation in facts.get("relations", []):
        if relation["kind"] == "both_documented":
            ids = [identity for cell in facts["cells"] if cell["id"] in relation["cell_ids"] for identity in cell["evidence_ids"]]
            add(relation["text"], relation["participants"], ids, relation["cell_ids"], detail=False)
    # JD-only answers have no candidate cells. Keep full short source predicates.
    if not facts.get("participants"):
        for item in evidence:
            source = item["context"].strip().rstrip(".")
            if item["owner_kind"] == "jd" and len(source) <= 180 and not re.search(r"[\n;.!?]", source):
                add(f"The role lists {source}", [item["owner"]], [item["id"]])
    return assertions


def build_context(snapshot, result, history=None):
    citations = _citations(snapshot, result)
    facts = build_facts(snapshot, result)
    labels = {row["cv_id"]: f"Candidate {row['rank']}" for row in snapshot.candidates}
    labels[snapshot.jd_id] = "The role"
    selected = result.get("context", {}).get("selected_candidates", [])
    evidence = [{"id": key, "owner": item["owner"], "label": labels[item["owner"]],
                 "owner_kind": item["owner_kind"], "context": item["quote"]}
                for key, item in citations.items() if item["owner"] in [snapshot.jd_id, *selected]]
    bundle = {"schema_version": 3, "snapshot_id": snapshot.snapshot_id, "jd_id": snapshot.jd_id,
              "scope_digest": scope_digest(snapshot), "labels": {owner: labels[owner] for owner in [snapshot.jd_id, *selected]},
              "selected_candidates": list(selected), "evidence": evidence,
              "intent": result.get("context", {}).get("intent"),
              "cells": facts["cells"], "checked_relations": facts["relations"],
              "comparison_advantages": [{key: item[key] for key in ("criterion", "stronger_owner", "weaker_owner", "basis")}
                                        for item in facts.get("advantages", [])],
              "history": scoped_history(snapshot, history)}
    bundle["required_comparison_cells"] = (list(dict.fromkeys(identity for item in facts.get("advantages", [])
                                               for identity in item["cell_ids"]))
                                            if result.get("context", {}).get("intent") == "comparison_explanation" else [])
    bundle["checked_paraphrases"] = _assertions(facts, evidence, labels)
    return bundle


_NEUTRAL = {
    "missing information doesn't mean candidate lacks qualification",
    "missing evidence does not establish absence",
    "this does not establish overall suitability",
    "this is tradeoff rather than clear overall advantage",
    "this is trade-off rather than clear overall advantage",
    "i can't establish that either candidate is better overall",
    "i cannot establish that either candidate is better overall",
}


def validate_prose(text, bundle):
    """Return claim/source mappings or a finite fallback reason.

Every factual clause must match an explicit registered paraphrase. Word overlap,
model confidence and being supplied a passage are deliberately insufficient.
"""
    if not isinstance(text, str) or not text.strip() or len(text) > 2400:
        return [], "invalid_response_size"
    if re.search(r"[`{}\[\]<>]|\b(?:chunk|evidence_id|relation_id|according to chunk)\b", text, re.I):
        return [], "invalid_prose_format"
    if redact_text(text) != text:
        return [], "redaction_required"
    allowed = {_key(item["text"]): item for item in bundle["checked_paraphrases"]}
    # Only non-causal conjunctions are removable. 'Because', 'therefore', etc.
    # introduce new reasoning and must themselves be registered, not ignored.
    clauses = re.split(r"(?<=[.!?])\s+|;\s*|,?\s+(?:while|whereas|and|but)\s+(?=Candidate\s+\d)", text.strip(), flags=re.I)
    mappings, covered, cells, all_cells = [], set(), set(), set()
    for clause in clauses:
        key = _key(clause)
        if key in _NEUTRAL:
            continue
        supported = allowed.get(key)
        if supported is None:
            return [], "unsupported_paraphrase"
        covered.update(supported["owners"])
        all_cells.update(supported["cell_ids"])
        if supported["detail"]:
            cells.update(supported["cell_ids"])
        mappings.append({"text": clause, "evidence_ids": supported["evidence_ids"],
                         "support_kind": "registered_source_paraphrase"})
    if not mappings or not set(bundle["selected_candidates"]) <= covered:
        return [], "participant_omission"
    critical = {cell["id"] for cell in bundle["cells"]
                if cell["support_status"] in {"qualified", "explicit_negative", "conflicting"}}
    if not critical <= cells:
        return [], "qualification_omission"
    if not set(bundle.get("required_comparison_cells", [])) <= all_cells:
        return [], "counterfact_omission"
    return mappings, None


def fallback_text(bundle, reason):
    if reason == "provider_unavailable":
        return "The conversational model is unavailable right now. You can still review the comparison and supporting sources."
    explanations, details, caveats = [], [], []
    advantages = bundle.get("comparison_advantages", [])
    if bundle.get("intent") == "comparison_explanation":
        for item in advantages:
            lead, other = (bundle["labels"][item[key]] for key in ("stronger_owner", "weaker_owner"))
            criterion = item["criterion"]
            if item["basis"] == "coverage":
                text = f"{lead} has clearer support for {criterion}, but I can't confirm it for {other}"
            elif item["basis"] == "completion":
                text = f"{lead} has a clearer match on {criterion} because their qualification is completed and {other}'s completion is not confirmed"
            elif item["basis"] == "duration":
                text = f"{lead} reports more experience for {criterion} than {other}"
                caveats.append("more experience alone doesn't establish proficiency")
            else:
                text = f"{lead} has stronger support for {criterion}, while {other} explicitly describes a limitation"
            explanations.append(text)
        caveats.append("this is a tradeoff rather than a clear overall advantage" if len({item["stronger_owner"] for item in advantages}) > 1
                       else "I can't establish that either candidate is better overall")
    # Select one complete registered summary per cell, retaining all participants
    # and qualification caveats. Combine claims rather than dumping CV paragraphs.
    for cell in bundle["cells"]:
        choices = [item for item in bundle["checked_paraphrases"]
                   if item["detail"] and item["cell_ids"] == [cell["id"]]]
        if not choices:
            continue
        def priority(item):
            text = item["text"]
            if re.search(r"\b(?:has completed|is pursuing)\b", text):
                return 0
            if re.search(r"\b(?:reports|lists|mentions)\b", text):
                return 2
            return 1
        selected = min(choices, key=priority)["text"].rstrip(".")
        if selected not in details:
            details.append(selected)
    if any(cell["support_status"] in {"not_found_in_retrieved_evidence", "partial_source", "unavailable"} for cell in bundle["cells"]):
        caveats.append("missing information doesn't mean the candidate lacks the qualification")
    paragraphs = ["; ".join(parts) + "." for parts in (explanations, details, list(dict.fromkeys(caveats))) if parts]
    if paragraphs:
        return " ".join(paragraphs)
    return "I don't have enough checked information to answer that confidently. You can review the supporting sources."
