"""Controlled comparative prose; factual text comes only from checked facts."""
from __future__ import annotations


def render_comparison(snapshot, facts, selected_relation_ids=None, *, include_source_context=True):
    """Render every requested participant without promoting unknown evidence."""
    relations = facts.get("relations", [])
    allowed = {item["id"] for item in relations}
    if selected_relation_ids is not None and (not isinstance(selected_relation_ids, list)
                                            or any(item not in allowed for item in selected_relation_ids)
                                            or len(set(selected_relation_ids)) != len(selected_relation_ids)):
        raise ValueError("Unknown comparison relation")
    critical = {"explicit_contrast", "completion_status_differs", "evidence_incomplete", "source_conflict", "comparison_advantage"}
    if selected_relation_ids is None:
        # Meaningful differences lead the default, rather than category order.
        priority = {"explicit_contrast": 0, "completion_status_differs": 1,
                    "duration_greater": 2, "both_documented": 3,
                    "source_conflict": 4, "evidence_incomplete": 5}
        chosen = sorted(relations, key=lambda item: priority.get(item["kind"], 6))
    else:
        by_id = {item["id"]: item for item in relations}
        chosen = [by_id[identity] for identity in selected_relation_ids]
        selected = set(selected_relation_ids)
        chosen += [item for item in relations if item["id"] not in selected and item["kind"] in critical]
    # Never drop a critical qualification to cap a list of summaries. The service
    # can omit whole source cells and rebuild facts under its shared byte budget.
    participants = facts.get("participants", facts.get("requested_owners", []))
    labels = facts.get("labels", {})
    lines = list(dict.fromkeys(item["text"] for item in chosen if item["kind"] != "evidence_incomplete"))
    cells_by_id = {cell["id"]: cell for cell in facts.get("cells", [])}
    limitations = {}
    for relation in chosen:
        if relation["kind"] != "evidence_incomplete":
            continue
        for identity in relation.get("cell_ids", []):
            cell = cells_by_id.get(identity)
            if cell:
                key = (cell["owner"], cell["support_status"])
                limitations.setdefault(key, []).append(cell["criterion"])
    for (owner, status), criteria in limitations.items():
        criterion_list = ", ".join(dict.fromkeys(criteria))
        message = ("the redacted source is unavailable" if status == "unavailable" else
                   "the saved evidence is partial or omitted" if status == "partial_source" else
                   "I could not establish the requested facts from retrieved CV evidence")
        lines.append(f"{labels.get(owner, 'Candidate')} — {criterion_list}: {message}.")
    if limitations:
        lines.append("Missing or limited evidence does not establish absence.")
    realized = {identity for relation in chosen if relation["kind"] in
                {"both_documented", "explicit_contrast", "completion_status_differs", "duration_greater"}
                for identity in relation.get("cell_ids", [])}
    for owner in participants:
        cells = [cell for cell in facts.get("cells", []) if cell["owner"] == owner]
        evidenced = [cell for cell in cells if cell.get("context")
                     and (include_source_context or cell["id"] not in realized)]
        if evidenced:
            focused_ids = list(dict.fromkeys(identity for relation in chosen
                                            for identity in relation.get("cell_ids", [])))
            focused_order = {identity: index for index, identity in enumerate(focused_ids)}
            # A broad comparison's first rows need not be its useful contrast.
            # Prefer the source contexts supporting the selected conclusions and
            # qualification caveats over arbitrary category order.
            evidenced.sort(key=lambda cell: (0, focused_order[cell["id"]]) if cell["id"] in focused_order else
                           (1 if cell.get("support_status") in {"qualified", "conflicting", "explicit_negative"} else 2, 0))
            # Whole source contexts retain late qualifiers; never clip the sentence.
            contexts = list(dict.fromkeys(cell["context"] for cell in evidenced))[:2]
            lines.append(f"{labels.get(owner, 'Candidate')}: " + " ".join(f'“{text}”' for text in contexts))
        elif not any(owner in item["participants"] for item in chosen):
            criteria = ", ".join(facts.get("criteria", [])) or "the requested criteria"
            lines.append(f"{labels.get(owner, 'Candidate')} — {criteria}: I could not establish the requested facts from retrieved CV evidence; this does not establish absence.")
    if not lines:
        return "I could not establish a comparison from the available source evidence."
    return "\n\n".join(lines)


def render_explanation(snapshot, facts, subject, reference, selected_relation_ids=None, *, ranked_above=False):
    """Explain a directed premise using all checked reasons and counterreasons.

    Selection may change emphasis, never erase a reverse advantage. Ranking facts
    come from the snapshot and are explicitly separate from these source claims.
    """
    labels = facts.get("labels", {})
    if subject == reference or any(owner not in facts.get("participants", []) or owner not in labels
                                   for owner in (subject, reference)):
        raise ValueError("Explain two distinct authorized comparison participants")
    allowed = {item["id"] for item in facts.get("relations", []) + facts.get("advantages", [])}
    if selected_relation_ids is not None and (not isinstance(selected_relation_ids, list)
                                            or any(item not in allowed for item in selected_relation_ids)
                                            or len(set(selected_relation_ids)) != len(selected_relation_ids)):
        raise ValueError("Unknown explanation relation")
    order = {identity: index for index, identity in enumerate(selected_relation_ids or [])}
    advantages = [item for item in facts.get("advantages", [])
                  if set(item["participants"]) == {subject, reference}]
    advantages.sort(key=lambda item: order.get(item["id"], len(order)))
    reasons = [item for item in advantages if item["stronger_owner"] == subject]
    counters = [item for item in advantages if item["stronger_owner"] == reference]
    lead, other = labels[subject], labels[reference]
    criteria = lambda rows: ", ".join(dict.fromkeys(item["criterion"] for item in rows))
    if reasons and counters:
        opening = f"The evidence shows a trade-off, rather than establishing that {lead} is more qualified overall than {other}."
    elif reasons:
        opening = (f"The comparison supports stronger documented evidence for {lead} on {criteria(reasons)}; "
                   "it does not establish an overall better-candidate verdict.")
    elif counters:
        opening = (f"I cannot support the premise that {lead} has stronger evidence here; "
                   f"the checked JD-linked points instead favor {other} on {criteria(counters)}.")
    else:
        opening = f"The comparison does not establish a directed advantage for {lead} over {other} against the recognized JD criteria."
    lines = [opening]
    if ranked_above:
        ranks = {item["cv_id"]: item["rank"] for item in snapshot.candidates}
        lines.append(f"The saved ranking places {lead} at {ranks[subject]} and {other} at {ranks[reference]}; these evidence contrasts do not prove why the ranking model ordered them.")
        if ranks[subject] > ranks[reference]:
            lines.append(f"{lead} is not ranked above {other} in the saved ranking.")
        elif ranks[subject] == ranks[reference]:
            lines.append(f"{lead} and {other} have the same saved rank; neither is ranked above the other.")
    for prefix, rows in (("Supporting evidence", reasons), ("Counterevidence", counters)):
        for item in rows:
            citations = " ".join(f"[{facts.get('citation_labels', {}).get(identity, identity)}]" for identity in item.get("evidence_ids", []))
            lines.append(f"{prefix}: {item['text']} {citations}".rstrip())
    realized = {identity for item in advantages for identity in item["cell_ids"]}
    unresolved = [cell for cell in facts.get("cells", []) if cell["owner"] in {subject, reference}
                  and cell["id"] not in realized]
    if unresolved:
        ids = {cell["id"] for cell in unresolved}
        remaining = {**facts, "cells": unresolved,
                     "participants": [subject, reference], "requested_owners": [subject, reference],
                     "relations": [item for item in facts.get("relations", [])
                                   if item["kind"] in {"source_conflict", "evidence_incomplete"}
                                   and set(item.get("cell_ids", [])) <= ids]}
        lines.append("Unresolved evidence: " + render_comparison(snapshot, remaining, include_source_context=False))
    return "\n\n".join(lines)
