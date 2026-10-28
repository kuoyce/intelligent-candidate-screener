"""One-line directed explanations from current source-checked comparison facts."""
from __future__ import annotations


def brief_explanation(snapshot, result, facts):
    """Keep direction and counterevidence without inventing employer weighting.

    ``facts`` must come from the current snapshot's checked fact builder. Only
    its registered JD-linked advantages can establish a directional strength;
    broad mentions, saved ranks and a user's preferred winner cannot do so.
    """
    state = result.get("context", {})
    subject, reference = state.get("explanation_subject"), state.get("explanation_reference")
    available = {row["cv_id"] for row in snapshot.candidates}
    labels = facts.get("labels", {})
    participants = set(facts.get("participants", []))
    if (subject == reference or any(owner not in available or owner not in labels
                                   or owner not in participants for owner in (subject, reference))):
        raise ValueError("Explain two distinct authorized comparison participants")
    cells = {cell["id"]: cell for cell in facts.get("cells", [])}
    advantages = [row for row in facts.get("advantages", [])
                  if set(row.get("participants", [])) == {subject, reference}
                  and {row.get("stronger_owner"), row.get("weaker_owner")} == {subject, reference}
                  and row.get("requirement_evidence_ids") and row.get("evidence_ids")
                  and row.get("cell_ids") and all(identity in cells for identity in row["cell_ids"])
                  and row.get("basis") in {"coverage", "duration", "completion", "explicit_contrast"}]
    forward = [row for row in advantages if row["stronger_owner"] == subject]
    reverse = [row for row in advantages if row["stronger_owner"] == reference]
    lead, other = labels[subject], labels[reference]

    def strengths(rows):
        kinds = {"coverage": "stronger documented support for",
                 "duration": "more explicitly linked experience for",
                 "completion": "clearer documented completion for",
                 "explicit_contrast": "stronger affirmative source evidence for"}
        phrases = []
        for basis, description in kinds.items():
            criteria = list(dict.fromkeys(row["criterion"] for row in rows if row["basis"] == basis))
            if criteria:
                phrases.append(description + " " + ", ".join(criteria))
        return " and ".join(phrases)

    if forward and reverse:
        text = (f"The evidence shows a trade-off: {lead} has {strengths(forward)}, while {other} has "
                f"{strengths(reverse)}; it does not establish an overall better candidate")
    elif forward:
        text = (f"{lead} has {strengths(forward)}, but the checked evidence does not establish "
                f"that they are better overall than {other}")
    elif reverse:
        text = (f"I cannot support the premise: the checked evidence instead favors {other}, with "
                f"{strengths(reverse)}; this does not establish overall suitability")
    else:
        text = f"I cannot establish that {lead} is better than {other} from the checked evidence"

    caveats = []
    for cell in cells.values():
        if cell["owner"] not in {subject, reference}:
            continue
        status, category = cell.get("support_status"), cell.get("category")
        owner = labels[cell["owner"]]
        if status == "qualified":
            caveats.append(f"{owner}'s degree completion is not confirmed" if category == "DEGREE" else
                           f"{owner}'s credential validity is not confirmed" if category == "CERTIFICATION" else
                           f"{owner}'s evidence retains its learning or training qualification")
        elif status == "conflicting":
            caveats.append(f"{owner}'s conflicting {cell['criterion']} evidence needs review")
    if any(cell.get("support_status") in {"not_found_in_retrieved_evidence", "partial_source", "unavailable"}
           for cell in cells.values()):
        caveats.append("missing evidence does not establish absence")
    if state.get("explanation_relation") == "ranked_above":
        caveats.append("saved ranks do not establish why the ranking model ordered the candidates")
    return "; ".join([text, *dict.fromkeys(caveats)]) + "."
