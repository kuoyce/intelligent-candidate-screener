"""Bounded single-candidate summaries of complete, authorized source statements."""
from __future__ import annotations

import re

_GENERIC = set("a an the and or of in on for with to as is are was were we our you your candidate applicant role job position developer engineer senior junior experience experienced skill skills knowledge proficiency programming development technology technologies software project projects work working years year months month required requirement requirements looking team professional commercial completed degree".split())
_QUALIFIER = re.compile(r"\b(?:no|not|never|without|lacks?|learning|studying|training|pursuing|hopes?|hoping|plans?|planned|planning|aspir\w*|wants?|wishes?|expired|pending|conditional|uncertain|provisional|unless|if|in progress|would like)\b", re.I)
_NAMED_TOOLS = frozenset("jira python ruby rails mysql postgresql sql javascript typescript java kotlin swift flutter react angular django flask docker kubernetes terraform ansible linux unix git github gitlab jenkins selenium cypress playwright postman airflow spark tableau powerbi excel aws azure gcp".split())
_PHRASE_GENERIC = _GENERIC | set("good excellent strong ability abilities relevant related similar various many several other more most some all any using used uses use has have had can could should will would be been being it its this that these those also including includes include general overall duties responsibilities responsible performed performing created creating seven six five four three two one".split())


def _literal_overlap(source, requirement):
    """Require an exact tool token or a contiguous nontrivial two-word phrase.

    This establishes only documented mention overlap, never proficiency or
    tenure. Callers separately enforce ownership, source status and qualifiers.
    A missing entity cache must not erase literal source evidence.
    """
    tokens = lambda text: re.findall(r"[\w+#]+(?:[.-][\w+#]+)*", text.casefold())
    cv, jd = tokens(source), tokens(requirement)
    if set(cv) & set(jd) & _NAMED_TOOLS:
        return True
    phrases = lambda words: {(left, right) for left, right in zip(words, words[1:])
                             if left not in _PHRASE_GENERIC and right not in _PHRASE_GENERIC
                             and not left.isdigit() and not right.isdigit()
                             and (len(left) >= 3 or left == "qa")
                             and (len(right) >= 3 or right == "qa")}
    return bool(phrases(cv) & phrases(jd))


def summary_conclusion(bundle):
    """A conditional relevance statement requires nontrivial, current JD/CV support."""
    if (bundle.get("summary_scope") != "broad_candidate_evidence" or not bundle.get("summary_alignment")
            or len(bundle.get("selected_candidates", [])) != 1):
        return None
    owner = bundle["selected_candidates"][0]
    evidence = {item["id"]: item for item in bundle.get("evidence", [])}

    def sources(ids, expected, kind):
        return [evidence[key] for key in ids if key in evidence
                and evidence[key]["owner"] == expected and evidence[key]["owner_kind"] == kind]

    def healthy(cell):
        records = sources(cell.get("evidence_ids", []), owner, "cv")
        context = " ".join(dict.fromkeys(item["context"] for item in records))
        return (cell.get("owner") == owner and cell.get("support_status") in {"explicit_positive", "unclear"}
                and bool(context) and context == cell.get("context") and not _QUALIFIER.search(context))

    broad = [cell for cell in bundle.get("summary_cells", []) if healthy(cell)]
    healthy_ids = {identity for cell in broad for identity in cell["evidence_ids"]}
    jd_sources = [item for item in evidence.values() if item["owner"] == bundle.get("jd_id")
                  and item["owner_kind"] == "jd" and not _QUALIFIER.search(item["context"])]
    support = []
    for cell in bundle.get("cells", []):
        if not healthy(cell) or cell["support_status"] != "explicit_positive":
            continue
        requirements = sources(cell.get("requirement_evidence_ids", []), bundle.get("jd_id"), "jd")
        terms = set(re.findall(r"[\w+#.-]+", cell["criterion"].casefold())) - _GENERIC
        if requirements and terms and set(cell["evidence_ids"]) <= healthy_ids:
            support.append((cell, [item["id"] for item in requirements if item in jd_sources]))
    for cell in broad:
        if cell.get("category") not in {"SKILL", "KNOWLEDGE", "TITLE"}:
            continue
        # Exact source-owned entity tokens establish mention overlap only. Generic
        # role/experience words never establish relevance by themselves.
        terms = {term.casefold() for mention in cell.get("mentions", [])
                 for term in re.findall(r"[\w+#.-]+", mention)} - _GENERIC
        for term in terms:
            pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
            if not re.search(pattern, cell["context"], re.I):
                continue
            ids = [item["id"] for item in jd_sources if re.search(pattern, item["context"], re.I)]
            if ids:
                support.append((cell, ids))
        ids = [item["id"] for item in jd_sources
               if _literal_overlap(cell["context"], item["context"])]
        if ids:
            support.append((cell, ids))
    support = [(cell, ids) for cell, ids in support if ids]
    if not support:
        return None
    return {"text": f"Hence, {bundle['labels'][owner]} is a promising candidate for further review, subject to confirming the remaining requirements.",
            "owners": [owner], "detail": False,
            "evidence_ids": list(dict.fromkeys(identity for cell, ids in support for identity in [*cell["evidence_ids"], *ids])),
            "cell_ids": list(dict.fromkeys(cell["id"] for cell, _ in support))}


def summary_fallback(bundle):
    """Attribute statements to the CV and preserve unresolved JD criteria."""
    statements, qualifications, gaps = [], [], []
    quotations = {}
    assertions = bundle.get("matrix_assertions", [])
    covered = set()
    priority = lambda text: (0 if re.search(r"\b(?:has completed|is pursuing)\b", text) else
                             2 if re.search(r"\b(?:reports|lists|mentions)\b", text) else 1)
    for cell in bundle.get("cells", []):
        if cell["support_status"] == "explicit_positive" and cell.get("requirement_evidence_ids"):
            choices = [item["text"] for item in bundle["checked_paraphrases"]
                       if item["detail"] and item["cell_ids"] == [cell["id"]]]
            if choices:
                statements.append(min(choices, key=priority))
                covered.update(cell["evidence_ids"])
    for cell in bundle.get("summary_cells", []):
        if not cell.get("evidence_ids"):
            continue
        if set(cell["evidence_ids"]) <= covered:
            continue
        label = bundle["labels"][cell["owner"]]
        source = cell.get("context", "").strip()
        choices = [item["text"] for item in assertions
                   if item["detail"] and item["cell_ids"] == [cell["id"]]]
        if source and len(source) <= 180:
            quotations.setdefault(cell["owner"], [])
            if source not in quotations[cell["owner"]]:
                quotations[cell["owner"]].append(source)
            continue
        elif cell.get("mentions") and cell["support_status"] in {"explicit_positive", "unclear"}:
            text = (f"{label}'s CV mentions {', '.join(cell['mentions'])}; "
                    "these mentions do not establish proficiency or skill-specific tenure.")
        elif choices:
            text = min(choices, key=priority)
        else:
            text = f"{label}'s CV contains a passage about {cell['criterion']}; review its full qualification context."
        if cell["support_status"] in {"qualified", "explicit_negative", "conflicting", "partial_source", "unavailable"}:
            qualifications.append(text)
        else:
            statements.append(text)
    for cell in bundle.get("cells", []):
        if (cell.get("requirement_evidence_ids")
                and cell["support_status"] in {"not_found_in_retrieved_evidence", "partial_source", "unavailable", "unclear"}):
            gaps.append(f"I can't confirm {cell['criterion']} for {bundle['labels'][cell['owner']]} from the reviewed evidence.")
    # Keep complete quotations/qualifiers; omit whole repeated passages only.
    grouped = [f"{bundle['labels'][owner]}'s CV states: " + " ".join(f"“{source}”" for source in sources)
               for owner, sources in quotations.items()]
    parts = list(dict.fromkeys([*statements, *grouped, *qualifications]))
    if gaps:
        gaps = [*dict.fromkeys(gaps), "Missing evidence does not establish absence."]
        parts.append(" ".join(gaps))
    conclusion = summary_conclusion(bundle)
    if not parts:
        return "I could not establish a summary from the reviewed CV evidence; this does not establish absence."
    body = " ".join(parts) + (" This shows that candidate's CV contains relevant experience and qualifications." if len(parts) < 3 else "  ")
    text = body + ("\n\n" + conclusion["text"] if conclusion else "")
    if len(text) > 2400:
        return "The source evidence is too large for a concise summary. Review the supporting sources for complete statements and qualification context."
    return text
