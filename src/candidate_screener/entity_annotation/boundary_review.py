"""Read-only human review of tokenizer alignment blockers; never edits labels."""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
from urllib.parse import urlencode

from .text import digest, prepare_text


def _visible(text: str, limit=120) -> str:
    return text[:limit].replace(" ", "·").replace("\t", "⇥").replace("\r", "␍").replace("\n", "↵")


def render_review(campaign: dict, records: list[dict], queue_path: str | Path,
                  annotator: str) -> str:
    """Render escaped, redacted excerpts with stale-revision/source warnings.

    The queue contains suggestions only. Human edits always use the existing
    annotation editor and create ordinary new review revisions.
    """
    if annotator not in campaign.get("annotators", [annotator]):
        raise ValueError("Annotator is not registered for this campaign")
    queue = json.loads(Path(queue_path).read_text(encoding="utf-8"))
    if annotator != queue.get("annotator", "annotator1"):
        raise ValueError("Boundary suggestions are available only to the snapshot annotator")
    issues = queue.get("issues", [])
    if not isinstance(issues, list) or len(issues) > 10000:
        raise ValueError("Invalid bounded alignment review queue")
    documents = {doc["doc_id"]: doc for doc in campaign["documents"]}
    latest = {}
    for record in records:
        if record["annotator"] == annotator:
            if record.get("campaign_id") != campaign["campaign_id"]:
                raise ValueError("Review record campaign mismatch")
            previous = latest.get(record["doc_id"])
            if previous is None or record["revision"] > previous["revision"]:
                latest[record["doc_id"]] = record
    counts = Counter(case.get("reason", "unknown") for case in issues)
    pieces = ['<!doctype html><html lang="en"><meta charset="utf-8">',
              '<title>Entity boundary review</title><style>body{font:16px system-ui;margin:24px}',
              'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}',
              'code{white-space:pre-wrap}.stale{color:#a21}small{display:block}</style>',
              '<h1>Tokenizer boundary review</h1>',
              '<p>Read-only suggestions. Review each label in the existing editor and save corrections as a new revision. ',
              'No suggestions are applied automatically. Token coverage can include unwanted text; ',
              'inside-word suggestions require a semantic decision.</p>',
              '<p>If the current entity is correct but token coverage adds a heading or other unwanted text, ',
              'leave the correct label unchanged. That case needs a tokenizer/preprocessing change, ',
              'not a misleading gold-label expansion.</p>',
              f'<p>{len(issues)} cases: {counts["whitespace_only"]} whitespace-only; {counts["inside_word_cut"]} inside-word cuts. ',
              'Visible whitespace: · space, ↵ newline, ⇥ tab. Excerpts pass the frozen redactor; names are not guaranteed removed.</p>',
              '<table><thead><tr><th>Case/document</th><th>Current saved span</th><th>Tentative token coverage</th></tr></thead><tbody>']
    stale_count = 0
    for index, case in enumerate(issues, 1):
        doc_id = case.get("doc_id")
        doc, review = documents.get(doc_id), latest.get(doc_id)
        reasons = []
        if doc is None or review is None:
            reasons.append("Document or annotator review is unavailable")
        elif doc.get("split", "train") != "train":
            reasons.append("Boundary suggestions are restricted to training documents")
        elif "assignments" in campaign and doc_id not in campaign["assignments"].get(annotator, []):
            reasons.append("Document is not assigned to this annotator")
        elif (doc.get("source_sha256") != case.get("source_sha256") or
              digest(doc["source_text"]) != case.get("source_sha256") or
              review.get("source_sha256") != case.get("source_sha256")):
            reasons.append("Source changed")
        elif review.get("revision") != case.get("revision"):
            reasons.append("Newer review revision exists")
        elif review.get("status", "reviewed") != "reviewed" or review.get("reviewed_all", True) is not True:
            reasons.append("Document is no longer fully reviewed")
        else:
            gold = case.get("gold", {})
            if not any(span.get("start") == gold.get("start") and span.get("end") == gold.get("end") and
                       span.get("label") == case.get("label") for span in review.get("spans", [])):
                reasons.append("Saved span changed")
        escaped_id = html.escape(str(doc_id))
        identity = f'{index}. <code>{escaped_id}</code><small>snapshot revision {html.escape(str(case.get("revision")))}</small>'
        if doc is not None and not reasons:
            link = "/?" + urlencode({"annotator": annotator, "doc_id": doc_id})
            identity += f'<a href="{html.escape(link, quote=True)}">Open document editor</a>'
        if reasons:
            stale_count += 1
            pieces.append(f'<tr><td>{identity}</td><td colspan="2" class="stale">STALE — {html.escape("; ".join(reasons))}. Rebuild the review queue; no old suggestion should be applied.</td></tr>')
            continue
        prepared = prepare_text(doc["source_text"])
        inverse = {value: key for key, value in enumerate(prepared["boundary_map"]) if value is not None}
        display = prepared["display_text"]

        def span_preview(span, context=False):
            if not isinstance(span, dict):
                return "No representable token coverage"
            a, b = span.get("start"), span.get("end")
            if type(a) is not int or type(b) is not int or a >= b or a not in inverse or b not in inverse:
                return "Boundary unavailable in redacted display"
            da, db = inverse[a], inverse[b]
            if any(da < end and start < db for start, end in prepared["masked_intervals"]):
                return "Span intersects a masked identifier"
            surface = html.escape(_visible(display[da:db]))
            result = f'<code>[{a}:{b}] {surface}</code>'
            if context:
                left = max(0, da - 35)
                excerpt = display[left:left + 120]
                result += f'<small>Context: <code>{html.escape(_visible(excerpt))}</code></small>'
            return result
        label = html.escape(str(case.get("label")))
        reason = html.escape(str(case.get("reason")))
        tentative = span_preview(case.get("proposal_for_human_review_only"))
        warning = "TOKEN COVERAGE ONLY — not automatically correct" if case.get("reason") == "inside_word_cut" else "Whitespace boundary suggestion — verify before saving"
        pieces.append(f'<tr><td>{identity}<small>{reason}</small></td><td>{label}<br>{span_preview(case.get("gold"), context=True)}</td><td>{tentative}<small>{warning}</small></td></tr>')
    pieces.append(f'</tbody></table><p>{stale_count} stale cases. Human labels have not been changed by this report.</p></html>')
    return "".join(pieces)
