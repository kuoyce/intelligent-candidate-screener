"""Source-grounded summaries and readable labels for Hiring Desk."""
from __future__ import annotations

import re

from candidate_screener.annotation.redact import redact_text


def source_summary(text, *, limit=220):
    """Use an exact source prefix; never invent requirements or missing text."""
    if not isinstance(text, str) or not text.strip():
        return "Full JD source is unavailable in this export."
    clean = re.sub(r"\s+", " ", redact_text(text)).strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rsplit(" ", 1)[0] + "…"


def evidence_text(value):
    """Present model mentions as readable prose without making them source gold."""
    if not isinstance(value, dict):
        return "Not available"
    mentions = value.get("mentions", [])
    if mentions:
        parts = []
        for mention in mentions:
            if isinstance(mention, dict):
                parts.append(str(mention.get("text", mention.get("value", "Unspecified mention"))))
            else:
                parts.append(str(mention))
        return redact_text("; ".join(parts))
    values = value.get("table_values", {})
    if any(item is not None for item in values.values()):
        return "; ".join(f"{key.replace('_', ' ')}: {item}" for key, item in values.items() if item is not None)
    return "Not specified in the available evidence"


def source_label(snapshot):
    return "Live ranking" if snapshot.source_flags.get("operational") else "CSV export"


STYLES = """<style>
:root {--hd-ink:#172b4d; --hd-muted:#43536b; --hd-accent:#174ea6;}
.stApp {background:#f6f8fc; color:var(--hd-ink);}
[data-testid="stMainBlockContainer"] {max-width:1320px; padding-top:2.2rem; padding-bottom:4rem;}
[data-testid="stSidebar"] {background:#eef2f8; border-right:1px solid #d4dce8;}
[data-testid="stSidebarUserContent"] {display:flex; flex-direction:column; min-height:100%;}
[data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] {flex:1; min-height:calc(100dvh - 6rem);}
[data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] > div:has(.st-key-portal_logout_footer) {margin-top:auto;}
.st-key-portal_sidebar_layout {min-height:calc(100dvh - 9rem);}
.st-key-portal_sidebar_layout > div:has(.st-key-portal_logout_footer) {margin-top:auto;}
.st-key-portal_logout_footer {margin-top:auto; padding-top:2rem;}
h1,h2,h3 {color:var(--hd-ink); letter-spacing:-.025em;}
h1 {font-size:2.25rem !important;} h2 {font-size:1.5rem !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaption"] {color:var(--hd-muted) !important;}
[data-testid="stCaptionContainer"] *, [data-testid="stCaption"] * {color:var(--hd-muted) !important;}
.stButton button {min-height:2.75rem; border-radius:.65rem;}
.stButton button:focus-visible, input:focus-visible, textarea:focus-visible, a:focus-visible {outline:3px solid #174ea6; outline-offset:3px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:1rem;}
.st-key-portal_jd_source {background:white; border:1px solid #d4dce8; border-radius:1rem; padding:1.25rem;}
.st-key-portal_jd_source [data-testid="stText"] {white-space:pre-wrap; overflow-wrap:anywhere; font-family:inherit; font-size:1rem; line-height:1.7; max-width:85ch;}
.st-key-portal_job_cards button {text-align:left;}
@media (max-width:640px) {
 [data-testid="stMainBlockContainer"] {padding:1rem .8rem 3rem;}
 h1 {font-size:1.8rem !important;}
 .st-key-portal_job_cards [data-testid="stColumn"]:not(:has([data-testid="stBaseButton-primary"])) {display:none;}
}
</style>"""
