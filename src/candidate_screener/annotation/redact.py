"""PII removal, applied to everything that reaches a human *(data rule 4)*.

The corpora this project labels are real documents, not synthetic ones: livecareer
(C1/C2) and A1 resumes are genuine CVs, and DataTurks (B1) annotates `Name` and
`Email Address` as entity classes precisely because they are present. `AGENTS.md`
requires redaction before display and before any commit.

**Deliberately conservative about what it claims.** This masks the mechanically
detectable identifiers — email addresses, phone numbers, URLs and long digit runs. It
does **not** claim to remove names: a person's name is not a regex, and a resume that
still says "Priya Nair, Senior Engineer at Acme" is still identifying after every
pattern here has run. So `redact_text` is a *reduction*, and the only complete control
is the one the guide states — annotators see documents under an access agreement and
nothing is redistributed.

Overstating this would be worse than not having it: a caller who believes the output
is anonymous will commit it.
"""
from __future__ import annotations

import re

import pandas as pd

#: Ordered — URLs before emails, so an email inside a `mailto:` link is not half-masked.
PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("url", re.compile(r"\b(?:https?://|www\.)\S+", re.I), "[URL]"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    # Phone numbers vary far more than emails; this targets the shapes that actually
    # appear in these corpora (+380…, (044) …, 555-123-4567) and accepts that an
    # unusual format survives. A looser pattern eats years, salaries and version
    # numbers, which damages the document the annotator has to read.
    ("phone", re.compile(r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?"
                         r"\d{2,4}(?:[\s.-]\d{2,4}){1,3}(?!\w)"), "[PHONE]"),
    ("digits", re.compile(r"(?<!\w)\d{7,}(?!\w)"), "[NUMBER]"),
)

#: What `verify --derived` sweeps for. Narrower than `PATTERNS` on purpose: this is the
#: set whose survival is a *failure*, not the set we attempt to mask.
RESIDUAL = {name: pattern for name, pattern, _ in PATTERNS if name in ("email", "url")}
RESIDUAL["phone_intl"] = re.compile(r"(?<!\w)\+\d{7,}(?!\w)")


def redact_text(text: object) -> str:
    """Mask every pattern in `PATTERNS`, in order. Idempotent — a `[EMAIL]` stays one."""
    out = str(text) if text is not None and not pd.isna(text) else ""
    for _, pattern, replacement in PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact(texts: pd.Series) -> pd.Series:
    return texts.map(redact_text)


def residual_pii(texts: pd.Series) -> dict[str, int]:
    """Count what survived, by class. Zero on every class is the acceptance criterion."""
    joined = "\n".join(texts.fillna("").astype(str))
    return {name: len(pattern.findall(joined)) for name, pattern in RESIDUAL.items()}


def assert_clean(texts: pd.Series, what: str) -> None:
    found = {k: v for k, v in residual_pii(texts).items() if v}
    if found:
        raise AssertionError(
            f"{what}: {found} survived redaction. Nothing reaches an annotator or a "
            "commit with detectable PII in it — see AGENTS.md data rule 4.")
