"""Draft text-only screening with validated extraction evidence; no fit classifier adopted."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from candidate_screener.evidence import evidence as ev
from candidate_screener.evidence import requirements as req

#: Declared draft baseline; no new stage-1 rung is adopted by this UI integration.
SELECTED = "TF-IDF (text_v2, draft)"
SELECTED_TEXT_ONLY = SELECTED


@dataclass
class Result:
    cv_id: str
    rank: int
    score: float
    fit_call: str | None
    confidence: float
    coverage: float | None
    years_required: float | None
    years_stated: float | None
    evidence: list[dict] = field(default_factory=list)

    entities: list[dict] = field(default_factory=list)
    jd_entities: list[dict] = field(default_factory=list)
    entity_status: str = "unavailable"
    jd_entity_status: str = "unavailable"
    entity_reason: str | None = None
    extractor_provenance: dict = field(default_factory=dict)
    ranker_name: str = SELECTED
    source_kind: str = "text"
    translation_provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def default_ranker(has_metadata: bool | None = None):
    """Declared draft text-only baseline; metadata never selects an inference path."""
    from candidate_screener.matching.lexical import Tfidf
    return Tfidf(text_variant="v2")


def _frame(jd_text: str, cvs: pd.DataFrame, jd_meta: dict | None, jd_title=None) -> pd.DataFrame:
    frame = pd.DataFrame({"cv_id": cvs.cv_id.astype(str).to_numpy(),
                          "cv_text": cvs.cv_text.to_numpy(),
                          "jd_id": str((jd_meta or {}).get("jd_id", "jd")), "jd_text": jd_text})
    if jd_title is not None:
        frame["jd_title"] = jd_title
    frame["pair_id"] = frame.jd_id + "__" + frame.cv_id
    return frame


def _entities(text, side, extractor):
    from candidate_screener.extraction.schema import validate_entity
    if extractor is None:
        return {"status": "unavailable", "entities": [], "provenance": {}}
    bundle = extractor(text, side)
    if not isinstance(bundle, dict) or bundle.get("status") not in {
            "complete", "completed_zero", "unavailable", "failed"}:
        raise ValueError("Explicit entity completion status required")
    rows = bundle.get("entities", [])
    if bundle["status"] in {"unavailable", "failed"} and rows:
        raise ValueError("Incomplete extraction cannot supply evidence")
    for entity in rows:
        validate_entity(entity, text)
    return bundle


def _concept(text, matcher):
    from candidate_screener.matching.entity_weighted import normalized_phrase
    matches = matcher.find(text)
    exact = [m.canonical for m in matches if m.start == 0 and m.end == len(text)]
    return exact[0] if exact else normalized_phrase(text)


def _entity_requirements(text, wanted, rows, matcher):
    from candidate_screener.extraction.rules import _context
    seen = {r.canonical for r in wanted}
    for entity in rows:
        if entity["label"] not in {"SKILL", "KNOWLEDGE"} or entity["context_status"] != "unqualified":
            continue
        if _context(text, entity["start"], entity["end"], "text") != "unqualified":
            continue
        canonical = _concept(entity["surface"], matcher)
        if canonical not in seen:
            wanted.append(req.Requirement(canonical, entity["surface"], entity["start"], entity["end"], "entity"))
            seen.add(canonical)
    return wanted


def _source_evidence(rows, text, provenance, source_kind):
    """Attach translation lineage without claiming an English quote was spoken French."""
    if not isinstance(provenance, dict):
        raise ValueError("Translation provenance must be structured")
    aligned = provenance.get("alignment_valid") is True
    segments = provenance.get("segments", []) if aligned else []
    for segment in segments:
        start, end = segment["english_start"], segment["english_end"]
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise ValueError("Translation alignment outside candidate text")
        if "english" in segment and text[start:end] != segment["english"]:
            raise ValueError("Translation alignment text differs from candidate text")
        if "english" not in segment:
            import hashlib
            expected = segment.get("english_sha256")
            if expected is None or hashlib.sha256(text[start:end].encode()).hexdigest() != expected:
                raise ValueError("Translation alignment digest differs from candidate text")
    out = []
    for row in rows:
        copied = dict(row)
        links = [{"audio_start": s["audio_start"], "audio_end": s["audio_end"],
                  "english_start": s["english_start"], "english_end": s["english_end"]}
                 for s in segments if row.get("start") is not None
                 and row["start"] < s["english_end"] and row["end"] > s["english_start"]]
        copied["audio_segments"] = links
        translated = bool(links) or (bool(provenance) and not aligned and source_kind != "combined")
        copied["quote_origin"] = "reviewed_english_translation" if translated else (
            "written_or_edited_text" if provenance and not aligned else "written_cv")
        out.append(copied)
    return out


def screen(jd_text: str, cvs: pd.DataFrame, jd_meta: dict | None = None, ranker=None,
           matcher=None, encoder=None, tagger=None, entity_extractor=None, jd_title=None) -> list[Result]:
    """Rank readable CV text; legacy metadata is ignored except the JD identifier."""
    from candidate_screener.extraction.experience import jd_experience, cv_experience
    if not isinstance(jd_text, str) or not jd_text.strip() or cvs.empty:
        raise ValueError("Nonempty JD and candidate pool required")
    if cvs.cv_text.isna().any() or not cvs.cv_text.map(lambda t: isinstance(t, str) and bool(t.strip())).all():
        raise ValueError("Every candidate requires readable nonempty text")
    if cvs.cv_id.isna().any() or cvs.cv_id.astype(str).duplicated().any():
        raise ValueError("Unique candidate identifiers required")
    frame = _frame(jd_text, cvs, jd_meta, jd_title)
    if ranker is None:
        ranker = default_ranker()
        ranker.fit(None)
    scores = ranker.score(frame).to_numpy()
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Ranker must return one finite score per candidate")
    if matcher is None:
        from candidate_screener.extraction.rules import default_matcher
        matcher = default_matcher()
    jd_bundle = _entities(jd_text, "jd", entity_extractor)
    wanted = _entity_requirements(jd_text, req.requirements(jd_text, matcher, tagger),
                                  jd_bundle["entities"], matcher)
    need_years = jd_experience(jd_text).min_years
    order = np.lexsort((frame.cv_id.to_numpy(), -scores))
    pct = pd.Series(scores).rank(pct=True).to_numpy()
    results = []
    for rank, i in enumerate(order, start=1):
        row = frame.iloc[i]
        cv_bundle = _entities(row.cv_text, "cv", entity_extractor)
        found = ev.gather(wanted, row.cv_text, matcher, encoder,
                          entity_matches=[(_concept(e["surface"], matcher), e)
                                          for e in cv_bundle["entities"]
                                          if e["label"] in {"SKILL", "KNOWLEDGE"}
                                          and e["context_status"] == "unqualified"])
        call = None
        source_kind = cvs.iloc[i].get("source_kind", "text")
        provenance = cvs.iloc[i].get("translation_provenance", {})
        entity_rows = _source_evidence(cv_bundle["entities"], row.cv_text, provenance, source_kind)
        evidence_rows = _source_evidence(ev.as_dicts(found), row.cv_text, provenance, source_kind)
        results.append(Result(
            cv_id=row.cv_id, rank=rank, score=round(float(scores[i]), 6), fit_call=call,
            confidence=round(float(pct[i]), 4), coverage=ev.coverage(found),
            years_required=need_years, years_stated=cv_experience(row.cv_text).years,
            evidence=evidence_rows, entities=entity_rows,
            jd_entities=jd_bundle["entities"], entity_status=cv_bundle["status"],
            jd_entity_status=jd_bundle["status"], entity_reason=cv_bundle.get("reason"),
            extractor_provenance={"cv": cv_bundle.get("provenance", {}),
                                  "jd": jd_bundle.get("provenance", {})},
            ranker_name=getattr(ranker, "name", type(ranker).__name__),
            source_kind=source_kind, translation_provenance=provenance))
    return results
