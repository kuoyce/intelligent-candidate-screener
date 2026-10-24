"""Read-only portal adapters for existing X2 ranks and unverified table mentions."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd
from candidate_screener.annotation.redact import redact_text
from candidate_screener.config import PROCESSED

CATEGORIES = ("KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS")
CV_COLUMNS = ("cv_id", "role", "role_prob", "top_roles", "skills", "skills_qualified", "years", "years_source", "seniority", "seniority_source")
JD_COLUMNS = ("jd_id", "stratum", "jd_title", "role", "role_prob", "top_roles", "skills", "skills_qualified", "min_years", "max_years", "seniority")
RANK_COLUMNS = ("pair_id", "stratum", "jd_id", "cv_id", "x2_score", "x2_rank")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _clean(value):
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {redact_text(str(k)): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite table aggregate")
    return value


def parse_mentions(value):
    """JSON or explicit delimiters only; no eval or ontology guessing."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return []
    if isinstance(value, (list, dict)):
        return _clean(value if isinstance(value, list) else [value])
    text = str(value).strip()
    if not text:
        return []
    if text.startswith(("[", "{")):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed structured extraction cell") from exc
        if not isinstance(decoded, (list, dict)):
            raise ValueError("Extraction cell must be a list or object")
        return _clean(decoded if isinstance(decoded, list) else [decoded])
    # CSV aggregates use semicolons; commas may be part of a skill name.
    return [redact_text(part.strip()) for part in re.split(r"\s*[;|]\s*", text) if part.strip()]


def _entities(row, *, jd=False):
    result = {label: {"status": "unavailable", "mentions": [], "verified_spans": [],
                      "reason": "This table does not provide this category with verified source offsets."}
              for label in CATEGORIES}
    skills = parse_mentions(row.get("skills"))
    qualified = parse_mentions(row.get("skills_qualified"))
    result["SKILL"] = {"status": "unverified_table_mentions" if skills or qualified else "unavailable",
                       "mentions": skills, "qualified_mentions": qualified, "verified_spans": [],
                       "source": "skills/skills_qualified aggregate columns; not source citations"}
    if jd and row.get("jd_title"):
        result["TITLE"] = {"status": "unverified_table_mentions", "mentions": [_clean(row["jd_title"])],
                            "verified_spans": [], "source": "jd_title table metadata; not a span"}
    years = {key: _clean(row.get(key)) for key in (("min_years", "max_years") if jd else ("years",))}
    if any(value is not None for value in years.values()):
        result["EXPERIENCE_YEARS"] = {"status": "unverified_table_mentions", "mentions": [],
            "table_values": years, "verified_spans": [], "source": "numeric extraction aggregate; not tool-specific tenure"}
    return result


def _profile(row):
    keys = ("role", "role_prob", "top_roles", "seniority", "seniority_source", "years_source")
    result = {key: _clean(row.get(key)) for key in keys if key in row}
    result["role_semantics"] = "predicted role family; not an extracted job title"
    return result


def load_source_documents(path: Path | None = None):
    """Only ID/text columns from an existing cache; never build or read labels."""
    path = Path(path) if path else PROCESSED / "matching/a2-pairs-v2-539ddf32117c.parquet"
    if not path.is_file():
        return {"jd": {}, "cv": {}}
    import pyarrow.parquet as pq
    before = (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
    frame = pq.read_table(path, columns=["jd_id", "cv_id", "jd_text", "cv_text"]).to_pandas()
    if before != (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError("Source document cache changed during the read")
    documents = {"jd": {}, "cv": {}}
    for side in documents:
        for identity, text in frame[[side + "_id", side + "_text"]].itertuples(index=False, name=None):
            if not isinstance(identity, str) or not isinstance(text, str):
                raise ValueError("Source cache requires string IDs and original text")
            if identity in documents[side] and documents[side][identity] != text:
                raise ValueError("Same source identity has conflicting document text")
            documents[side][identity] = text
    return documents


@dataclass(frozen=True)
class PortalSnapshot:
    snapshot_id: str
    run_id: str
    jd_id: str
    jd_title: str
    digest: str
    _payload_json: str
    model_id: str = "existing-x2-table"

    @property
    def candidates(self):
        return tuple(json.loads(self._payload_json)["candidates"])

    @property
    def jd_entities(self):
        return json.loads(self._payload_json)["jd_entities"]

    @property
    def jd_text(self):
        return json.loads(self._payload_json)["jd_text"]

    @property
    def source_flags(self):
        return json.loads(self._payload_json)["source_flags"]


@dataclass(frozen=True)
class PortalDataset:
    run_id: str
    digest: str
    _jobs_json: str
    _snapshots_json: str
    is_demo: bool = False

    @classmethod
    def from_folder(cls, folder: Path | str = PROCESSED / "phase2-rerank-output", *,
                    source_documents=None, source_loader=None, _frames=None, _is_demo=False):
        folder = Path(folder)
        # Whitelists are essential: final_label must never be read into memory.
        if _frames is None:
            if _is_demo:
                raise ValueError("Real source folders cannot be labelled synthetic demos")
            paths = [folder / name for name in ("a2-cv-extraction.csv", "a2-jd-extraction.csv", "a2-x2-scores.csv")]
            versions = [(path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
            cv = pd.read_csv(folder / "a2-cv-extraction.csv", usecols=list(CV_COLUMNS))
            jd = pd.read_csv(folder / "a2-jd-extraction.csv", usecols=list(JD_COLUMNS))
            ranks = pd.read_csv(folder / "a2-x2-scores.csv", usecols=list(RANK_COLUMNS))
            if versions != [(path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in paths]:
                raise ValueError("Source tables changed while loading; retry a consistent snapshot")
        else:
            cv, jd, ranks = (frame.loc[:, list(columns)].copy(deep=True) for frame, columns in zip(_frames, (CV_COLUMNS, JD_COLUMNS, RANK_COLUMNS), strict=True))
        if not jd.jd_title.map(lambda text: isinstance(text, str) and bool(text.strip())).all():
            raise ValueError("JD titles must be explicit nonempty table strings")
        for frame, identity in ((cv, "cv_id"), (jd, "jd_id"), (ranks, "pair_id")):
            if frame[identity].isna().any() or not frame[identity].map(lambda x: isinstance(x, str) and bool(x.strip())).all() or frame[identity].duplicated().any():
                raise ValueError("Missing or duplicate stable table identity")
        if ranks[["jd_id", "cv_id", "stratum"]].isna().any().any() or ranks[["jd_id", "cv_id"]].duplicated().any():
            raise ValueError("Ranking rows contain missing or duplicate document joins")
        if set(ranks.jd_id) != set(jd.jd_id) or set(ranks.cv_id) != set(cv.cv_id):
            raise ValueError("Ranked document IDs do not exactly match extraction tables")
        scores = pd.to_numeric(ranks.x2_score, errors="coerce")
        rank_values = pd.to_numeric(ranks.x2_rank, errors="coerce")
        if not scores.map(lambda v: pd.notna(v) and math.isfinite(float(v))).all():
            raise ValueError("Ranking scores must be finite")
        if not rank_values.map(lambda v: pd.notna(v) and math.isfinite(float(v)) and float(v).is_integer()).all():
            raise ValueError("Ranking ranks must be finite integers")
        ranks = ranks.assign(x2_score=scores, x2_rank=rank_values.astype(int))
        for _, group in ranks.groupby("jd_id", sort=False):
            if len(group) != 10 or sorted(group.x2_rank.tolist()) != list(range(1, 11)):
                raise ValueError("Every JD must contain exactly ten existing ranks1–10; no skipped rows or backfill")
        jd_records = {row["jd_id"]: row for row in jd.to_dict("records")}
        cv_records = {row["cv_id"]: row for row in cv.to_dict("records")}
        if any(row.stratum != jd_records[row.jd_id]["stratum"] for row in ranks.itertuples()):
            raise ValueError("Ranking and JD strata disagree")
        if source_documents is not None and source_loader is not None:
            raise ValueError("Choose source documents or a source loader")
        sources = source_documents if source_documents is not None else (source_loader or load_source_documents)()
        if not isinstance(sources, dict) or any(not isinstance(sources.get(side), dict) for side in ("jd", "cv")):
            raise ValueError("Source documents require JD/CV ID-to-text mappings")
        if any(not isinstance(k, str) or not isinstance(v, str) for side in sources.values() for k, v in side.items()):
            raise ValueError("Source document identities/text must be strings")
        loaded = {"cv": _clean(cv.to_dict("records")), "jd": _clean(jd.to_dict("records")),
                  "ranks": _clean(ranks.to_dict("records"))}
        run_digest = _digest(loaded)
        run_id = ("demo-" if _is_demo else "phase2-x2:") + run_digest[:24]
        jobs, snapshots = [], {}
        for raw_id, job in sorted(jd_records.items()):
            safe_id, title = redact_text(raw_id), redact_text(str(job["jd_title"]))
            if safe_id in snapshots:
                raise ValueError("Redaction made distinct JD IDs ambiguous")
            candidates = []
            for row in ranks[ranks.jd_id == raw_id].sort_values("x2_rank").to_dict("records"):
                raw_cv_id = row["cv_id"]
                source = sources["cv"].get(raw_cv_id)
                preview = redact_text(source) if source is not None else None
                entities = _entities(cv_records[raw_cv_id])
                flags = {"verified_spans_available": False, "table_mentions_only": True,
                         "is_demo": bool(_is_demo),
                         "preview_status": "available_redacted_text" if source is not None else "unavailable",
                         "page_metadata": "unavailable", "pdf_preview": "unavailable"}
                candidates.append({"rank": int(row["x2_rank"]), "cv_id": redact_text(raw_cv_id),
                    "candidate_id": redact_text(raw_cv_id), "score": float(row["x2_score"]),
                    "score_semantics": "existing X2 ordering score; not a Fit probability",
                    "entities": entities, "profile": entities, "table_aggregates": _profile(cv_records[raw_cv_id]),
                    "preview_text": preview, "cv_text": preview, "preview_status": flags["preview_status"],
                    "source_digest": hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
                    "source_flags": flags, "coverage": flags})
            if len({c["cv_id"] for c in candidates}) != len(candidates):
                raise ValueError("Redaction made distinct candidate IDs ambiguous")
            jd_source = sources["jd"].get(raw_id)
            payload = {"jd_entities": _entities(job, jd=True), "candidates": candidates,
                       "jd_text": redact_text(jd_source) if jd_source is not None else None,
                       "source_flags": {"preview_status": "available_redacted_text" if jd_source is not None else "unavailable",
                                        "is_demo": bool(_is_demo),
                                        "source_digest": hashlib.sha256(jd_source.encode()).hexdigest() if jd_source is not None else None,
                                        "verified_spans_available": False, "table_mentions_only": True,
                                        "run_identity_semantics": "digest of allowed table columns; producer timestamp unavailable"}}
            snapshot_digest = _digest({"run_id": run_id, "jd_id": safe_id, "jd_title": title, **payload})
            snapshots[safe_id] = {"snapshot_id": snapshot_digest[:24], "run_id": run_id,
                                 "jd_id": safe_id, "jd_title": title, "digest": snapshot_digest,
                                 "_payload_json": _json(payload),
                                 "model_id": "synthetic-demo-ranks" if _is_demo else "existing-x2-table"}
            jobs.append({"jd_id": safe_id, "jd_title": title})
        return cls(run_id, run_digest, _json(jobs), _json(snapshots), bool(_is_demo))

    load = from_folder

    def jobs(self):
        return json.loads(self._jobs_json)

    @classmethod
    def synthetic_demo(cls):
        return synthetic_demo()

    def snapshot(self, jd_id: str) -> PortalSnapshot:
        data = json.loads(self._snapshots_json).get(jd_id)
        if data is None:
            raise ValueError("Unknown JD in the validated portal dataset")
        return PortalSnapshot(**data)


load_portal_dataset = PortalDataset.from_folder


def synthetic_demo() -> PortalDataset:
    """Invented in-memory documents/ranks; never opens any local real corpus."""
    cvs, jobs, ranks = [], [], []
    sources = {"jd": {}, "cv": {}}
    for job_index, topic in enumerate(("Python", "SQL"), 1):
        jd_id = f"synthetic-jd-{job_index}"
        job = dict.fromkeys(JD_COLUMNS)
        job.update(jd_id=jd_id, stratum="synthetic", jd_title=f"Synthetic {topic} role",
                   role="synthetic family", skills=topic, min_years=2)
        jobs.append(job)
        sources["jd"][jd_id] = f"Synthetic job: {topic} development and software testing."
        for rank in range(1, 11):
            cv_id = f"synthetic-cv-{job_index}-{rank}"
            cv = dict.fromkeys(CV_COLUMNS)
            cv.update(cv_id=cv_id, role="synthetic family", skills=topic + ";software testing", years=3)
            cvs.append(cv)
            sources["cv"][cv_id] = f"Invented applicant {job_index}-{rank}. {topic} development and software testing. Three years of experience."
            ranks.append(dict(pair_id=f"synthetic-pair-{job_index}-{rank}", stratum="synthetic",
                              jd_id=jd_id, cv_id=cv_id, x2_score=float(11-rank), x2_rank=rank))
    return PortalDataset.from_folder(source_documents=sources,
        _frames=(pd.DataFrame(cvs), pd.DataFrame(jobs), pd.DataFrame(ranks)), _is_demo=True)
