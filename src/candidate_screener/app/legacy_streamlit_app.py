"""Session-only recruiter QA over frozen screening results and reviewed evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from candidate_screener.annotation.redact import redact_text
from candidate_screener.app import files, intents
from candidate_screener.app.chat_schema import ChatState
from candidate_screener.app.chat_service import answer_question
from candidate_screener.app.rag_service import DialogueState, RagService
from candidate_screener.llm.extractive_qa import QaClient
from candidate_screener.llm.qa_assets import ROOT as QA_ROOT
from candidate_screener.app.screening_service import load_reranker, screen
from candidate_screener.app.rerank_client import load_service_reranker
from candidate_screener.app.session import input_key, invalidate
from candidate_screener.app.submissions import DEMO_JD, DEMO_CVS, Submission, applicant_id
from candidate_screener.llm.base import ProviderUnavailable
from candidate_screener.llm.qwen import QwenConfig, QwenProvider
from candidate_screener.llm.qwen_assets import DEFAULT_MANIFEST, local_settings

st.set_page_config(page_title="Candidate screening", layout="wide")

def _extractor():
    from candidate_screener.extraction.live import GlinerEntityExtractor
    if "entity_extractor" not in st.session_state:
        st.session_state.entity_extractor = GlinerEntityExtractor()
    return st.session_state.entity_extractor


def _settings():
    values = dict(os.environ)
    try:
        values.update({k: str(v) for k, v in st.secrets.get("chatbot", {}).items()})
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    if values.get("SCREENER_QWEN_DISABLED", "false").lower() == "true":
        values = {key: value for key, value in values.items() if not key.startswith("SCREENER_QWEN_")}
    elif not values.get("SCREENER_QWEN_BASE_URL") and not values.get("SCREENER_QWEN_MODEL"):
        try:
            for key, value in local_settings().items():
                values.setdefault(key, value)
        except (OSError, ValueError, KeyError):
            pass
    default_ce = DEFAULT_MANIFEST.parent.parent / "reranker.json"
    if default_ce.is_file():
        values.setdefault("SCREENER_CE_MANIFEST", str(default_ce))
        if DEFAULT_MANIFEST.is_file():
            values.setdefault("SCREENER_CE_SERVICE_URL", "http://127.0.0.1:8000/v1")
    return values


@st.cache_resource(show_spinner=False)
def _load_backend(path, artifact_key, service_url=""):
    return load_service_reranker(path, service_url) if service_url else load_reranker(path)


def _artifact_key(path):
    """Reload changed manifests/assets; the loader verifies content digests."""
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    assets = []
    for key in ("checkpoint", "tokenizer"):
        root = (manifest_path.parent / manifest[key]).resolve()
        for asset in sorted(root.rglob("*")):
            if asset.is_file():
                stat = asset.stat()
                assets.append((str(asset), stat.st_mtime_ns, stat.st_size))
    return hashlib.sha256(raw + json.dumps(assets).encode()).hexdigest()


def _provider(values):
    try:
        config = QwenConfig.from_mapping(values)
        if config is None:
            st.session_state.pop("qwen", None)
            st.session_state.pop("provider_key", None)
            st.caption("Qwen is not configured; grounded template answers are available.")
            return None
        # Provider carries a tokenizer and immutable config, never applicant context.
        key = hashlib.sha256(repr(config).encode() + config.api_key.encode()).hexdigest()
        if st.session_state.get("provider_key") != key:
            st.session_state.qwen = QwenProvider(config)
            st.session_state.provider_key = key
        st.caption(f"QA model: {redact_text(config.model)} · {redact_text(config.revision)}")
        return st.session_state.qwen
    except (ValueError, OSError, ImportError, ProviderUnavailable):
        st.warning("Qwen configuration or local tokenizer is unavailable; using grounded template answers.")
        st.session_state.pop("qwen", None)
        st.session_state.pop("provider_key", None)
        return None


def _invalidate(documents, source_description):
    current = input_key(jd_text, jd_title, source_description, documents,
                        {"backend": backend_mode, "artifact": backend_key, "shortlist": shortlist_size,
                         "ce_service": settings.get("SCREENER_CE_SERVICE_URL", ""),
                         "provider": st.session_state.get("provider_key", "templates"),
                         "qa": hashlib.sha256(qa_manifest_path.read_bytes()).hexdigest()
                         if qa_manifest_path.is_file() else "unavailable"})
    if invalidate(st.session_state, current):
        st.info("Inputs changed. Screen the current documents to start a new conversation.")


def _run(jd_text, texts, source_kind="written_cv", provenance=None, illustration=False, candidate_frame=None):
    from candidate_screener.extraction.rules import SkillMatcher
    # Small explicit illustration vocabulary, not a substitute for GLiNER predictions.
    if illustration:
        terms = ["Python", "SQL", "software testing", "project management"]
        matcher = SkillMatcher(pd.DataFrame({"canonical": terms, "alias": terms}))
    else:
        from candidate_screener.extraction.rules import default_matcher
        try:
            matcher = default_matcher()
        except (FileNotFoundError, OSError):
            # Explicitly empty rules evidence; GLiNER remains a separate stage.
            st.warning("Full skill vocabulary is unavailable; rule-based skill evidence is disabled.")
            matcher = SkillMatcher(pd.DataFrame({"canonical": [], "alias": []}))
    cvs = pd.DataFrame({"cv_id": [applicant_id(i + 1) for i in range(len(texts))],
                        "cv_text": texts, "source_kind": source_kind})
    if provenance:
        cvs["translation_provenance"] = [provenance for _ in texts]
    if candidate_frame is not None:
        cvs = candidate_frame.copy(deep=True)
    if backend_mode == "Configured cross-encoder" and backend is None:
        st.error("The configured frozen cross-encoder is unavailable. Restore its manifest and weights before screening.")
        return
    with st.spinner("Screening candidate text and collecting evidence…"):
        snapshot = screen(jd_text, cvs, jd_title=jd_title, reranker=backend,
                          shortlist_size=shortlist_size, matcher=matcher, entity_extractor=_extractor())
    st.session_state.snapshot = snapshot
    st.session_state.results = list(snapshot.results)
    st.session_state.cvs = cvs
    st.session_state.chat = []
    st.session_state.chat_state = DialogueState(snapshot.snapshot_id)
    if old_service := st.session_state.get("rag_service"):
        old_service.close()
    st.session_state.rag_service = RagService(snapshot, provider=st.session_state.get("qwen"), extractor=QaClient(qa_manifest_path))
    st.session_state.pop("last_answer", None)
    st.session_state.pop("filter_description", None)


st.title("HR screening assistant")
st.caption("Explore the active role, compare applicants and inspect cited CV evidence. Scores are not fit probabilities.")
settings = _settings()
provider = _provider(settings)
qa_manifest_path = Path(settings.get("SCREENER_QA_MANIFEST", str(QA_ROOT / "manifest.json"))).expanduser().resolve()
st.caption("Factual QA: " + ("local extractive model configured; its files are verified when loaded"
           if qa_manifest_path.is_file() else "unavailable; source retrieval remains available"))
if settings.get("SCREENER_QWEN_TOKENIZER") == "service":
    st.caption("Local Qwen2.5 workshop service: start it separately; unavailable or invalid answers use templates.")
try:
    availability = _extractor().availability()
    st.caption(f"Five-label GLiNER: {availability.get('status', 'unknown')} · {availability.get('reason', '')}")
except (ImportError, FileNotFoundError, RuntimeError) as exc:
    st.warning(f"Five-label GLiNER unavailable ({type(exc).__name__}). Screening prerequisites must be restored.")

with st.sidebar:
    source = st.radio("Source", ["Illustration demo", "Upload CVs", "Applicant interview"])
    jd_title = st.text_input("Job title (optional)")
    jd_text = st.text_area("Job description", value=DEMO_JD, height=160)
    manifest_path = settings.get("SCREENER_CE_MANIFEST", "")
    backend_mode = st.selectbox("Screening backend", ["Illustration TF-IDF", "Configured cross-encoder"],
                               index=1 if manifest_path else 0)
    shortlist_size = int(st.number_input("Reranked shortlist size", min_value=1, max_value=100, value=50))
    backend, backend_key = None, "illustration"
    if backend_mode == "Configured cross-encoder":
        try:
            if not manifest_path:
                raise ValueError("No manifest configured")
            backend_key = _artifact_key(manifest_path)
            backend = _load_backend(manifest_path, backend_key, settings.get("SCREENER_CE_SERVICE_URL", ""))
            st.caption("Cross-encoder: " + redact_text(backend.name))
            st.caption("Configured application model; research adoption is recorded separately.")
        except (ValueError, OSError, ImportError, RuntimeError, KeyError):
            backend_key = "unavailable"
            st.warning("Frozen cross-encoder manifest or weights unavailable. No replacement model is selected.")
    else:
        st.caption("Explicit pool-based TF-IDF illustration; Fit / No Fit classification is unavailable.")
    if st.button("Clear session"):
        if service := st.session_state.get("rag_service"):
            service.close()
        if "entity_extractor" in st.session_state:
            st.session_state.entity_extractor.close()
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()
    if source == "Illustration demo":
        _invalidate(DEMO_CVS, source)
        st.caption("Two invented CVs. This demonstrates the interface, not benchmark performance.")
        if st.button("Screen demonstration", type="primary", disabled=not jd_text.strip()):
            try:
                _run(jd_text, DEMO_CVS, illustration=True)
            except (ValueError, OSError, ImportError, RuntimeError):
                st.error("Screening failed. Check the configured model and input prerequisites.")
    elif source == "Upload CVs":
        uploads = st.file_uploader("Written CVs", type=["txt", "pdf", "html"], accept_multiple_files=True)
        _invalidate([u.getvalue() for u in uploads], source)
        if st.button("Screen CVs", disabled=not (jd_text.strip() and uploads)):
            try:
                _run(jd_text, [files.to_text(u.name, u.getvalue()) for u in uploads])
            except (ValueError, OSError, ImportError, RuntimeError) as exc:
                st.error(f"Could not read the CVs ({type(exc).__name__}); try a text file.")

if source == "Applicant interview":
    st.subheader("Applicant conversation")
    st.info("Submit a French recording, review its English translation, then confirm it. A short introduction may omit qualifications.")
    mode = st.radio("Use this interview as", ["Standalone interview", "CV supplement"])
    written = st.text_area("Written CV text (required for supplement)") if mode == "CV supplement" else ""
    nonce = st.session_state.get("media_upload_nonce", 0)
    upload = st.file_uploader("French audio/video · maximum 20 MB and 120 seconds", type=["mp4", "mov", "m4a", "mp3", "wav", "webm", "ogg"], key=f"media_upload_{nonce}")
    _invalidate([written, st.session_state.get("review_text", ""), upload.getvalue() if upload else b""], source + mode)
    if st.button("Transcribe and translate", disabled=upload is None):
        from candidate_screener.multimodal.submission import process_media, MediaError
        try:
            with st.spinner("Transcribing French and translating with MarianMT…"):
                processed = process_media(upload.name, upload.getvalue(), language="fr")
            segments = processed["segments"]
            st.session_state.submission = Submission(segments=segments,
                english_text="\n".join(s["english"] for s in segments),
                provenance=processed.get("provenance", {}), state="translated")
            st.session_state.review_text = redact_text(st.session_state.submission.english_text)
            st.session_state.applicant_messages = [("assistant", "Your recording has been transcribed and translated. Review the English below, especially durations, technology names and negation.")]
            st.session_state.media_upload_nonce = nonce + 1
            st.rerun()
        except MediaError as exc:
            st.error(redact_text(str(exc)))
    submission = st.session_state.get("submission")
    for role, message in st.session_state.get("applicant_messages", [("assistant", "Welcome. Submit a French introduction or use a written CV, then review your evidence before screening.")]):
        with st.chat_message(role):
            st.markdown(redact_text(message))
    if submission:
        st.caption("Review technology names, durations and negation. Display redaction removes contact details, not every personal name.")
        with st.expander("French transcript with audio timestamps"):
            for segment in submission.segments:
                st.write(f"{segment['start']:.1f}–{segment['end']:.1f}s: " + redact_text(segment.get("french", "")))
        reviewed = st.text_area("Review/edit English translation", key="review_text", height=220)
        submission.written_text = written
        if reviewed.strip() != submission.english_text:
            submission.state = "reviewed"
        if st.button("Confirm reviewed English", disabled=not reviewed.strip()):
            submission.confirm(reviewed)
            st.session_state.applicant_messages = [("user", "I confirm the reviewed English transcript."),
                ("assistant", "Your submission is ready. You can screen it against the job description; missing information in an introduction does not establish No Fit.")]
        st.caption(f"Submission state: {submission.state}")
        st.download_button("Download reviewed English", reviewed, file_name="reviewed-interview.txt")
        if st.button("Screen reviewed submission", disabled=submission.state != "ready" or not jd_text.strip()):
            try:
                from candidate_screener.multimodal.entity_bridge import reviewed_document
                linked = reviewed_document(submission.segments, submission.english_text, confirmed=True)
                if mode == "CV supplement":
                    shift = len(f"Written CV:\n{submission.written_text.strip()}\n\nReviewed interview:\n")
                    for segment in linked["segments"]:
                        segment["english_start"] += shift
                        segment["english_end"] += shift
                linked["segments"] = [{k: v for k, v in segment.items()
                                       if k not in {"french", "english"}}
                                      for segment in linked["segments"]]
                provenance = {**submission.provenance, **{k: v for k, v in linked.items()
                              if k not in {"text", "source_segments"}}}
                _run(jd_text, [submission.scoring_text(mode)],
                     "combined" if mode == "CV supplement" else "interview", provenance)
            except (ValueError, OSError, ImportError, RuntimeError):
                st.error("Screening failed. Confirm the reviewed text and check model prerequisites.")
        if st.button("Delete interview transcript"):
            _extractor().clear()
            for key in ("submission", "review_text", "results", "cvs", "chat", "applicant_messages",
                        "snapshot", "chat_state", "last_answer"):
                st.session_state.pop(key, None)
            st.rerun()

results = st.session_state.get("results")
if results:
    st.caption("Ranker: " + redact_text(str(getattr(results[0], "ranker_name", "Illustration TF-IDF"))))
    st.caption("Reranked shortlist first; retrieval-only candidates retain retrieval order. Classification coverage is shown per candidate.")
    st.dataframe(pd.DataFrame([{"rank": r.rank, "applicant": r.cv_id,
        "retrieval score": r.retrieval_score, "cross-encoder score": r.rerank_score,
        "reranking": r.rerank_status, "fit call": r.fit_call or "not assessed",
        "classification": r.classification_status,
        "requirement coverage": r.coverage, "years stated": r.years_stated,
        "entity extraction": getattr(r, "entity_status", "unavailable")} for r in results]), hide_index=True)
    chosen_rank = st.selectbox("Evidence for applicant rank", [r.rank for r in results])
    chosen = next(r for r in results if r.rank == chosen_rank)
    with st.expander("Five-label extraction and requirement evidence"):
        st.markdown(redact_text(intents.answer(intents.Intent("entities", (str(chosen_rank),)), results)))
        st.markdown(redact_text(intents.answer(intents.Intent("explain", (str(chosen_rank),)), results)))
        jd_status = getattr(chosen, "jd_entity_status", "unavailable")
        st.caption(f"JD entity evidence · {jd_status}")
        labels = ("KNOWLEDGE", "SKILL", "TITLE", "DEGREE", "EXPERIENCE_YEARS")
        st.write({label: sum(e.get("label") == label for e in chosen.jd_entities)
                  for label in labels} if jd_status in {"complete", "completed_zero"}
                 else "JD counts are unavailable; missing extraction is not zero predictions.")
    for role, message in st.session_state.get("chat", []):
        with st.chat_message(role):
            st.markdown(redact_text(message))
    if last_answer := st.session_state.get("last_answer"):
        if last_answer.fallback:
            st.caption("Answer mode: source excerpts with validated backend facts. Generator selection was unavailable or not required.")
        st.caption(f"Question route: {getattr(last_answer, 'route', 'legacy')} · factual QA: {getattr(last_answer, 'qa_status', 'not_requested')}")
        if getattr(last_answer, "comparison", ()):
            with st.expander("Requirement evidence comparison"):
                st.dataframe(pd.DataFrame(last_answer.comparison), hide_index=True)
        if last_answer.citations:
            with st.expander("Answer sources"):
                for citation in last_answer.citations:
                    st.write(redact_text(f"{citation.evidence_id} · {citation.candidate_id}: {citation.quote}"))
                    st.caption(f"Original text offsets: {citation.original_start}–{citation.original_end}. Display offsets refer to the redacted document.")
        pending = getattr(st.session_state.get("chat_state"), "pending_filter", "")
        if pending:
            service = st.session_state.rag_service
            matching = service.filter_preview(pending)
            st.info(f"Proposed evidence filter: {redact_text(pending)}. {len(matching)} applicants have unambiguous matching excerpts. Review before applying.")
            if st.button("Apply evidence filter and screen again", disabled=not matching):
                current_cvs = st.session_state.cvs
                filtered = current_cvs[current_cvs.cv_id.astype(str).isin(matching)]
                previous = st.session_state.snapshot
                _run(previous.jd_text, filtered.cv_text.tolist(), illustration=source == "Illustration demo", candidate_frame=filtered)
                st.session_state.filter_description = pending
                st.rerun()
    if st.session_state.get("filter_description"):
        st.caption("Active evidence filter: " + redact_text(st.session_state.filter_description))
    with st.expander("Questions to try"):
        st.write("Who best suits this role? · Compare 1 and 2 · What certifications does applicant 1 hold? · What about applicant 3?")
else:
    st.info("Press Screen demonstration to try the chatbox without downloading benchmark data.")
st.caption(intents.HELP)
if prompt := st.chat_input("Ask about the shortlist, compare candidates or request supporting evidence", disabled=not results):
    with st.spinner("Preparing a grounded answer…"):
        service = st.session_state.get("rag_service")
        if service is None or service.snapshot.snapshot_id != st.session_state.snapshot.snapshot_id:
            service = RagService(st.session_state.snapshot, provider=provider, extractor=QaClient(qa_manifest_path))
            st.session_state.rag_service = service
        service.provider = provider
        answer = service.answer(prompt, state=st.session_state.get("chat_state"))
    reply = answer.text
    st.session_state.chat_state = answer.state
    st.session_state.last_answer = answer
    st.session_state.setdefault("chat", []).extend([("user", redact_text(prompt)), ("assistant", redact_text(reply))])
    st.session_state.chat = st.session_state.chat[-50:]
    st.rerun()
