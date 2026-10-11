"""Small session-only screening and applicant-review chat illustration."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from candidate_screener import pipeline
from candidate_screener.annotation.redact import redact_text
from candidate_screener.app import files, intents
from candidate_screener.app.submissions import DEMO_JD, DEMO_CVS, Submission, applicant_id

st.set_page_config(page_title="Candidate screening", layout="wide")

def _extractor():
    from candidate_screener.extraction.live import GlinerEntityExtractor
    if "entity_extractor" not in st.session_state:
        st.session_state.entity_extractor = GlinerEntityExtractor()
    return st.session_state.entity_extractor


def _run(jd_text, texts, source_kind="written_cv", provenance=None, illustration=False):
    from candidate_screener.matching.demo import IllustrationTfidf
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
    ranker = IllustrationTfidf()
    ranker.fit([jd_text, *texts])
    st.session_state.results = pipeline.screen(jd_text, cvs, ranker=ranker,
                                               matcher=matcher, entity_extractor=_extractor())
    st.session_state.cvs = cvs
    st.session_state.chat = []


st.title("Intelligent candidate screener · chat illustration")
st.caption("Text evidence only. Scores are relative to this submission pool; no three-class fit prediction is made.")
try:
    availability = _extractor().availability()
    st.caption(f"Five-label GLiNER: {availability.get('status', 'unknown')} · {availability.get('reason', '')}")
except (ImportError, FileNotFoundError, RuntimeError) as exc:
    st.warning(f"Five-label GLiNER unavailable ({type(exc).__name__}). Screening prerequisites must be restored.")

with st.sidebar:
    source = st.radio("Source", ["Illustration demo", "Upload CVs", "Applicant interview"])
    jd_text = st.text_area("Job description", value=DEMO_JD, height=160)
    if st.button("Clear session"):
        if "entity_extractor" in st.session_state:
            st.session_state.entity_extractor.close()
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()
    if source == "Illustration demo":
        st.caption("Two invented CVs. This demonstrates the interface, not benchmark performance.")
        if st.button("Screen demonstration", type="primary", disabled=not jd_text.strip()):
            _run(jd_text, DEMO_CVS, illustration=True)
    elif source == "Upload CVs":
        uploads = st.file_uploader("Written CVs", type=["txt", "pdf", "html"], accept_multiple_files=True)
        if st.button("Screen CVs", disabled=not (jd_text.strip() and uploads)):
            try:
                _run(jd_text, [files.to_text(u.name, u.getvalue()) for u in uploads])
            except (ValueError, OSError, ImportError) as exc:
                st.error(f"Could not read the CVs ({type(exc).__name__}); try a text file.")

if source == "Applicant interview":
    st.subheader("Applicant conversation")
    st.info("Submit a French recording, review its English translation, then confirm it. A short introduction may omit qualifications.")
    mode = st.radio("Use this interview as", ["Standalone interview", "CV supplement"])
    written = st.text_area("Written CV text (required for supplement)") if mode == "CV supplement" else ""
    nonce = st.session_state.get("media_upload_nonce", 0)
    upload = st.file_uploader("French audio/video · maximum 20 MB and 120 seconds", type=["mp4", "mov", "m4a", "mp3", "wav", "webm", "ogg"], key=f"media_upload_{nonce}")
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
            except ValueError as exc:
                st.error(redact_text(str(exc)))
        if st.button("Delete interview transcript"):
            _extractor().clear()
            for key in ("submission", "review_text", "results", "cvs", "chat", "applicant_messages"):
                st.session_state.pop(key, None)
            st.rerun()

results = st.session_state.get("results")
if results:
    st.caption("Ranker: " + str(getattr(results[0], "ranker_name", "Illustration TF-IDF")))
    st.dataframe(pd.DataFrame([{"rank": r.rank, "applicant": r.cv_id,
        "score": r.score, "relative percentile": r.confidence,
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
else:
    st.info("Press Screen demonstration to try the chatbox without downloading benchmark data.")
st.caption(intents.HELP)
if prompt := st.chat_input("Try: top 5, explain 1, entities 1, source 1, at least 3 years"):
    reply = intents.answer(intents.parse(prompt), results or [])
    st.session_state.setdefault("chat", []).extend([("user", redact_text(prompt)), ("assistant", redact_text(reply))])
    st.rerun()
