"""Guided JD authoring transport, separate from evidence review chat."""
from __future__ import annotations

import math
import os

import streamlit as st

from .portal_jobs import GUIDED_FIELDS, REQUIREMENT_FIELDS, guided_question
from .portal_store import RevisionConflict


def _answer(field, value):
    text = value.strip()
    unspecified = text.casefold() in {"", "skip", "not specified", "unspecified", "none"}
    if field in {"min_years", "max_years"}:
        if unspecified:
            return None
        years = float(text)
        if not math.isfinite(years) or not 0 <= years <= 100:
            raise ValueError("Enter years between 0 and 100, or 'unspecified'.")
        return years
    if field in REQUIREMENT_FIELDS:
        result = {"required": [], "preferred": []}
        if not unspecified:
            for line in text.splitlines():
                level = "required"
                line = line.strip().lstrip("-• ")
                if not line:
                    continue
                for prefix in ("required", "preferred"):
                    if line.casefold().startswith(prefix + ":"):
                        level, line = prefix, line[len(prefix)+1:].strip()
                        break
                if line:
                    result[level].append(line)
        return result
    if unspecified:
        return ""
    return text


def _display(value):
    if isinstance(value, dict):
        return "\n".join(level.title() + ": " + "; ".join(items) for level, items in value.items() if items) or "Not specified"
    return "Not specified" if value is None or value == "" else str(value)


def render_authoring(controller):
    service = controller.job_service
    st.header("Create a new job description")
    st.write("Build the requirements one question at a time, review the full JD, then rank candidates with the live model.")
    jobs = [job for job in service.list() if job.get("can_edit", True)]
    choices = {job["job_id"]: f"{job['title'] or 'Untitled JD'} · {job['status']} · revision {job['revision']}" for job in jobs}
    if st.session_state.pop("portal_authoring_clear_resume", False):
        st.session_state.portal_authoring_resume = None
    selected = st.selectbox("Resume a JD", [None] + list(choices), format_func=lambda value: "Choose a saved JD" if value is None else choices[value], key="portal_authoring_resume")
    if selected:
        st.session_state.portal_authoring_job = selected
    if st.button("Start a new JD", type="primary"):
        job = service.create(workspace=controller.department)
        st.session_state.portal_authoring_job = job["job_id"]
        st.session_state.portal_authoring_clear_resume = True
        st.rerun()
    job_id = st.session_state.get("portal_authoring_job")
    if not job_id:
        st.info("Start a new JD or resume one above. Your answers are saved after each accepted response.")
        return
    job = service.get(job_id)
    st.caption(f"Revision {job['revision']} · {job['status'].title()} · saved locally")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Guided requirements")
        st.caption("For lists, use one item per line. Prefix a line with 'Preferred:' to mark a preference; other items are required.")
        for field in GUIDED_FIELDS:
            if field not in job["confirmed_fields"]:
                continue
            with st.chat_message("assistant"):
                st.write(guided_question(job["fields"], [f for f in GUIDED_FIELDS if f != field])["question"])
            with st.chat_message("user"):
                st.text(_display(job["fields"][field]))
        question = service.next_question(job_id)
        if job["status"] == "draft" and question:
            with st.chat_message("assistant"):
                st.write(question["question"])
            if question["optional"] and st.button("Leave this requirement unspecified", key="portal_skip_" + job_id):
                service.update(job_id, job["revision"], {question["field"]: _answer(question["field"], "unspecified")})
                st.rerun()
            response = st.chat_input("Answer the JD question", key="portal_authoring_chat_" + job_id, max_chars=2000)
            if response:
                try:
                    value = _answer(question["field"], response)
                    required = question["field"] in {"title", "department", "responsibilities"}
                    empty = not any(value.values()) if isinstance(value, dict) else not value
                    if required and empty:
                        raise ValueError("This requirement is needed before publication.")
                    service.update(job_id, job["revision"], {question["field"]: value})
                except (ValueError, RevisionConflict) as exc:
                    st.warning(str(exc))
                else:
                    st.rerun()
        elif job["status"] == "draft":
            st.success("The guided questions are complete. Review your preview before publishing.")
        with st.expander("Edit a previous answer"):
            field = st.selectbox("Requirement to edit", GUIDED_FIELDS, format_func=lambda value: value.replace("_", " ").title(), key="portal_edit_field_" + job_id)
            st.caption("Current value: " + _display(job["fields"][field]))
            with st.form("portal_edit_answer_" + job_id):
                updated = st.text_area("Replacement answer", key="portal_edit_value_" + job_id + field)
                if st.form_submit_button("Save answer"):
                    try:
                        service.update(job_id, job["revision"], {field: _answer(field, updated)})
                    except (ValueError, RevisionConflict) as exc:
                        st.warning(str(exc))
                    else:
                        st.rerun()
    with right:
        st.subheader("Full JD preview")
        st.text(job["jd_text"])
        if job["status"] == "draft" and st.button("Confirm JD preview", key="portal_review_jd_" + job_id):
            try:
                service.review(job_id, job["revision"])
            except (ValueError, RevisionConflict) as exc:
                st.warning(str(exc))
            else:
                st.rerun()
        elif job["status"] == "reviewed":
            st.info("Preview confirmed. Publish to add this JD to the job browser and enable ranking.")
            if st.button("Publish JD", key="portal_publish_jd_" + job_id, type="primary"):
                service.publish(job_id, job["revision"])
                st.rerun()
        elif job["status"] == "published":
            st.success("JD published in Hiring Desk.")
            _ranking_controls(controller, job)
            if st.button("Open published JD"):
                from .portal_ui import _switch_job
                _switch_job(controller, job_id)
                st.session_state.portal_next_navigation = "Review jobs"
                st.rerun()


def _ranking_controls(controller, job):
    if not job.get("can_edit", True):
        st.caption("Published job shared with your workspace. Its author manages requirements and ranking requests.")
        return
    service = controller.ranking_service
    availability = service.availability
    st.caption("Ranking status: " + job["ranking_status"].replace("_", " "))
    if not availability["available"]:
        st.info("Live ranking is not connected yet. Your JD is saved; existing CSV reviews remain available.")
        with st.expander("Live integration details"):
            st.write(availability["reason"])
        return
    request_id = job.get("active_request_id")
    if request_id:
        request = service.get(request_id)
        if request["status"] in {"queued", "running"}:
            st.caption("Ranking runs in the background; the D64 reranker takes a few minutes. Use Refresh ranking status to fetch the result.")
            if st.button("Refresh ranking status", key="portal_refresh_rank_" + job["job_id"]):
                with st.spinner("Checking the ranking service…"):
                    service.refresh(request_id)
                st.rerun()
        elif request["status"] in {"failed", "service_unavailable"}:
            st.warning(request.get("error") or "The ranking request could not complete. Retry when the service is available.")
    if not request_id or job["ranking_status"] in {"ready", "failed", "service_unavailable"}:
        label = "Rank candidates" if not request_id else "Re-rank candidates" if job["ranking_status"] == "ready" else "Retry ranking"
        onnx = os.environ.get("HR_PORTAL_EMBEDDING_BACKEND", "torch").strip().lower() == "onnx"
        refine = st.checkbox("Refine the top 50 with the reranker (D64, a few minutes)",
                             key="portal_rank_d64_" + job["job_id"], disabled=onnx)
        if onnx:
            st.caption("Local ONNX mode runs D63. The Torch-based D64 reranker is unavailable in this mode.")
            refine = False
        if st.button(label, key="portal_rank_" + job["job_id"], type="primary"):
            with st.spinner("Submitting the confirmed JD for ranking…"):
                service.rank(job["job_id"], job["revision"], rerank=bool(request_id), system="D64" if refine else "D63")
            st.rerun()


def render_published_job(controller, jd_id):
    job = controller.job_service.get(jd_id)
    st.header(job["title"])
    st.subheader("Full job description")
    st.text(job["jd_text"])
    _ranking_controls(controller, job)
    st.caption("Candidate review and Ask Hiring Desk become available when this JD has a ready ranking.")
