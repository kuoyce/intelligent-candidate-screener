"""Chat and dropdown HR review transport. Local tokens are not company SSO."""
from __future__ import annotations

import hashlib
import base64
from dataclasses import replace
import html
import json
import os
import re
from pathlib import Path
import secrets
import sqlite3
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from candidate_screener.annotation.redact import redact_text
from candidate_screener.config import PROCESSED
from .portal_auth import (DEFAULT_ACCOUNT_HOURS, DEFAULT_ACCOUNTS, MAX_TOKEN_CHARS, AccountsExist, TokenAuthenticator,
                          has_accounts, submit_first_account)
from .portal_controller import PortalController, StaleReview
from .portal_data import CATEGORIES, PortalDataset
from .portal_store import PortalStore, RevisionConflict, _opaque_identity
from .portal_presentation import STYLES, evidence_text, source_label, source_summary


DATA = PROCESSED / "phase2-rerank-output"


def _stamp(folder):
    paths = [folder / name for name in ("a2-cv-extraction.csv", "a2-jd-extraction.csv", "a2-x2-scores.csv")]
    paths.append(PROCESSED / "matching/a2-pairs-v2-539ddf32117c.parquet")
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else (str(p), None, None) for p in paths)


def _clear_review():
    for key in list(st.session_state):
        if key.startswith("portal_"):
            del st.session_state[key]


def _say(text):
    with st.chat_message("assistant"):
        st.write(redact_text(text))


def _artifact_download(label, content, filename):
    """Deliver authorized bytes in this session, without a shared media-file URL."""
    encoded = base64.b64encode(content.encode() if isinstance(content, str) else content).decode("ascii")
    st.markdown(f'<a download="{html.escape(filename, quote=True)}" '
                f'href="data:application/octet-stream;base64,{encoded}">{html.escape(label)}</a>', unsafe_allow_html=True)


def _optional_provider():
    """Only explicitly configured inference; absent runtime never blocks review."""
    if os.environ.get("SCREENER_QWEN_DISABLED", "false").lower() == "true":
        return None, "disabled"
    try:
        from candidate_screener.llm.qwen import QwenConfig, QwenProvider
        from candidate_screener.llm.qwen_assets import local_settings
        values = dict(os.environ)
        if not values.get("SCREENER_QWEN_BASE_URL") and not values.get("SCREENER_QWEN_MODEL"):
            try:
                values.update(local_settings())
            except (OSError, ValueError, KeyError):
                pass
        config = QwenConfig.from_mapping(values)
        if config is None:
            return None, "unconfigured"
        key = hashlib.sha256(repr(config).encode()).hexdigest()
        if st.session_state.get("portal_provider_key") != key:
            st.session_state.portal_provider = QwenProvider(config)
            st.session_state.portal_provider_key = key
        if config.tokenizer_path != "service" and not config.expected_sha256:
            # A local tokenizer cannot establish remote service availability.
            return st.session_state.portal_provider, "configured:" + key
        probe = st.session_state.get("portal_provider_probe", {})
        if probe.get("key") != key or time.monotonic() - probe.get("checked_at", 0) >= 10:
            try:
                # Verify metadata/tokenization with a short deadline, without
                # generating an answer or sending any applicant text.
                QwenProvider(replace(config, timeout_seconds=2)).count_tokens([
                    {"role": "user", "content": "Service availability check."}])
                available = True
            except (OSError, ValueError, RuntimeError):
                available = False
            probe = {"key": key, "checked_at": time.monotonic(), "available": available}
            st.session_state.portal_provider_probe = probe
        if not probe["available"]:
            return None, "unavailable"
        return st.session_state.portal_provider, key
    except (ImportError, ValueError, OSError, RuntimeError):
        return None, "unavailable"


@st.dialog("Read-only CV preview", width="large")
def _preview(controller, session_id, candidate_id):
    # Fetch/authorize again when the dialog runs, not only at dropdown selection.
    try:
        candidate = controller.preview(session_id, candidate_id)
    except (PermissionError, ValueError, OSError):
        st.error("Preview access is no longer available. Sign in or refresh the review.")
        return
    st.caption("Redacted text derivative. Original PDF and page layout are unavailable in this export.")
    if candidate["preview_status"] == "unavailable":
        st.warning("No authorized source preview is available.")
    else:
        st.text(candidate["preview_text"])


def _first_account(accounts):
    """Shown only while no account exists. Accepted risk (owner, plan 2026-10-27-docker S10):
    whoever opens the page first creates the first account; the Docker port is host-loopback
    only. If the port is ever exposed on a network, add a one-time setup code first.
    Further accounts use the portal_auth command; there is no administrator role."""
    st.subheader("Create the first account")
    st.info("Local reviewer accounts are not configured yet. Create the first account here; further "
            "accounts are created with the command below. This is local token authentication, not company SSO.")
    st.code("uv run python -m candidate_screener.app.portal_auth --create-local-user reviewer --department benchmark", language="bash")
    with st.form("local_first_account"):
        login_id = st.text_input("Login ID", max_chars=100)
        departments = st.text_input("Departments (comma-separated)", value="benchmark",
                                    help="The departments whose jobs this account may open, e.g. benchmark.")
        token = st.text_input("Access token", type="password", max_chars=MAX_TOKEN_CHARS,
                              help="24 to 512 characters. It is not displayed or stored in plain text.")
        confirmation = st.text_input("Confirm access token", type="password", max_chars=MAX_TOKEN_CHARS)
        days = st.number_input("Account expiry (days)", min_value=1, max_value=DEFAULT_ACCOUNT_HOURS // 24,
                               value=DEFAULT_ACCOUNT_HOURS // 24, step=1)
        submitted = st.form_submit_button("Create the first account")
    if submitted:
        try:
            submit_first_account(accounts, login_id, departments, token, confirmation, days)
        except AccountsExist:
            st.error("An account already exists. Reload the page and sign in.")
        except (ValueError, OSError) as error:
            st.error(str(error) if isinstance(error, ValueError) else "The account could not be saved.")
        else:
            st.session_state.portal_first_account_created = True
            st.rerun()


def _login():
    accounts = Path(os.environ.get("HR_PORTAL_ACCOUNTS", str(DEFAULT_ACCOUNTS)))
    authenticator = TokenAuthenticator(accounts)
    principal = st.session_state.get("portal_principal")
    version = st.session_state.get("portal_auth_version", "")
    if principal is not None:
        try:
            valid = authenticator.current(principal, version)
        except (OSError, ValueError, KeyError):
            valid = False
        if valid:
            return principal, lambda: authenticator.current(principal, version)
        _clear_review()
        st.warning("Your sign-in expired or access changed. Sign in again.")
    if not has_accounts(accounts):
        _first_account(accounts)
        return None, None
    st.subheader("Local reviewer sign-in")
    st.caption("This uses an explicitly provisioned local login ID and access token. Company SSO is not configured.")
    if st.session_state.pop("portal_first_account_created", None):
        st.success("The first account was created. Sign in with its login ID and access token.")
    with st.form("local_reviewer_sign_in"):
        login_id = st.text_input("Login ID", max_chars=100)
        token = st.text_input("Access token", type="password", max_chars=MAX_TOKEN_CHARS)
        submitted = st.form_submit_button("Sign in")
    if submitted:
        try:
            principal, version = authenticator.authenticate(login_id, token)
        except (PermissionError, OSError, ValueError, KeyError):
            st.error("Sign-in failed. Check your credentials or contact the administrator.")
        else:
            st.session_state.portal_principal = principal
            st.session_state.portal_auth_version = version
            st.rerun()
    return None, None


def _controller():
    principal, guard = _login()
    if principal is None:
        return None
    folder = Path(os.environ.get("HR_PORTAL_DATA", str(DATA)))
    stamp = _stamp(folder)
    if st.session_state.get("portal_data_stamp") != stamp:
        st.session_state.portal_dataset = PortalDataset.from_folder(folder)
        st.session_state.portal_data_stamp = stamp
    store = PortalStore(PROCESSED / "hr-portal/workspace.sqlite")
    return PortalController(st.session_state.portal_dataset, store, principal, auth_guard=guard,
                            freshness_guard=lambda: _stamp(folder) == stamp)


def _history_once(controller, session_id, action, answer, *, menu=None, extra=None):
    marker = (session_id, action)
    if st.session_state.get("portal_last_action") != marker:
        controller.remember(session_id, action, answer, menu=menu or action, extra=extra)
        st.session_state.portal_last_action = marker


def _chat_answer(answer, *, source_key=None):
    st.write(redact_text(answer["text"]))
    if answer.get("status") == "out_of_scope":
        return
    mode = answer.get("mode", "deterministic_evidence")
    if mode == "conversation_v3":
        generation = answer.get("generation_mode", "controlled")
        if generation == "qwen_prose":
            st.caption("Hiring Desk v3")
        elif generation in {"model_fallback", "qwen_fallback", "prose_fallback"}:
            st.caption("Hiring Desk v3")
        else:
            st.caption("Hiring Desk v3")
        with st.expander("Answer details", expanded=False):
            st.write({"mode": generation, "model": answer.get("model"),
                      "provenance": answer.get("generation_provenance", {})})
    elif mode == "jd_search":
        st.caption("Authorized job descriptions · source-based search")
    elif mode == "conversation_v2":
        generation = answer.get("generation_mode", "controlled")
        if generation == "qwen_answer_plan":
            st.caption("Hiring Desk v2 · Qwen selected validated comparison points · application-written factual clauses")
        elif generation in {"model_fallback", "qwen_fallback"}:
            st.caption("Hiring Desk v2 · Qwen unavailable or rejected · application comparison")
        else:
            st.caption("Hiring Desk v2 · evidence-grounded application conversation")
        if answer.get("comparison_scope") == "recognized_jd_criteria":
            st.caption("Comparing recognized JD criteria; this is not an exhaustive review of every requirement. Ask about a specific criterion to explore it.")
    elif answer.get("status") == "insufficient_evidence" and not answer.get("citations"):
        st.caption("No matching source evidence · no validated model answer")
    elif mode == "extractive_qa":
        st.caption("RoBERTa-SQuAD2 · extracted answer spans validated against source")
    elif mode == "entity_mentions":
        st.caption("GLiNER medium · source-linked entity mentions")
    elif mode == "comparison":
        st.caption("Extractive evidence · application-built comparison")
    elif mode == "source_only":
        st.caption("Source excerpts · no validated QA answer")
    elif mode == "llm_validated_excerpts":
        st.caption(redact_text(str(answer.get("model") or "Configured model")) + " · validated extractive answer")
    elif answer.get("status") == "model_fallback":
        st.caption("Configured model failed or returned invalid output · evidence-only response")
    elif mode == "deterministic_evidence":
        st.caption("Source excerpts · deterministic retrieval")
    else:
        st.caption("Review guidance · deterministic response")
    if (mode in {"conversation_v3", "jd_search"} and answer.get("citations")
            and answer.get("generation_mode") != "citation_reopen"):
        identity = source_key or hashlib.sha256(json.dumps(answer, sort_keys=True).encode()).hexdigest()
        if not st.checkbox("Show supporting sources", key="portal_sources_" + identity):
            return
    for citation in answer.get("citations", []):
        try:
            owner = answer.get("candidate_labels", {}).get(citation["owner"]) or _opaque_identity(citation["owner"])
        except ValueError:
            owner = redact_text(str(citation["owner"]))
        evidence_id = str(citation["evidence_id"])
        if not re.fullmatch(r"qa:[a-f0-9]{24}", evidence_id):
            evidence_id = redact_text(evidence_id)
        evidence_id = answer.get("citation_labels", {}).get(citation["evidence_id"], evidence_id)
        heading = (f"View supporting source: {owner}" if mode in {"conversation_v3", "jd_search"}
                   else f"Supporting source: {owner} · {evidence_id}")
        with st.expander(heading, expanded=False):
            st.text(redact_text(citation["quote"]))
            st.caption(f"Redacted text offsets {citation['start']}–{citation['end']}; original PDF pages unavailable.")


def _comparison_table(result, *, labels=None):
    rows = result.get("comparison_rows", result.get("comparison", []))
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    if not rows:
        return
    labels = labels or result.get("candidate_labels", {})
    def cell_text(cell):
        if not isinstance(cell, dict):
            return str(cell)
        answer = cell.get("answer", cell.get("text")) or "Not found in reviewed text"
        context = cell.get("context") or ""
        if context and context not in answer:
            # answer += "
            answer += ""
        return answer
    display = []
    for row in rows:
        output = {"Category": str(row.get("category", "Evidence")).replace("_", " ").title(),
                  "Requirement": row.get("requirement", row.get("label", row.get("category", "Evidence")))}
        if "criterion" in row:
            output["JD evidence"] = cell_text(row["criterion"])
        answers = row.get("answers")
        if isinstance(answers, dict):
            for identity, cell in answers.items():
                output[labels.get(identity, identity)] = cell_text(cell)
        else:
            for cell in row.get("candidates", []):
                identity = cell.get("owner", cell.get("candidate_id", "Applicant"))
                output[labels.get(identity, identity)] = cell_text(cell)
        display.append(output)
    st.dataframe(pd.DataFrame(display), hide_index=True, width="stretch", row_height=100)
    entity_status = result.get("entity_status")
    if isinstance(entity_status, str) and entity_status not in {"complete", "completed_zero"}:
        st.caption("Entity extraction: " + entity_status.replace("_", " ") + ". Available source evidence is shown while processing completes.")


def _chat(controller, session_id, snapshot):
    # Inline, independently scrollable chat keeps the JD visible on initial load.
    with st.container(height=500, key="portal_review_chat_" + session_id):
        _chat_content(controller, session_id, snapshot)


def _chat_content(controller, session_id, snapshot):
    st.subheader("Ask Hiring Desk")
    version = os.environ.get("HR_PORTAL_CHAT_VERSION", "v1").lower()
    v2 = version in {"v2", "v3"}
    st.caption("Hiring Desk v3 · evidence-backed conversation · experimental" if version == "v3" else
               "Hiring Desk v2 · conversational comparisons with source evidence" if v2 else
               "Extractive QA: RoBERTa-SQuAD2 · entity evidence: GLiNER medium · models load locally on demand")
    history = controller.chat_history(session_id)
    if not history:
        with st.chat_message("assistant"):
            st.write(f"You're reviewing {redact_text(snapshot.jd_title)}. I can explain the requirements or compare the selected candidates. Try ‘Compare candidates 1 and 2’, then ‘What about their qualifications?’" if v2 else
                     "Ask me about this role and its ranked applicants. For example: ‘What does this role require?’, ‘Compare candidates 1 and 2’, or ‘What degree does candidate 1 have?’")
    for turn in history:
        with st.chat_message("user"):
            st.write(redact_text(turn["question"]))
        with st.chat_message("assistant"):
            _chat_answer(turn["answer"], source_key=turn.get("turn_id"))
            if turn["answer"].get("mode") == "comparison" or turn["answer"].get("show_comparison_table"):
                if turn["answer"].get("comparison_table_expandable"):
                    with st.expander("View comparison table", expanded=False):
                        _comparison_table(turn["answer"])
                else:
                    _comparison_table(turn["answer"])
    pending_key = "portal_pending_chat_" + session_id
    pending = st.session_state.get(pending_key)
    if pending:
        identity = hashlib.sha256(pending["turn_id"].encode()).hexdigest().translate(
            str.maketrans("0123456789abcdef", "abcdefghijklmnop"))
        if any(turn.get("turn_id") == identity and turn.get("question") == pending["question"] for turn in history):
            st.session_state.pop(pending_key, None)
            pending = None
    state = controller.load_session(session_id)["state"]
    context_marker = [state.get("qa_context"), state.get("qa_dropdown_context"), state.get("qa_dropdown_revision")]
    if pending and (pending["snapshot_digest"] != snapshot.digest or pending["context_marker"] != context_marker):
        st.session_state.pop(pending_key, None)
        pending = None
        st.warning("The review or candidate context changed. The pending question was discarded; ask again for the current selection.")
    retry = False
    if pending:
        st.warning("Your question could not be confirmed as saved. Retry it or discard it. Review dropdowns remain available.")
        st.write(redact_text(pending["question"]))
        retry = st.button("Retry question", key="portal_retry_chat_" + session_id)
        if st.button("Discard question", key="portal_discard_chat_" + session_id):
            st.session_state.pop(pending_key, None)
            st.rerun()
    prompt = st.chat_input("Ask about the selected JD, applicants or supporting evidence", max_chars=2000,
                           disabled=bool(pending),
                           key="portal_chat_input_" + session_id)
    if prompt:
        pending = {"question": redact_text(prompt.strip()), "turn_id": secrets.token_hex(16),
                   "snapshot_digest": snapshot.digest, "context_marker": context_marker}
        st.session_state[pending_key] = pending
    if prompt or retry:
        try:
            with st.spinner("Reviewing the available evidence…"):
                controller.chat(session_id, pending["question"], pending["turn_id"])
        except (PermissionError, StaleReview):
            st.session_state.pop(pending_key, None)
            raise
        except (OSError, sqlite3.Error, RevisionConflict):
            # A rerun renders recovery controls without automatically resubmitting.
            st.rerun()
        except ValueError:
            st.session_state.pop(pending_key, None)
            st.warning("This question could not be accepted. Rephrase it and submit again.")
        else:
            st.session_state.pop(pending_key, None)
            st.rerun()


def _review(controller, session_id):
    try:
        snapshot = controller.snapshot(session_id)
    except StaleReview:
        _say("This saved review is historical. What would you like to do?")
        old = controller.load_session(session_id)
        try:
            controller.dataset.snapshot(old["jd_id"])
            can_refresh = True
        except ValueError:
            can_refresh = False
        _capture_history(controller, session_id)
        choices = [None] + (["Refresh rankings"] if can_refresh else []) + ["View historical report", "Return home"]
        choice = st.selectbox("Recover review", choices)
        if choice == "Refresh rankings":
            old = controller.load_session(session_id)
            refreshed = controller.open_job(old["jd_id"])["session_id"]
            _switch_job(controller, old["jd_id"], session_id=refreshed)
            st.session_state.portal_next_navigation = "Review jobs"
            st.rerun()
        elif choice == "View historical report":
            report = controller.historical_report(session_id)
            st.text(report["text"] if report else "No saved report is available.")
        elif choice == "Return home":
            st.session_state.pop("portal_session_id", None)
            st.session_state.portal_next_navigation = "Review jobs"
            st.session_state.portal_browse_only = True
            st.rerun()
        return
    _job_description(snapshot)
    if hasattr(controller.dataset, "exports") and snapshot.jd_id not in {job["jd_id"] for job in controller.dataset.exports.jobs()}:
        from .portal_authoring_ui import _ranking_controls
        _ranking_controls(controller, controller.job_service.get(snapshot.jd_id))
    st.subheader("Candidate review")
    st.caption(f"{len(snapshot.candidates)} applicants · {source_label(snapshot)}")
    with st.expander("Ranking details"):
        st.caption(f"Snapshot {snapshot.snapshot_id} · {snapshot.model_id}")
        st.write(snapshot.candidates[0].get("score_semantics", "Backend ordering score") if snapshot.candidates else "No candidates returned by the ranking service.")
    actions = ["Top10 ranked CVs", "Open CV",
               "Generate session report", "Send to hiring department", "Review another job"]
    # not required these actions -  "Requirement evidence", "Compare candidates", "Prepare top10 package", "Save session"
    if len(snapshot.candidates) < 2 and "Compare candidates" in actions:
        actions.remove("Compare candidates")
    if not snapshot.candidates:
        actions = ["Generate session report", "Save session", "Review another job"]
    saved_menu = controller.load_session(session_id)["state"].get("menu")
    action = st.selectbox("Review action", actions, index=actions.index(saved_menu) if saved_menu in actions else 0,
                         key="portal_review_action_" + session_id)
    if action not in {"Save session", "Review another job", "Send to hiring department", "Requirement evidence", "Open CV", "Compare candidates"}:
        _history_once(controller, session_id, action, "Opened the selected review action for the current ranking snapshot.")
    candidates = snapshot.candidates
    by_id = {c["cv_id"]: c for c in candidates}
    ids = list(by_id)
    label = lambda identity: f"#{by_id[identity]['rank']} · {identity}"
    if action == "Top10 ranked CVs":
        st.dataframe(pd.DataFrame([{"Rank": c["rank"], "Applicant": c["cv_id"], "Ranking score": c["score"],
                                   "CV preview": "Available" if c["preview_status"] != "unavailable" else "Unavailable"} for c in candidates]), hide_index=True)
        st.caption("See Ranking details for their meaning; inspect source evidence for context.")
    elif action == "Requirement evidence":
        category = st.selectbox("Which requirement category?", CATEGORIES, key="portal_category_" + session_id)
        st.write("**JD mentions:** " + evidence_text(snapshot.jd_entities[category]))
        selected = st.selectbox("Which applicant?", ids, format_func=label, key="portal_candidate_" + session_id)
        _history_once(controller, session_id, f"Requirement evidence: {category} / {selected}",
                      "Displayed unverified table mentions; source preview is available for manual context review.",
                      menu=action, extra={"selected_category": category, "selected_candidates": [selected]})
        st.write("**Applicant mentions:** " + evidence_text(by_id[selected]["entities"][category]))
        _say("These extraction-table mentions have no verified source spans. Use the CV preview to check the context; missing evidence is not a confirmed gap.")
        if controller.qa_service is not None:
            query = {"KNOWLEDGE": "What technologies are mentioned", "SKILL": "What professional skills are mentioned",
                     "TITLE": "What job titles are mentioned", "DEGREE": "What degree is mentioned",
                     "EXPERIENCE_YEARS": "How many years of experience are stated"}[category]
            evidence = controller.extractive_evidence(session_id, f"{query} for candidate {by_id[selected]['rank']}?")
            _chat_answer(evidence)
    elif action == "Open CV":
        selected = st.selectbox("Which CV would you like to open?", [None] + ids,
                                format_func=lambda value: "Select an applicant" if value is None else label(value), key="portal_preview_" + session_id)
        if selected is not None:
            _history_once(controller, session_id, "Open CV: " + selected, "Opened the authorized redacted text preview.",
                          menu=action, extra={"selected_candidates": [selected]})
            _preview(controller, session_id, selected)
    elif action == "Compare candidates":
        first = st.selectbox("First applicant", ids, format_func=label, key="portal_first_" + session_id)
        second = st.selectbox("Second applicant", [identity for identity in ids if identity != first], format_func=label, key="portal_second_" + session_id)
        _history_once(controller, session_id, f"Compare: {first} / {second}", "Requested a source-linked candidate comparison.",
                      menu=action, extra={"selected_candidates": [first, second]})
        comparison = controller.compare(session_id, [first, second])
        _comparison_table(comparison, labels={identity: label(identity) for identity in (first, second)})
        st.caption("GLiNER medium extracts source mentions once per document. Missing mentions do not prove absence; source context remains available.")
        if st.button("Refresh extraction", key="portal_refresh_entities_" + session_id):
            st.rerun()
        with st.expander("Comparison source evidence"):
            _chat_answer(comparison)
    elif action == "Generate session report":
        report = controller.report(session_id)
        st.text(report["text"])
        _artifact_download("Download redacted comparison report", report["text"], "comparison-report.txt")
        st.success("The report is saved for this session.")
    elif action == "Prepare top10 package":
        try:
            report, package = controller.package(session_id)
        except ValueError:
            st.warning("The complete package requires ten ranked applicants with all ten source CVs available.")
        else:
            st.write(f"{len(report['candidate_ids'])} ranked applicants in the comparison.")
            st.caption("Ten redacted CV text files, ten profiles and the comparison report. Original PDFs are unavailable.")
            _artifact_download("Download reviewed package", package, "shortlist-package.zip")
    elif action == "Send to hiring department":
        _handoff(controller, session_id, snapshot)
    elif action == "Save session":
        _history_once(controller, session_id, action, "Review saved; rankings remain bound to this snapshot.")
        st.success("Session saved. You can resume it from Saved sessions.")
    elif action == "Review another job":
        st.session_state.portal_clear_search = True
        st.session_state.portal_browse_only = True
        st.session_state.portal_next_navigation = "Review jobs"
        st.rerun()
    _chat(controller, session_id, snapshot)


def _job_description(snapshot):
    st.header(snapshot.jd_title)
    with st.container(border=True):
        st.subheader("JD summary")
        st.write(source_summary(snapshot.jd_text))
        left, right = st.columns(2)
        left.markdown("**Skills mentioned**")
        left.write(evidence_text(snapshot.jd_entities.get("SKILL")))
        right.markdown("**Experience mentioned**")
        right.write(evidence_text(snapshot.jd_entities.get("EXPERIENCE_YEARS")))
        st.caption("Source excerpt and available mentions; unspecified requirements have not been inferred.")
    with st.container(key="portal_jd_source", height=360):
        st.subheader("Full job description")
        if snapshot.jd_text:
            st.text(snapshot.jd_text)
        else:
            st.info("Full JD source is unavailable. The export mentions above do not replace the original description.")


def _switch_job(controller, jd_id, *, session_id=None):
    """Resume only this owner's compatible snapshot; discard pending cross-job work."""
    controller._guard(fresh=True)
    st.session_state.portal_selected_job = jd_id
    st.session_state.portal_browse_only = False
    for key in list(st.session_state):
        if key.startswith(("portal_pending_chat_", "portal_explanation")) or key in {"portal_last_action", "portal_mail_preview"}:
            del st.session_state[key]
    st.session_state.pop("portal_session_id", None)
    try:
        snapshot = controller.dataset.snapshot(jd_id)
    except ValueError:
        return
    if session_id is None:
        compatible = [r for r in controller.sessions() if r["jd_id"] == jd_id and r["source_fingerprint"] == snapshot.digest]
        session_id = compatible[0]["session_id"] if compatible else controller.open_job(jd_id)["session_id"]
    else:
        record = controller.load_session(session_id)
        if record["jd_id"] != jd_id:
            raise PermissionError("Saved review belongs to another job")
    st.session_state.portal_session_id = session_id


def _job_browser(controller):
    st.subheader("Find your next review")
    if os.environ.get("HR_PORTAL_CHAT_VERSION", "v1").lower() == "v3":
        with st.expander("Find jobs by requirement", expanded=False):
            with st.form("portal_jd_requirement_search"):
                requirement_question = st.text_input("Ask about job requirements", max_chars=2000,
                                                    placeholder="Which JD requires Flutter?")
                search_requirements = st.form_submit_button("Search requirements")
            if search_requirements:
                try:
                    _chat_answer(controller.search_job_requirements(requirement_question))
                except (ValueError, PermissionError) as exc:
                    st.warning(str(exc))
    if st.session_state.pop("portal_clear_search", False):
        st.session_state.portal_job_search = ""
    search = st.text_input("Search job titles", key="portal_job_search", placeholder="Search by role or job title")
    all_jobs = controller.jobs()
    jobs = [job for job in all_jobs if search.casefold() in job["jd_title"].casefold()]
    if not jobs:
        st.info("No jobs match this search. Clear the search or create a new JD.")
        return
    selected = st.session_state.get("portal_selected_job")
    if selected not in {job["jd_id"] for job in jobs}:
        _switch_job(controller, jobs[0]["jd_id"])
        selected = jobs[0]["jd_id"]
    index = next(i for i, job in enumerate(jobs) if job["jd_id"] == selected)
    previous, position, following = st.columns([1, 2, 1])
    if previous.button("Previous JD", disabled=index == 0, use_container_width=True):
        _switch_job(controller, jobs[index-1]["jd_id"])
        st.rerun()
    position.caption(f"Job {index+1} of {len(jobs)} · select a card to switch reviews")
    if following.button("Next JD", disabled=index == len(jobs)-1, use_container_width=True):
        _switch_job(controller, jobs[index+1]["jd_id"])
        st.rerun()
    with st.container(key="portal_job_cards"):
        for column, job in zip(st.columns(min(3, len(jobs))), jobs[index//3*3:index//3*3+3]):
            with column, st.container(border=True):
                st.caption("SELECTED" if job["jd_id"] == selected else "JOB DESCRIPTION")
                st.markdown("**" + job["jd_title"] + "**")
                try:
                    snapshot = controller.dataset.snapshot(job["jd_id"])
                    st.caption(source_label(snapshot) + f" · {len(snapshot.candidates)} applicants")
                    st.write(source_summary(snapshot.jd_text, limit=125))
                except ValueError:
                    st.caption("New JD · " + job.get("ranking_status", "Awaiting screening").replace("_", " "))
                    st.write(source_summary(job.get("jd_text"), limit=125))
                if st.button("Review this JD", key="portal_card_" + job["jd_id"], type="primary" if job["jd_id"] == selected else "secondary", use_container_width=True):
                    _switch_job(controller, job["jd_id"])
                    st.rerun()
    if st.session_state.get("portal_browse_only"):
        st.info("Choose a JD card above to open its review. Your previous review is saved.")
        return
    session_id = st.session_state.get("portal_session_id")
    if not session_id:
        _switch_job(controller, selected)
        session_id = st.session_state.get("portal_session_id")
    if session_id and hasattr(controller.dataset, "exports") and selected not in {job["jd_id"] for job in controller.dataset.exports.jobs()}:
        try:
            controller.dataset.snapshot(selected)
        except ValueError:
            # A re-ranking request supersedes the active view, not its saved history.
            _operational_review(controller, selected)
            return
    if session_id:
        _review(controller, session_id)
    else:
        _operational_review(controller, selected)


def _events(controller):
    events = [{**item, "event_source": "review"} for item in controller.events()]
    service = getattr(controller, "job_service", None)
    if service is not None:
        events += [{**item, "event_source": "jobs"} for item in service.events()]
    return sorted(events, key=lambda item: item.get("created_at", 0), reverse=True)


def _alerts(controller, events):
    st.header("Alerts")
    st.caption("Updates from published jobs, live ranking and your email outbox.")
    unread_only = st.checkbox("Unread only")
    if st.button("Refresh alerts"):
        st.rerun()
    visible = [event for event in events if not unread_only or not event.get("read_at")]
    if not visible:
        st.info("You're all caught up. New job and handoff updates will appear here.")
    for event in visible:
        event_id = event["event_id"]
        key = event["event_source"] + event_id
        with st.container(border=True):
            st.markdown("**" + event["kind"].replace("_", " ").title() + (" · Unread**" if not event.get("read_at") else "**"))
            body = event.get("body", {})
            st.caption(datetime.fromtimestamp(event["created_at"]).strftime("%d %b %Y, %H:%M"))
            st.write(redact_text(str(body.get("message", body.get("title", event.get("jd_id", "Job update"))))))
            col1, col2 = st.columns(2)
            if col1.button("Mark unread" if event.get("read_at") else "Mark read", key="portal_event_read_" + key):
                if event["event_source"] == "jobs":
                    controller.job_service.mark_read(event_id, read=not bool(event.get("read_at")))
                elif event.get("read_at"):
                    controller.store.mark_event_unread(controller.principal, event_id)
                else:
                    controller.read_event(event_id)
                st.rerun()
            if col2.button("Open job", key="portal_event_open_" + key):
                if event["jd_id"] not in {job["jd_id"] for job in controller.jobs()}:
                    st.warning("This job is no longer available in the current sources.")
                else:
                    _switch_job(controller, event["jd_id"])
                    st.session_state.portal_next_navigation = "Review jobs"
                    st.rerun()


def _handoff(controller, session_id, snapshot):
    from .portal_mail import DemoMailer, department_directory
    st.subheader("Send to hiring department")
    st.info("Email demonstration — captured locally. This flow does not deliver to a real mailbox.")
    _capture_history(controller, session_id)
    if len(snapshot.candidates) != 10:
        st.warning("A complete ten-applicant ranking is required before preparing this handoff.")
        return
    recipient = st.selectbox("Hiring department", ["tech", "business"], format_func=lambda value: {"tech": "Tech Department", "business": "Business Department"}[value], key="portal_mail_recipient_" + session_id)
    directory = department_directory(snapshot.jd_id, authorization_department=controller.department)
    mailer = DemoMailer(controller, directory=directory)
    identity = (session_id, snapshot.digest, recipient, controller.load_session(session_id)["revision"])
    saved = st.session_state.get("portal_mail_preview")
    if saved and saved["identity"] != identity:
        st.session_state.pop("portal_mail_preview", None)
        saved = None
        st.info("The review changed. Preview the current email again before sending.")
    if st.button("Preview email", key="portal_prepare_mail_" + session_id):
        try:
            package = controller.prepare_handoff(session_id, recipient, directory)
            preview = mailer.preview(session_id, package["package_id"])
        except (ValueError, OSError) as exc:
            st.warning("The complete package is unavailable. Check that all ten source CVs are present and refresh the review.")
            return
        saved = {"identity": identity, "package_id": package["package_id"], "preview": preview,
                 "request_id": secrets.token_hex(16)}
        st.session_state.portal_mail_preview = saved
    if not saved:
        return
    preview = saved["preview"]
    with st.container(border=True):
        st.write("**To:** " + str(preview["recipient"]))
        st.write("**Subject:** " + preview["subject"])
        st.text(preview["body"])
        st.caption("Attachment: ten redacted CVs/profiles and the comparison report as a ZIP package.")
    if st.button("Send to demo outbox", key="portal_send_mail_" + session_id, type="primary"):
        try:
            result = mailer.capture(session_id, saved["package_id"], saved["request_id"], preview_id=preview["preview_id"])
            if result["status"] == "captured":
                st.success("Email captured in demo outbox.")
                _artifact_download("Download captured email", mailer.download(saved["package_id"]), "department-shortlist.eml")
            else:
                st.error("Email capture failed. Review the package and retry.")
        except (ValueError, OSError):
            st.warning("Email preview is no longer current or capture failed. Preview again before retrying.")


def _capture_history(controller, session_id):
    from .portal_mail import DemoMailer
    captures = controller.store.list_captures(controller.principal, session_id=session_id)
    if captures:
        with st.expander("Captured emails"):
            for package in captures:
                st.caption("Local capture · " + package["recipient_id"].title() + " Department")
                _artifact_download("Download captured email", DemoMailer(controller).download(package["package_id"]), "department-shortlist.eml")


def _operational_review(controller, jd_id):
    from .portal_authoring_ui import render_published_job
    render_published_job(controller, jd_id)
    if os.environ.get("HR_PORTAL_CHAT_VERSION", "v1").lower() == "v3":
        with st.expander("Ask about this job description", expanded=False):
            st.caption("Ask a standalone question about the JD. Candidate discussion becomes available after ranking.")
            with st.form("portal_jd_only_" + jd_id):
                question = st.text_input("Question about this role", max_chars=2000)
                submitted = st.form_submit_button("Ask about the JD")
            if submitted:
                try:
                    _chat_answer(controller.ask_job(jd_id, question))
                except (ValueError, PermissionError) as exc:
                    st.warning(str(exc))


def main():
    st.set_page_config(page_title="Hiring Desk", layout="wide")
    st.markdown(STYLES, unsafe_allow_html=True)
    st.title("Hiring Desk")
    st.caption("Understand the role. Review the evidence. Move the right applicants forward.")
    try:
        controller = _controller()
        if controller is None:
            return
        from .portal_workspace import attach_services
        attach_services(controller)
        events = _events(controller)
        unread = sum(not item.get("read_at") for item in events)
        if st.session_state.get("portal_next_navigation"):
            st.session_state.portal_navigation = st.session_state.pop("portal_next_navigation")
        with st.sidebar.container(key="portal_sidebar_layout"):
            st.title("Hiring Desk")
            st.caption("Signed in: " + redact_text(controller.principal.subject))
            navigation = st.radio("Navigation", ["Review jobs", "Create new JD", "Alerts", "Saved reviews"], key="portal_navigation")
            st.caption(f"{unread} unread alert{'s' if unread != 1 else ''}")
            with st.expander("Service details"):
                st.caption("Local authentication · CSV review works independently of live modeling.")
                st.caption("Email demonstration captures messages locally. Original PDFs are unavailable.")
            with st.container(key="portal_logout_footer"):
                logout = st.button("Log out", key="portal_logout", use_container_width=True)
        if logout:
            _clear_review()
            st.rerun()
        if navigation == "Create new JD":
            from .portal_authoring_ui import render_authoring
            render_authoring(controller)
            return
        if navigation == "Alerts":
            _alerts(controller, events)
            return
        if navigation == "Saved reviews":
            st.subheader("Saved reviews")
            # Stable ordering/labels prevent a saved turn from resetting selection
            # when updated_at changes during review activity.
            sessions = sorted(controller.sessions(), key=lambda record: record["session_id"])
            if not sessions:
                st.info("Your saved reviews will appear here after you open a job.")
                return
            titles = {job["jd_id"]: job["jd_title"] for job in controller.jobs()}
            labels = {record["session_id"]: titles.get(record["jd_id"], "Historical job") + " · Review " + record["session_id"][:6] for record in sessions}
            selected = st.selectbox("Which saved review?", [None] + list(labels), format_func=lambda value: "Choose a saved review" if value is None else labels[value], key="portal_saved_selection")
            if selected:
                record = controller.load_session(selected)
                if st.session_state.get("portal_saved_active") != selected:
                    _switch_job(controller, record["jd_id"], session_id=selected)
                    st.session_state.portal_saved_active = selected
                if st.button("Continue in job browser"):
                    _switch_job(controller, record["jd_id"], session_id=selected)
                    st.session_state.portal_next_navigation = "Review jobs"
                    st.rerun()
                _review(controller, selected)
            return
        _job_browser(controller)
    except PermissionError:
        _clear_review()
        st.error("Access expired or was revoked. Sign in again.")
    except (OSError, ValueError, KeyError) as exc:
        st.error("The requested review is unavailable. No ranking, saved state or delivery success has been fabricated.")
        st.caption("Service status: " + type(exc).__name__)


if __name__ == "__main__":
    main()
