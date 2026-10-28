"""Real Streamlit transport with offline evidence models and synthetic sources."""
from test_portal_v2_ui import workspace, app


def test_v3_comparison_then_why_keeps_table_and_sources(workspace, monkeypatch):
    from candidate_screener.app.conversation_v3 import ConversationV3Service
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v3")
    at = app()
    assert isinstance(workspace.qa_service, ConversationV3Service)
    at.chat_input[0].set_value("Compare candidates 1 and 2").run()
    assert not at.exception
    assert any({"Candidate 1", "Candidate 2"} <= set(item.value.columns) for item in at.dataframe)
    at.chat_input[0].set_value("Why is 1 better than 2?").run()
    assert not at.exception
    assert any(item.label == "View comparison table" for item in at.expander)
    assert not any("qa:" in item.label for item in at.expander)


def test_v3_jd_search_first_does_not_break_followup(workspace, monkeypatch):
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v3")
    at = app()
    at.chat_input[0].set_value("Which JD requires Python?").run()
    assert not at.exception
    at.chat_input[0].set_value("What degree does candidate 1 have?").run()
    assert not at.exception
    session = at.session_state["portal_session_id"]
    assert len(workspace.chat_history(session)) == 2
    assert workspace.chat_history(session)[-1]["answer"]["mode"] == "conversation_v3"


def test_owner_happy_and_negative_flows_are_available_in_v3_ui(workspace, monkeypatch):
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v3")
    at = app()
    questions = ["Summarise why candidate 1 is a good fit to this JD",
                 "Compare candidate 1 and 2",
                 "Why canddiate 1 is better than candidate 2?",
                 "Which JD has this specific skills -Flutter?",
                 "Any relevant job experiences for any of these candidates?",
                 "Give me a recipe of apple pie.",
                 "What is the nationality of all the top 10 candidates for this JD?"]
    for index, question in enumerate(questions):
        at.chat_input[0].set_value(question).run()
        assert not at.exception
        session = at.session_state["portal_session_id"]
        answer = workspace.chat_history(session)[-1]["answer"]
        assert answer["status"] != "clarification", (question, answer["text"])
        if index < 3 or index == 4:
            assert answer["citations"], (question, answer["text"])
        if index == 3:
            assert answer["mode"] == "jd_search"
        if index >= 5:
            assert not answer["citations"]
            assert answer["text"] == "I am not able to answer this question. It is out of my scope."


def test_v3_sources_are_only_rendered_after_user_requests_them(workspace, monkeypatch):
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v3")
    at = app()
    at.chat_input[0].set_value("Compare candidates 1 and 2").run()
    assert not at.exception
    assert not any("View supporting source:" in item.label for item in at.expander)
    show = next(item for item in at.checkbox if item.label == "Show supporting sources")
    show.check().run()
    assert not at.exception
    assert any("View supporting source:" in item.label for item in at.expander)


def test_v3_comparison_followup_renders_both_tables_without_opening_sources(workspace, monkeypatch):
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v3")
    at = app()
    at.chat_input[0].set_value("Compare candidates 1 and 2").run()
    assert not at.exception
    tables = lambda: [item.value for item in at.dataframe
                      if {"Candidate 1", "Candidate 2"} <= set(item.value.columns)]
    assert len(tables()) == 1
    assert not any("View supporting source:" in item.label for item in at.expander)
    assert all(not item.value for item in at.checkbox if item.label == "Show supporting sources")

    at.chat_input[0].set_value("What about their qualifications?").run()
    assert not at.exception
    history = workspace.chat_history(at.session_state["portal_session_id"])
    assert len(history) == 2
    followup = history[-1]["answer"]
    assert followup["context"]["intent"] == "comparison"
    assert followup["context"]["category"] == "DEGREE"
    assert len(tables()) == 2
    source_controls = [item for item in at.checkbox if item.label == "Show supporting sources"]
    assert len(source_controls) == 2
    assert all(not item.value for item in source_controls)
    assert not any("View supporting source:" in item.label for item in at.expander)


def test_v3_display_table_has_broad_categories_and_separate_jd_criteria(workspace, monkeypatch):
    monkeypatch.setenv("HR_PORTAL_CHAT_VERSION", "v3")
    at = app()
    at.chat_input[0].set_value("Compare candidates 1 and 2").run()
    assert not at.exception
    table = next(item.value for item in at.dataframe
                 if {"Candidate 1", "Candidate 2"} <= set(item.value.columns))
    assert len(table) >= 7
    assert set(table["Scope"]) >= {"Broad evidence", "JD criterion"}
    assert set(table.iloc[:6]["Category"]) == {
        "Knowledge", "Skill", "Title", "Degree", "Experience Years", "Certification"}
    assert any("in progress" in value for value in table["Candidate 2"])
